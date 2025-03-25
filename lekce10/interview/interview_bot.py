from typing import TypedDict, List, Dict, Optional, Union, Annotated
from langgraph.graph import MessageGraph
from langgraph.graph import StateGraph
START = "__start__"
END = "__end__"
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
import os
import csv
from dotenv import load_dotenv
import time
from random import uniform, random, choice
from dataclasses import dataclass
from uuid import uuid4
from pydantic import BaseModel, Field
import json
import logging
from langsmith import Client
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from humanity.humanity_manager import HumanityManager
from state import State, HumanityState, Topic, FactorInsight


# Add these Pydantic models at the top of the file
class Finding(BaseModel):
    detail: str = Field(..., description="Specific information found in the response")
    quote: str = Field(..., description="Exact quote from the text supporting this finding")
    relevance: float = Field(..., ge=0.0, le=1.0, description="How relevant this finding is")

class FactorAnalysis(BaseModel):
    factor: str = Field(..., description="Name of the factor being analyzed")
    score: float = Field(..., ge=0.0, le=1.0, description="Overall coverage score for this factor")
    found_info: List[Finding] = Field(..., description="List of specific findings for this factor")
    summary: str = Field(..., description="Overall summary of what was found")
    missing: str = Field(..., description="What information is still needed")

class AnalysisResponse(BaseModel):
    analysis: List[FactorAnalysis] = Field(..., description="Analysis for each factor")

class ChatState(TypedDict):
    messages: List[HumanMessage | AIMessage]

class GeneratedResponse(BaseModel):
    acknowledgment: str = Field(..., description="Genuine, empathetic acknowledgment in Czech")
    response: str = Field(..., description="Very gentle follow-up or supportive statement in Czech")
    approach: str = Field(..., pattern="^(careful|silent|supportive)$", description="Approach to take with the response")

class ResearchQuestion(BaseModel):
    """Research question generation model"""
    potvrzeni: str = Field(..., description="Krátké potvrzení předchozí odpovědi")
    otazka: str = Field(..., description="Výzkumná otázka zaměřená na konkrétní faktor")
    cilovy_faktor: str = Field(..., description="Faktor, na který se otázka zaměřuje")
    pristup: str = Field(..., pattern="^(opatrny|primy|podporujici)$", description="Způsob položení otázky")

class FactorEvidence(BaseModel):
    quote: str = Field(..., description="Přímá citace z odpovědi učitele")
    context: str = Field(..., description="Kontext citace")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance k faktoru")
    interpretation: str = Field(..., description="Interpretace vzhledem k faktoru")

class DetailedFactorAnalysis(BaseModel):
    factor_name: str = Field(..., description="Název zkoumaného faktoru")
    coverage_score: float = Field(..., ge=0.0, le=1.0, description="Míra pokrytí faktoru")
    questions_asked: int = Field(..., ge=0, description="Počet položených otázek k faktoru")
    evidence: List[FactorEvidence] = Field(..., description="Důkazy z odpovědi")
    summary: str = Field(..., description="Shrnutí pokrytí faktoru")
    missing_aspects: str = Field(..., description="Chybějící aspekty faktoru")

class InterviewBot:
    def __init__(self):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('interview_debug.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('InterviewBot')
        
        # Load environment variables
        load_dotenv()
        self.logger.info("Environment variables loaded")
        
        # Initialize state
        try:
            self.state = self.get_default_state()
            self.is_initialized = False
            self.logger.info("State initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize state: {str(e)}")
            raise

        # Define system prompt
        self.system_prompt = """You are having a natural conversation with a teacher in Czech. 

Key rules:
- Ask exactly ONE question at a time
- Be direct but warm
- Never evaluate or judge their responses
- Don't add disclaimers like "if you don't want to talk about it..."
- Keep responses short and natural
- Briefly acknowledge what they said, then ask your next question
- Use conversational Czech

Good examples:
"Jak jste na to reagoval?"
"Co se stalo potom?"
"Jak to vnímali ostatní studenti?"

Bad examples:
❌ "To muselo být těžké. Pokud o tom nechcete mluvit, nemusíte, ale zajímalo by mě..."
❌ "Chápu, že to není jednoduché téma. Můžete mi říct..."
❌ "To je velmi zajímavé. A když se zamyslíte nad tím, co se stalo..."
"""

        # Initialize intents BEFORE creating graph
        self.intents = {
            "asking_purpose": self.handle_purpose_question,
            "seeking_clarification": self.handle_clarification,
            "expressing_concern": self.handle_concern,
            "meta_discussion": self.handle_meta_discussion,
            "challenging_interview": self.handle_challenge,
            "off_topic": self.handle_off_topic
        }
        
        # Build graph
        try:
            self.graph = self.create_graph()
            self.logger.info("Graph created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create graph: {str(e)}")
            raise

        # Initialize LangSmith tracing
        try:
            self.model = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=0.7,
                streaming=False
            )
            self.logger.info("LangSmith tracing initialized")
            
            # Add humanity manager with model
            self.humanity = HumanityManager(self.model)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangSmith: {str(e)}")
            raise

    def load_topics(self, file_path: str = "topics.csv") -> Dict[str, Topic]:
        """Load interview topics and factors from CSV."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, file_path)
        
        topics = {}
        current_topic = None
        
        with open(full_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['type'] == 'topic':
                    # Create Topic as a dictionary since it's now a TypedDict
                    current_topic = {
                        "id": row['id'],
                        "question": row['content'],
                        "factors": {},
                        "covered_factors": {},
                        "question_attempts": {},
                        "factor_insights": {}
                    }
                    topics[row['id']] = current_topic
                elif row['type'] == 'factor' and current_topic and row['id'] == current_topic["id"]:
                    factor_name = row['content']
                    # Initialize all tracking for this factor
                    current_topic["factors"][factor_name] = row['factor']
                    current_topic["covered_factors"][factor_name] = 0.0
                    current_topic["question_attempts"][factor_name] = 0
                
        return topics

    def get_default_state(self) -> State:
        """Initialize default state with all required keys."""
        topics = self.load_topics()
        humanity = {
            "emotional": {
                "emotional_weight": 0.0,
                "trauma_indicators": False,
                "emotional_cues": [],
                "key_emotions": [],
                "requires_support": False,
                "last_emotional_acknowledgment": None,
                "emotional_complexity": 0.0,
                "vulnerability_level": 0.0
            },
            "cognitive": {
                "response_times": [],
                "complexity_scores": [],
                "current_load": 0.0,
                "response_lengths": [],
                "fatigue_indicators": 0,
                "mental_effort_level": 0.0,
                "complexity_indicators": [],
                "processing_patterns": {
                    "hesitations": 0,
                    "self_corrections": 0,
                    "memory_access_issues": False
                }
            },
            "engagement": {
                "engagement_level": 0.0,
                "short_answers_count": 0,
                "repetitive_phrases": {},
                "dismissive_responses": 0,
                "last_engagement_check": None
            },
            "self_disclosure": {
                "disclosure_level": 0.0,
                "personal_shares": [],
                "last_acknowledgment": None,
                "sensitive_topics": set()
            },
            "formality": {
                "current_level": "semi_formal",
                "formality_scores": {},
                "ty_vy_ratio": 0.0,
                "slang_count": 0,
                "style_consistency": 0.5,
                "formality_markers": []
            },
            "persona": {
                "current_persona": "researcher",
                "persona_history": [],
                "sentiment_trend": 0.0,
                "self_disclosure_level": 0.0,
                "interaction_style": {
                    "formality_preference": 0.5,
                    "support_needed": 0.5,
                    "expertise_level": 0.5
                },
                "rapport_indicators": [],
                "adaptation_needed": False
            }
        }
        
        return {
            "current_question": None,
            "user_message": None,
            "conversation_history": [],
            "topics": topics,
            "current_topic_id": "T1",
            "questions_in_topic": 0,
            "topic_exhausted": False,
            "introduction_done": False,
            "interview_complete": False,
            "is_complete": False,
            "detected_intent": None,
            "humanity": humanity,
            "messages": [],
            "interview_metadata": {}
        }

    def process_message(self, state: Dict) -> Dict:
        """Process the incoming message and generate a response."""
        # Only process messages after the first response
        user_message = state["messages"][-1].content
        print("\n=== PROCESSING NEW MESSAGE ===")
        print(f"User message: {user_message}")
        
        # Set up for graph processing
        state["user_message"] = user_message
        
        print("\n=== STARTING GRAPH EXECUTION ===")
        try:
            # Run through the graph
            processed_state = self.graph.invoke(state)
            print("\n=== GRAPH EXECUTION COMPLETED ===")
            
            if not processed_state.get("current_question"):
                print("ERROR: No question generated!")
            else:
                print(f"Generated question: {processed_state['current_question']}")
            
            return processed_state
            
        except Exception as e:
            print(f"\nERROR IN GRAPH EXECUTION: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return state

    def create_graph(self) -> StateGraph:
        """Create conversation flow with proper intent handling."""
        builder = StateGraph(State)

        def log_node(name: str, func):
            def wrapped(state: Dict) -> Dict:
                print(f"\n=== EXECUTING NODE: {name} ===")
                print(f"Input message: {state.get('user_message', 'None')}")
                result = func(state)
                print(f"Node {name} completed")
                return result
            return wrapped

        # Core flow nodes
        builder.add_node("process_response", log_node("process_response", self.process_response))
        builder.add_node("analyze_intent", log_node("analyze_intent", self.analyze_intent))
        builder.add_node("generate_response", log_node("generate_response", self._generate_base_question))

        # Force emotional analysis first, then intent analysis
        print("\n=== CREATING GRAPH FLOW ===")
        builder.add_edge(START, "process_response")
        builder.add_edge("process_response", "analyze_intent")
        builder.add_edge("analyze_intent", "generate_response")
        builder.add_edge("generate_response", END)

        return builder.compile()

    def chat(self, message: str = None) -> Dict:
        """Process chat message and return response."""

        
        try:
            if not self.is_initialized:
                print("\n=== INTRODUCTION ===")
                self.state = self.introduce_interview(self.get_default_state())
                self.is_initialized = True
                return {"response": self.state["current_question"]}

            if not self.state.get("messages"):
                print("=== INITIALIZING MESSAGES LIST ===")
                self.state["messages"] = []
            self.state["messages"].append(HumanMessage(content=message))

            user_message_count = sum(1 for msg in self.state.get("messages", []) if isinstance(msg, HumanMessage))
            print(f"User message count: {user_message_count}")
        
            # CRITICAL FIX: Don't try to append None message
            if not message:
                return {"response": self.state.get("current_question", "")}

            if user_message_count <= 1:  
                print("=== NO RESPONSE ===") 
                return {"response": ""}

            # CRITICAL FIX: Count user messages before processing
            elif message:
                print("\n=== PROCESSING MESSAGE ===")
                # Process through state machine
                self.state = self.process_message(self.state)
                
                # Get the current question and analysis
                question = self.state.get("current_question", "")
                
                # Create analysis output
                analysis = {
                    "emotional": {
                        "emotional_weight": self.state["humanity"]["emotional"]["emotional_weight"],
                        "key_emotions": self.state["humanity"]["emotional"]["key_emotions"],
                        "emotional_complexity": self.state["humanity"]["emotional"]["emotional_complexity"],
                        "vulnerability_level": self.state["humanity"]["emotional"]["vulnerability_level"]
                    },
                    "cognitive": {
                        "current_load": self.state["humanity"]["cognitive"]["current_load"],
                        "mental_effort_level": self.state["humanity"]["cognitive"]["mental_effort_level"]
                    },
                    "engagement": {
                        "engagement_level": self.state["humanity"]["engagement"]["engagement_level"],
                        "dimensions": self.state["humanity"]["engagement"].get("dimensions", {})
                    },
                    "formality": {
                        "current_level": self.state["humanity"]["formality"]["current_level"],
                        "style_consistency": self.state["humanity"]["formality"].get("style_consistency", 0.5)
                    }
                }

                # Generate topic coverage stats
                current_topic = self.state["topics"][self.state["current_topic_id"]]
                topic_stats = {
                    "topic_id": self.state["current_topic_id"],
                    "topic_question": current_topic["question"],
                    "questions_asked": self.state["questions_in_topic"],
                    "factor_coverage": {
                        factor: {
                            "coverage": score,
                            "questions": current_topic["question_attempts"].get(factor, 0)
                        }
                        for factor, score in current_topic["covered_factors"].items()
                    }
                }
                
                return {
                    "response": question,
                    "humanity_analysis": analysis,
                    "topic_stats": topic_stats
                }

            return {"response": "Omlouvám se, ale nerozuměl/a jsem vaší odpovědi."}

        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            raise

    def introduce_interview(self, state: State) -> State:
        """Generate a focused introduction about classroom behavior challenges."""
        print("\n=== INTRODUCING INTERVIEW ===")

        intro_prompt = """Generate a professional introduction in Czech that covers:

        1. CLEAR RESEARCH PURPOSE:
        - Professional greeting
        - Explain this is research specifically about challenging classroom situations
        - We're studying how teachers handle problematic student behavior
        - Goal is to understand real difficulties teachers face and improve support systems
        
        2. SPECIFIC FOCUS:
        - Looking for concrete examples of problematic situations
        - Interest in understanding what makes these situations difficult
        - Want to learn about their emotional and practical challenges
        - Focus on specific incidents and their impact

        3. INTERVIEW STRUCTURE:
        - 20-30 minute conversation
        - Will ask about specific challenging situations
        - Interested in details about what happened and how they handled it
        - No judgment, focus is on understanding their experience

        4. CONFIDENTIALITY & SUPPORT:
        - All responses are confidential
        - Used to develop better support for teachers
        - Can be completely honest about difficulties
        - No need to minimize challenges

        Then ask first question specifically about challenging situations they've faced.
        Use natural but professional Czech.
        Make it clear we want to hear about the difficult aspects of teaching.
        """
        
        response = self.model.invoke([SystemMessage(content=intro_prompt)])
        
        # Store introduction metadata
        state["interview_metadata"] = {
            "expected_duration": "20-30 minut",
            "purpose": "výzkum problematických situací ve třídě",
            "focus": "náročné situace a chování studentů",
            "goal": "zlepšení podpory učitelů v obtížných situacích",
            "confidentiality": True
        }

        return {
            **state,
            "current_question": response.content,
            "introduction_done": True
            }
        

    def print_factor_analysis(self, state: Dict):
        """Print detailed analysis of all factors."""
        current_topic = state["topics"][state["current_topic_id"]]
        
        print("\n=== DETAILNÍ ANALÝZA FAKTORŮ ===")
        print(f"Aktuální téma: {current_topic['question']}\n")
        
        for factor_name, factor_desc in current_topic["factors"].items():
            print(f"\nFAKTOR: {factor_name}")
            print(f"Popis: {factor_desc}")
            print(f"Pokrytí: {current_topic['covered_factors'].get(factor_name, 0.0):.2f}")
            print(f"Počet otázek: {current_topic['question_attempts'].get(factor_name, 0)}")
            
            # Print evidence if exists
            if factor_name in current_topic.get("factor_insights", {}):
                insights = current_topic["factor_insights"][factor_name]
                print("\nNalezené důkazy:")
                for evidence in insights.get("evidence", []):
                    print(f"- Citace: \"{evidence.quote}\"")
                    print(f"  Kontext: {evidence.context}")
                    print(f"  Relevance: {evidence.relevance:.2f}")
                    print(f"  Interpretace: {evidence.interpretation}")
                print(f"\nShrnutí: {insights.get('summary', 'Není k dispozici')}")
                print(f"Chybějící aspekty: {insights.get('missing', 'Není k dispozici')}")
            else:
                print("\nŽádné důkazy zatím nebyly nalezeny.")

    def analyze_response(self, response: str, topic: Topic) -> Dict[str, float]:
        """Analyze response with detailed factor evidence extraction in a single prompt."""
        print("\n=== ANALÝZA ODPOVĚDI UČITELE ===")
        print(f"Analyzuji odpověď: {response}\n")
        
        # Create a combined prompt for all factors
        factors_info = "\n".join([
            f"FAKTOR {name}:\n{desc}\n"
            for name, desc in topic["factors"].items()
        ])
        
        class MultiFactorAnalysis(BaseModel):
            """Model for analyzing multiple factors at once"""
            factors: Dict[str, DetailedFactorAnalysis] = Field(
                ..., 
                description="Analysis results for each factor"
            )

        parser = PydanticOutputParser(pydantic_object=MultiFactorAnalysis)
        
        prompt = PromptTemplate(
            template="""Analyzujte odpověď učitele pro VŠECHNY uvedené faktory najednou.
            DŮLEŽITÉ: Pokud učitel odmítne odpovědět na otázku o traumatické události, zachovejte předchozí skóre pokrytí.

            ODPOVĚĎ UČITELE:
            {response}

            FAKTORY K ANALÝZE:
            {factors_info}
            
            POŽADAVKY PRO KAŽDÝ FAKTOR:
            1. Najděte POUZE EXPLICITNĚ zmíněné relevantní citace
            2. Pro každou citaci uveďte:
               - Přesnou citaci z textu
               - Kontext citace
               - Míru relevance (0-1) - buďte přísní!
               - Interpretaci vzhledem k faktoru
            3. Určete celkovou míru pokrytí (0-1):
               - 0.0-0.2: Žádné nebo minimální zmínky
               - 0.2-0.4: Základní, ale neúplné informace
               - 0.4-0.6: Částečné pokrytí s důležitými mezerami
               - 0.6-0.8: Dobré pokrytí, chybí některé detaily
               - 0.8-1.0: Velmi detailní a kompletní pokrytí
            4. Shrňte POUZE skutečně nalezené informace
            5. Detailně vypište všechny chybějící aspekty
            
            PRAVIDLA:
            - Analyzujte všechny faktory v kontextu celé odpovědi
            - Neodvozujte informace, které nejsou přímo řečeny
            - Nepředpokládejte kontext, který není explicitně zmíněn
            - Buďte skeptičtí k náznakům - počítejte pouze jasné zmínky
            - Zohledněte vzájemné souvislosti mezi faktory
            
            {format_instructions}""",
            input_variables=["response", "factors_info"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        try:
            # Get and parse analysis for all factors at once
            result = self.model.invoke([SystemMessage(content=prompt.format(
                response=response,
                factors_info=factors_info
            ))])
            
            analysis = parser.parse(result.content)
            coverage = {}
            
            # Process results for each factor
            for factor_name, factor_analysis in analysis.factors.items():
                print(f"\nAnalýza faktoru: {factor_name}")
                
                # Update coverage more conservatively
                current_coverage = topic["covered_factors"].get(factor_name, 0.0)
                if factor_analysis.coverage_score > (current_coverage + 0.2):
                    topic["covered_factors"][factor_name] = factor_analysis.coverage_score
                else:
                    topic["covered_factors"][factor_name] = min(
                        current_coverage + 0.1,
                        max(current_coverage, factor_analysis.coverage_score)
                    )
                
                coverage[factor_name] = factor_analysis.coverage_score
                
                # Store insights
                topic["factor_insights"][factor_name] = {
                    "evidence": factor_analysis.evidence,
                    "summary": factor_analysis.summary,
                    "missing": factor_analysis.missing_aspects
                }
                
                # Print detailed results
                print(f"- Pokrytí: {topic['covered_factors'][factor_name]:.2f}")
                print(f"- Počet cílených otázek: {topic['question_attempts'].get(factor_name, 0)}")
                if topic['covered_factors'][factor_name] >= 0.5:
                    print("  ✓ Dostatečně pokryto")
                else:
                    print("  ⚠ Potřebuje další prozkoumání")
                
                if factor_analysis.evidence:
                    print("\nNalezené důkazy:")
                    for evidence in factor_analysis.evidence:
                        print(f"- Citace: \"{evidence.quote}\"")
                        print(f"  Kontext: {evidence.context}")
                        print(f"  Relevance: {evidence.relevance:.2f}")
                        print(f"  Interpretace: {evidence.interpretation}")
                
                print(f"\nShrnutí: {factor_analysis.summary}")
                print(f"Chybějící aspekty: {factor_analysis.missing_aspects}")
            
            return coverage
            
        except Exception as e:
            print(f"Chyba při analýze faktorů: {str(e)}")
            return {factor: 0.0 for factor in topic["factors"]}

    def process_response(self, state: Dict) -> Dict:
        """Process response with mandatory emotional analysis and factor analysis."""
        if not state["user_message"]:
            return state

        print("\n=== PROCESSING RESPONSE ===")
        print(f"Message: {state['user_message']}")

        # Get current topic
        current_topic = state["topics"][state["current_topic_id"]]
        
        # Do factor analysis first
        print("\n=== FAKTOROVÁ ANALÝZA ===")
        coverage = self.analyze_response(state["user_message"], current_topic)
        
        # Update state with factor analysis results
        state["topics"][state["current_topic_id"]] = current_topic
        
        # Then do emotional analysis (just once)
        try:
            processed_state = self.humanity.process_response(state, state["user_message"])
            
            # Check for critical content
            critical_words = ['smrt', 'sebevražd', 'zemřel', 'zemřela', 'zabil', 'zabila', 'killing']
            message_lower = state['user_message'].lower()
            if any(word in message_lower for word in critical_words):
                processed_state["humanity"]["emotional"].update({
                    "emotional_weight": 0.9,
                    "trauma_indicators": True,
                    "requires_support": True,
                    "evidence": {"severity_level": "critical"}
                })
            
            return processed_state

        except Exception as e:
            print(f"Error in process_response: {str(e)}")
            return state

    def end_step(self, state: Dict) -> Dict:
        """Handle the end of conversation."""
        self.logger.debug("Ending conversation")
        if not state.get("messages") or not isinstance(state["messages"][-1], AIMessage):
            state["messages"].append(AIMessage(content="Děkuji za rozhovor. Na shledanou!"))
        state["is_complete"] = True  # Ensure we mark the conversation as complete
        return state

    def reset_state(self):
        """Reset the bot's state to start a new conversation."""
        self.state = self.get_default_state()
        self.is_initialized = False
        self.logger.info("State reset - starting new conversation")

    def analyze_intent(self, state: State) -> State:
        """Analyze the intent of user's response."""
        intent_prompt = f"""Analyze the intent of this response in the context of an interview about teaching experiences.

Context: This is a research interview about classroom experiences. The last question was: "{state['current_question']}"

Response: "{state['user_message']}"

Possible intents:
- direct_response: Directly answers the question or tells something that is emotionally significant
- refuses_question: Indicates they've already answered or don't want to discuss this aspect further
- asking_purpose: Asks about interview purpose/goals
- seeking_clarification: Needs clarification about question
- expressing_concern: Shows worry/discomfort about being interviewed
- meta_discussion: Wants to discuss interview process
- challenging_interview: Questions interview validity
- off_topic: Unrelated to question/interview

Return only the intent name that best matches."""

        result = self.model.invoke([SystemMessage(content=intent_prompt)])
        intent = result.content.strip()

        # Update the state dictionary directly
        state["detected_intent"] = intent
        return state

    def handle_purpose_question(self, state: Dict) -> Dict:
        """Handle questions about interview purpose."""
        # No need to convert dict to State object anymore
        purpose_context = """Tento rozhovor je součástí výzkumu, který se snaží porozumět reálným 
        zkušenostem učitelů ve třídách. Zajímají nás vaše osobní zážitky a postřehy, 
        protože věříme, že právě pohled učitelů je klíčový pro zlepšení podpory ve školství."""
        
        response = self.model.invoke([
            SystemMessage(content=f"""
            Respond empathetically to a question about interview purpose.
            Context: {purpose_context}
            History: {state['conversation_history']}
            Current question: {state['current_question']}
            User concern: {state['user_message']}
            """)
        ])

        return {**state, "current_question": response.content}

    def handle_clarification(self, state: Dict) -> Dict:
        """Handle requests for clarification."""
        clarification_context = """Snažíme se porozumět vašim zkušenostem z vlastní perspektivy. 
        Není zde žádná správná nebo špatná odpověď, zajímá nás váš pohled."""
        
        response = self.model.invoke([
            SystemMessage(content=f"""
            Respond to a request for clarification in Czech.
            Context: {clarification_context}
            Current question: {state['current_question']}
            User asks: {state['user_message']}
            
            Explain the question clearly and naturally, then ask it again in a slightly different way.
            Keep it short and friendly.
            """)
        ])

        return {
            **state,
            "current_question": response.content
        }

    def handle_concern(self, state: Dict) -> Dict:
        """Handle expressed concerns about privacy/safety."""
        concern_context = """Vaše bezpečí a pohodlí jsou pro nás prioritou. 
        Veškeré informace jsou důvěrné a používají se pouze pro výzkumné účely."""
        
        response = self.model.invoke([
            SystemMessage(content=f"""
            Address privacy/safety concerns in Czech.
            Context: {concern_context}
            User concern: {state['user_message']}
            
            First acknowledge their concern, then explain privacy/confidentiality, 
            and gently return to the topic with a simple question.
            """)
        ])

        return {
            **state,
            "current_question": response.content
        }

    def handle_meta_discussion(self, state: Dict) -> Dict:
        """Handle questions about the interview process."""
        meta_context = """Rozhovor je polostrukturovaný, vedený přirozeně a bez hodnocení. 
        Zajímají nás vaše zkušenosti a pohled na různé situace ve třídě."""
        
        response = self.model.invoke([
            SystemMessage(content=f"""
            Address questions about the interview process in Czech.
            Context: {meta_context}
            User question: {state['user_message']}
            
            Explain briefly how the interview works, then return to the current topic.
            """)
        ])

        return {
            **state,
            "current_question": response.content
        }

    def handle_challenge(self, state: Dict) -> Dict:
        """Handle challenges to interview validity."""
        challenge_context = """Tento výzkum vznikl ve spolupráci s učiteli a jeho cílem je 
        zlepšit podporu pro pedagogy v náročných situacích."""
        
        response = self.model.invoke([
            SystemMessage(content=f"""
            Address challenges to the interview in Czech.
            Context: {challenge_context}
            Challenge: {state['user_message']}
            
            Acknowledge their perspective, explain the research value briefly,
            then return to the topic with a simple question.
            """)
        ])

        return {
            **state,
            "current_question": response.content
        }

    def handle_off_topic(self, state: Dict) -> Dict:
        """Handle responses that are off-topic."""
        response = self.model.invoke([
            SystemMessage(content=f"""
            Gently redirect an off-topic response in Czech.
            Current topic: {state['current_question']}
            Off-topic response: {state['user_message']}
            
            Briefly acknowledge what they said, then return to the topic
            with a gentle reminder and the question.
            """)
        ])
        
        return {
            **state,
            "current_question": response.content
        }

    def evaluate_topic_value(self, state: Dict) -> bool:
        """Evaluate if the conversation is still providing value after 5+ questions."""
        current_topic = state["topics"][state["current_topic_id"]]
        
        # Check coverage scores
        coverage_scores = list(current_topic["covered_factors"].values())
        avg_coverage = sum(coverage_scores) / len(coverage_scores)
        
        # Check last 2 responses for new information
        recent_responses = state["conversation_history"][-2:]
        
        prompt = f"""Analyze if recent responses add new valuable information:

        Topic: {current_topic["question"]}
        Last responses:
        {chr(10).join(f'- {r["user"]}' for r in recent_responses)}

        Return JSON:
        {{
            "adds_value": bool,
            "reason": "brief explanation"
        }}"""

        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            analysis = json.loads(result.content)
            
            # Consider both coverage and new information
            return avg_coverage > 0.7 or analysis["adds_value"]
        except:
            return True  # Default to continue if analysis fails

    def check_topic_progress(self, state: Dict) -> Dict:
        """Check if we should continue with current topic or move on."""
        # Make sure questions_in_topic exists
        if "questions_in_topic" not in state:
            self.logger.warning("questions_in_topic missing from state, initializing to 0")
            state["questions_in_topic"] = 0
        
        state["questions_in_topic"] += 1
        
        # Debug print AFTER increment
        self.logger.debug(f"STATE AFTER INCREMENT: {json.dumps(state, default=str, indent=2)}")
        
        if state["questions_in_topic"] >= 5:
            # After 5 questions, evaluate if continuing adds value
            if not self.evaluate_topic_value(state):
                # Ask if they want to add anything before moving on
                response = self.model.invoke([
                    SystemMessage(content=f"""
                    Generate a polite message in Czech that:
                    1. Acknowledges their responses
                    2. Notes we've covered several aspects
                    3. Asks if they have anything important to add
                    4. If not, mentions we'll move to next topic
                    """)
                ])
                
                state["current_question"] = response.content
                state["topic_exhausted"] = True
                
        return state

    def _generate_base_question(self, state: Dict) -> Dict:
        """Generate research-focused response in Czech."""
        # Handle refusal to answer first
        if state["detected_intent"] == "refuses_question":
            return self.handle_topic_refusal(state)
        
        current_topic = state["topics"][state["current_topic_id"]]
        humanity = state["humanity"]
        emotional = humanity["emotional"]
        
        print("\n=== GENEROVÁNÍ VÝZKUMNÉ OTÁZKY ===")
        
        parser = PydanticOutputParser(pydantic_object=ResearchQuestion)

        # Get factors that still need coverage (less than 50%)
        uncovered = [f for f, score in current_topic["covered_factors"].items() 
                    if score < 0.5]

        # Format prompt
        formatted_prompt = PromptTemplate(
            template="""Vygenerujte výzkumnou otázku v češtině.

            AKTUÁLNÍ TÉMA: {topic_question}
            NEPOKRYTÉ FAKTORY: {uncovered_factors}
            POSLEDNÍ ODPOVĚĎ: {last_response}
            
            EMOČNÍ KONTEXT:
            - Intenzita: {weight}
            - Emoce: {emotions}
            - Potřeba podpory: {support}
            
            POŽADAVKY:
            1. Držte se výzkumného tématu
            2. Zaměřte se na nepokryté faktory
            3. Ptejte se na konkrétní zkušenosti
            4. Formulujte otázku jasně a přirozeně
            5. Zachovejte profesionální výzkumný kontext
            
            {format_instructions}""",
            input_variables=["topic_question", "uncovered_factors", "last_response", 
                            "weight", "emotions", "support"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
                # If previous factor was refused, ensure we don't ask about it again
        if state["detected_intent"] == "refuses_question" and state.get("last_asked_factor"):
            uncovered = [f for f in uncovered if f != state["last_asked_factor"]]

        state["current_question_factors"] = [] 

        try:
            # Get and parse response
            result = self.model.invoke([SystemMessage(content=formatted_prompt.format(
                topic_question=current_topic["question"],
                uncovered_factors=", ".join(uncovered) if uncovered else "All essential factors covered",
                last_response=state.get('user_message', ''),
                weight=emotional['emotional_weight'],
                emotions=', '.join(emotional['key_emotions']),
                support=emotional['requires_support']
            ))])
            response = parser.parse(result.content)
            
            # Store ALL factors this question is targeting
            target_factor = response.cilovy_faktor
            state["current_question_factors"].append(target_factor)
            
            # Also store any related factors this question might cover
            related_factors = self._identify_related_factors(response.otazka, current_topic["factors"])
            state["current_question_factors"].extend(related_factors)
            
            print(f"\nGenerated Research Response:")
            print(f"- Primary Factor: {target_factor}")
            print(f"- Related Factors: {related_factors}")
            print(f"- All Targeted Factors: {state['current_question_factors']}")
            
            # Update question counts for all targeted factors
            for factor in state["current_question_factors"]:
                if factor in current_topic["factors"]:
                    current_topic["question_attempts"][factor] = current_topic["question_attempts"].get(factor, 0) + 1
            
            state["questions_in_topic"] = state.get("questions_in_topic", 0) + 1
            
            full_response = f"{response.potvrzeni} {response.otazka}"
            return {**state, "current_question": full_response}
            
        except Exception as e:
            print(f"\nERROR generating question: {str(e)}")
            return {**state, "current_question": current_topic["question"]}

    def _enhance_question(self, state: Dict) -> Dict:
        """Pass through the question since enhancement is now built into generation."""
        return state  # No enhancement needed, just pass through

    def handle_topic_refusal(self, state: Dict) -> Dict:
        """Handle when user refuses to discuss a topic further."""
        print("\n=== HANDLING TOPIC REFUSAL ===")
        current_topic = state["topics"][state["current_topic_id"]]
        
        # Get ALL factors that were targeted by the last question
        refused_factors = state.get("current_question_factors", [])
        
        # CRITICAL FIX: If no factors are tracked, get them from the question attempts
        if not refused_factors:
            refused_factors = [
                factor for factor, attempts in current_topic["question_attempts"].items()
                if attempts > 0 and current_topic["covered_factors"].get(factor, 0.0) < 0.5
            ]
        
        print(f"\nFactors being marked as refused: {refused_factors}")
        
        # Mark ALL targeted factors as covered at EXACTLY 50%
        for factor in refused_factors:
            if factor in current_topic["factors"]:
                print(f"Setting factor '{factor}' coverage to EXACTLY 50%")
                current_topic["covered_factors"][factor] = 0.5  # Force exactly 50%
                current_topic["refused_factors"] = current_topic.get("refused_factors", set())
                current_topic["refused_factors"].add(factor)
                
                # Debug print to verify the update
                print(f"Verified: Factor {factor} coverage is now: {current_topic['covered_factors'][factor]}")
        
        # Change intent to direct_response to avoid refusal loop
        state["detected_intent"] = "direct_response"
        
        # Clear the tracked factors for the next question
        state["current_question_factors"] = []
        state["last_asked_factor"] = None
        
        # Get remaining available factors (not refused and under 50%)
        available_factors = [f for f in current_topic["factors"].keys() 
                            if f not in current_topic.get("refused_factors", set())
                            and current_topic["covered_factors"].get(f, 0.0) < 0.5]
        
        print(f"\nRemaining available factors: {available_factors}")
        
        if not available_factors:
            # Move to next topic if no more factors to ask about
            current_topic["topic_completed"] = True
            print("\nNo more available factors - marking topic as completed")
            return self._move_to_next_topic(state)
        
        print("\n=== GENERATING NEW QUESTION FOR REMAINING FACTORS ===")
        return self._generate_base_question(state)

    def _identify_related_factors(self, question: str, factors: Dict[str, str]) -> List[str]:
        """Identify which factors a question might cover based on its content."""
        prompt = f"""Analyze which factors this question might cover:

        Question: {question}
        
        Available Factors:
        {json.dumps(factors, indent=2)}
        
        Return a list of factor names that this question could potentially address.
        Consider both direct and indirect coverage of factors."""
        
        try:
            result = self.model.invoke([SystemMessage(content=prompt)])
            related_factors = json.loads(result.content)
            return related_factors
        except:
            return []  # Return empty list if analysis fails
   