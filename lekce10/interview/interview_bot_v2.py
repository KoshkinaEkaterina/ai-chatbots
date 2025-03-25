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
from classes import Topic, FactorInsight, State
from uuid import uuid4
from pydantic import BaseModel, Field
import json
import logging
from langsmith import Client
from langchain.prompts import SystemMessage


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
    analysis: List[FactorAnalysis]

class ChatState(TypedDict):
    messages: List[HumanMessage | AIMessage]

class InterviewBot:
    def __init__(self):
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG,
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
            "direct_response": self.handle_direct_response,
            "asking_purpose": self.handle_purpose_question,
            "seeking_clarification": self.handle_clarification,
            "expressing_concern": self.handle_concern,
            "meta_discussion": self.handle_meta_discussion,
            "emotional_support": self.handle_emotional_support,
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
            # Update the model to use callbacks
            self.model = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=0.7,
                streaming=False
            )
            self.logger.info("LangSmith tracing initialized")
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
                    current_topic = Topic(row['id'], row['content'])
                    topics[row['id']] = current_topic
                elif row['type'] == 'factor':
                    if current_topic and row['id'] == current_topic.id:
                        factor_name = row['content']
                        current_topic.factors[factor_name] = row['factor']
                        current_topic.covered_factors[factor_name] = 0.0
        return topics

    def get_default_state(self) -> State:
        """Initialize default state with all required keys."""
        return {
            "current_question": None,
            "user_message": None,
            "conversation_history": [],
            "topics": self.load_topics(),
            "current_topic_id": "T1",
            "introduction_done": False,
            "interview_complete": False
        }

    def process_message(self, state: Dict) -> Dict:
        """Process the incoming message and generate a response."""
        if not state.get("messages"):
            # Initial greeting
            current_topic = self.state["topics"][self.state["current_topic_id"]]
            greeting = "Dobrý den, jsem tady, abych s vámi vedl/a rozhovor o vašich zkušenostech ve třídě."
            state["messages"] = [
                AIMessage(content=f"{greeting} {current_topic.question}")
            ]
            return state
        
        # Get the last message
        last_message = state["messages"][-1]
        
        if isinstance(last_message, HumanMessage):
            # Process the message using the interview logic
            current_topic = self.state["topics"][self.state["current_topic_id"]]
            
            if "konec" in last_message.content.lower():
                state["messages"].append(
                    AIMessage(content="Děkuji za rozhovor. Na shledanou!")
                )
                return state
            
            # Analyze response if not a greeting
            if self.state["introduction_done"]:
                coverage = self.analyze_response(last_message.content, current_topic)
                
                # Generate next question based on coverage
                next_question = self.generate_next_question(current_topic, coverage)
                state["messages"].append(AIMessage(content=next_question))
            else:
                # Handle initial greeting
                self.state["introduction_done"] = True
                state["messages"].append(
                    AIMessage(content=f"Děkuji za odpověď. Nyní bych se vás rád/a zeptal/a: {current_topic.question}")
                )
        
        return state

    def create_graph(self) -> StateGraph:
        """Create conversation flow with intent analysis."""
        builder = StateGraph(State)

        # Add intent analysis as first step
        builder.add_node("analyze_intent", self.analyze_intent)
        
        # Add all intent handlers and regular nodes
        for intent in self.intents.keys():
            builder.add_node(f"handle_{intent}", self.intents[intent])
        builder.add_node("process_response", self.process_response)
        builder.add_node("generate_question", self.generate_question)

        # Create dynamic flow based on intent
        builder.add_edge(START, "analyze_intent")
        
        # Create a router function to handle all intents
        def route_by_intent(state: State) -> str:
            intent = state.get("detected_intent")
            if intent in self.intents:
                return f"handle_{intent}"
            return "process_response"

        # Add single conditional edge with router
        builder.add_conditional_edges(
            "analyze_intent",
            route_by_intent,
            [f"handle_{intent}" for intent in self.intents.keys()] + ["process_response"]
        )

        # Continue normal flow
        for intent in self.intents.keys():
            builder.add_edge(f"handle_{intent}", "generate_question")
        builder.add_edge("process_response", "generate_question")
        builder.add_edge("generate_question", END)

        return builder.compile()

    def chat(self, message: str = None) -> Dict:
        """Process chat message and return response."""
        try:
            # Initialize only once at the very start
            if not self.is_initialized and not message:
                self.state = self.introduce_interview(self.state)
                self.is_initialized = True
                return {
                    "response": self.state["current_question"]
                }

            # Process normal message
            if message:
                # Update state with user message and history
                self.state["user_message"] = message
                if not self.state.get("conversation_history"):
                    self.state["conversation_history"] = []
                self.state["conversation_history"].append({
                    "user": message,
                    "current_question": self.state.get("current_question")
                })
                
                # Process through graph
                self.state = self.graph.invoke(self.state)
                
                # Get current topic and analysis
                current_topic = self.state["topics"][self.state["current_topic_id"]]
                
                # Create analysis text
                analysis = f"\n=== ANALÝZA ROZHOVORU ===\n\n"
                analysis += f"Téma {self.state['current_topic_id']}: {current_topic.question}\n\n"
                analysis += "Pokrytí faktorů:\n"
                for factor, score in current_topic.covered_factors.items():
                    analysis += f"- {factor}: {score:.2f}\n"
                
                # Combine analysis and question with spacing
                response = f"{analysis}\n\n\n{self.state['current_question']}"
                
                return {
                    "response": response
                }

            return {
                "response": "Omlouvám se, ale nerozuměl/a jsem vaší odpovědi. Můžete to prosím zopakovat?"
            }

        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}", exc_info=True)
            raise

    def introduce_interview(self, state: State) -> State:
        """Generate a focused introduction about classroom behavior challenges."""
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

    def analyze_response(self, response: str, topic: Topic) -> Dict[str, float]:
        """Analyze response with handling for off-topic or chaotic answers."""
        # First, check if the response is completely off-topic
        relevance_prompt = f"""Analyze if this response is relevant to the topic:
        TOPIC: {topic.question}
        RESPONSE: {response}
        
        Return JSON:
        {{
            "is_relevant": bool,
            "reason": "brief explanation"
        }}"""
        
        relevance_check = self.model.invoke([
            SystemMessage(content="Determine if the response is on-topic."),
            SystemMessage(content=relevance_prompt)
        ])
        
        try:
            relevance = json.loads(relevance_check.content.strip())
            if not relevance["is_relevant"]:
                print(f"\nUPOZORNĚNÍ: Odpověď je mimo téma - {relevance['reason']}")
                return {factor: 0.0 for factor in topic.factors.keys()}
        except:
            pass  # If relevance check fails, continue with normal analysis

        prompt = f"""Analyze this Czech teacher's response and extract specific information.
        
        RESPONSE: {response}
        
        Analyze EXACTLY these factors (use these exact names):
        {chr(10).join(f'- {factor}' for factor in topic.factors.keys())}
        
        Return a structured analysis following this exact schema:
        {{
            "analysis": [
                {{
                    "factor": "EXACT_FACTOR_NAME_FROM_LIST_ABOVE",
                    "score": 0.8,
                    "found_info": [
                        {{
                            "detail": "specific information found",
                            "quote": "exact quote from text",
                            "relevance": 0.9
                        }}
                    ],
                    "summary": "overall summary of findings",
                    "missing": "what information is still needed"
                }}
            ]
        }}"""
        
        result = self.model.invoke([
            SystemMessage(content="You are a precise JSON generator. Use EXACT factor names as provided."),
            SystemMessage(content=prompt)
        ])
        
        try:
            # Clean up the response to ensure it's valid JSON
            response_text = result.content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Use model_validate_json instead of parse_raw for Pydantic v2
            analysis = AnalysisResponse.model_validate_json(response_text)
            coverage = {}
            
            for factor_analysis in analysis.analysis:
                factor = factor_analysis.factor
                if factor not in topic.factors:
                    print(f"Warning: Unknown factor {factor}")
                    continue
                
                coverage[factor] = factor_analysis.score
                topic.covered_factors[factor] = max(
                    topic.covered_factors.get(factor, 0.0),
                    factor_analysis.score
                )
                
                # Store insights for each finding
                for finding in factor_analysis.found_info:
                    insight = FactorInsight(
                        answer_id=str(uuid4()),
                        content=finding.detail,
                        source_answer=response,
                        relevance_score=finding.relevance,
                        evidence=factor_analysis.summary,
                        quote=finding.quote
                    )
                    topic.add_insight(factor, insight)
            
            return coverage
            
        except Exception as e:
            print(f"Error parsing analysis: {e}")
            print("Raw response:")
            print(response_text)
            return {factor: 0.0 for factor in topic.factors.keys()}



    def analyze_emotional_content(self, response: str) -> dict:
        """Analyze the emotional weight and trauma level of a response."""
        prompt = f"""Analyze the emotional content of this teacher's response:
        RESPONSE: {response}
        
        Return JSON:
        {{
            "emotional_weight": float,  # 0.0-1.0, how emotionally heavy is the content
            "trauma_indicators": bool,  # whether the response indicates traumatic experience
            "emotional_cues": [str],  # list of emotional indicators in the text (pauses, voice changes, etc.)
            "key_emotions": [str],  # main emotions expressed
            "requires_support": bool  # whether the response needs emotional acknowledgment
        }}"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            return json.loads(result.content.strip())
        except:
            return {
                "emotional_weight": 0.0,
                "trauma_indicators": False,
                "emotional_cues": [],
                "key_emotions": [],
                "requires_support": False
            }

    def generate_question(self, state: State) -> State:
        """Generate next question based on conversation history."""
        # Get conversation history in consistent format
        history = state.get("conversation_history", [])
        
        # Convert history to text format, handling both old and new formats
        history_text = "\n\n".join([
            f"Q: {exchange.get('current_question', exchange.get('question', ''))}\n"
            f"A: {exchange.get('user', exchange.get('answer', ''))}"
            for exchange in history[-3:]  # Get last 3 exchanges
        ])

        # Get current topic
        current_topic = state["topics"][state["current_topic_id"]]
        
        # Create prompt for next question
        prompt = f"""Based on this conversation history:
        {history_text}

        Current topic: {current_topic.question}
        Factors we need to explore:
        {chr(10).join(f'- {factor}: {description}' for factor, description in current_topic.factors.items())}

        Generate ONE follow-up question in Czech that:
        1. Directly relates to their last response
        2. Helps explore uncovered factors
        3. Is specific and focused
        4. Uses natural, conversational language
        5. Avoids evaluative statements or disclaimers
        """

        response = self.model.invoke([SystemMessage(content=prompt)])

        return {
            **state,
            "current_question": response.content
        }

    def process_response(self, state: State) -> State:
        """Process the user's response and update factor coverage."""
        preserved_state = {
            **state,
            "conversation_history": state.get("conversation_history", []),
            "topics": state.get("topics", self.load_topics()),
            "current_topic_id": state.get("current_topic_id", "T1"),
            "introduction_done": state.get("introduction_done", False),
            "interview_complete": state.get("interview_complete", False)
        }

        if not preserved_state.get("user_message"):
            return preserved_state

        # Analyze the current response
        current_topic = preserved_state["topics"][preserved_state["current_topic_id"]]
        analysis = self.analyze_response(preserved_state["user_message"], current_topic)
        
        # Create analysis summary
        analysis_text = "\n=== ANALÝZA ROZHOVORU ===\n\n"
        analysis_text += f"Téma {preserved_state['current_topic_id']}: {current_topic.question}\n"
        analysis_text += f"- Odpověď: {preserved_state['user_message']}\n\n"
        analysis_text += "Pokrytí faktorů:\n"
        
        uncovered_factors = []
        for factor, score in analysis.items():
            analysis_text += f"- {factor}: {score:.2f}\n"
            if score < 0.5:  # Track factors that need more exploration
                uncovered_factors.append(factor)
        
        analysis_text += "\nPotřebujeme lépe porozumět:\n"
        for factor in uncovered_factors:
            analysis_text += f"- {current_topic.factors[factor]}\n"

        # Add extra spacing between analysis and question
        analysis_text += "\n\n\n"  # Triple newline for clear separation

        # Get last few exchanges for context
        recent_history = preserved_state["conversation_history"][-3:] 
        history_context = "\n".join([
            f"Q: {ex['current_question']}\nA: {ex['user']}"
            for ex in recent_history
        ])

        # Generate contextual response with analysis and spacing
        response = self.model.invoke([
            SystemMessage(content=f"""
            Recent conversation:
            {history_context}

            Analysis:
            {analysis_text}

            Generate ONE follow-up question in Czech that:
            1. Directly relates to what they just said
            2. Helps explore uncovered factors (especially: {', '.join(uncovered_factors)})
            3. Is specific and focused
            4. Uses natural, conversational language
            5. Avoids evaluative statements or disclaimers

            Format your response EXACTLY like this:
            {analysis_text}

            [your question here]
            """)
        ])

        return {
            **preserved_state,
            "current_question": response.content,
            "last_analysis": analysis
        }

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
- direct_response: Directly answers the question
- asking_purpose: Asks about interview purpose/goals
- seeking_clarification: Needs clarification about question
- expressing_concern: Shows worry/discomfort
- meta_discussion: Wants to discuss interview process
- emotional_support: Needs emotional acknowledgment
- challenging_interview: Questions interview validity
- off_topic: Unrelated to question/interview

Return only the intent name that best matches."""

        result = self.model.invoke([SystemMessage(content=intent_prompt)])
        intent = result.content.strip()

        return {**state, "detected_intent": intent}

    def handle_purpose_question(self, state: State) -> State:
        """Handle questions about interview purpose."""
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

        return {
            **state,
            "current_question": response.content
        }

    def handle_clarification(self, state: State) -> State:
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

    def handle_concern(self, state: State) -> State:
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

    def handle_meta_discussion(self, state: State) -> State:
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

    def handle_emotional_support(self, state: State) -> State:
        """Provide emotional support when needed."""
        support_context = """Uvědomujeme si, že některá témata mohou být citlivá. 
        Váš pocit bezpečí je pro nás důležitý."""
        
        response = self.model.invoke([
            SystemMessage(content=f"""
            Provide emotional support in Czech.
            Context: {support_context}
            User message: {state['user_message']}
            History: {state['conversation_history']}
            
            Acknowledge their feelings with empathy, then gently continue with a simple question.
            """)
        ])

        return {
            **state,
            "current_question": response.content
        }

    def handle_challenge(self, state: State) -> State:
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

    def handle_off_topic(self, state: State) -> State:
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

    def handle_direct_response(self, state: State) -> State:
        """Handle direct responses to questions - pass through to normal processing."""
        # For direct responses, we just pass through to the normal processing
        # but we need to preserve the state
        return {
            **state,
            "current_question": state.get("current_question"),
            "user_message": state.get("user_message"),
            "conversation_history": state.get("conversation_history", []),
            "topics": state.get("topics", self.load_topics()),
            "current_topic_id": state.get("current_topic_id", "T1"),
            "introduction_done": state.get("introduction_done", False),
            "interview_complete": state.get("interview_complete", False)
        }

class TeacherBotV2:
    def __init__(self):
        # Core knowledge from the dialogue
        self.context = {
            "student": {
                "gender": "female",
                "behavior": "self-harm",
                "signs": [
                    "wears long sleeves almost always",
                    "avoids PE class",
                    "claims to be sick often",
                    "quiet personality",
                    "doesn't seek contact",
                    "visible cuts on arms (trying to hide them)"
                ],
                "social": [
                    "generally quiet",
                    "doesn't talk much to anyone",
                    "other students don't pay much attention to her",
                    "no conflicts with others"
                ]
            },
            "teacher": {
                "approach": [
                    "discussing with colleagues",
                    "planning to contact parents",
                    "unsure about proper response",
                    "emphasizes being teacher, not psychologist"
                ],
                "planned_actions": [
                    "contact parents",
                    "discuss self-harm with parents",
                    "find ways to help effectively"
                ],
                "class_impact": "minimal, isolated issue with one student"
            }
        }

    def generate_response(self, question: str) -> str:
        """Generate contextually appropriate response based on the teacher's perspective."""
        
        # Convert question to lowercase for easier matching
        question_lower = question.lower()
        
        # Define response patterns based on the dialogue
        if "spouštěč" in question_lower or "proč" in question_lower:
            return "To vůbec netuším"
            
        if "tělocvik" in question_lower or "sport" in question_lower:
            return "Tělocviku se většinou vyhývá s tím, že je nachlazená"
            
        if "konflikty" in question_lower or "problémy" in question_lower:
            return "Nene, nic takového, celkově je spíš tišší a úplně kontakty nevyhledává"
            
        if "ostatní" in question_lower or "spolužáci" in question_lower:
            return "Ostatní si ji spíš taky nevšímají"
            
        if "atmosféra" in question_lower or "třída" in question_lower:
            return "Ne, to se spíš neděje"
            
        if "reagovat" in question_lower or "řešit" in question_lower:
            return "Nevím, jak na to mám reagovat, nejsem psycholog ale učitel"
            
        if "postup" in question_lower or "plán" in question_lower:
            return "Probírám to s kolegy a snažím se sbírat jejich zkušenosti. Aktuálně se shodujeme, že se máme spojit s rodiči"
            
        if "rodiče" in question_lower:
            if "plán" in question_lower:
                return "Chceme s nimi probrat, že se jejich dcera sebepoškozuje a jak je možné jejich dceři efektivně pomoci"
            else:
                return "Zatím nijak, teprve to máme v plánu"

        # Default response for unmatched questions
        return "To nedokážu říct, celkově je těžké to hodnotit"

    def chat(self, message: str) -> Dict:
        """Process chat message and return response."""
        response = self.generate_response(message)
        return {
            "response": response
        }
