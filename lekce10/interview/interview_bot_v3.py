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
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


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
                    current_topic = Topic(
                        id=row['id'],
                        question=row['content'],
                        factors={},
                        covered_factors={},
                        question_attempts={}
                    )
                    topics[row['id']] = current_topic
                elif row['type'] == 'factor' and current_topic and row['id'] == current_topic.id:
                    factor_name = row['content']
                    # Initialize all tracking for this factor
                    current_topic.factors[factor_name] = row['factor']
                    current_topic.covered_factors[factor_name] = 0.0
                    current_topic.question_attempts[factor_name] = 0
                
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
            "interview_complete": False,
            "questions_in_topic": 0,
            "topic_exhausted": False
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
                next_question = self.generate_question(current_topic, coverage)
                state["messages"].append(AIMessage(content=next_question))
            else:
                # Handle initial greeting
                self.state["introduction_done"] = True
                state["messages"].append(
                    AIMessage(content=f"Děkuji za odpověď. Nyní bych se vás rád/a zeptal/a: {current_topic.question}")
                )
        
        return state

    def create_graph(self) -> StateGraph:
        """Create conversation flow with topic progress tracking."""
        builder = StateGraph(State)

        # Add nodes
        builder.add_node("analyze_intent", self.analyze_intent)
        builder.add_node("check_topic_progress", self.check_topic_progress)
        builder.add_node("process_response", self.process_response)
        builder.add_node("generate_question", self.generate_question)
        
        for intent in self.intents.keys():
            builder.add_node(f"handle_{intent}", self.intents[intent])

        # Create flow
        builder.add_edge(START, "analyze_intent")
        builder.add_edge("analyze_intent", "check_topic_progress")
        
        # Route based on intent and topic progress
        def route_by_state(state: State) -> str:
            if state.get("topic_exhausted"):
                return "process_response"
            intent = state.get("detected_intent")
            return f"handle_{intent}" if intent in self.intents else "process_response"

        builder.add_conditional_edges(
            "check_topic_progress",
            route_by_state,
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
                self.state = self.introduce_interview(self.get_default_state())  # Use fresh state
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
                
                # Create analysis text with coverage AND question attempts
                analysis = f"\n=== ANALÝZA ROZHOVORU ===\n\n"
                analysis += f"Téma {self.state['current_topic_id']}: {current_topic.question}\n\n"
                analysis += "Pokrytí faktorů a počet dotazů:\n"
                for factor, score in current_topic.covered_factors.items():
                    attempts = current_topic.question_attempts[factor]
                    analysis += f"- {factor}:\n"
                    analysis += f"  Pokrytí: {score:.2f}\n"
                    analysis += f"  Počet dotazů: {attempts}/2\n"
                
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
        """Analyze response using structured output parsing."""
        
        # Create parser
        parser = PydanticOutputParser(pydantic_object=AnalysisResponse)

        # Create prompt template
        template = """Analyze this Czech teacher's response and extract specific information.

        RESPONSE: {response}

        Analyze EXACTLY these factors (use these exact names):
        {factors}

        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["response", "factors"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Format prompt
        formatted_prompt = prompt.format(
            response=response,
            factors="\n".join(f"- {factor}" for factor in topic.factors.keys())
        )

        # Get structured output
        result = self.model.invoke([SystemMessage(content=formatted_prompt)])
        
        try:
            # Parse the response into our schema
            analysis = parser.parse(result.content)
            coverage = {}
            
            for factor_analysis in analysis.analysis:
                factor = factor_analysis.factor
                if factor not in topic.factors:
                    self.logger.warning(f"Unknown factor {factor}")
                    continue
                
                coverage[factor] = factor_analysis.score
                topic.covered_factors[factor] = max(
                    topic.covered_factors.get(factor, 0.0),
                    factor_analysis.score
                )
                
                # Store insights
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
            self.logger.error(f"Failed to parse analysis: {str(e)}")
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
        """Generate next question based on coverage and attempts."""
        current_topic = state["topics"][state["current_topic_id"]]
        
        # Get conversation history
        history = state.get("conversation_history", [])
        history_text = "\n\n".join([
            f"Q: {exchange.get('current_question', '')}\n"
            f"A: {exchange.get('user', '')}"
            for exchange in history[-3:]
        ])
        
        # Get factors we NEED to ask about (low coverage AND < 2 attempts)
        available_factors = {
            factor: details 
            for factor, details in current_topic.factors.items()
            if (current_topic.covered_factors[factor] < 0.7  # Low coverage
                and current_topic.question_attempts[factor] < 2)  # Haven't asked twice
        }
        
        if not available_factors:
            # If we've covered everything well or tried twice, wrap up
            response = self.model.invoke([
                SystemMessage(content="""
                Generate a polite message in Czech that:
                1. Acknowledges we've covered several aspects
                2. Asks if they have anything important to add
                3. If not, mentions we'll move to next topic
                """)
            ])
            state["topic_exhausted"] = True
            return {**state, "current_question": response.content}

        # Create parser for structured output
        class QuestionOutput(BaseModel):
            question: str = Field(..., description="The follow-up question to ask")
            targeted_factors: List[str] = Field(..., description="List of factors this question addresses")

        parser = PydanticOutputParser(pydantic_object=QuestionOutput)

        # Create prompt template focusing on UNCOVERED factors
        template = """Based on this conversation history:
        {history_text}
        
        Current topic: {topic}
        Last response: {last_response}
        
        FOCUS ON THESE UNCOVERED ASPECTS:
        {factors}
        
        Generate a natural follow-up question in Czech that:
        1. Directly relates to their last response
        2. Asks about one or more of the UNCOVERED aspects listed above
        3. Does NOT ask about information we already have
        4. Uses natural, conversational language
        
        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["history_text", "topic", "last_response", "factors"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        formatted_prompt = prompt.format(
            history_text=history_text,
            topic=current_topic.question,
            last_response=state['user_message'],
            factors="\n".join(f"- {factor} (coverage: {current_topic.covered_factors[factor]:.2f}): {details}" 
                             for factor, details in available_factors.items())
        )

        response = self.model.invoke([SystemMessage(content=formatted_prompt)])
        result = parser.parse(response.content)
        
        # Update attempt counters
        for factor in result.targeted_factors:
            current_topic.question_attempts[factor] += 1
        
        return {**state, "current_question": result.question}

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

        # Check topic progress before generating next question
        state = self.check_topic_progress(state)
        
        if state.get("topic_exhausted"):
            if "ne" in state["user_message"].lower() or "nemám" in state["user_message"].lower():
                # Move to next topic
                next_topic = self.get_next_topic(state["current_topic_id"])
                state["current_topic_id"] = next_topic
                state["questions_in_topic"] = 0
                state["topic_exhausted"] = False
                
                # Introduce new topic
                current_topic = state["topics"][next_topic]
                return {
                    **state,
                    "current_question": f"Děkuji. Nyní bych se rád/a zeptal/a na další téma: {current_topic.question}"
                }
        
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
        return {
            **state,
            "current_question": state.get("current_question"),
            "user_message": state.get("user_message"),
            "conversation_history": state.get("conversation_history", []),
            "topics": state.get("topics", self.load_topics()),
            "current_topic_id": state.get("current_topic_id", "T1"),
            "introduction_done": state.get("introduction_done", False),
            "interview_complete": state.get("interview_complete", False),
            "questions_in_topic": state.get("questions_in_topic", 0),
            "topic_exhausted": state.get("topic_exhausted", False)
        }

    def evaluate_topic_value(self, state: State) -> bool:
        """Evaluate if the conversation is still providing value after 5+ questions."""
        current_topic = state["topics"][state["current_topic_id"]]
        
        # Check coverage scores
        coverage_scores = list(current_topic.covered_factors.values())
        avg_coverage = sum(coverage_scores) / len(coverage_scores)
        
        # Check last 2 responses for new information
        recent_responses = state["conversation_history"][-2:]
        
        prompt = f"""Analyze if recent responses add new valuable information:

        Topic: {current_topic.question}
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

    def check_topic_progress(self, state: State) -> State:
        """Check if we should continue with current topic or move on."""
        # Debug print the state BEFORE incrementing
        
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
