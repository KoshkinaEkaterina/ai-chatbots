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
        """Create the conversation flow graph."""
        # Build graph
        builder = StateGraph(State)

        # Add nodes
        builder.add_node("process_response", self.process_response)
        builder.add_node("generate_question", self.generate_question)

        # Create simple linear flow: START -> process_response -> generate_question -> END
        builder.add_edge(START, "process_response")
        builder.add_edge("process_response", "generate_question")
        builder.add_edge("generate_question", END)

        return builder.compile()

    def chat(self, message: Optional[str] = None) -> Dict:
        """Process a chat message and return the response."""
        try:
            self.logger.debug(f"Processing chat message: {message}")
            
            # Initialize state if first call
            if not self.state.get("introduction_done"):
                self.state = self.introduce_interview(self.state)
                return {
                    "response": self.state["current_question"],
                    "question": self.state["current_question"],
                    "is_complete": False,
                    "covered_factors": self.state["topics"][self.state["current_topic_id"]].covered_factors
                }
            print(self.graph.nodes) 
            print(message)
            # Update state with user message
            if message:
                self.state["user_message"] = message
            
            # Process through graph
            self.logger.debug(f"Current state before processing: {self.state}")
            print("DEBUG: state before invoke:", self.state)
            result = self.graph.invoke(self.state)
            self.logger.debug(f"Graph processing result: {result}")
            
            # Update state
            self.state = result
            
            return {
                "response": self.state["current_question"],
                "question": self.state["current_question"],
                "is_complete": self.state.get("interview_complete", False),
                "covered_factors": self.state["topics"][self.state["current_topic_id"]].covered_factors
            }
            
        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}", exc_info=True)
            return {
                "response": "Omlouvám se, došlo k chybě v zpracování.",
                "question": "Omlouvám se, došlo k chybě v zpracování.",
                "is_complete": True,
                "covered_factors": {}
            }

    def introduce_interview(self, state: State) -> State:
        """Generate the interview introduction in Czech."""
        intro_prompt = """You are a professional interviewer starting a conversation in Czech. 
        Introduce yourself and explain that you'll be conducting an interview about classroom experiences 
        and student behavior. Be welcoming and friendly, but maintain a professional tone.
        
        Write a brief introduction in Czech followed by the first topic question."""
        
        response = self.model.invoke([SystemMessage(content=intro_prompt)])
        current_topic = state["topics"][state["current_topic_id"]]
        
        return {
            **state,
            "current_question": f"{response.content}\n\n{current_topic.question}",
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

    def print_topic_status(self, topic: Topic):
        """Print detailed status of topic coverage and insights."""
        print(f"\n{'='*100}")
        print(f"DETAILNÍ ANALÝZA ODPOVĚDI PRO TÉMA: {topic.question}")
        print(f"{'='*100}")
        
        for factor, description in topic.factors.items():
            print(f"\n{'='*50}")
            print(f"FAKTOR: {factor}")
            print(f"POPIS: {description}")
            print(f"{'='*50}")
            
            if factor in topic.factor_insights and topic.factor_insights[factor]:
                print("\nNALEZENÉ INFORMACE:")
                for insight in topic.factor_insights[factor]:
                    print(f"\n• DETAIL: {insight['key_info']}")
                    if 'evidence' in insight:
                        print(f"  DŮKAZ: {insight['evidence']}")
                    if 'quote' in insight:
                        print(f"  CITACE: \"{insight['quote']}\"")
                    print(f"  RELEVANCE: {insight.get('score', 0.0):.2f}")
                
                print(f"\nCELKOVÉ POKRYTÍ: {topic.covered_factors.get(factor, 0.0):.2f}")
            else:
                print("\nŽÁDNÉ INFORMACE NEBYLY NALEZENY")
                print(f"POKRYTÍ: {topic.covered_factors.get(factor, 0.0):.2f}")
            
            print(f"\n{'-'*50}")
        
        print(f"\n{'='*100}")

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
        """Generate naturally flowing, empathetic follow-up questions."""
        current_topic = state["topics"][state["current_topic_id"]]
        
        # Get recent conversation history to track our empathetic responses
        recent_history = state.get("conversation_history", [])[-3:]
        previous_responses = [
            exchange.get("interviewer_response", "")
            for exchange in recent_history
        ]
        
        last_response = state.get("user_message")
        if last_response:
            emotional_analysis = self.analyze_emotional_content(last_response)
            
            if emotional_analysis["emotional_weight"] > 0.6 or emotional_analysis["trauma_indicators"]:
                support_prompt = (
                    f"""The teacher just shared a deeply emotional experience in an ongoing conversation.

                    Recent conversation history:
                    {chr(10).join(f'Q: {ex["question"]}, A: {ex["answer"]}' for ex in recent_history)}
                    
                    Latest response: "{last_response}"
                    
                    Emotional context:
                    - Weight: {emotional_analysis['emotional_weight']}
                    - Emotions: {', '.join(emotional_analysis['key_emotions'])}
                    - Previous empathetic responses used: {previous_responses}
                    
                    Generate a natural, flowing response in Czech that:
                    1. Shows you're listening by referencing specific details they shared
                    2. Asks ONE clear follow-up question
                    
                    IMPORTANT GUIDELINES:
                    - Only offer to stop/change topic if trauma indicators are very high
                    - Don't use generic empathy phrases
                    - Stay focused on what they're actually telling you
                    - Let them guide the emotional depth
                    
                    BAD EXAMPLES (too generic/repetitive):
                    - "Samozřejmě, pokud byste o tom raději nemluvila, plně to respektuji."
                    - "Chápu, že je to těžké téma."
                    - "Děkuji za vaši otevřenost."
                    
                    GOOD EXAMPLES (specific to their story):
                    - "Ten moment s tím nožem musel být opravdu intenzivní... Jak jste dokázala zachovat klid?"
                    - "Zmínila jste, že vás to dodnes pronásleduje. Co vám pomáhá se s tím vyrovnat?"
                    
                    Current emotional weight: {emotional_analysis['emotional_weight']}"""
                )
                
                response = self.model.invoke([SystemMessage(content=support_prompt)])
                
                return {
                    **state,
                    "current_question": response.content,
                    "emotional_context": emotional_analysis
                }
        
        # If not emotionally heavy, proceed with normal question generation
        # Ensure all required state keys exist
        if not isinstance(state, dict):
            state = state.copy()
        
        # Initialize required state keys with defaults
        preserved_state = {
            "current_question": state.get("current_question"),
            "user_message": state.get("user_message"),
            "conversation_history": state.get("conversation_history", []),
            "topics": state.get("topics", self.load_topics("topics.csv")),
            "current_topic_id": state.get("current_topic_id", "T1"),
            "introduction_done": state.get("introduction_done", False),
            "interview_complete": state.get("interview_complete", False)
        }
        state = preserved_state
        
        if not state.get("introduction_done"):
            return self.introduce_interview(state)
        
        current_topic = state["topics"][state["current_topic_id"]]
        uncovered_factors = [
            (factor, desc) for factor, desc in current_topic.factors.items()
            if current_topic.covered_factors[factor] < 0.7
        ]
        
        # Format conversation history
        history_text = "\n\n".join([
            f"Q: {exchange['question']}\nA: {exchange['answer']}"
            for exchange in state.get("conversation_history", [])[-3:]
        ])
        
        prompt = f"""You are conducting an interview in Czech about classroom experiences.
        Current topic: {current_topic.question}
        
        Recent conversation:
{history_text}

        IMPORTANT GUIDELINES:
        1. If the teacher shares traumatic or emotionally heavy experiences:
           - Acknowledge the emotional weight first
           - Show empathy and understanding
           - Give space for processing
           - Only gently proceed with follow-up if appropriate
           
        2. When asking follow-up questions:
           - Stay with difficult topics rather than rushing forward
           - Show you're listening and care about their experience
           - Validate their feelings and experiences
           
        3. Question formulation:
           - Keep it gentle and supportive when needed
           - Allow space for "no" or "I'd rather not discuss this further"
           - Focus on understanding their experience
        
        Uncovered aspects to explore:
        {chr(10).join(f'- {desc}' for _, desc in uncovered_factors)}
        
        Generate an appropriate follow-up response or question in Czech that puts being human first and gathering information second."""
        
        response = self.model.invoke([SystemMessage(content=prompt)])
        
        return {
            **state,
            "current_question": response.content
        }

    def print_topic_summary(self, topic: Topic, file_path: str = "interview_analysis.txt"):
        """Create a detailed summary of all insights gathered for each factor in the topic."""
        # First print to console
        print(f"\n{'='*100}")
        print(f"SOUHRNNÁ ANALÝZA TÉMATU: {topic.question}")
        print(f"{'='*100}\n")
        
        for factor, description in topic.factors.items():
            print(f"\nFAKTOR: {factor}")
            print(f"POPIS: {description}")
            print(f"CELKOVÉ POKRYTÍ: {topic.covered_factors.get(factor, 0.0):.2f}")
            
            if factor in topic.factor_insights and topic.factor_insights[factor]:
                print("\nVŠECHNA ZJIŠTĚNÍ:")
                for insight in topic.factor_insights[factor]:
                    print(f"\n• INFORMACE: {insight['key_info']}")
                    print(f"  CITACE: \"{insight['quote']}\"")
                    print(f"  RELEVANCE: {insight.get('score', 0.0):.2f}")
                
                # Generate an overall summary
                summary_prompt = f"""Create a concise summary of these insights about {factor}:
                {chr(10).join(f'- {i["key_info"]}' for i in topic.factor_insights[factor])}
                
                Return a 2-3 sentence summary in Czech."""
                
                summary = self.model.invoke([SystemMessage(content=summary_prompt)])
                print(f"\nSOUHRN FAKTORU:\n{summary.content}")
            else:
                print("\nŽÁDNÉ INFORMACE NEBYLY ZÍSKÁNY")
            
            print(f"\n{'-'*50}")
        
        # Then write to file
        with open(file_path, "a", encoding='utf-8') as f:
            f.write(f"\n{'='*100}\n")
            f.write(f"SOUHRNNÁ ANALÝZA TÉMATU: {topic.question}\n")
            f.write(f"{'='*100}\n\n")
            
            for factor, description in topic.factors.items():
                f.write(f"\nFAKTOR: {factor}\n")
                f.write(f"POPIS: {description}\n")
                f.write(f"CELKOVÉ POKRYTÍ: {topic.covered_factors.get(factor, 0.0):.2f}\n")
                
                if factor in topic.factor_insights and topic.factor_insights[factor]:
                    f.write("\nVŠECHNA ZJIŠTĚNÍ:\n")
                    for insight in topic.factor_insights[factor]:
                        f.write(f"\n• INFORMACE: {insight['key_info']}\n")
                        f.write(f"  CITACE: \"{insight['quote']}\"\n")
                        f.write(f"  RELEVANCE: {insight.get('score', 0.0):.2f}\n")
                
                    # Generate an overall summary
                    summary_prompt = f"""Create a concise summary of these insights about {factor}:
                    {chr(10).join(f'- {i["key_info"]}' for i in topic.factor_insights[factor])}
                    
                    Return a 2-3 sentence summary in Czech."""
                    
                    summary = self.model.invoke([SystemMessage(content=summary_prompt)])
                    f.write(f"\nSOUHRN FAKTORU:\n{summary.content}\n")
                else:
                    f.write("\nŽÁDNÉ INFORMACE NEBYLY ZÍSKÁNY\n")
                
                f.write(f"\n{'-'*50}\n")

    def process_response(self, state: State) -> State:
        """Process the user's response and update factor coverage."""
        # Ensure all required state keys exist
        if not isinstance(state, dict):
            state = state.copy()
        
        # Initialize required state keys with defaults
        preserved_state = {
            "current_question": state.get("current_question"),
            "user_message": state.get("user_message"),
            "conversation_history": state.get("conversation_history", []),
            "topics": state.get("topics", self.load_topics("topics.csv")),
            "current_topic_id": state.get("current_topic_id", "T1"),
            "introduction_done": state.get("introduction_done", False),
            "interview_complete": state.get("interview_complete", False)
        }
        state = preserved_state
        
        if not state.get("user_message"):
            return state
        
        # Print the teacher's response immediately
        print(f"\nUčitel: {state['user_message']}")
        
        current_topic = state["topics"][state["current_topic_id"]]
        
        # Analyze factor coverage
        coverage = self.analyze_response(state["user_message"], current_topic)
        for factor, score in coverage.items():
            current_topic.covered_factors[factor] = max(
                current_topic.covered_factors[factor],
                score
            )
        
        # Update conversation history
        history = state.get("conversation_history", [])
        history.append({
            "question": state["current_question"],
            "answer": state["user_message"],
            "coverage": coverage
        })
        
        # Check if we should move to next topic
        if all(score >= 0.7 for score in current_topic.covered_factors.values()):
            print("\nTÉMA DOKONČENO - Generuji souhrnnou analýzu...\n")
            self.print_topic_summary(current_topic)
            
            # Find next topic
            topic_ids = list(state["topics"].keys())
            current_index = topic_ids.index(state["current_topic_id"])
            
            # Check if this was the last topic
            if current_index >= len(topic_ids) - 1:
                print("\nVŠECHNA TÉMATA DOKONČENA!")
                return {
                    **state,
                    "interview_complete": True,
                    "conversation_history": history,
                    "user_message": None
                }
            else:
                state["current_topic_id"] = topic_ids[current_index + 1]
        
        return {
            **state,
            "conversation_history": history,
            "user_message": None
        }

    def log_detailed_analysis(self, topic: Topic, response: str, analysis_data: Dict, file_path: str = "interview_analysis.txt"):
        """Write detailed analysis to a file."""
        with open(file_path, "a", encoding='utf-8') as f:
            f.write(f"\n{'='*100}\n")
            f.write(f"TÉMA: {topic.question}\n")
            f.write(f"ODPOVĚĎ: {response}\n\n")
            
            for factor, data in analysis_data.items():
                f.write(f"\nFAKTOR: {factor}\n")
                f.write(f"POPIS: {topic.factors[factor]}\n")
                f.write(f"SKÓRE: {topic.covered_factors[factor]:.2f}\n")
                
                if factor in topic.factor_insights:
                    for insight in topic.factor_insights[factor]:
                        f.write(f"\n• DETAIL: {insight['key_info']}\n")
                        f.write(f"  CITACE: \"{insight['quote']}\"\n")
                        f.write(f"  DŮKAZ: {insight['evidence']}\n")
                
                f.write(f"\n{'-'*50}\n")

    def print_brief_status(self, old_state: State, answer: str, next_question: str):
        """Print status with emotional awareness."""
        current_topic = old_state["topics"][old_state["current_topic_id"]]
        
        # Check for emotional content
        emotional_analysis = self.analyze_emotional_content(answer)
        
        # If the response was emotionally significant, acknowledge before analysis
        if emotional_analysis["emotional_weight"] > 0.6:
            print("\n" + "-"*50)
            print("EMOČNÍ KONTEXT:")
            print("Učitel sdílel velmi citlivou zkušenost. Dejme prostor pro zpracování...")
            print("-"*50 + "\n")
        
        # Only print analysis and next question (removed answer printing)
        print("\nANALÝZA:")
        covered = {f: s for f, s in current_topic.covered_factors.items() if s > 0}
        if covered:
            for factor, score in covered.items():
                print(f"✓ {factor}: {score:.2f}")
        else:
            print("❌ Odpověď neposkytla žádné relevantní informace k tématu.")
        
        print("\nDALŠÍ OTÁZKA:")
        print(next_question)
        print("\nZDŮVODNĚNÍ:")
        if not covered:
            print("Předchozí odpověď byla mimo téma. Zkusíme otázku položit jinak.")
        else:
            uncovered = [f for f, s in current_topic.covered_factors.items() if s < 0.7]
            if uncovered:
                print(f"Potřebujeme více informací o: {', '.join(uncovered)}")
            else:
                print("Přecházíme k dalšímu tématu.")

    def end_step(self, state: Dict) -> Dict:
        """Handle the end of conversation."""
        self.logger.debug("Ending conversation")
        if not state.get("messages") or not isinstance(state["messages"][-1], AIMessage):
            state["messages"].append(AIMessage(content="Děkuji za rozhovor. Na shledanou!"))
        state["is_complete"] = True  # Ensure we mark the conversation as complete
        return state
