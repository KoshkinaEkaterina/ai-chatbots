from typing import Dict, Optional
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import logging
from enum import Enum, auto

from financialagent.core.state import FinancialState
from financialagent.core.parameter_extractor import ParameterExtractor
from financialagent.agents.home_purchase.agent import HomePurchaseAgent
from financialagent.agents.retirement.agent import RetirementAgent
from financialagent.core.explanation_handler import ExplanationHandler

# Update logging configuration
logging.basicConfig(level=logging.WARNING)  # Change DEBUG to WARNING
logger = logging.getLogger(__name__)

# Load environment variables and profiles
load_dotenv()

# Get the directory containing this file
current_dir = Path(__file__).parent
profiles_path = current_dir / "financial_profiles.csv"
profiles_df = pd.read_csv(profiles_path)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

class Intent(Enum):
    RETIREMENT = "retirement"
    HOME_PURCHASE = "home_purchase"
    EXPLANATION = "explanation"
    PARAMETER_CHANGE = "parameter_change"
    UNKNOWN = "unknown"

def create_intent_prompt(message: str, current_domain: Optional[str] = None) -> str:
    """Create prompt for intent classification"""
    return f"""Classify the user's intent. Message: "{message}"
    Current domain: {current_domain or "None"}

    RESPOND WITH ONE WORD:
    - retirement (if about retirement, pension, 401k, retiring, old age, savings for later)
    - home_purchase (if about buying house, mortgage, down payment)
    - explanation (if asking why/how/explain about current topic)
    - parameter_change (if changing any numbers or settings)
    - unknown (if unclear)

    Examples:
    "I want to retire" -> retirement
    "Let's talk about retirement" -> retirement
    "I need to plan for later" -> retirement
    "Change my age" -> parameter_change
    "Why so much?" -> explanation

    Intent:"""

def conversation_handler(state: Dict) -> Dict:
    """Route conversation based on intent and current domain"""
    message = state["messages"][-1].content.lower()
    current_domain = state.get("domain")
    
    # Get intent using LLM
    intent_prompt = SystemMessage(content=create_intent_prompt(message, current_domain))
    intent_response = llm.invoke([intent_prompt]).content.strip().lower()
    
    print("\nMessage Analysis:")
    print(f"Your message: '{message}'")
    print(f"Current domain: {current_domain or 'None'}")
    print(f"Detected intent: {intent_response}")
    
    # Set new domain if starting fresh or switching domains
    new_domain = current_domain
    if intent_response in ["retirement", "home_purchase"]:
        new_domain = intent_response
        print(f"Starting new {new_domain} planning session")
    elif current_domain == "unknown" and intent_response == "parameter_change":
        print("Cannot process parameters without an active planning session")
        new_domain = "unknown"
    
    return {
        **state,
        "intent": intent_response,
        "domain": new_domain
    }

def continue_conversation(state: Dict) -> Dict:
    """Pass through state for continuing the conversation"""
    return state

def end_conversation(state: Dict) -> Dict:
    """End the conversation gracefully"""
    # Only say goodbye if they explicitly said goodbye
    message = state["messages"][-1].content.lower()
    if any(word in message for word in ['quit', 'exit', 'bye', 'goodbye']):
        state["messages"].append(AIMessage(content="Goodbye! Have a great day!"))
    return state

def unknown_handler(state: Dict) -> Dict:
    """Handle unknown intents by giving clear instructions"""
    message = state["messages"][-1].content
    
    # Use LLM to give a contextual response that acknowledges what they said
    response_prompt = SystemMessage(content=f"""You are a financial advisor assistant. The user said: "{message}"

    Respond in this format:
    1. Briefly acknowledge what they said
    2. Explain you can ONLY help with home purchase or retirement planning
    3. Ask them to choose one of these topics if they're interested

    Keep it direct and clear. Don't apologize - just be straightforward about what you can help with.
    
    Example:
    "I see you're asking about student loans. I'm specifically designed to help with:
    1. Home Purchase Planning (mortgages, down payments)
    2. Retirement Planning (401k, pension, savings)
    
    Would you like to discuss either of these topics?"
    
    Response:""")
    
    response = llm.invoke([response_prompt])
    state["messages"].append(AIMessage(content=response.content))
    return state

def create_graph():
    """Create the financial planning graph"""
    workflow = StateGraph(FinancialState)
    
    # Add nodes
    workflow.add_node("conversation_handler", conversation_handler)
    workflow.add_node("parameter_extractor", ParameterExtractor(llm))
    workflow.add_node("home_purchase", HomePurchaseAgent())
    workflow.add_node("retirement", RetirementAgent())
    workflow.add_node("unknown", unknown_handler)
    workflow.add_node("explanation", ExplanationHandler(llm))
    workflow.add_node("end", end_conversation)
    
    workflow.set_entry_point("conversation_handler")

    # Main routing from conversation handler
    workflow.add_conditional_edges(
        "conversation_handler",
        lambda x: x.get("intent", "unknown"),  # Route based on INTENT
        {
            "explanation": "explanation",       # Send explanation intent directly to explanation handler
            "parameter_change": "parameter_extractor",
            "retirement": "parameter_extractor",
            "home_purchase": "parameter_extractor",
            "unknown": "unknown"
        }
    )
    
    # After explanation, return to appropriate agent based on domain
    workflow.add_conditional_edges(
        "explanation",
        lambda x: x.get("domain", "unknown"),
        {
            "retirement": "end",        # Go straight to end after explanation
            "home_purchase": "end",
            "unknown": "unknown"
        }
    )
    
    # Route from handlers to agents
    workflow.add_conditional_edges(
        "parameter_extractor",
        lambda x: x.get("domain", "unknown"),
        {
            "retirement": "retirement",
            "home_purchase": "home_purchase",
            "unknown": "unknown"
        }
    )
    
    # All paths lead to end
    workflow.add_edge("home_purchase", "end")
    workflow.add_edge("retirement", "end")
    workflow.add_edge("unknown", "end")
    
    workflow.set_finish_point("end")
    
    return workflow.compile()

# Create the graph instance for the API
graph = create_graph()

def get_initial_state() -> FinancialState:
    """Create initial state with conversation tracking"""
    return {
        "messages": [
            SystemMessage(content="You are a focused financial advisor. Help users achieve their financial goals."),
            AIMessage(content="What financial goal would you like to work towards? I can help with:\n1. Home Purchase Planning\n2. Retirement Planning")
        ],
        "profile": profiles_df[profiles_df["profile_type"] == "moderate"].to_dict("records")[0],
        "current_agent": None,
        "domain": None,
        "analysis": None,
        "strategy": None,
        "context": {},
        "calculation_context": None,
        "parameters": {}
    }

# Main loop handles the conversation flow instead of the graph
if __name__ == "__main__":
    logger.info("Starting Financial Planning Assistant...")
    
    state = get_initial_state()
    logger.debug("Initial state created")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            logger.debug(f"User input: {user_input}")
            
            state["messages"].append(HumanMessage(content=user_input))
            
            logger.debug("Invoking graph")
            state = graph.invoke(state)
            logger.debug(f"Graph returned. Domain: {state.get('domain')}, Agent: {state.get('current_agent')}")
            
            print("\nAI:", state["messages"][-1].content)
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}", exc_info=True)
            print("\nAI: I encountered an error. Please try again.") 