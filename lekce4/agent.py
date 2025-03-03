from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]

# Define LLM with bound tools
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()

def get_initial_state() -> MessagesState:
    """Create initial state with system message"""
    return {"messages": [sys_msg]}

if __name__ == "__main__":
    logger.info("Starting Math Assistant...")
    print("\nMath Assistant ready! Type 'quit' to exit.")
    print("Example commands: 'add 5 and 3', 'multiply 6 by 4', 'divide 10 by 2'\n")
    
    state = get_initial_state()
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                print("\nGoodbye!")
                break
                
            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))
            
            # Process through graph
            state = graph.invoke(state)
            
            # Print assistant's response
            print(f"\nAssistant: {state['messages'][-1].content}")
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            print("\nSorry, I encountered an error. Please try again.")
            state = get_initial_state()  # Reset state on error
