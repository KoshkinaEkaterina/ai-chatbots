from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
import logging
from typing import Union, Dict
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def calculate_monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    """Calculate monthly payment for a loan.
    
    Args:
        principal: Loan amount
        annual_rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        years: Loan term in years
    """
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    
    if monthly_rate == 0:
        return principal / num_payments
        
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    return monthly_payment

def calculate_total_interest(principal: float, monthly_payment: float, years: int) -> float:
    """Calculate total interest paid over loan term."""
    total_paid = monthly_payment * years * 12
    return total_paid - principal

def calculate_loan(principal: float, annual_rate: float, years: int) -> Dict:
    """Calculate loan details and return comprehensive information."""
    monthly_payment = calculate_monthly_payment(principal, annual_rate, years)
    total_interest = calculate_total_interest(principal, monthly_payment, years)
    
    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_interest": round(total_interest, 2),
        "total_paid": round(principal + total_interest, 2),
        "principal": principal,
        "annual_rate": annual_rate,
        "years": years
    }

# Define tools
tools = [
    calculate_loan,
    calculate_monthly_payment,
    calculate_total_interest
]

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="""You are a helpful loan calculator assistant. You can:
1. Calculate monthly payments for loans
2. Calculate total interest paid
3. Explain how loan calculations work

Always explain your calculations and why the numbers make sense.
If the user asks about APR vs APY or other concepts, explain them clearly.

Example interaction:
User: "Calculate loan payment for $200,000 at 5% for 30 years"
Assistant: Let me calculate that for you.

Monthly payment: $1,073.64
Total interest: $186,511.57
Total amount paid: $386,511.57

Here's why:
- The monthly interest rate is 5%/12 = 0.417%
- Over 30 years, you'll make 360 payments
- The formula accounts for compound interest, which is why the total interest is close to the principal
- Each payment goes partly to principal and partly to interest, with more going to principal over time

Would you like me to break down how the payments change over time?""")

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
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()

def get_initial_state() -> MessagesState:
    """Create initial state with system message"""
    return {"messages": [sys_msg]}

if __name__ == "__main__":
    logger.info("Starting Loan Calculator Assistant...")
    print("\nLoan Calculator ready! Type 'quit' to exit.")
    print("Example commands:")
    print("- Calculate loan for $200,000 at 5% for 30 years")
    print("- Explain amortization")
    print("- What's the difference between APR and APY?\n")
    
    state = get_initial_state()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                print("\nGoodbye!")
                break
                
            state["messages"].append(HumanMessage(content=user_input))
            state = graph.invoke(state)
            print(f"\nAssistant: {state['messages'][-1].content}")
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            print("\nSorry, I encountered an error. Please try again.")
            state = get_initial_state() 