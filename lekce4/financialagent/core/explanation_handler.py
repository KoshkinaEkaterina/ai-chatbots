from typing import Dict, List
import asyncio
from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
import logging

class ExplanationHandler:
    def __init__(self, llm: ChatOpenAI):
        self.logger = logging.getLogger(__name__)
        self.llm = llm
        # This is just reference material for the LLM, not for key matching
        self.knowledge_base = """
        Retirement Planning Concepts:
        
        1. Target Retirement Amount
        - Uses the 4% safe withdrawal rule
        - To get target amount: multiply your future annual need by 25
        - Future annual need: take current income, multiply by income replacement percent, then account for inflation over the years
        - Example: $100k income, 80% replacement, 3% inflation for 30 years
        - This grows your needed income to account for future prices
        - Assumes you'll need the money to last 30+ years
        
        2. Monthly Savings Required
        - Calculates how much you need to save each month
        - Takes into account: target amount, current savings, years until retirement, expected return rate
        - Your current savings will grow with investment returns
        - Additional monthly savings also grow with compound interest
        - Adjusts for the time value of money
        
        3. Investment Returns
        - Typical long-term market returns: 7-10% per year
        - Real return is what you get after subtracting inflation
        - Money grows faster over time due to compound interest
        - Higher returns usually mean more risk
        - Past performance doesn't guarantee future returns
        
        4. Inflation Impact
        - Historical average around 3% per year
        - Makes things more expensive over time
        - Example: $100 today might buy only $50 worth in 20 years
        - Critical for long-term planning
        - Why we need investment returns above inflation
        
        Home Purchase Planning Concepts:
        
        1. Maximum Mortgage Amount
        - Based on your debt-to-income ratio (DTI)
        - Monthly payment can't exceed a certain percent of monthly income
        - Usually 28-36% of your income for all housing costs
        - Includes principal, interest, taxes, insurance
        - Other debts reduce how much you can borrow
        
        2. Down Payment Requirements
        - Usually need 3.5-20% of house price
        - Bigger down payment means smaller monthly payments
        - Less than 20% usually requires PMI (extra insurance)
        - Time to save = Down Payment Amount divided by Monthly Savings
        
        3. Monthly Payment Calculation
        - Depends on: loan amount, interest rate, loan term
        - Longer terms mean lower payments but more total interest
        - Also need to add property taxes and insurance
        - PMI adds extra cost if down payment is low
        - Interest rates greatly affect monthly payment
        """

    def get_context_from_messages(self, messages: List) -> Dict:
        """Extract relevant context from previous messages"""
        context = {
            "domain": None,
            "last_calculation": None,
            "parameters": {}
        }
        
        for msg in messages:
            if isinstance(msg, AIMessage) and "Analysis:" in msg.content:
                if "Retirement Analysis:" in msg.content:
                    context["domain"] = "retirement"
                elif "Purchase Potential:" in msg.content:
                    context["domain"] = "home_purchase"
                context["last_calculation"] = msg.content
                break
        
        return context

    def create_explanation_prompt(self, user_question: str, context: Dict) -> str:
        return f"""The user asked: "{user_question}"

        Their current situation:
        - Current Age: {context['inputs']['current_age']}
        - Retirement Age: {context['inputs']['retirement_age']}
        - Current Income: ${context['inputs']['income']:,.2f} per year
        - Current Savings: ${context['inputs']['savings']:,.2f}
        - Years to Retirement: {context['results']['years_to_retirement']}
        
        Knowledge Base:
        {self.knowledge_base}

        Explain what they asked about in plain conversational English:
        1. What it means in their specific situation
        2. How we calculated it
        3. Why it matters
        4. What could change it
        
        Use their actual numbers and explain like you're talking to a friend. No formulas or technical terms.
        Focus specifically on what they asked about, but connect it to the bigger picture if relevant.
        """

    def explain(self, state: Dict) -> Dict:
        try:
            context = state.get("calculation_context")
            if not context:
                state["messages"].append(AIMessage(content="I don't have any calculations to explain yet. Let's do some planning first."))
                return state

            user_question = state["messages"][-1].content
            
            # Build comprehensive context
            full_context = {
                "inputs": context.inputs,
                "results": context.results,
                "assumptions": context.assumptions
            }
            
            # Get explanation from LLM
            prompt = self.create_explanation_prompt(user_question, full_context)
            explanation = self.llm.invoke([SystemMessage(content=prompt)])
            
            # Replace the standard retirement analysis with the explanation
            state["messages"].append(AIMessage(content=explanation.content))
            
            return {
                **state,
                "domain": state.get("domain")
            }
            
        except Exception as e:
            state["messages"].append(AIMessage(content="I encountered an error while explaining. Could you rephrase your question?"))
            return state

    def __call__(self, state: Dict) -> Dict:
        message = state["messages"][-1].content
        
        print("\nExplanation Request Analysis:")
        print(f"Your question: '{message}'")
        print("I'll explain this concept in detail in my next response")
        
        return self.explain(state) 