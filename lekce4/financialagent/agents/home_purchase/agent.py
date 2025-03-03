from typing import Dict
from ...core.state import FinancialState, Strategy, FinancialAnalysis, CalculationContext
from langchain_core.messages import AIMessage
from .calculator import HomePurchaseCalculator
from .prompts import HomePurchasePrompts

class HomePurchaseAgent:
    def __init__(self):
        self.calculator = HomePurchaseCalculator()
        self.prompts = HomePurchasePrompts()

    def __call__(self, state: FinancialState) -> FinancialState:
        params = state.get("parameters", {})
        profile = state["profile"]
        
        context = CalculationContext(
            inputs={
                "income": params.get("income", profile["income"]),
                "savings": params.get("savings", profile["savings"]),
                "monthly_expenses": sum(params.get("expenses", {}).values()) or 
                    sum(v for k, v in profile.items() if k.startswith("expense_")),
                "dti_ratio": params.get("dti_ratio", 0.43),
                "down_payment_percent": params.get("down_payment_percent", 0.20),
                "mortgage_rate": params.get("mortgage_rate", 0.065)
            },
            steps=[],
            results={},
            assumptions={
                "dti_ratio": 0.43,
                "down_payment": 0.20,
                "mortgage_years": 30,
                "property_tax_rate": 0.012,
                "insurance_rate": 0.005,
                "maintenance_percent": 0.01
            }
        )

        # Calculate with detailed tracking
        monthly_income = context.inputs["income"] / 12
        monthly_expenses = context.inputs["monthly_expenses"]
        
        max_mortgage = self.calculator.calculate_max_mortgage(
            monthly_income, 
            monthly_expenses,
            context.inputs["dti_ratio"]
        )
        
        months_to_down = self.calculator.calculate_down_payment_timeline(
            max_mortgage,
            monthly_income - monthly_expenses,
            context.inputs["savings"],
            context.inputs["down_payment_percent"]
        )

        monthly_payment = self.calculator.calculate_monthly_payment(
            max_mortgage * (1 - context.inputs["down_payment_percent"]),
            context.inputs["mortgage_rate"]
        )

        context.results.update({
            "max_home_price": max_mortgage,
            "down_payment": max_mortgage * context.inputs["down_payment_percent"],
            "monthly_payment": monthly_payment,
            "months_to_down": months_to_down,
            "monthly_income": monthly_income,
            "available_monthly": monthly_income - monthly_expenses
        })

        response = self.prompts.format_response({
            **context.inputs,
            **context.results,
            "next_steps": [
                f"Save ${(context.results['down_payment'] / months_to_down):.0f} monthly for down payment",
                "Improve credit score to get better rates",
                "Research neighborhoods and property values"
            ]
        })
        
        return {
            **state,
            "current_agent": "home_purchase",
            "calculation_context": context,
            "analysis": FinancialAnalysis(
                metrics=context.results,
                assessment=f"Can afford ${max_mortgage:,.0f} home in {months_to_down} months",
                required_actions=[
                    f"Save ${context.results['down_payment']/months_to_down:.0f} monthly for down payment",
                    "Maintain or improve credit score",
                    "Keep debt-to-income ratio low"
                ]
            ),
            "messages": list(state["messages"]) + [AIMessage(content=response)]
        } 