from typing import Dict
from ...core.state import FinancialState, Strategy, FinancialAnalysis, CalculationContext
from langchain_core.messages import AIMessage
from .calculator import RetirementCalculator
from .prompts import RetirementPrompts

class RetirementAgent:
    def __init__(self):
        self.calculator = RetirementCalculator()
        self.prompts = RetirementPrompts()

    def __call__(self, state: Dict) -> Dict:
        """Process retirement planning request"""
        params = state.get("parameters", {})
        
        # Set default values if not provided
        context = CalculationContext(
            inputs={
                "current_age": params.get("current_age", 35),  # Use actual params
                "retirement_age": params.get("retirement_age", 65),
                "income": params.get("income", 95000),
                "savings": params.get("savings", 25000),
                "inflation_rate": params.get("inflation_rate", 0.03),
                "return_rate": params.get("return_rate", 0.07),
                "income_replacement": params.get("income_replacement", 0.80)
            },
            steps=[],
            results={},
            assumptions={
                "inflation_rate": 0.03,
                "return_rate": 0.07,
                "income_replacement": 0.80
            },
            formulas={
                "future_annual_need": "current_income * income_replacement * (1 + inflation)^years",
                "target_amount": "future_annual_need * 25",  # 4% rule
                "monthly_savings": "(target - current_savings * (1 + return)^years) * (r/12) / ((1 + r/12)^(n*12) - 1)"
            }
        )

        # Calculate years to retirement using actual current age
        years_to_retirement = context.inputs["retirement_age"] - context.inputs["current_age"]
        
        # Calculate retirement needs
        target_amount = self.calculator.calculate_retirement_needs(
            context.inputs["current_age"],  # Use actual current age
            context.inputs["retirement_age"],
            context.inputs["income"],
            context.inputs["income_replacement"],
            context.inputs["inflation_rate"]
        )
        
        monthly_needed = self.calculator.calculate_monthly_savings_needed(
            target_amount,
            context.inputs["savings"],
            years_to_retirement,
            context.inputs["return_rate"]
        )
        
        future_value = self.calculator.calculate_future_value(
            context.inputs["savings"],
            monthly_needed,
            years_to_retirement,
            context.inputs["return_rate"]
        )

        # Update results with actual values
        context.results.update({
            "target_amount": target_amount,
            "monthly_savings": monthly_needed,
            "years_to_retirement": years_to_retirement,
            "future_value": future_value
        })

        response = self.prompts.format_response({
            **context.inputs,  # This now includes the updated parameters
            **context.results,
            "steps": [
                f"Years to retirement: {years_to_retirement}",
                f"Target retirement amount: ${target_amount:,.2f}",
                f"Required monthly savings: ${monthly_needed:,.2f}"
            ]
        })
        
        return {
            **state,
            "current_agent": "retirement",
            "calculation_context": context,
            "analysis": FinancialAnalysis(
                metrics=context.results,
                assessment=f"Need ${target_amount:,.2f} for retirement in {years_to_retirement} years",
                required_actions=[
                    f"Save ${monthly_needed:.0f} monthly",
                    "Review investment allocation",
                    "Consider tax-advantaged accounts"
                ]
            ),
            "messages": list(state["messages"]) + [AIMessage(content=response)]
        } 