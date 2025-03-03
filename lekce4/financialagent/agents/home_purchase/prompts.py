from typing import Dict

class HomePurchasePrompts:
    @staticmethod
    def create_analysis_prompt(context: Dict) -> str:
        return f"""Analyze home purchase potential based on:
        - Monthly Income: ${context['monthly_income']:,.2f}
        - Current Savings: ${context['savings']:,.2f}
        - Monthly Expenses: ${context['monthly_expenses']:,.2f}
        - Debt-to-Income Ratio: {context['dti_ratio']*100:.1f}%
        - Down Payment Required: {context['down_payment_percent']*100:.1f}%
        
        Consider:
        1. Maximum affordable home price
        2. Down payment timeline
        3. Monthly payment estimates
        4. Additional costs (taxes, insurance, maintenance)
        """

    @staticmethod
    def format_response(context: Dict) -> str:
        return f"""Home Purchase Analysis:

Current Financial Situation:
- Monthly Income: ${context['monthly_income']:,.2f}
- Available for Housing: ${context['available_monthly']:,.2f}
- Current Savings: ${context['savings']:,.2f}

Purchase Potential:
- Maximum Home Price: ${context['max_home_price']:,.2f}
- Required Down Payment: ${context['down_payment']:,.2f}
- Estimated Monthly Payment: ${context['monthly_payment']:,.2f}
- Time to Down Payment: {context['months_to_down']} months

Next Steps:
1. {context['next_steps'][0]}
2. {context['next_steps'][1]}
3. {context['next_steps'][2]}

Would you like to:
1. Adjust your target home price?
2. Explore different down payment options?
3. See how changes in income would affect affordability?""" 