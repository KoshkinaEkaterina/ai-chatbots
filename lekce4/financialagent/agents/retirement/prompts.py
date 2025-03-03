from typing import Dict

class RetirementPrompts:
    @staticmethod
    def create_analysis_prompt(context: Dict) -> str:
        return f"""Analyze retirement planning based on:
        - Current Age: {context['current_age']}
        - Retirement Age: {context['retirement_age']}
        - Annual Income: ${context['income']:,.2f}
        - Current Savings: ${context['savings']:,.2f}
        - Inflation Rate: {context['inflation_rate']*100:.1f}%
        - Expected Return: {context['return_rate']*100:.1f}%
        
        Consider:
        1. Total amount needed for retirement
        2. Monthly savings required
        3. Investment strategy
        4. Tax implications
        """

    @staticmethod
    def format_response(context: Dict) -> str:
        return f"""Retirement Analysis:

Input Parameters:
- Current Age: {context['current_age']}
- Retirement Age: {context['retirement_age']}
- Annual Income: ${context['income']:,.2f}
- Current Savings: ${context['savings']:,.2f}
- Inflation Rate: {context['inflation_rate']*100:.1f}%
- Expected Return: {context['return_rate']*100:.1f}%
- Income Replacement: {context['income_replacement']*100:.1f}%

Calculations:
{chr(10).join(f'- {step}' for step in context.get('steps', []))}

Results:
- Target Retirement Amount: ${context['target_amount']:,.2f}
  (Adjusted for {context['inflation_rate']*100:.1f}% annual inflation)
- Required Monthly Savings: ${context['monthly_savings']:,.2f}
  (Assuming {context['return_rate']*100:.1f}% annual return)
- Years to Retirement: {context['years_to_retirement']}

Would you like to:
1. Adjust any of these parameters?
2. See how different returns would affect the numbers?
3. Understand how these calculations work?""" 