from typing import Dict
import numpy as np

class RetirementCalculator:
    @staticmethod
    def calculate_retirement_needs(current_age: int, retirement_age: int, 
                                 current_income: float, income_replacement: float = 0.8,
                                 inflation_rate: float = 0.03) -> float:
        """Calculate total amount needed for retirement"""
        years_to_retirement = retirement_age - current_age
        future_annual_need = current_income * income_replacement * ((1 + inflation_rate) ** years_to_retirement)
        return future_annual_need * 25  # 4% withdrawal rule

    @staticmethod
    def calculate_monthly_savings_needed(target_amount: float, current_savings: float, 
                                      years: int, return_rate: float = 0.07) -> float:
        """Calculate required monthly savings to reach retirement goal"""
        r = return_rate / 12
        n = years * 12
        future_savings = current_savings * (1 + return_rate) ** years
        monthly_payment = (target_amount - future_savings) * r / ((1 + r)**n - 1)
        return monthly_payment

class DebtFreeCalculator:
    @staticmethod
    def calculate_optimal_payoff(debts: Dict[str, Dict[str, float]], 
                               available_monthly: float) -> Dict[str, float]:
        """Calculate optimal debt payoff schedule using avalanche method"""
        sorted_debts = sorted(
            [(name, info["rate"], info["balance"]) 
             for name, info in debts.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        schedule = {}
        remaining = available_monthly
        
        for name, rate, balance in sorted_debts:
            min_payment = balance * 0.02
            if name == sorted_debts[0][0]:
                schedule[name] = min(remaining, balance)
            else:
                schedule[name] = min_payment
                remaining -= min_payment
        
        return schedule

class HomePurchaseCalculator:
    @staticmethod
    def calculate_max_mortgage(monthly_income: float, monthly_debts: float, 
                             dti_ratio: float = 0.43) -> float:
        available_monthly = monthly_income * dti_ratio - monthly_debts
        return (available_monthly * 12) * 25

    @staticmethod
    def calculate_down_payment_timeline(target_house_price: float, monthly_savings: float, 
                                     current_savings: float, min_down_percent: float = 0.20) -> int:
        down_payment_needed = target_house_price * min_down_percent
        remaining_needed = max(0, down_payment_needed - current_savings)
        return int(np.ceil(remaining_needed / monthly_savings)) 