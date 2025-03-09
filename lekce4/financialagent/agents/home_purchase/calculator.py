from typing import Dict, Tuple
import numpy as np

class HomePurchaseCalculator:
    @staticmethod
    def calculate_max_mortgage(monthly_income: float, monthly_debts: float, 
                             dti_ratio: float = 0.43) -> float:
        """Calculate maximum mortgage amount based on income and debts"""
        available_monthly = monthly_income * dti_ratio - monthly_debts
        return (available_monthly * 12) * 25  # 25-year mortgage approximation

    @staticmethod
    def calculate_down_payment_timeline(target_house_price: float, monthly_savings: float, 
                                     current_savings: float, min_down_percent: float = 0.20) -> int:
        """Calculate months needed to save for down payment"""
        down_payment_needed = target_house_price * min_down_percent
        remaining_needed = max(0, down_payment_needed - current_savings)
        return int(np.ceil(remaining_needed / monthly_savings))

    @staticmethod
    def calculate_monthly_payment(loan_amount: float, rate: float, years: int = 30) -> float:
        """Calculate monthly mortgage payment"""
        monthly_rate = rate / 12
        n_payments = years * 12
        return loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1) 