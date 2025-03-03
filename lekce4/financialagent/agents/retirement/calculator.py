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

    @staticmethod
    def calculate_future_value(current_savings: float, monthly_contribution: float,
                             years: int, return_rate: float = 0.07) -> float:
        """Calculate future value of savings with regular contributions"""
        r = return_rate / 12
        n = years * 12
        future_savings = current_savings * (1 + return_rate) ** years
        future_contributions = monthly_contribution * ((1 + r)**n - 1) / r
        return future_savings + future_contributions 