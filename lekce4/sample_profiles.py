import pandas as pd

# Create sample financial profiles
profiles = {
    "conservative": {
        "income": 75000,
        "monthly_expenses": {
            "housing": 2000,
            "utilities": 300,
            "food": 600,
            "transport": 400,
            "insurance": 300,
            "entertainment": 400
        },
        "savings": 15000,
        "debts": {
            "student_loan": {"balance": 20000, "rate": 0.045},
            "car_loan": {"balance": 15000, "rate": 0.039}
        },
        "risk_tolerance": "low",
        "investment_horizon": "long"
    },
    "moderate": {
        "income": 95000,
        "monthly_expenses": {
            "housing": 2500,
            "utilities": 350,
            "food": 800,
            "transport": 500,
            "insurance": 400,
            "entertainment": 600
        },
        "savings": 25000,
        "debts": {
            "mortgage": {"balance": 300000, "rate": 0.035},
            "credit_card": {"balance": 5000, "rate": 0.159}
        },
        "risk_tolerance": "medium",
        "investment_horizon": "medium"
    }
}

# Save to CSV
def save_profiles():
    # Flatten the data for CSV format
    flat_data = []
    for profile_type, data in profiles.items():
        row = {
            "profile_type": profile_type,
            "income": data["income"],
            "savings": data["savings"],
            "risk_tolerance": data["risk_tolerance"],
            "investment_horizon": data["investment_horizon"]
        }
        
        # Add expenses
        for expense, amount in data["monthly_expenses"].items():
            row[f"expense_{expense}"] = amount
            
        # Add debts
        for debt, details in data["debts"].items():
            row[f"debt_{debt}_balance"] = details["balance"]
            row[f"debt_{debt}_rate"] = details["rate"]
            
        flat_data.append(row)
    
    df = pd.DataFrame(flat_data)
    df.to_csv("financial_profiles.csv", index=False)

if __name__ == "__main__":
    save_profiles() 