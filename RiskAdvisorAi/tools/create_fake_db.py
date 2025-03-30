import sqlite3

DB_PATH = "data/fake_db.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create table for VaR
cursor.execute("""
CREATE TABLE IF NOT EXISTS historical_var (
    date TEXT PRIMARY KEY,
    var_value REAL
)
""")

# Insert sample VaR data
cursor.executemany(
    "INSERT OR REPLACE INTO historical_var (date, var_value) VALUES (?, ?)",
    [   ("2025-03-01", 1.08),
        ("2025-03-02", 1.10),
        ("2025-03-03", 1.25),
        ("2025-03-04", 1.05),
        ("2025-03-05", 1.08),
        ("2025-03-06", 1.28),
        ("2025-03-07", 1.30),
        ("2025-03-08", 1.35),
        ("2025-03-09", 1.05),
        ("2025-03-10", 1.07),
        ("2025-03-11", 1.09),
        ("2025-03-12", 1.15),
        ("2025-03-13", 1.38),
        ("2025-03-14", 1.50),
        ("2025-03-15", 1.16),
        ("2025-03-16", 1.26),
        ("2025-03-17", 1.28),
        ("2025-03-18", 1.15),
        ("2025-03-19", 1.07),
        ("2025-03-20", 1.20),
        ("2025-03-21", 1.09),
        ("2025-03-22", 1.35),
        ("2025-03-23", 1.25),
        ("2025-03-24", 1.25),
        ("2025-03-25", 1.17),
        ("2025-03-26", 1.04),
        ("2025-03-27", 1.15),
        ("2025-03-28", 1.35),
        ("2025-03-29", 1.37),
    ]
)
# Create table for allowed products
cursor.execute("""
CREATE TABLE IF NOT EXISTS allowed_products (
    desk TEXT,
    product TEXT
)
""")

# Insert sample products
cursor.executemany(
    "INSERT INTO allowed_products (desk, product) VALUES (?, ?)",
    [
        ("Desk A", "Electricity Futures"),
        ("Desk A", "German Power"),
        ("Desk A", "French Power"),
        ("Desk A", "Gas Options"),
        ("Desk A", "Dutch Gas"),
        ("Desk B", "Carbon Allowances"),
        ("Desk C", "Agricultural Futures and Options"),
        ("Desk C", "Corn Futures"),
        ("Desk C", "Milk Futures"),
        ("Desk C", "Grains Futures"),
        
        
    ]
)

conn.commit()
conn.close()

print("Fake database created.")