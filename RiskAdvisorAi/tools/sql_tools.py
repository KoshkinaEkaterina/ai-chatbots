import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import base64

DB_PATH = "data/fake_db.db"

def get_var_by_date(date: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT var_value FROM historical_var WHERE date = ?", (date,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return float(row[0])
    return None

def get_allowed_products(desk: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT product FROM allowed_products WHERE desk = ?", (desk,))
    rows = cursor.fetchall()
    conn.close()

    return [r[0] for r in rows]

def get_var_trend_chart():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT date, var_value FROM historical_var ORDER BY date ASC")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    dates = [r[0] for r in rows]
    values = [r[1] for r in rows]

    # Create plot
    plt.figure(figsize=(6, 3))
    plt.plot(dates, values, marker='o', color='steelblue')
    plt.title("VaR Trend")
    plt.xlabel("Date")
    plt.ylabel("VaR (mil. EUR)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save as base64
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return image_base64