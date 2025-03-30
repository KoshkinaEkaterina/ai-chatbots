from tools.sql_tools import get_var_by_date, get_allowed_products, get_var_trend_chart
import re

async def handle_sql_query(message: str) -> dict:
    msg_lower = message.lower()

    if "var" in msg_lower:
        # Extract date from message
        match = re.search(r"\d{4}-\d{2}-\d{2}", message)
        if match:
            date = match.group(0)
        else:
            date = "2025-03-14"  # default for testing

        var_value = get_var_by_date(date)
        if var_value:
            return {
                "text": f"VaR value as of {date} is {var_value:.2f} mEUR.",
                "source": "SQL",
                "raw": {"var": var_value, "date": date}
            }
        else:
            return {
                "text": f"For date {date} no values were found.",
                "source": "SQL",
                "raw": {}
            }

    elif "product" in msg_lower or "trading" in msg_lower:
        # Example: “What products my desk can trade?”
        desk = "Desk A"
        products = get_allowed_products(desk)
        return {
            "text": f"Allowed products for {desk}: {', '.join(products)}",
            "source": "SQL",
            "raw": {"desk": desk, "products": products}
        }
    elif "graf" in msg_lower or "chart" in msg_lower:
        chart_base64 = get_var_trend_chart()
        if chart_base64:
            return {
                "text": "VaR chart:",
                "source": "SQL",
                "chart_base64": chart_base64,
                "raw": {"type": "chart", "name": "VaR Trend"}
            }
        else:
            return {
                "text": "Chart is not availiable.",
                "source": "SQL",
                "raw": {}
            }

    return {
        "text": "SQL cannot answer this question",
        "source": "SQL",
        "raw": {}
    }