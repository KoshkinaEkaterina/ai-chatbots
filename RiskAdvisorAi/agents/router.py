from agents.doc_agent import handle_doc_query
from agents.sql_agent import handle_sql_query
from agents.gpt_agent import handle_gpt_fallback
from agents.evaluator import evaluate_confidence

async def route_message(message: str, user_id: str, role: str):
    # Naivní klasifikace záměru (můžeš později vylepšit)
    if "var" in message.lower() or "pnl" in message.lower():
        result = await handle_sql_query(message)
    elif "kde najdu" in message.lower() or "proces" in message.lower():
        result = await handle_doc_query(message)
    else:
        result = await handle_gpt_fallback(message)
    result["confidence"] = evaluate_confidence(result)
    return result