def classify_intent(message: str) -> str:
    message = message.lower()
    if "schedule" in message:
        return "ASK_SCHEDULE"
    elif "certificate" in message:
        return "REQUEST_CERTIFICATE"
    elif "payment" in message or "pay" in message:
        return "ASK_PAYMENT"
    elif "what is" in message or "explain" in message:
        return "EXPLAIN_TERM"
    elif "recommend" in message or "next course" in message:
        return "RECOMMEND_COURSE"
    elif "homework" in message or "assignment" in message:
        return "ASK_HOMEWORK_STATUS"
    elif "can't log in" in message or "login problem" in message or "trouble logging in" in message:
        return "TROUBLE_LOGIN"
    else:
        return "UNKNOWN"