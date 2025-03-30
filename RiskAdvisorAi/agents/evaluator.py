def evaluate_confidence(result: dict) -> dict:
    source = result.get("source", "GPT")
    raw = result.get("raw", {})

    score = 1.0

    # 🛢 Source penalty
    if source == "GPT":
        score -= 0.4
    elif source == "Docs":
        score -= 0.2
    elif source == "SQL":
        score -= 0.1

    # 🕓 Data recency penalty (if date exists)
    if "date" in raw:
        # Optional: check how old the date is
        pass

    # 🔍 Missing values penalty
    if not raw or len(raw) == 0:
        score -= 0.3

    # 🎯 Final confidence level
    if score >= 0.8:
        level = "high"
        icon = "✅"
    elif score >= 0.5:
        level = "medium"
        icon = "⚠️"
    else:
        level = "low"
        icon = "❌"

    return {
        "score": round(score, 2),
        "level": level,
        "icon": icon
    }