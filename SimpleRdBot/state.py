_memory = {}


def get_state(user_id: str) -> dict:
    return _memory.get(user_id, {
        "user_id": user_id,
        "current_course": "AI Chatbots",
        "progress": 67
    })


def update_state(user_id: str, state: dict):
    _memory[user_id] = state