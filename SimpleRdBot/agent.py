from SimpleRdBot.intents import classify_intent
from SimpleRdBot.handlers import handle_intent
from SimpleRdBot.state import get_state, update_state


async def process_user_message(user_id: str, message: str) -> str:
    state = get_state(user_id)
    intent = classify_intent(message)
    state["current_intent"] = intent
    update_state(user_id, state)
    return await handle_intent(intent, state, message)