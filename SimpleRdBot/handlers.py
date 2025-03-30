from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


explanations = {
    "langchain": "LangChain is a Python framework for building applications powered by language models.",
    "prompt engineering": "Prompt engineering is the practice of crafting inputs to get the best responses from LLMs.",
    "llm": "LLM stands for Large Language Model, like GPT-4.",
}


async def handle_intent(intent: str, state: dict, message: str) -> str:
    if intent == "ASK_SCHEDULE":
        return f"Your next class is on March 28 at 6:00 PM for the course {state.get('current_course', 'unknown')}."
    elif intent == "REQUEST_CERTIFICATE":
        progress = state.get("progress", 0)
        if progress >= 100:
            return "Your certificate is available in your profile."
        else:
            return "You need to complete all lessons and homework before getting the certificate."
    elif intent == "ASK_PAYMENT":
        return "You can pay for the course by card or bank transfer. More info at robotdreams.cz/payment"
    elif intent == "EXPLAIN_TERM":
        for key in explanations:
            if key in message.lower():
                return explanations[key]
        # Fallback to LLM via LangChain
        messages = [
            SystemMessage(content="You are a helpful teaching assistant for online tech courses."),
            HumanMessage(content=f"Explain the following term simply and clearly: {message}")
        ]
        response = llm(messages)
        return response.content.strip()
    elif intent == "RECOMMEND_COURSE":
        return "Based on your progress in AI Chatbots, we recommend 'AI Agents with LangChain' as your next course."
    elif intent == "ASK_HOMEWORK_STATUS":
        return "Your last assignment was submitted and is waiting to be reviewed. The next homework is due on April 2."
    elif intent == "TROUBLE_LOGIN":
        return "If you're having trouble logging in, try resetting your password at robotdreams.cz/reset. If that doesn't help, contact support@robotdreams.cz."
    else:
        return "Sorry, I didn't understand your question"