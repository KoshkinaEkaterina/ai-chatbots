from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv(".env", usecwd=True)
load_dotenv(dotenv_path)
import os
my_openai_api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    openai_api_key=my_openai_api_key
)

async def handle_gpt_fallback(message: str) -> dict:
    system_prompt = (
        "You are an expert AI assistant helping people in the area of Risk Management "
        "and Trading. Answer factually, clearly, and concisely—ideally in one paragraph. "
        "If you're not sure about something, it's better to admit it."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message)
    ]

    response = llm(messages)

    return {
        "text": response.content,
        "source": "GPT",
        "raw": {"model": "gpt-4"}
    }