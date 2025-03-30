from dotenv import load_dotenv
load_dotenv(".env")
import os
my_openai_api_key = os.getenv("OPENAI_API_KEY")
from fastapi import FastAPI, Request
from pydantic import BaseModel
from agents.router import route_message
from tools.sql_tools import get_var_trend_chart

app = FastAPI()
class ChatInput(BaseModel):
    message: str
    user_id: str
    role: str  # trader / risk / compliance

@app.post("/chat")
async def chat(input: ChatInput):
    response = await route_message(input.message, input.user_id, input.role)
    return {
        "response": response["text"],
        "source": response["source"],
        "confidence": response.get("confidence"),
        "chart_base64": response.get("chart_base64")  # ⬅ if present
    }

@app.get("/var-chart")
def get_chart():
    image = get_var_trend_chart()
    if not image:
        return {"error": "No data found"}
    return {"image_base64": image}