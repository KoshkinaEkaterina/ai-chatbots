from fastapi import FastAPI, Request
from pydantic import BaseModel
from SimpleRdBot.agent import process_user_message

app = FastAPI()


class UserInput(BaseModel):
    message: str
    user_id: str


@app.post("/chat")
async def chat(input: UserInput):
    response = await process_user_message(input.user_id, input.message)
    return {"response": response}


@app.get("/")
def health_check():
    return {"status": "OK"}