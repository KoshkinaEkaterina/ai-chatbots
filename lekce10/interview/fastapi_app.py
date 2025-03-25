from fastapi import FastAPI, HTTPException, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.messaging_response import MessagingResponse
from pydantic import BaseModel
import logging
import os
from interview_bot import InterviewBot
import socket

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware - ALLOW EVERYTHING
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize bot
bot = InterviewBot()

class ChatRequest(BaseModel):
    message: str | None = None

@app.post("/chat")
async def chat(request: Request):
    """Handle chat messages."""
    try:
        # Debug: Log raw request
        body = await request.json()
        logger.debug(f"Received request body: {body}")
        
        # Parse request
        chat_request = ChatRequest(**body)
        logger.debug(f"Parsed request: {chat_request}")
        
        # Process with bot
        response = bot.chat(chat_request.message)
        logger.debug(f"Bot response: {response}")
        
        return response

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return {
            "response": "Omlouvám se, došlo k chybě. Zkuste to prosím znovu."
        }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "alive", "message": "Interview Bot is running"}

@app.post("/twilio")
async def twilio_webhook(request: Request):
    """Chat endpoint that uses both InterviewBot and Twilio"""
    try:
        # Get form data
        form_data = await request.form()
        message = form_data.get('Body', 'No message')
        from_number = form_data.get('From', 'Unknown')
        
        logging.info(f"Chat received message: '{message}' from {from_number}")
        
        # Process through interview bot
        result = bot.chat(message)
        
        # Create Twilio response
        resp = MessagingResponse()
        
        # Add bot's question as main response
        resp.message(result["question"])
        
        # Add coverage info in a separate message if available
        if result.get("covered_factors"):
            coverage_text = "\nPokrytí témat:"
            for factor, score in result["covered_factors"].items():
                coverage_text += f"\n- {factor}: {score:.2f}"
            resp.message(coverage_text)
        
        logging.info(f"Chat sending response: {result['question']}")
        
        return Response(
            content=str(resp),
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        resp = MessagingResponse()
        resp.message("🤖 Omlouvám se, došlo k chybě. Zkuste to prosím znovu.")
        return Response(
            content=str(resp),
            media_type="application/xml",
            status_code=500
        )

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    try:
        # Get form data
        form_data = await request.form()
        message = form_data.get('Body', 'No message')
        from_number = form_data.get('From', 'Unknown')
        
        logging.info(f"Received message: '{message}' from {from_number}")
        
        # Create response
        resp = MessagingResponse()
        resp.message(f"👋 Got your message!\nFrom: {from_number}\nMessage: {message}")
        
        return Response(
            content=str(resp),
            media_type="application/xml"
        )
        
    except Exception as e:
        logging.error(f"Error in whatsapp webhook: {str(e)}", exc_info=True)
        return {"error": "Internal server error"}, 500

@app.post("/whatsapp")
async def chat(Body: str = Form(...), From: str = Form(...)):
    result = bot.chat(Body)
    return {
        "response": result["response"],
        "question": result["question"],
        "is_complete": result["is_complete"]
    }

def find_free_port(start_port=8000, max_port=8020):
    """Find a free port starting from start_port."""
    for port in range(start_port, max_port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('0.0.0.0', port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port between {start_port} and {max_port}")

if __name__ == "__main__":
    import uvicorn
    port = find_free_port()
    print(f"\nStarting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 