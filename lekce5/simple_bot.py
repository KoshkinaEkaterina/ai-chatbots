from typing import Dict
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from twilio.twiml.messaging_response import MessagingResponse
import logging
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import parse_qs
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import time
from collections import deque
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RateLimiter:
    def __init__(self, max_requests: int = 20, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = deque()

    def can_make_request(self) -> bool:
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window)
        
        # Remove old requests
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        
        # Check if we can make a new request
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

class ChatBot:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Using cheaper model
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.conversation_history = {}
        self.rate_limiter = RateLimiter(max_requests=20, time_window=60)
        self.fallback_responses = [
            "👋 Hi! I'm currently taking a short break to recharge. Try again in a minute!",
            "🔄 I'm a bit busy right now. Please send your message again shortly!",
            "⏳ Just need a moment to process everything. Mind trying again?",
            "🤔 Hmm, let me think about that... Could you ask again in a minute?"
        ]
        self.fallback_index = 0

    def get_fallback_response(self) -> str:
        response = self.fallback_responses[self.fallback_index]
        self.fallback_index = (self.fallback_index + 1) % len(self.fallback_responses)
        return response

    def get_response(self, message: str, phone_number: str) -> str:
        if not self.rate_limiter.can_make_request():
            return self.get_fallback_response()

        # Get or initialize conversation history
        if phone_number not in self.conversation_history:
            self.conversation_history[phone_number] = []
        
        history = self.conversation_history[phone_number]
        history.append({"role": "user", "content": message})
        
        try:
            # Create prompt with conversation history
            prompt = f"""You are a helpful and friendly AI assistant.
            Be concise and use emojis appropriately.
            Keep responses under 100 words.
            
            Previous conversation:
            {history[-3:]}  # Only use last 3 messages for context
            
            Current message: {message}
            """
            
            response = self.llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=message)
            ])
            
            # Store response in history
            history.append({"role": "assistant", "content": response.content})
            
            # Keep history limited to last 10 messages
            if len(history) > 10:
                history = history[-10:]
            
            return response.content

        except Exception as e:
            logging.error(f"ChatGPT error: {str(e)}")
            return self.get_fallback_response()

# Initialize FastAPI and ChatBot
app = FastAPI(title="ChatGPT WhatsApp Bot")
chat_bot = ChatBot()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/test")
async def test():
    """Test endpoint"""
    return {"message": "ChatGPT bot is running!"}

@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming WhatsApp messages"""
    try:
        # Debug logging
        logging.debug("Received webhook request")
        logging.debug(f"Headers: {request.headers}")
        
        # Get raw body and parse it
        body = await request.body()
        body_str = body.decode('utf-8')
        logging.debug(f"Raw body: {body_str}")
        
        # Parse form data manually
        form_data = parse_qs(body_str)
        logging.debug(f"Parsed form data: {form_data}")
        
        # Extract message and phone number
        message = form_data.get('Body', [''])[0].strip()
        from_number = form_data.get('From', [''])[0].replace('whatsapp:', '')
        
        logging.debug(f"Message from {from_number}: {message}")
        
        # Get ChatGPT response
        response_text = chat_bot.get_response(message, from_number)
        
        # Create response
        twiml = MessagingResponse()
        twiml.message(response_text)
        
        response = Response(content=str(twiml), media_type="application/xml")
        logging.debug(f"Sending response: {str(twiml)}")
        return response

    except Exception as e:
        logging.error(f"Webhook error: {str(e)}", exc_info=True)
        # Create error response
        twiml = MessagingResponse()
        twiml.message("🤖 Beep boop... I encountered an error. Please try again!")
        return Response(content=str(twiml), media_type="application/xml")

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "simple_bot:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
        access_log=True
    ) 