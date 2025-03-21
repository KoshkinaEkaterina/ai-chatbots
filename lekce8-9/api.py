from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from product_agent import ProductAgent
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
import logging
from rich.console import Console
import uvicorn
import uuid
from rich.logging import RichHandler
from rich.traceback import install

# Load environment variables
load_dotenv()

# Setup rich traceback
install(show_locals=True)

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ProductAgent
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)
product_agent = ProductAgent(llm=llm, debug=True)

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    products: List[Dict] = []
    criteria: Dict = {}

    @validator('message')
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise HTTPException(
                status_code=400, 
                detail="Message cannot be empty"
            )
        return v.strip()

class Product(BaseModel):
    id: str
    name: str
    price: float
    category: List[str]
    description: str
    score: float
    dimensions: Dict[str, float] = {}
    weight: Optional[float] = None
    confidence: float

class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    products: List[Product] = []
    criteria: Optional[Dict] = None

class ChatState:
    def __init__(self):
        self.conversations: Dict[str, Dict] = {}

    def get_or_create_conversation(self, conversation_id: str) -> Dict:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "messages": [],
                "scored_products": [],  # Store products here
                "current_criteria": {},
                "last_query": ""
            }
        return self.conversations[conversation_id]

chat_state = ChatState()

@app.post("/chat")
async def chat(request: ChatMessage) -> ChatResponse:
    try:
        logger.info("=== NEW CHAT REQUEST ===")
        logger.info(f"Incoming request: {request.dict()}")

        if not request.message:
            logger.warning("Empty message received")
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )

        # Get or create conversation state
        conversation_id = request.conversation_id or str(uuid.uuid4())
        logger.info(f"Using conversation ID: {conversation_id}")
        
        state = chat_state.get_or_create_conversation(conversation_id)
        logger.debug(f"Current state before processing: {state}")
        
        # Add user message to state
        state["messages"].append({
            "role": "user",
            "content": request.message
        })
        logger.debug("Added user message to state")
        
        # Process the message and get updated state
        try:
            logger.info("Processing message with product agent...")
            state = await product_agent.process_message(request.message, state)
            logger.debug(f"State after processing: {state}")
        except Exception as e:
            logger.exception("Error in product agent processing")
            state["messages"].append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error processing your request."
            })
        
        # Get the latest assistant message
        latest_message = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "assistant":
                latest_message = msg["content"]
                break
        
        logger.debug(f"Latest assistant message: {latest_message[:200]}...")
        
        if not latest_message:
            logger.warning("No assistant message found in state")
            latest_message = "I apologize, but I couldn't generate a response."
        
        # Include products in response if they exist
        products = state.get("scored_products", [])
        criteria = state.get("current_criteria", {})
        
        logger.info(f"Found {len(products)} products in state")
        logger.debug(f"Current criteria: {criteria}")
        
        response = ChatResponse(
            message=latest_message,
            conversation_id=conversation_id,
            products=products,
            criteria=criteria
        )
        
        logger.info("=== SENDING RESPONSE ===")
        logger.debug(f"Response object: {response.dict()}")
        
        return response
        
    except Exception as e:
        logger.exception("!!! CRITICAL ERROR IN CHAT ENDPOINT !!!")
        logger.error(f"Error details: {str(e)}")
        logger.error(f"Request that caused error: {request.dict()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/")
def read_root():
    logger.info("Health check request received")
    return {"message": "Product Agent API is running"}

if __name__ == "__main__":
    logger.info("=== STARTING API SERVER ===")
    logger.info(f"Debug mode: {os.getenv('DEBUG', 'False')}")
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
