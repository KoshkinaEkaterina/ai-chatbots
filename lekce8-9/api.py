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

        conversation_id = request.conversation_id or str(uuid.uuid4())
        logger.info(f"Using conversation ID: {conversation_id}")
        
        state = chat_state.get_or_create_conversation(conversation_id)
        state["last_query"] = request.message
        
        try:
            # Process the message
            state = await product_agent.process_message(request.message, state)
            
            # Get the products directly from state
            products = state.get("scored_products", [])
            logger.info(f"Found {len(products)} products")
            
            # Get the last assistant message
            messages = state.get("messages", [])
            last_message = next((msg["content"] for msg in reversed(messages) 
                               if msg["role"] == "assistant"), "No response generated")
            
            response = ChatResponse(
                message=last_message,
                conversation_id=conversation_id,
                products=products[:5],  # Take top 5 products
                criteria=state.get("current_criteria", {})
            )
            
            logger.info("=== SENDING RESPONSE ===")
            logger.debug(f"Response object: {response.dict()}")
            
            return response
            
        except Exception as e:
            logger.exception("Error in product agent processing")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
        
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
