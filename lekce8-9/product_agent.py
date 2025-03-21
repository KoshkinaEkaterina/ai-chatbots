from typing import Dict, List, Optional
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import logging
from intent_classifier import IntentClassifier, UserIntent
from intent_resolver import IntentResolver
from product_recommendation import ProductRecommender
from product_classes import ConversationState, ProductNode, QueryAnalysis
import os
from dotenv import load_dotenv
from rich.console import Console
import pinecone
from rich.markdown import Markdown

logger = logging.getLogger(__name__)
console = Console()

class ProductAgent:
    def __init__(self, llm: AzureChatOpenAI, debug: bool = False):
        self.llm = llm
        self.intent_classifier = IntentClassifier(llm)
        self.intent_resolver = IntentResolver(llm)
        self.product_recommender = ProductRecommender(llm)
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            
        # Setup embeddings and vector store
        load_dotenv()
        self.embeddings = AzureOpenAIEmbeddings()
        self.pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX")
        
        # Initialize state
        self.sessions = {}

    def _init_state(self) -> Dict:
        """Initialize conversation state"""
        return {
            "messages": [],
            "current_intent": None,
            "current_intents": [],
            "current_criteria": {},
            "scored_products": [],
            "search_context": {},
            "current_product": None,
            "selected_products": [],
            "last_query": None
        }

    async def handle_message(self, message: str, state: Optional[Dict] = None) -> Dict:
        """Process user message through intent classification and resolution"""
        if not state:
            state = self._init_state()
            
        # Add message to history
        state["messages"].append({"role": "user", "content": message})
        state["last_query"] = message

        try:
            # 1. Classify intent
            intent = await self.intent_classifier.classify_intent(message)
            state["current_intent"] = intent
            logger.info(f"Classified intent: {intent}")

            # 2. For comparison intent, check if we have products
            if intent == UserIntent.COMPARISON:
                if not state.get("scored_products"):
                    state["messages"].append({
                        "role": "assistant",
                        "content": "I don't have any products to compare yet. Could you tell me what kind of products you're interested in?"
                    })
                    return state
                # Don't analyze query or search for new products if we're just comparing
                handler = getattr(self.intent_resolver, "handle_comparison")
                state = await handler(state, self.product_recommender.search_and_score_products)
                return state

            # 3. For other intents, analyze query and search products
            state = self.product_recommender.analyze_query(state)
            logger.info("Search criteria found:", state.get("current_criteria"))
            
            if intent in [UserIntent.PURCHASE_NEW, UserIntent.PURCHASE_REPLACEMENT, 
                         UserIntent.PURCHASE_UPGRADE]:
                state = self.product_recommender.search_and_score_products(state)

            # 4. Get appropriate handler for the intent
            handler_name = self.intent_classifier.get_handler_name(intent)
            if not handler_name:
                raise ValueError(f"No handler found for intent {intent}")

            # 5. Execute the handler
            handler = getattr(self.intent_resolver, handler_name)
            if handler_name in ["handle_comparison", "handle_replacement", "handle_upgrade", "handle_purchase"]:
                state = await handler(state, self.product_recommender.search_and_score_products)
            else:
                state = await handler(state)

            return state

        except Exception as e:
            logger.exception("Error handling message")
            state["messages"].append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error. Could you try rephrasing that?"
            })
            return state

    async def process_message(self, message: str, state: Dict) -> Dict:
        try:
            self.logger.info(f"Processing message: {message}")
            self.logger.debug(f"Initial state: {state}")

            # Get intent
            intent = await self.intent_classifier.classify_intent(message)
            self.logger.info(f"Classified intent: {intent}")

            # Update state with the message
            state["last_query"] = message

            # Process based on intent
            if intent == UserIntent.PURCHASE_NEW:
                self.logger.info("Processing PURCHASE_NEW intent")
                # Analyze query for search criteria
                criteria = await self.product_recommender.analyze_query(state)
                state["current_criteria"] = criteria
                
                # Search and score products - FIXED: pass state instead of criteria
                state = self.product_recommender.search_and_score_products(state)
                
                # Generate recommendations
                state = await self.product_recommender.generate_recommendations(state)

            elif intent == UserIntent.COMPARISON:
                self.logger.info("Processing COMPARISON intent")
                state = await self.intent_resolver.handle_comparison(state)

            elif intent == UserIntent.FIRST_USE:
                self.logger.info("Processing FIRST_USE intent")
                state = await self.intent_resolver.handle_first_use(state)

            elif intent == UserIntent.REQUIREMENTS:
                self.logger.info("Processing REQUIREMENTS intent")
                state = await self.intent_resolver.handle_requirements(state)

            else:
                self.logger.warning(f"Unhandled intent: {intent}")
                state["messages"].append({
                    "role": "assistant",
                    "content": "I'm not sure how to help with that. Could you rephrase your request?"
                })

            self.logger.debug(f"Final state after processing: {state}")
            return state

        except Exception as e:
            self.logger.exception("Error in process_message")
            state["messages"].append({
                "role": "assistant",
                "content": "I encountered an error processing your request. Please try again."
            })
            return state

    async def chat(self):
        """Interactive chat interface"""
        state = self._init_state()
        console.print("\n[bold blue]👋 Hi! I'm your product assistant. How can I help you today?[/]")
        
        while True:
            message = input("\nYou: ").strip()
            if message.lower() in ['quit', 'exit', 'bye']:
                console.print("\n[bold blue]👋 Goodbye! Have a great day![/]")
                break
                
            try:
                state = await self.handle_message(message, state)
                latest_message = next((m for m in reversed(state["messages"]) 
                                    if m["role"] == "assistant"), None)
                if latest_message:
                    console.print("\n[bold blue]Assistant:[/]")
                    console.print(Markdown(latest_message["content"]))
            except Exception as e:
                logger.exception("Error in chat loop")
                console.print(f"\n[bold red]Error: {str(e)}[/]")

async def main():
    load_dotenv()
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )
    agent = ProductAgent(llm=llm, debug=True)
    await agent.chat()

if __name__ == "__main__":
    asyncio.run(main())
