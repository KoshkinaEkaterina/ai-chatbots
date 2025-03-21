from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)

class UserIntent(Enum):
    """Supported user intents that map to specific handlers"""
    
    REQUIREMENTS = "requirements and compatibility questions"
    FIRST_USE = "how to use for the first time"
    COMPARISON = "compare with similar products"
    PURCHASE_NEW = "looking to buy a new product"
    PURCHASE_REPLACEMENT = "need to replace existing product"
    PURCHASE_UPGRADE = "want to upgrade current product"
    SUPPORT = "need help with product issues"
    WARRANTY = "warranty and return questions"

class PurchaseContext(BaseModel):
    """Structured context about a purchase intent"""
    primary_need: str = Field(description="Main reason for the purchase")
    use_frequency: str = Field(description="How often product will be used")
    environment: str = Field(description="Where product will be used")
    constraints: Dict[str, str] = Field(description="Key limitations/requirements")
    preferences: Dict[str, float] = Field(description="Feature preferences with weights")
    timeline: str = Field(description="When purchase is needed")
    budget_flexibility: str = Field(description="Flexibility on price")
    experience_level: str = Field(description="User's experience with product type")

class IntentClassifier:
    """Analyzes user input to determine intent and gather context"""
    
    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        # Updated prompt to better handle our specific intents
        self.prompt = PromptTemplate(
            template="""Analyze this user query and determine their primary intent.

User Query: {query}

Available intents:
{intents}

Choose the SINGLE most appropriate intent. Consider:
- The main action the user wants to take
- The stage in the customer journey
- The type of information they need

Return JSON with:
{{
    "intent": "INTENT_NAME",
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}

You must respond with valid JSON only. No other text.""",
            input_variables=["query", "intents"]
        )
        self.chain = LLMChain(llm=llm, prompt=self.prompt)

    async def classify_intent(self, query: str) -> UserIntent:
        """Classify user intent from query"""
        try:
            prompt = f"""Analyze this user query and determine their primary intent.

User Query: "{query}"

Available intents:
- FIRST_USE: Questions about product setup, usage instructions, getting started guides
- PURCHASE_NEW: Searching for new products to purchase
- COMPARISON: Requesting comparison between products, asking about differences
- REQUIREMENTS: Questions about specifications, compatibility, or system requirements
- UPGRADE: Inquiries about upgrading or improving existing products
- SUPPORT: Technical support or troubleshooting assistance
- WARRANTY: Warranty or return policy questions
- REPLACEMENT: Seeking to replace existing products

Common patterns:
- "How do I use/setup/start" -> FIRST_USE
- "Which is better/compare" -> COMPARISON
- "Will this work with" -> REQUIREMENTS
- "Looking for new" -> PURCHASE_NEW
- "Need to replace" -> PURCHASE_REPLACEMENT
- "Want to upgrade" -> PURCHASE_UPGRADE
- "Having issues" -> SUPPORT
- "Warranty coverage" -> WARRANTY

Return only the intent name (e.g. 'FIRST_USE')."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            intent_str = response.content.strip().upper()
            
            # Clean up response
            intent_str = intent_str.replace('INTENT.', '')
            intent_str = intent_str.replace('USERINTENT.', '')
            intent_str = intent_str.split('\n')[0]  # Take only first line
            
            try:
                return UserIntent[intent_str]
            except KeyError:
                self.logger.error(f"Invalid intent returned: {intent_str}, query: {query}")
                # Smarter fallback logic
                query_lower = query.lower()
                if any(word in query_lower for word in ["how", "guide", "use", "setup", "start"]):
                    return UserIntent.FIRST_USE
                if any(word in query_lower for word in ["compare", "better", "difference"]):
                    return UserIntent.COMPARISON
                return UserIntent.PURCHASE_NEW
                
        except Exception as e:
            self.logger.exception(f"Error classifying intent for query: {query}")
            return UserIntent.PURCHASE_NEW

    def get_handler_name(self, intent: UserIntent) -> str:
        """Get the corresponding handler function name for an intent"""
        handler_map = {
            UserIntent.REQUIREMENTS: "handle_requirements",
            UserIntent.FIRST_USE: "handle_first_use",
            UserIntent.COMPARISON: "handle_comparison",
            UserIntent.PURCHASE_NEW: "handle_purchase",
            UserIntent.PURCHASE_REPLACEMENT: "handle_replacement",
            UserIntent.PURCHASE_UPGRADE: "handle_upgrade",
            UserIntent.SUPPORT: "handle_support",
            UserIntent.WARRANTY: "handle_warranty",
        }
        return handler_map.get(intent)

    async def gather_purchase_context(self, initial_query: str) -> PurchaseContext:
        """Interactive dialogue to understand purchase needs"""
        try:
            self.logger.info("\n=== Starting Purchase Dialog ===")
            self.logger.info(f"Initial query: {initial_query}")
            
            # Initialize context with defaults
            context = {
                "primary_need": initial_query,
                "use_frequency": "Unknown",
                "environment": "Not specified",
                "constraints": {},
                "preferences": {},
                "timeline": "Not specified",
                "budget_flexibility": "Not specified",
                "experience_level": "Not specified"
            }

            # Sequential questions for toys
            questions = [
                "What age is the toy for?",
                "What's your maximum budget? (Our toys range from $5 to $50)",
                "Do you prefer educational toys or purely fun toys?",
                "Any specific materials to avoid (e.g., plastic, metal)?",
                "Does size matter? We can focus on small toys under 10cm if needed.",
                "Any favorite colors or themes?"
            ]

            # Start with first question
            response_lines = [
                "I'll help you find the perfect toy. Let's start with the first question:",
                "",
                questions[0]
            ]

            # Store questions in context for follow-up
            context["pending_questions"] = questions[1:]  # Store remaining questions
            context["asked_questions"] = [questions[0]]  # Track what we've asked
            
            self.logger.info("\n=== Initial Context ===")
            for key, value in context.items():
                self.logger.info(f"{key}: {value}")

            return PurchaseContext(**context), "\n".join(response_lines)
            
        except Exception as e:
            self.logger.exception("❌ Error in gather_purchase_context")
            return PurchaseContext(
                primary_need=initial_query,
                use_frequency="Unknown",
                environment="Not specified",
                constraints={},
                preferences={},
                timeline="Not specified",
                budget_flexibility="Not specified",
                experience_level="Not specified"
            ), "I apologize, but I encountered an error. Let's start with: what age is the toy for?"

    async def _ask_question(self, question: str) -> str:
        """In real app, this would interact with user. Here we simulate."""
        return f"Simulated user response to: {question}"

    def _determine_follow_up(self, follow_ups: Dict[str, str], answer: str) -> Optional[str]:
        """Determine which follow-up question to ask based on answer"""
        # Use LLM to analyze answer and pick appropriate follow-up
        return None  # For now