from typing import Dict, List, Tuple, Optional, TypedDict, Annotated, NotRequired, Union
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import pinecone
from pinecone import ServerlessSpec, Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
import os
from dotenv import load_dotenv
import numpy as np
from langgraph.graph import StateGraph, END
import json
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
import logging
from langchain.chains import LLMChain
from pydantic import validator

console = Console()

class Intent(BaseModel):
    """Shopping intent with weight and explanation"""
    intent: str = Field(
        description="Type of shopping intent (e.g., SEARCH, DETAILS, PURCHASE)"
    )
    weight: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this intent classification"
    )
    explanation: str = Field(
        description="Detailed explanation of why this intent was identified"
    )

class DimensionConstraint(BaseModel):
    """Single dimension constraint that can handle both direct values and min/max"""
    min: Optional[float] = Field(default=None, description="Minimum value in cm")
    max: Optional[float] = Field(default=None, description="Maximum value in cm")
    value: Optional[float] = Field(default=None, description="Exact value in cm")

class ProductCriteria(BaseModel):
    """Product search criteria and constraints"""
    price_range: Optional[Dict[str, float]] = Field(
        default=None,
        description="Price constraints in USD"
    )
    category: Optional[str] = Field(
        default=None,
        description="Product category or department"
    )
    dimensions: Optional[Dict[str, Union[DimensionConstraint, float]]] = Field(
        default=None,
        description="Size requirements with min/max constraints per dimension"
    )
    weight: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weight requirements in specified units"
    )
    explanation: str = Field(
        description="Reasoning behind the identified criteria"
    )

    @validator('dimensions')
    def validate_dimensions(cls, v):
        if v is None:
            return v
        result = {}
        for key, value in v.items():
            if isinstance(value, (int, float)):
                # Convert simple numbers to constraint objects
                result[key] = DimensionConstraint(value=float(value))
            elif isinstance(value, dict):
                # Convert dicts to constraint objects
                result[key] = DimensionConstraint(**value)
            else:
                result[key] = value
        return result

class QueryAnalysis(BaseModel):
    """Complete analysis of user's shopping query"""
    intents: List[Intent] = Field(
        description="List of identified shopping intents"
    )
    criteria: ProductCriteria = Field(
        description="Extracted product search criteria"
    )

class ProductIntent(Enum):
    # Value-focused intents
    BUDGET_CONSCIOUS = "Looking for affordable options"
    PREMIUM_QUALITY = "Seeking high-end, premium products"
    BEST_VALUE = "Wanting best value for money"
    
    # Usage-focused intents
    BEGINNER_FRIENDLY = "Suitable for beginners/entry-level"
    PROFESSIONAL_USE = "For professional/heavy use"
    CASUAL_USE = "For occasional/casual use"
    
    # Feature-focused intents
    FEATURE_RICH = "Seeking products with many features"
    SIMPLICITY = "Preferring simple, straightforward options"
    INNOVATIVE = "Looking for innovative/unique solutions"
    
    # Physical attributes
    COMPACT = "Needs to be space-efficient"
    LIGHTWEIGHT = "Needs to be easily portable"
    DURABILITY = "Must be long-lasting/durable"
    
    # Experience-focused
    EASE_OF_USE = "Must be user-friendly"
    PERFORMANCE = "Prioritizing high performance"
    RELIABILITY = "Must be dependable"
    
    # Specific needs
    GIFT = "Shopping for a gift"
    URGENT_NEED = "Need immediate solution"
    REPLACEMENT = "Looking to replace existing product"
    
    # Style/Aesthetic
    DESIGN_FOCUSED = "Emphasizing aesthetics"
    TRADITIONAL = "Preferring classic/traditional style"
    MODERN = "Seeking contemporary design"
    
    # Environmental/Social
    ECO_FRIENDLY = "Environmentally conscious choice"
    SUSTAINABLE = "Seeking sustainable options"
    ETHICAL = "Concerned with ethical production"
    
    # Special considerations
    SAFETY = "Prioritizing safety features"
    COMPATIBILITY = "Must work with existing items"

class IntentQuery:
    def __init__(self, intent: ProductIntent, weight: float = 1.0):
        self.intent = intent
        self.weight = weight
        self.query_templates = self._get_query_templates()
    
    def _get_query_templates(self) -> List[str]:
        """Get search queries that represent this intent"""
        templates = {
            ProductIntent.BUDGET_CONSCIOUS: [
                "affordable quality product with good value",
                "budget-friendly option that meets basic needs",
                "cost-effective solution without compromising quality"
            ],
            ProductIntent.PREMIUM_QUALITY: [
                "high-end premium product with superior quality",
                "luxury product with exceptional craftsmanship",
                "top-tier professional grade product"
            ],
            # ... (similar templates for other intents)
        }
        return templates.get(self.intent, ["general purpose product"])

class ConversationState(TypedDict):
    """Track the state of our shopping conversation"""
    messages: List[Dict[str, str]]  # Conversation history
    current_intents: List[Dict]  # Current shopping intents
    current_criteria: Dict  # Current product criteria
    recommended_products: List[Dict]  # Current recommendations
    satisfaction_level: float  # How satisfied user is (0-1)
    purchase_intent: bool  # Whether user wants to purchase
    needs_clarification: bool  # Whether we need more info
    conversation_stage: str  # Current stage of conversation
    filtered_products: NotRequired[List[str]]  # Products after filtering
    scored_products: NotRequired[List[Dict]]  # Products after scoring
    current_intent: NotRequired[Dict]  # Current classified intent
    identified_category: NotRequired[str]  # Identified product category

class ConversationIntent(Enum):
    SEARCH = "search_products"  # User wants to search/modify search
    PURCHASE = "make_purchase"  # User wants to buy
    DETAILS = "get_details"    # User wants more details about specific product
    IRRELEVANT = "cant_help"   # User's request is not shopping related
    
class ProductNode(BaseModel):
    """Product details with dimensions and weight"""
    id: str
    name: str
    price: float
    category: List[str]
    dimensions: Dict[str, float] = Field(
        description="Product dimensions in cm",
        default_factory=lambda: {
            "length": None,
            "width": None,
            "height": None
        }
    )
    weight: Optional[float] = Field(
        description="Product weight in kg",
        default=None
    )
    description: str = Field(description="Product description")
    score: float
    intent_matches: Dict[str, float]  # How well it matches each intent
    criteria_matches: Dict[str, bool]  # How well it meets criteria
    explanation: str  # Detailed explanation of recommendation
    confidence: float  # Confidence in recommendation (0-1)

class ProductRecommendation(BaseModel):
    """Product recommendation with explanation"""
    product_id: str = Field(description="Unique identifier of the product")
    name: str = Field(description="Product name")
    price: float = Field(description="Current price in USD")
    category: List[str] = Field(description="Category hierarchy")
    dimensions: Dict[str, float] = Field(
        description="Product dimensions in cm",
        default_factory=lambda: {
            "length": None,
            "width": None,
            "height": None
        }
    )
    weight: Optional[float] = Field(
        description="Product weight in kg",
        default=None
    )
    score: float = Field(ge=0.0, le=1.0, description="Overall match score")
    intent_matches: Dict[str, float] = Field(
        description="How well product matches each intent"
    )
    explanation: str = Field(
        description="Detailed explanation of why this product is recommended"
    )

class ProductAgent:
    def __init__(self, debug: bool = False):
        """Initialize the agent"""
        # Set up logging
        self.logger = logging.getLogger("Fiddley")
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        
        self.logger.debug("Initializing ProductAgent with debug mode ON...")
        
        # Load environment variables
        load_dotenv()
        
        # Initialize embeddings with correct deployment
        self.logger.debug("Setting up Azure OpenAI embeddings...")
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        # Initialize chat model with correct deployment
        self.logger.debug("Setting up Azure OpenAI LLM...")
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        
        self.logger.debug("Setting up Pinecone...")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX")
        
        self.logger.debug("Initializing conversation state...")
        self.state = self._init_state()
        
        self.logger.debug("Creating conversation graph...")
        self.graph = self._create_graph()
        
        self.logger.debug("Initialization complete!")
    
    def _init_state(self) -> Dict:
        """Initialize conversation state"""
        return {
            "messages": [],  # Conversation history
            "current_intent": None,  # Current classified intent
            "current_intents": [],  # List of identified intents
            "current_criteria": {},  # Current search criteria
            "scored_products": [],  # Products matching criteria
            "satisfaction_level": 0.0,  # How satisfied user is
            "purchase_intent": False,  # Whether user wants to buy
            "needs_clarification": False,  # Whether we need more info
            "conversation_stage": "start"  # Current stage of conversation
        }
    
    def _create_graph(self) -> StateGraph:
        """Create the conversation flow graph"""
        workflow = StateGraph(ConversationState)
        
        # Define nodes with new category identification step
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("score_products", self.search_and_score_products)
        workflow.add_node("generate_recommendations", self.generate_recommendations)
        
        # Update flow to include category identification
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_intent,
            {
                "search": "analyze_query",  # Changed from analyze_query
                "details": "generate_recommendations",
                "purchase": END,
                "irrelevant": END
            }
        )
        
        # Updated linear flow
        workflow.add_edge("analyze_query", "score_products")
        workflow.add_edge("score_products", "generate_recommendations")
        
        workflow.set_entry_point("classify_intent")
        
        return workflow.compile()
    
    def classify_intent(self, state: Dict) -> Dict:
        """Classify user intent"""
        self.logger.debug("Classifying user intent...")
        self.logger.debug(f"User message: {state['messages'][-1]['content']}")
        
        # Create prompt using ChatPromptTemplate instead of PromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a shopping assistant. Classify the user's intent into one of these categories:
- SEARCH: User wants to find or browse products (including by size, dimensions, weight, etc.)
- DETAILS: User wants more information about specific products
- PURCHASE: User wants to buy something
- IRRELEVANT: Not related to shopping at all

Return ONLY a JSON object in this exact format (nothing else):
{
    "intent": "SEARCH/DETAILS/PURCHASE/IRRELEVANT",
    "confidence": 0.0 to 1.0,
    "explanation": "why this intent was chosen"
}

IMPORTANT: Always return valid JSON, even if the user's message contains profanity or is rude."""),
            HumanMessage(content=state["messages"][-1]["content"])
        ])
        
        try:
            # Use direct LLM invocation instead of chain
            response = self.llm.invoke(prompt.format_messages())
            
            # Clean up response - remove any markdown formatting
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            # Parse response
            classification = json.loads(content)
            state["current_intent"] = classification
            
            return state
            
        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            # Fallback classification
            state["current_intent"] = {
                "intent": "SEARCH",
                "confidence": 1.0,
                "explanation": "Fallback classification due to error"
            }
            return state
    
    def route_intent(self, state: Dict) -> str:
        """Route to appropriate node based on classified intent"""
        return state["current_intent"]["intent"].lower()
    
    def _fallback_analysis(self, state: Dict) -> Dict:
        """Fallback analysis when parsing fails"""
        self.logger.debug("Using fallback analysis...")
        
        # Create basic analysis
        state["current_intents"] = [{
            "intent": "SEARCH",
            "weight": 1.0,
            "explanation": "Fallback search intent"
        }]
        
        state["current_criteria"] = {
            "price_range": None,
            "category": None,
            "dimensions": None,
            "weight": None,
            "explanation": "Unable to parse specific criteria"
        }
        
        return state

    def analyze_query(self, state: Dict) -> Dict:
        """Analyze query using structured schema"""
        self.logger.debug("Analyzing query for intents and criteria...")
        
        # Create prompt template with better context handling
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""Analyze the shopping query and determine if it's:
1. A refinement of previous search (updating size/price/etc. for same type of product)
2. A completely new search (different type of product)

Previous criteria: {json.dumps(state.get('current_criteria', {}), indent=2)}

Return a JSON object that matches this schema:
{json.dumps(QueryAnalysis.model_json_schema(), indent=2)}

Examples:
- "Show me smaller ones" -> Refinement, keep product type but update size
- "No, I want toys instead" -> New search, different product type
- "Make it under 10cm" -> Refinement, update size only
- "Actually, show me books" -> New search, different product type

IMPORTANT: For size constraints:
- "smaller than X" -> {{"dimensions": {{"max": X}}}}
- "larger than X" -> {{"dimensions": {{"min": X}}}}
- "width less than X" -> {{"dimensions": {{"width": X}}}}"""),
            *[AIMessage(content=m["content"]) if m["role"] == "assistant" else HumanMessage(content=m["content"]) 
              for m in state["messages"][-3:]], # Include last few messages for context
            HumanMessage(content=state["messages"][-1]["content"])
        ])
        
        try:
            # Get response from LLM
            response = self.llm.invoke(prompt.format_messages())
            
            # Clean up response content
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            # Parse and validate with Pydantic
            analysis = QueryAnalysis.model_validate_json(content)
            
            # Update state with validated data
            state["current_intents"] = [intent.model_dump() for intent in analysis.intents]
            
            # If this is a refinement, merge with existing criteria
            if state.get("current_criteria") and "refine" in analysis.criteria.explanation.lower():
                self.logger.debug("Refining existing search criteria...")
                new_criteria = analysis.criteria.model_dump()
                old_criteria = state["current_criteria"]
                
                # Merge criteria, new values override old ones
                merged = old_criteria.copy()
                for key, value in new_criteria.items():
                    if value is not None:
                        merged[key] = value
                state["current_criteria"] = merged
            else:
                # New search, use new criteria
                self.logger.debug("Using new search criteria...")
                state["current_criteria"] = analysis.criteria.model_dump()
            
            # Print debug info
            self._print_analysis_summary(analysis)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            if 'response' in locals():
                self.logger.error(f"LLM response was: {response.content}")
            
            # Fallback analysis for dimension queries
            if any(word in state["messages"][-1]["content"].lower() 
                   for word in ["cm", "meter", "inch", "foot", "smaller", "larger"]):
                state["current_criteria"] = {
                    "dimensions": {"max": 100.0},  # Default to 1 meter
                    "explanation": "Fallback for dimension query"
                }
            return state
    
    def search_and_score_products(self, state: Dict) -> Dict:
        """Search and score products using metadata filtering"""
        self.logger.debug("Searching and scoring products...")
        
        criteria = state.get("current_criteria", {})
        filter_parts = []
        filter_conditions = {}
        
        # Handle dimension filters
        if criteria.get("dimensions"):
            dims = criteria["dimensions"]
            self.logger.debug(f"Dimension criteria: {dims}")
            
            dimension_filters = []
            for dim_name, constraint in dims.items():
                if isinstance(constraint, dict):
                    conditions = {"$exists": True}
                    
                    # For "larger than X", we want ANY dimension to be larger
                    if dim_name.lower() == "min":
                        min_val = constraint.get("min") or constraint.get("value")
                        if min_val:
                            # Create an OR condition for any dimension being larger
                            dimension_filters.append({
                                "$or": [
                                    {"length": {"$gte": float(min_val)}},
                                    {"width": {"$gte": float(min_val)}},
                                    {"height": {"$gte": float(min_val)}}
                                ]
                            })
                    # For specific dimensions
                    else:
                        if constraint.get("min"):
                            conditions["$gte"] = float(constraint["min"])
                        if constraint.get("max"):
                            conditions["$lte"] = float(constraint["max"])
                        dimension_filters.append({dim_name: conditions})
            
            if dimension_filters:
                # If we have multiple filters, combine with AND
                if len(dimension_filters) > 1:
                    filter_parts.append({"$and": dimension_filters})
                else:
                    filter_parts.extend(dimension_filters)
        
        # Handle category filter - use $text search instead of regex
        if criteria.get("category"):
            category = criteria["category"].lower()
            self.logger.debug(f"Searching for category: {category}")
            filter_parts.append({
                "$or": [
                    {"category_l1": category},
                    {"category_l2": category},
                    {"category_l3": category},
                    {"category_l4": category}
                ]
            })
        
        # Handle price range filter
        if criteria.get("price_range"):
            price_range = criteria["price_range"]
            self.logger.debug(f"Price criteria: {price_range}")
            
            price_conditions = {"$exists": True}
            if price_range.get("min"):
                price_conditions["$gte"] = float(price_range["min"])
            if price_range.get("max"):
                price_conditions["$lte"] = float(price_range["max"])
            
            filter_parts.append({"price": price_conditions})
        
        # Combine all filters with AND if we have any
        if filter_parts:
            filter_conditions = {"$and": filter_parts}
        
        self.logger.debug(f"Filter conditions: {json.dumps(filter_conditions, indent=2)}")
        
        # Generate embeddings for search
        query_text = state["messages"][-1]["content"]
        query_embedding = self.embeddings.embed_query(query_text)
        
        # Search with filters
        try:
            # First check if we have any products matching the filters
            stats = self.pc.Index(self.index_name).describe_index_stats()
            self.logger.debug(f"Total products in index: {stats.total_vector_count}")
            
            results = self.pc.Index(self.index_name).query(
                vector=query_embedding,
                filter=filter_conditions,
                top_k=5,
                include_metadata=True
            )
            
            self.logger.debug(f"Found {len(results.matches)} matching products")
            if len(results.matches) == 0:
                self.logger.debug("No products found with current filters")
                # Try without dimension filters to see if that's the issue
                filter_conditions = {k: v for k, v in filter_conditions.items() 
                                  if not any(d in str(v) for d in ["length", "width", "height"])}
                self.logger.debug(f"Retrying without dimension filters: {json.dumps(filter_conditions, indent=2)}")
                results = self.pc.Index(self.index_name).query(
                    vector=query_embedding,
                    filter=filter_conditions if filter_conditions else None,
                    top_k=5,
                    include_metadata=True
                )
                self.logger.debug(f"Found {len(results.matches)} products without dimension filters")
            
            state["scored_products"] = [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            state["scored_products"] = []
        
        return state
    
    def _print_analysis_summary(self, analysis: QueryAnalysis) -> None:
        """Print a debug summary of the query analysis"""
        self.logger.debug("Query Analysis Summary:")
        for intent in analysis.intents:
            self.logger.debug(f"Intent: {intent.intent} (confidence: {intent.weight})")
            self.logger.debug(f"Explanation: {intent.explanation}")
        self.logger.debug(f"Criteria: {analysis.criteria.model_dump_json(indent=2)}")

    def generate_recommendations(self, state: Dict) -> Dict:
        """Generate recommendations using structured schema"""
        # Create messages list for chat
        messages = [
            SystemMessage(content="""You are a shopping assistant. Generate product recommendations based on the provided 
            intents, criteria, and available products. Focus on explaining why each product matches the user's needs."""),
            HumanMessage(content=f"""
            User intents: {json.dumps(state['current_intents'], indent=2)}
            Search criteria: {json.dumps(state['current_criteria'], indent=2)}
            Available products: {json.dumps(state['scored_products'], indent=2)}
            
            Generate recommendations that explain how each product matches the user's requirements.
            Focus especially on any specified dimensions, categories, or other criteria.
            """)
        ]
        
        try:
            # Get recommendations from LLM
            response = self.llm.invoke(messages)
            
            # Format the response into markdown
            recommendations = f"""Here are some products that match your requirements:

{response.content}

Would you like more details about any of these products?"""
            
            # Add to conversation
            state["messages"].append({
                "role": "assistant",
                "content": recommendations
            })
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {str(e)}")
            state["messages"].append({
                "role": "assistant",
                "content": "I apologize, but I had trouble finding matching products. Could you try rephrasing your requirements?"
            })
            return state
    
    def chat(self):
        """Main chat loop"""
        self.logger.debug("Starting chat session...")
        console.print("\n[bold blue]🤖 Hi! I'm Fiddley, your shopping assistant. How can I help you today?[/bold blue]")
        
        while True:
            user_input = input("\nYou: ").strip()
            self.logger.debug(f"Received user input: {user_input}")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print("\n[bold blue]👋 Thanks for shopping! Have a great day![/bold blue]")
                break
            
            self.logger.debug("Adding message to conversation history...")
            self.state["messages"].append({"role": "user", "content": user_input})
            
            try:
                self.logger.debug("Invoking conversation graph...")
                self.state = self.graph.invoke(dict(self.state))  # Convert to dict
                
                self.logger.debug("Finding latest assistant message...")
                latest_message = next((m for m in reversed(self.state["messages"]) 
                                    if m["role"] == "assistant"), None)
                if latest_message:
                    console.print(f"\n[bold blue]🤖 Fiddley:[/bold blue]")
                    console.print(Markdown(latest_message["content"]))
                
                if self.state["purchase_intent"]:
                    self.logger.debug("Purchase intent detected, ending conversation...")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
                console.print(f"\n[bold red]🤖 Oops! Something went wrong: {str(e)}[/bold red]")
                console.print("Let's try again!")

    def identify_category(self, state: Dict) -> Dict:
        """Identify product category from user query"""
        self.logger.debug("Identifying product category...")
        
        # Create prompt template
        prompt = PromptTemplate(
            template="""Based on the user's query, identify the most likely product category.
            
            User query: {query}
            Previous context: {context}
            
            Return a JSON object with:
            {
                "category": "identified category",
                "confidence": 0.0 to 1.0,
                "explanation": "why this category was chosen"
            }
            """,
            input_variables=["query", "context"]
        )
        
        # Create chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Get context from previous messages
        context = "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in state["messages"][:-1]
        ])
        
        # Run chain
        response = chain.run({
            "query": state["messages"][-1]["content"],
            "context": context
        })
        
        try:
            # Parse response
            result = json.loads(response)
            state["current_category"] = result["category"]
            state["category_confidence"] = result["confidence"]
            
            self.logger.debug(f"Identified category: {result['category']} (confidence: {result['confidence']})")
            self.logger.debug(f"Explanation: {result['explanation']}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Category identification failed: {str(e)}")
            state["current_category"] = None
            state["category_confidence"] = 0.0
            return state

def main():
    # Create agent with debug mode on
    agent = ProductAgent(debug=True)
    agent.chat()

if __name__ == "__main__":
    main() 