from typing import Dict, List, Optional, Union, TypedDict, NotRequired
from pydantic import BaseModel, Field, validator
from enum import Enum
from intent_classifier import UserIntent

class Intent(BaseModel):
    """Shopping intent with weight and explanation"""
    intent: str
    weight: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this intent classification"
    )
    explanation: str = Field(
        description="Detailed explanation of why this intent was identified"
    )

class DimensionConstraint(BaseModel):
    """Single dimension constraint that can handle both direct values and min/max"""
    min: Optional[float] = None
    max: Optional[float] = None
    value: Optional[float] = None

class WeightConstraint(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None

class PriceRange(BaseModel):
    """Price range constraints"""
    min: Optional[float] = None
    max: Optional[float] = None

class ProductCriteria(BaseModel):
    """Product search criteria and constraints"""
    price_range: PriceRange = Field(default_factory=PriceRange)
    category: Optional[str] = None
    dimensions: Dict[str, DimensionConstraint] = Field(default_factory=dict)
    weight: WeightConstraint = Field(default_factory=WeightConstraint)
    features: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    explanation: str = ""

    @validator('price_range', pre=True)
    def validate_price_range(cls, v):
        if v is None:
            return PriceRange()
        if isinstance(v, dict):
            # Handle case where price range comes as a dict
            return PriceRange(**v)
        return v

    @validator('dimensions')
    def validate_dimensions(cls, v):
        if v is None:
            return {}
        result = {}
        for key, value in v.items():
            if isinstance(value, (int, float)):
                result[key] = DimensionConstraint(value=float(value))
            elif isinstance(value, dict):
                result[key] = DimensionConstraint(**value)
            else:
                result[key] = value
        return result

class QueryAnalysis(BaseModel):
    """Complete analysis of user's shopping query"""
    intents: List[Intent] = Field(
        default_factory=list,
        description="List of identified shopping intents"
    )
    criteria: ProductCriteria = Field(
        default_factory=ProductCriteria,
        description="Extracted product search criteria"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Overall confidence in the analysis"
    )

class ProductNode(BaseModel):
    """Product details with scoring and matching"""
    id: str = Field(description="Unique product identifier")
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    category: Union[List[str], str] = Field(description="Product category")
    description: str = Field(description="Product description")
    score: float = Field(default=0.0, description="Overall match score")
    dimensions: Dict[str, float] = Field(
        default_factory=dict,
        description="Product dimensions"
    )
    weight: Optional[float] = Field(
        default=None,
        description="Product weight"
    )
    intent_matches: Dict[str, float] = Field(
        default_factory=dict,
        description="Intent matching scores"
    )
    criteria_matches: Dict[str, float] = Field(
        default_factory=dict,
        description="Criteria matching scores"
    )
    explanation: str = Field(
        default="",
        description="Recommendation explanation"
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence in recommendation"
    )

class ProductComparison(BaseModel):
    """Product comparison structure"""
    name: str
    price: float
    category: List[str]
    description: str
    dimensions: Dict[str, float] = Field(default_factory=dict)
    weight: Optional[float] = None
    score: float
    key_features: List[str] = Field(default_factory=list)
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)

class ComparisonResponse(BaseModel):
    """Structured comparison response"""
    products: List[ProductComparison] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    best_value: str = ""
    best_features: str = ""
    explanation: str = ""

class ConversationState(TypedDict):
    """Track the state of our shopping conversation"""
    messages: List[Dict[str, str]]  # Conversation history
    current_intent: Optional[UserIntent]  # Current classified intent
    current_intents: List[Dict]  # Current shopping intents
    current_criteria: Dict  # Current product criteria
    scored_products: List[Dict]  # Products after scoring
    search_context: Dict  # Search context and parameters
    current_product: Optional[Dict]  # Currently discussed product
    selected_products: List[Dict]  # Products selected for comparison
    last_query: Optional[str]  # Last user query

