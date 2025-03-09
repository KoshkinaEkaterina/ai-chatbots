from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from decimal import Decimal

class ExtractedParameter(BaseModel):
    """Represents a single extracted parameter"""
    key: str = Field(..., description="Parameter key (e.g., 'return_rate')")
    value: float = Field(..., description="Normalized parameter value")
    original_text: str = Field(..., description="Original text from user message")
    confidence: float = Field(default=1.0, description="Confidence in extraction (0-1)")

class ParameterExtractionResult(BaseModel):
    """Results of parameter extraction"""
    parameters: List[ExtractedParameter] = Field(default_factory=list)
    raw_message: str = Field(..., description="Original message")
    unmatched_text: Optional[str] = Field(None, description="Text that couldn't be parsed") 