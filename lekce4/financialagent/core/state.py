from typing import TypedDict, List, Dict, Optional
from dataclasses import dataclass, field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

@dataclass
class CalculationContext:
    """Stores all calculation details for explanation"""
    inputs: Dict[str, any]  # All input parameters
    steps: List[str]        # Calculation steps
    results: Dict[str, any] # All calculated results
    assumptions: Dict[str, any]
    formulas: Dict[str, str]
    history: List[Dict[str, any]] = field(default_factory=list)  # Track parameter changes

@dataclass
class Strategy:
    steps: List[str]
    calculations: Dict[str, float]
    timeline: str
    monthly_targets: Dict[str, float]

@dataclass
class FinancialAnalysis:
    metrics: Dict[str, float]
    assessment: str
    required_actions: List[str]

class FinancialState(TypedDict):
    messages: List
    profile: Dict
    current_agent: Optional[str]
    domain: Optional[str]
    analysis: Optional[FinancialAnalysis]
    strategy: Optional[Strategy]
    context: Dict
    calculation_context: Optional[CalculationContext]
    parameters: Dict[str, any]  # Current parameters
    parameter_history: List[Dict[str, any]]  # Track all parameter changes 