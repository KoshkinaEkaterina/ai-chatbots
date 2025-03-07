from typing import TypedDict, Dict, List, Optional
from dataclasses import dataclass
from uuid import uuid4


@dataclass
class FactorInsight:
    """Represents a single piece of information gathered about a factor"""
    answer_id: str
    content: str
    source_answer: str
    relevance_score: float
    evidence: str
    quote: str

class Topic:
    """Manages memory for a specific topic"""
    def __init__(self, id: str, question: str):
        self.id = id
        self.question = question
        self.factors: Dict[str, str] = {}
        self.covered_factors: Dict[str, float] = {}
        self.factor_insights: Dict[str, List[Dict]] = {}
        self.factor_summaries: Dict[str, Dict] = {}
    
    def add_insight(self, factor: str, insight: FactorInsight):
        """Add a new insight for a factor"""
        if factor not in self.factor_insights:
            self.factor_insights[factor] = []
        self.factor_insights[factor].append({
            "key_info": insight.content,
            "evidence": insight.evidence,
            "quote": insight.quote,
            "score": insight.relevance_score
        })

    def get_factor_coverage(self, factor: str) -> float:
        """Get the current coverage score for a factor"""
        return self.covered_factors.get(factor, 0.0)

    def summarize_insights(self, factor: str) -> str:
        """Generate a summary of all insights for a specific factor"""
        if factor not in self.factor_insights:
            return f"No insights gathered for factor: {factor}"
        
        insights = self.factor_insights[factor]
        summary = f"\nFactor: {factor}\n"
        summary += f"Description: {self.factors[factor]}\n"
        summary += f"Current Coverage: {self.get_factor_coverage(factor):.2f}\n\n"
        
        for insight in sorted(insights, key=lambda x: x["score"], reverse=True):
            summary += f"[{insight['key_info'][:8]}] ({insight['score']:.2f})\n"
            summary += f"Content: {insight['key_info']}\n"
            summary += f"Evidence: {insight['evidence']}\n"
            summary += f"Quote: \"{insight['quote']}\"\n\n"
        
        return summary

class State(TypedDict, total=False):
    """Internal state for the interview flow."""
    current_question: Optional[str]
    user_message: Optional[str]
    conversation_history: List[Dict]
    topics: Dict[str, Topic]
    current_topic_id: str
    introduction_done: bool
    interview_complete: bool
