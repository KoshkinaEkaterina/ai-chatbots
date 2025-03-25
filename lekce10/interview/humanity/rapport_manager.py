from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class RapportMetrics(BaseModel):
    trust_level: float = Field(0.0, ge=0.0, le=1.0)
    comfort_level: float = Field(0.0, ge=0.0, le=1.0)
    openness: float = Field(0.0, ge=0.0, le=1.0)
    engagement: float = Field(0.0, ge=0.0, le=1.0)

class RapportManager:
    def __init__(self, model):
        self.model = model
        self.metrics = RapportMetrics()
        self.rapport_history = []

    def analyze_rapport(self, response: str) -> RapportMetrics:
        """Analyze rapport indicators in response."""
        prompt = f"""Analyze rapport indicators in this response:
        {response}
        
        Return JSON with rapport metrics (0-1 scale):
        {{
            "trust_level": float,
            "comfort_level": float,
            "openness": float,
            "engagement": float
        }}
        
        Consider:
        - Personal disclosure level
        - Emotional expression
        - Response depth
        - Interaction quality"""

        result = self.model.invoke(prompt)
        metrics = RapportMetrics.parse_raw(result.content)
        self.rapport_history.append(metrics)
        return metrics

    def get_rapport_strategy(self, metrics: RapportMetrics) -> str:
        """Get appropriate rapport building strategy."""
        if metrics.trust_level < 0.3:
            return "trust_building"
        elif metrics.comfort_level < 0.3:
            return "comfort_increasing"
        elif metrics.openness < 0.3:
            return "encouraging_disclosure"
        elif metrics.engagement < 0.3:
            return "engagement_boosting"
        return "maintain_rapport"

    def enhance_question(self, question: str, strategy: str) -> str:
        """Enhance question based on rapport strategy."""
        prompt = f"""Enhance this interview question using {strategy} strategy:
        Question: {question}
        
        Rules:
        - Keep it natural and conversational
        - Don't make it obvious you're building rapport
        - Maintain professional research context
        - Use appropriate Czech language
        - Keep the core question intent
        
        Return enhanced question in Czech."""

        result = self.model.invoke(prompt)
        return result.content.strip() 