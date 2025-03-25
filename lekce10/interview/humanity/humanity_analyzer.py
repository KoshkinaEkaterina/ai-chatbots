from typing import Dict, List, Optional
from langchain.schema import SystemMessage
import json
from pydantic import BaseModel, Field, conlist, ValidationError
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

class EmotionalEvidence(BaseModel):
    quotes: List[str] = Field(
        ...,
        description="Direct quotes from the text showing emotional content",
        min_items=1
    )
    context: str = Field(
        ...,
        description="Explanation of why this content is emotionally significant"
    )
    severity_level: str = Field(
        ...,
        description="Severity level of emotional content",
        pattern="^(critical|severe|moderate|light)$"
    )

class EmotionalAnalysis(BaseModel):
    emotional_weight: float = Field(
        ...,
        description="Emotional intensity on 0-1 scale. MUST be >0.7 for severe content",
        ge=0.0,
        le=1.0
    )
    trauma_indicators: bool = Field(
        ...,
        description="Whether trauma/crisis indicators are present"
    )
    emotional_cues: List[str] = Field(
        ...,
        description="Specific phrases or indicators showing emotional content",
        min_items=1
    )
    key_emotions: List[str] = Field(
        ...,
        description="Primary emotions detected in the response",
        min_items=1
    )
    requires_support: bool = Field(
        ...,
        description="Whether emotional support is needed based on content"
    )
    emotional_complexity: float = Field(
        ...,
        description="Complexity of emotional expression on 0-1 scale",
        ge=0.0,
        le=1.0
    )
    evidence: EmotionalEvidence = Field(
        ...,
        description="Supporting evidence for emotional analysis"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "emotional_weight": 0.9,
                "trauma_indicators": True,
                "emotional_cues": ["feeling of failure", "loss expression"],
                "key_emotions": ["grief", "guilt"],
                "requires_support": True,
                "emotional_complexity": 0.8,
                "evidence": {
                    "quotes": ["Direct quote showing emotion"],
                    "context": "Why this is emotionally significant",
                    "severity_level": "critical"
                }
            }
        }

class HumanityAnalyzer:
    def __init__(self, model):
        self.model = model

    def analyze_emotional_content(self, response: str) -> Dict:
        """Analyze emotional content with focus on severe/traumatic content."""
        print("\nDEBUG: Starting emotional content analysis...")
        
        # Create parser
        parser = PydanticOutputParser(pydantic_object=EmotionalAnalysis)
        
        # Create prompt template
        prompt = PromptTemplate(
            template="""Analyze the emotional weight and trauma in this response. Pay special attention to:

            RESPONSE TO ANALYZE: {response}

            CRITICAL INDICATORS:
            - Death, suicide, violence
            - Severe trauma or crisis
            - Strong emotional impact
            - Personal involvement
            - Distressing situations

            EMOTIONAL MARKERS:
            - Direct mentions of emotions
            - Descriptive emotional language
            - Impact statements
            - Personal reactions

            {format_instructions}""",
            input_variables=["response"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        try:
            # Get and parse response
            result = self.model.invoke([SystemMessage(content=prompt.format(response=response))])
            analysis = parser.parse(result.content)
            
            # Force high emotional weight for critical content
            if analysis.evidence.severity_level == "critical":
                analysis.emotional_weight = max(0.8, analysis.emotional_weight)
                analysis.requires_support = True
                analysis.trauma_indicators = True
                
            print(f"\nDEBUG: Parsed and validated analysis: {analysis.model_dump_json(indent=2)}")
            return analysis.model_dump()
            
        except ValidationError as e:
            print(f"\nDEBUG: Validation error: {str(e)}")
            return self._get_default_emotional_analysis()

    def _get_default_emotional_analysis(self) -> Dict:
        """Return default emotional analysis structure with valid data."""
        return EmotionalAnalysis(
            emotional_weight=0.0,
            trauma_indicators=False,
            emotional_cues=["standard response"],  # Ensure non-empty list
            key_emotions=["neutral"],  # Ensure non-empty list
            requires_support=False,
            emotional_complexity=0.0,
            evidence=EmotionalEvidence(
                quotes=["-"],  # Ensure non-empty list
                context="No emotional content detected",
                severity_level="light"
            )
        ).model_dump()

    def assess_cognitive_load(self, response: str) -> Dict:
        """Assess cognitive load from response."""
        prompt = f"""Analyze cognitive load indicators in this response:
        {response}
        
        Return JSON with:
        {{
            "current_load": float,  # 0-1 scale
            "complexity_indicators": List[str],
            "processing_patterns": {{
                "hesitations": int,
                "self_corrections": int,
                "memory_access_issues": bool
            }},
            "mental_effort_level": float,  # 0-1 scale
            "fatigue_indicators": int
        }}"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            return json.loads(result.content.strip())
        except json.JSONDecodeError:
            return {
                "current_load": 0.5,
                "complexity_indicators": [],
                "processing_patterns": {
                    "hesitations": 0,
                    "self_corrections": 0,
                    "memory_access_issues": False
                },
                "mental_effort_level": 0.5,
                "fatigue_indicators": 0
            }

    def calculate_engagement(self, response: str) -> Dict:
        """Calculate engagement metrics from response."""
        prompt = f"""Analyze engagement level in this response:
        {response}
        
        Return JSON:
        {{
            "dimensions": {{
                "elaboration": float,
                "investment": float,
                "emotional": float,
                "interactive": float,
                "depth": float,
                "hooks": float,
                "richness": float,
                "question_engagement": float,
                "flow": float,
                "self_initiated": float
            }},
            "overall_score": float,
            "engagement_patterns": {{
                "consistent_engagement": bool,
                "topic_interest": float,
                "interaction_quality": float
            }}
        }}"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            return json.loads(result.content.strip())
        except json.JSONDecodeError:
            return {
                "dimensions": {k: 0.5 for k in [
                    "elaboration", "investment", "emotional", "interactive",
                    "depth", "hooks", "richness", "question_engagement",
                    "flow", "self_initiated"
                ]},
                "overall_score": 0.5,
                "engagement_patterns": {
                    "consistent_engagement": True,
                    "topic_interest": 0.5,
                    "interaction_quality": 0.5
                }
            }

    def assess_formality(self, response: str) -> Dict:
        """Assess formality level and patterns in response."""
        prompt = f"""Analyze formality level in this response:
        {response}
        
        Return JSON:
        {{
            "current_level": str,  # "formal", "semi_formal", or "informal"
            "ty_vy_ratio": float,
            "formality_markers": List[str],
            "style_consistency": float,  # 0-1 scale
            "register_shifts": List[Dict]
        }}"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            return json.loads(result.content.strip())
        except json.JSONDecodeError:
            return {
                "current_level": "semi_formal",
                "ty_vy_ratio": 0.5,
                "formality_markers": [],
                "style_consistency": 0.5,
                "register_shifts": []
            }

    def assess_disclosure(self, response: str) -> Dict:
        """Assess self-disclosure patterns in response."""
        prompt = f"""Analyze self-disclosure in this response:
        {response}
        
        Return JSON:
        {{
            "disclosure_level": float,  # 0-1 scale
            "personal_topics": List[str],
            "vulnerability_markers": List[str],
            "disclosure_patterns": {{
                "depth": float,
                "comfort_level": float,
                "boundary_maintenance": float
            }}
        }}"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            return json.loads(result.content.strip())
        except json.JSONDecodeError:
            return {
                "disclosure_level": 0.5,
                "personal_topics": [],
                "vulnerability_markers": [],
                "disclosure_patterns": {
                    "depth": 0.5,
                    "comfort_level": 0.5,
                    "boundary_maintenance": 0.5
                }
            }

    def determine_persona(self, response: str) -> Dict:
        """Determine appropriate persona and interaction patterns."""
        prompt = f"""Analyze appropriate interviewer persona for this response:
        {response}
        
        Return JSON:
        {{
            "current_persona": str,  # "researcher", "peer", or "curious_learner"
            "interaction_style": {{
                "formality_preference": float,
                "support_needed": float,
                "expertise_level": float
            }},
            "rapport_indicators": List[str],
            "adaptation_needed": bool
        }}"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            return json.loads(result.content.strip())
        except json.JSONDecodeError:
            return {
                "current_persona": "researcher",
                "interaction_style": {
                    "formality_preference": 0.5,
                    "support_needed": 0.5,
                    "expertise_level": 0.5
                },
                "rapport_indicators": [],
                "adaptation_needed": False
            }

    def calculate_similarity(self, text1: str, text2: str) -> Dict:
        """Calculate semantic similarity and patterns between two texts."""
        prompt = f"""Analyze similarity between these two texts:
        Text 1: {text1}
        Text 2: {text2}
        
        Return JSON:
        {{
            "similarity_score": float,  # 0-1 scale
            "shared_concepts": List[str],
            "key_differences": List[str],
            "semantic_overlap": {{
                "intent": float,
                "content": float,
                "style": float
            }}
        }}"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            return json.loads(result.content.strip())
        except json.JSONDecodeError:
            return {
                "similarity_score": 0.0,
                "shared_concepts": [],
                "key_differences": [],
                "semantic_overlap": {
                    "intent": 0.0,
                    "content": 0.0,
                    "style": 0.0
                }
            }