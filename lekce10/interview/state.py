from typing import TypedDict, List, Dict, Optional, Set, Union
from enum import Enum
from datetime import datetime

class PersonaType(Enum):
    RESEARCHER = "researcher"
    PEER = "peer"
    CURIOUS_LEARNER = "curious_learner"

class FormalityLevel(Enum):
    FORMAL = "formal"
    SEMI_FORMAL = "semi_formal"
    INFORMAL = "informal"

class CognitiveLoad(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

class FactorInsight(TypedDict):
    answer_id: str
    content: str
    source_answer: str
    relevance_score: float
    evidence: str
    quote: str

class Topic(TypedDict):
    id: str
    question: str
    factors: Dict[str, str]
    covered_factors: Dict[str, float]
    question_attempts: Dict[str, int]
    factor_insights: Dict[str, List[Dict]]

class EmotionalState(TypedDict):
    emotional_weight: float
    trauma_indicators: bool
    emotional_cues: List[str]
    key_emotions: List[str]
    requires_support: bool
    last_emotional_acknowledgment: Optional[datetime]

class FormalityState(TypedDict):
    current_level: FormalityLevel
    formality_scores: Dict[str, float]
    ty_vy_ratio: float
    slang_count: int

class CognitiveState(TypedDict):
    response_times: List[float]
    complexity_scores: List[float]
    current_load: float
    response_lengths: List[int]
    fatigue_indicators: int

class PersonaState(TypedDict):
    current_persona: PersonaType
    persona_history: List[PersonaType]
    sentiment_trend: float
    self_disclosure_level: float

class EngagementState(TypedDict):
    engagement_level: float
    short_answers_count: int
    repetitive_phrases: Dict[str, int]
    dismissive_responses: int
    last_engagement_check: Optional[datetime]

class SelfDisclosureState(TypedDict):
    disclosure_level: float
    personal_shares: List[Dict[str, str]]
    last_acknowledgment: Optional[datetime]
    sensitive_topics: Set[str]

class ConsistencyState(TypedDict):
    memory_summary: Dict[str, List[str]]
    contradictions: List[Dict[str, str]]
    topic_mentions: Dict[str, int]
    key_statements: Dict[str, str]

class TransitionState(TypedDict):
    confidence_to_transition: float
    covered_aspects: Set[str]
    topic_exhaustion: float
    last_transition_type: Optional[str]

class ReferenceState(TypedDict):
    reference_bank: Dict[str, str]
    named_entities: Dict[str, str]
    key_moments: List[str]
    emotions_expressed: Dict[str, int]

class LanguageStyleState(TypedDict):
    metaphors: List[Dict[str, str]]
    idioms: Set[str]
    emotional_expressions: Dict[str, str]
    style_markers: Dict[str, int]

class HumanityState(TypedDict):
    emotional: EmotionalState
    cognitive: CognitiveState
    reference: ReferenceState
    formality: FormalityState
    persona: PersonaState
    engagement: EngagementState
    self_disclosure: SelfDisclosureState
    consistency: ConsistencyState
    transition: TransitionState
    language_style: LanguageStyleState

class State(TypedDict):
    current_question: Optional[str]
    user_message: Optional[str]
    conversation_history: List[Dict]
    topics: Dict[str, Topic]
    current_topic_id: str
    questions_in_topic: int
    topic_exhausted: bool
    introduction_done: bool
    interview_complete: bool
    is_complete: bool
    detected_intent: Optional[str]
    humanity: HumanityState
    messages: Optional[List]
    interview_metadata: Optional[Dict[str, Union[str, bool]]] 