from typing import Dict, Optional, List
from state import State, HumanityState
from humanity.emotional_sensitivity import EmotionalSensitivityModule
from humanity.formality_matcher import FormalityMatcher
from humanity.self_disclosure import SelfDisclosureModule
from humanity.consistency_tracker import ConsistencyTracker
from humanity.adaptive_transition import AdaptiveTransitionModule
from humanity.micro_references import MicroReferencesModule
from humanity.language_style import LanguageStyleModule
from humanity.cognitive_load import CognitiveLoadModule
from humanity.engagement_tracker import EngagementTracker
from humanity.question_quality import QuestionQualityControl
from humanity.uncertainty_handler import UncertaintyHandler
from humanity.reflection_generator import ReflectionGenerator
from langchain.schema import SystemMessage
import json

class HumanityManager:
    def __init__(self, model):
        self.model = model
        self.emotional = EmotionalSensitivityModule()
        self.formality = FormalityMatcher()
        self.self_disclosure = SelfDisclosureModule()
        self.consistency = ConsistencyTracker()
        self.transition = AdaptiveTransitionModule()
        self.references = MicroReferencesModule()
        self.language = LanguageStyleModule()
        self.cognitive = CognitiveLoadModule()
        self.engagement = EngagementTracker()
        self.question_quality = QuestionQualityControl()
        self.uncertainty = UncertaintyHandler()
        self.reflection = ReflectionGenerator()

    def process_response(self, state: Dict, response: str) -> Dict:
        """Process response through humanity modules."""
        humanity = state["humanity"]
        
        # Update emotional state
        emotional_analysis = self._analyze_emotional_content(response)
        humanity["emotional"].update({
            "emotional_weight": emotional_analysis["emotional_weight"],
            "trauma_indicators": emotional_analysis["trauma_indicators"],
            "emotional_cues": emotional_analysis["emotional_cues"],
            "key_emotions": emotional_analysis["key_emotions"],
            "requires_support": emotional_analysis["requires_support"]
        })
        
        # Update cognitive state
        humanity["cognitive"].update({
            "current_load": self._assess_cognitive_load(response),
            "response_lengths": humanity["cognitive"]["response_lengths"] + [len(response.split())],
            "fatigue_indicators": humanity["cognitive"]["fatigue_indicators"] + self._count_fatigue_indicators(response)
        })
        
        # Update engagement
        humanity["engagement"].update({
            "engagement_level": self._calculate_engagement(response),
            "short_answers_count": humanity["engagement"]["short_answers_count"] + (1 if len(response.split()) < 10 else 0)
        })
        
        # Update self-disclosure
        humanity["self_disclosure"].update({
            "disclosure_level": self._assess_disclosure(response),
            "personal_shares": humanity["self_disclosure"]["personal_shares"] + [{"content": response}]
        })
        
        # Update formality
        humanity["formality"].update({
            "current_level": self._assess_formality(response),
            "ty_vy_ratio": self._calculate_ty_vy_ratio(response)
        })
        
        # Update persona
        humanity["persona"].update({
            "current_persona": self._determine_persona(response),
            "sentiment_trend": self._calculate_sentiment(response)
        })
        
        return {**state, "humanity": humanity}

    def enhance_question(self, state: Dict, question: str) -> str:
        """Enhance question based on humanity state."""
        humanity = state["humanity"]
        
        # Check cognitive load
        if humanity["cognitive"]["current_load"] == "HIGH":
            question = self._simplify_question(question)
        
        # Adjust formality
        if humanity["formality"]["current_level"] == "informal":
            question = self._make_informal(question)
        elif humanity["formality"]["current_level"] == "formal":
            question = self._make_formal(question)
        
        # Add emotional support if needed
        if humanity["emotional"]["requires_support"]:
            question = self._add_emotional_support(question, humanity["emotional"]["key_emotions"])
        
        # Adjust engagement style using detailed metrics
        engagement_analysis = self._calculate_engagement(state.get("user_message", ""))
        if isinstance(engagement_analysis, dict) and engagement_analysis.get("overall_score", 1.0) < 0.5:
            question = self._make_more_engaging(question, engagement_analysis)
        
        # Match persona
        if humanity["persona"]["current_persona"] == "peer":
            question = self._adapt_to_peer(question)
        elif humanity["persona"]["current_persona"] == "curious_learner":
            question = self._adapt_to_learner(question)
        
        return question

    def _simplify_question(self, question: str) -> str:
        """Simplify a question when cognitive load is high."""
        prompt = f"""Simplify this complex question while maintaining its core meaning:
        {question}
        
        Rules:
        - Break into shorter sentences if needed
        - Use simpler vocabulary
        - Remove subordinate clauses
        - Keep it in Czech
        - Maintain the key information
        - Add clarifying context if needed
        
        Return the simplified version."""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return result.content.strip()

    def _incorporate_reference(self, question: str, reference: str) -> str:
        """Incorporate a micro-reference into the question naturally."""
        prompt = f"""Incorporate this reference into the question naturally:
        
        Question: {question}
        Reference to include: {reference}
        
        Rules:
        - Make it feel natural and conversational
        - Don't make it obvious you're referencing
        - Keep the flow smooth
        - Maintain the original question's intent
        - Keep it in Czech
        
        Return the enhanced question."""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return result.content.strip()

    def _incorporate_style(self, question: str, style: Dict) -> str:
        """Incorporate language style elements into the question."""
        style_elements = json.dumps(style, ensure_ascii=False)
        prompt = f"""Adapt this question to match the specified language style:
        
        Question: {question}
        Style elements: {style_elements}
        
        Rules:
        - Match the metaphor usage pattern
        - Use similar emotional expressions
        - Maintain consistent formality
        - Keep the meaning intact
        - Keep it in Czech
        
        Return the styled question."""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return result.content.strip()

    def get_contradiction_prompt(self, state: State) -> Optional[str]:
        """Get a prompt to resolve contradictions if any exist."""
        if state["humanity"].consistency.contradictions:
            contradiction = state["humanity"].consistency.contradictions[-1]
            return self.consistency.generate_clarification_prompt(contradiction)
        return None

    def check_repetition(self, question: str, state: State) -> bool:
        """Check if question is too similar to recent questions."""
        recent_questions = [
            exchange.get("current_question", "")
            for exchange in state.conversation_history[-5:]
        ]
        
        for prev_question in recent_questions:
            if self._calculate_similarity(question, prev_question) > 0.6:
                return True
        return False

    def _make_more_engaging(self, question: str, engagement_state: Dict) -> str:
        """Make the question more engaging based on specific engagement metrics."""
        dimensions = engagement_state.get("dimensions", {})
        
        prompt = f"""Enhance this question's engagement based on these specific metrics:
        {question}

        Current engagement metrics:
        - Elaboration: {dimensions.get('elaboration', 0.0)} (how detailed their responses are)
        - Personal Investment: {dimensions.get('investment', 0.0)} (emotional/personal stake)
        - Interactive Elements: {dimensions.get('interactive', 0.0)} (conversation reciprocity)
        - Topic Depth: {dimensions.get('depth', 0.0)} (exploration of subject)
        - Follow-up Hooks: {dimensions.get('hooks', 0.0)} (conversation continuity)
        - Narrative Richness: {dimensions.get('richness', 0.0)} (story/example sharing)
        - Question Engagement: {dimensions.get('question_engagement', 0.0)} (direct answer quality)
        - Conversational Flow: {dimensions.get('flow', 0.0)} (natural dialogue)
        - Self-initiated Elaboration: {dimensions.get('self_initiated', 0.0)} (voluntary expansion)

        Adapt the question by:
        {self._get_engagement_strategies(dimensions)}
        
        Rules:
        - Keep it in Czech
        - Maintain professional tone
        - Preserve core question intent
        - Make adjustments natural and subtle
        
        Return the enhanced question."""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return result.content.strip()

    def _get_engagement_strategies(self, dimensions: Dict) -> str:
        """Generate specific strategies based on engagement metrics."""
        strategies = []
        
        if dimensions.get('elaboration', 0.0) < 0.4:
            strategies.append("- Add specific details they can elaborate on")
        
        if dimensions.get('investment', 0.0) < 0.4:
            strategies.append("- Include personal relevance hooks")
        
        if dimensions.get('interactive', 0.0) < 0.4:
            strategies.append("- Add interactive elements that invite dialogue")
        
        if dimensions.get('depth', 0.0) < 0.4:
            strategies.append("- Include prompts for deeper exploration")
        
        if dimensions.get('hooks', 0.0) < 0.4:
            strategies.append("- Add natural follow-up opportunities")
        
        if dimensions.get('richness', 0.0) < 0.4:
            strategies.append("- Invite storytelling or examples")
        
        if dimensions.get('question_engagement', 0.0) < 0.4:
            strategies.append("- Make the question more direct and clear")
        
        if dimensions.get('flow', 0.0) < 0.4:
            strategies.append("- Improve conversational naturalness")
        
        if dimensions.get('self_initiated', 0.0) < 0.4:
            strategies.append("- Add open-ended elements that encourage voluntary expansion")
        
        # If all dimensions are low, add general engagement strategies
        if not strategies:
            strategies = [
                "- Make the question more personally relevant",
                "- Add specific details to respond to",
                "- Include gentle prompts for examples",
                "- Improve conversational flow",
                "- Add natural follow-up hooks"
            ]
        
        return "\n".join(strategies)

    def _make_informal(self, question: str) -> str:
        """Make the question more informal."""
        prompt = f"""Make this question more informal in Czech:
        {question}
        
        Use:
        - More casual language
        - "Ty" form instead of "Vy"
        - Natural expressions"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return result.content.strip()

    def _make_formal(self, question: str) -> str:
        """Make the question more formal."""
        prompt = f"""Make this question more formal in Czech:
        {question}
        
        Use:
        - Professional language
        - "Vy" form
        - Respectful tone"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return result.content.strip()

    def _add_emotional_support(self, question: str, emotions: List[str]) -> str:
        """Add emotional support to the question."""
        emotions_str = ", ".join(emotions)
        prompt = f"""Add subtle emotional support to this question while acknowledging these emotions: {emotions_str}
        
        Question: {question}
        
        Rules:
        - Keep it in Czech
        - Be empathetic but not patronizing
        - Don't make assumptions
        - Keep it professional"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return result.content.strip()

    def _adapt_to_peer(self, question: str) -> str:
        """Adapt question to peer persona."""
        prompt = f"""Adapt this question to sound more like a peer/colleague:
        {question}
        
        Rules:
        - Keep it in Czech
        - Use collaborative language
        - Show shared understanding
        - Keep professional boundaries"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return result.content.strip()

    def _adapt_to_learner(self, question: str) -> str:
        """Adapt question to curious learner persona."""
        prompt = f"""Adapt this question to sound more like a curious learner:
        {question}
        
        Rules:
        - Keep it in Czech
        - Show genuine curiosity
        - Acknowledge their expertise
        - Keep it respectful"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return result.content.strip()

    def _analyze_emotional_content(self, response: str) -> Dict:
        """Analyze emotional content of response."""
        prompt = f"""Perform detailed emotional content analysis of this response:
        {response}
        
        Return JSON with these fields:
        {{
            "emotional_weight": float,  # 0-1 scale
            "trauma_indicators": bool,
            "emotional_cues": List[str],  # verbal and contextual cues
            "key_emotions": List[str],  # primary emotions detected
            "requires_support": bool,
            "emotional_complexity": float,  # 0-1 scale
            "defense_mechanisms": List[str],  # observed psychological defenses
            "emotional_patterns": {{
                "escalation": bool,
                "suppression": bool,
                "ambivalence": bool
            }},
            "vulnerability_level": float,  # 0-1 scale
            "emotional_regulation": {{
                "level": float,  # 0-1 scale
                "strategies": List[str]
            }}
        }}"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            return json.loads(result.content.strip())
        except json.JSONDecodeError:
            return {
                "emotional_weight": 0.0,
                "trauma_indicators": False,
                "emotional_cues": [],
                "key_emotions": [],
                "requires_support": False,
                "emotional_complexity": 0.0,
                "defense_mechanisms": [],
                "emotional_patterns": {
                    "escalation": False,
                    "suppression": False,
                    "ambivalence": False
                },
                "vulnerability_level": 0.0,
                "emotional_regulation": {
                    "level": 0.5,
                    "strategies": []
                }
            }

    def _assess_cognitive_load(self, response: str) -> float:
        """Assess cognitive load from response."""
        prompt = f"""Analyze cognitive load indicators in this response:
        {response}
        
        Consider these factors:
        - Sentence complexity (structure, length)
        - Vocabulary sophistication
        - Response organization
        - Hesitation markers
        - Self-corrections
        - Processing time indicators
        - Memory access patterns
        - Abstract thinking level
        - Task switching indicators
        - Mental effort expressions
        
        Return a float 0-1 representing cognitive load level."""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            return float(result.content.strip())
        except ValueError:
            return 0.5

    def _calculate_engagement(self, response: str) -> float:
        """Calculate engagement level from response."""
        prompt = f"""Analyze engagement level in this response:
        {response}
        
        Consider these dimensions:
        - Response elaboration (0-1)
        - Personal investment (0-1)
        - Emotional involvement (0-1)
        - Interactive elements (0-1)
        - Topic exploration depth (0-1)
        - Follow-up hooks (0-1)
        - Narrative richness (0-1)
        - Question engagement (0-1)
        - Conversational flow (0-1)
        - Self-initiated elaboration (0-1)
        
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
            "overall_score": float  # weighted average
        }}"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            analysis = json.loads(result.content.strip())
            return analysis["overall_score"]
        except (json.JSONDecodeError, KeyError):
            return 0.5

    def _assess_disclosure(self, response: str) -> float:
        """Assess level of self-disclosure in response."""
        prompt = f"""Assess self-disclosure level in this response:
        {response}
        
        Consider:
        - Personal details shared
        - Emotional openness
        - Vulnerability level
        
        Return a float 0-1"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return float(result.content.strip())

    def _assess_formality(self, response: str) -> str:
        """Assess formality level of response."""
        prompt = f"""Assess formality level in this response:
        {response}
        
        Return one of: "formal", "semi_formal", "informal" """
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return result.content.strip()

    def _calculate_ty_vy_ratio(self, response: str) -> float:
        """Calculate ratio of informal to formal pronouns."""
        ty_forms = len([w for w in response.lower().split() if w in ["ty", "tebe", "tobě", "tebou"]])
        vy_forms = len([w for w in response.lower().split() if w in ["vy", "vás", "vám", "vámi"]])
        return ty_forms / (vy_forms + 1)  # Add 1 to avoid division by zero

    def _determine_persona(self, response: str) -> str:
        """Determine appropriate persona based on response."""
        prompt = f"""Determine best interviewer persona for this response:
        {response}
        
        Return one of: "researcher", "peer", "curious_learner" """
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return result.content.strip()

    def _calculate_sentiment(self, response: str) -> float:
        """Calculate sentiment trend from response."""
        prompt = f"""Calculate sentiment value from this response:
        {response}
        
        Return float -1 to 1"""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        return float(result.content.strip())

    def _count_fatigue_indicators(self, response: str) -> int:
        """Count indicators of cognitive fatigue in response."""
        indicators = [
            "nevím",
            "už si nepamatuji",
            "je to složité",
            "těžko říct",
            "je to náročné",
            "musím přemýšlet"
        ]
        return sum(1 for ind in indicators if ind in response.lower())

    def _calculate_similarity(self, question1: str, question2: str) -> float:
        """Calculate semantic similarity between two questions."""
        prompt = f"""Calculate the semantic similarity between these two questions:
        
        Question 1: {question1}
        Question 2: {question2}
        
        Consider:
        - Core meaning
        - Key concepts
        - Intent
        - Information sought
        
        Return a float 0-1 representing similarity."""
        
        result = self.model.invoke([SystemMessage(content=prompt)])
        try:
            return float(result.content.strip())
        except ValueError:
            return 0.0 