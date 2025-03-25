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
from .humanity_analyzer import HumanityAnalyzer
import json

class HumanityManager:
    def __init__(self, model):
        self.model = model
        self.analyzer = HumanityAnalyzer(model)

    def process_response(self, state: Dict, response: str) -> Dict:
        """Process response through humanity modules and update all dimensions."""
        humanity = state["humanity"]
        
        print("\n=== STARTING HUMANITY ANALYSIS ===")
        
        # Add debug prints
        print("\nDEBUG: Calling emotional analysis...")
        emotional_analysis = self.analyzer.analyze_emotional_content(response)
        print(f"\nDEBUG: Raw emotional analysis result: {json.dumps(emotional_analysis, indent=2)}")
        
        # Update emotional state with ALL emotional metrics
        humanity["emotional"].update({
            "emotional_weight": emotional_analysis["emotional_weight"],
            "trauma_indicators": emotional_analysis["trauma_indicators"],
            "emotional_cues": emotional_analysis["emotional_cues"],
            "key_emotions": emotional_analysis["key_emotions"],
            "requires_support": emotional_analysis["requires_support"],
            "emotional_complexity": emotional_analysis.get("emotional_complexity", 0.0),
            "vulnerability_level": emotional_analysis.get("vulnerability_level", 0.0)
        })
        
        print(f"\nDEBUG: Updated humanity state emotional values: {json.dumps(humanity['emotional'], indent=2)}")
        
        # Update cognitive state with ALL cognitive metrics
        cognitive_analysis = self.analyzer.assess_cognitive_load(response)
        humanity["cognitive"].update({
            "current_load": cognitive_analysis["current_load"],
            "complexity_indicators": cognitive_analysis["complexity_indicators"],
            "processing_patterns": cognitive_analysis["processing_patterns"],
            "mental_effort_level": cognitive_analysis["mental_effort_level"],
            "fatigue_indicators": cognitive_analysis["fatigue_indicators"],
            "response_lengths": humanity["cognitive"]["response_lengths"] + [len(response.split())]
        })
        
        # Update engagement with ALL engagement dimensions
        engagement_analysis = self.analyzer.calculate_engagement(response)
        humanity["engagement"].update({
            "engagement_level": engagement_analysis["overall_score"],
            "dimensions": engagement_analysis["dimensions"],
            "engagement_patterns": engagement_analysis["engagement_patterns"],
            "short_answers_count": humanity["engagement"]["short_answers_count"] + 
                                 (1 if len(response.split()) < 10 else 0)
        })
        
        # Update self-disclosure with ALL disclosure metrics
        disclosure_analysis = self.analyzer.assess_disclosure(response)
        humanity["self_disclosure"].update({
            "disclosure_level": disclosure_analysis["disclosure_level"],
            "personal_topics": disclosure_analysis["personal_topics"],
            "vulnerability_markers": disclosure_analysis["vulnerability_markers"],
            "disclosure_patterns": disclosure_analysis["disclosure_patterns"],
            "personal_shares": humanity["self_disclosure"]["personal_shares"] + [{"content": response}]
        })
        
        # Update formality with ALL formality metrics
        formality_analysis = self.analyzer.assess_formality(response)
        humanity["formality"].update({
            "current_level": formality_analysis["current_level"],
            "ty_vy_ratio": formality_analysis["ty_vy_ratio"],
            "formality_markers": formality_analysis["formality_markers"],
            "style_consistency": formality_analysis["style_consistency"],
            "register_shifts": formality_analysis["register_shifts"]
        })
        
        # Update persona with ALL persona metrics
        persona_analysis = self.analyzer.determine_persona(response)
        humanity["persona"].update({
            "current_persona": persona_analysis["current_persona"],
            "interaction_style": persona_analysis["interaction_style"],
            "rapport_indicators": persona_analysis["rapport_indicators"],
            "adaptation_needed": persona_analysis["adaptation_needed"],
            "sentiment_trend": persona_analysis["interaction_style"]["support_needed"]
        })
        
        return {**state, "humanity": humanity}

    def _build_enhancement_prompt(self, question: str, state: Dict) -> str:
        """Build a comprehensive prompt using all analyzed dimensions."""
        humanity = state["humanity"]
        formality = humanity["formality"]
        
        # Use the already analyzed engagement dimensions from the state
        engagement_dims = humanity["engagement"].get("dimensions", {})
        engagement_patterns = humanity["engagement"].get("engagement_patterns", {})
        
        # Build enhancement requirements
        requirements = []
        
        # Cognitive load adaptations with detailed metrics
        cognitive = humanity["cognitive"]
        if cognitive.get("current_load", 0.0) > 0.7 or cognitive.get("mental_effort_level", 0.0) > 0.7:
            requirements.append(f"""
            SIMPLIFY (Current Load: {cognitive.get("current_load", 0.0):.2f}):
            - Break into shorter sentences
            - Use simpler vocabulary (Complexity: {len(cognitive.get("complexity_indicators", []))})
            - Remove subordinate clauses
            - Add clarifying context if needed
            - Address processing issues: {cognitive.get("processing_patterns", {}).get("hesitations", 0)} hesitations noted""")
        
        # Formality adaptations with style consistency
        if formality["current_level"] == "informal":
            requirements.append(f"""
            MAKE INFORMAL (Style Consistency: {formality.get('style_consistency', 0.5):.2f}):
            - Use more casual language
            - Switch to "ty" form (Current ratio: {formality.get('ty_vy_ratio', 0.5):.2f})
            - Use natural expressions
            - Match formality markers: {', '.join(formality.get('formality_markers', [])[:3] or ['neutral'])}""")
        elif formality["current_level"] == "formal":
            requirements.append(f"""
            MAKE FORMAL (Style Consistency: {formality.get('style_consistency', 0.5):.2f}):
            - Use professional language
            - Maintain "vy" form
            - Keep respectful tone
            - Match formality markers: {', '.join(formality.get('formality_markers', [])[:3] or ['neutral'])}""")
        
        # Emotional support with detailed patterns
        if humanity["emotional"]["requires_support"]:
            requirements.append(f"""
            ADD EMOTIONAL SUPPORT (Vulnerability: {humanity["emotional"]["vulnerability_level"]:.2f}):
            - Acknowledge emotions: {', '.join(humanity["emotional"]["key_emotions"])}
            - Match emotional complexity: {humanity["emotional"]["emotional_complexity"]:.2f}
            - Consider defense mechanisms: {', '.join(humanity["emotional"]["defense_mechanisms"])}
            - Support emotional regulation (level: {humanity["emotional"]["emotional_regulation"]["level"]:.2f})""")
        
        # Engagement improvements with all dimensions
        if engagement_dims:
            requirements.append(f"""
            IMPROVE ENGAGEMENT (Overall: {humanity["engagement"]["engagement_level"]:.2f}):
            - Elaboration level: {engagement_dims["elaboration"]:.2f}
            - Investment level: {engagement_dims["investment"]:.2f}
            - Emotional engagement: {engagement_dims["emotional"]:.2f}
            - Interactive elements: {engagement_dims["interactive"]:.2f}
            - Topic depth: {engagement_dims["depth"]:.2f}
            - Response hooks: {engagement_dims["hooks"]:.2f}
            - Narrative richness: {engagement_dims["richness"]:.2f}
            - Question engagement: {engagement_dims["question_engagement"]:.2f}
            - Conversational flow: {engagement_dims["flow"]:.2f}
            - Self-initiated elements: {engagement_dims["self_initiated"]:.2f}
            
            Patterns:
            - Consistency: {engagement_patterns["consistent_engagement"]}
            - Topic interest: {engagement_patterns["topic_interest"]:.2f}
            - Interaction quality: {engagement_patterns["interaction_quality"]:.2f}""")
        
        # Persona matching with interaction style
        persona_data = humanity["persona"]
        interaction_style = persona_data.get("interaction_style", {
            "formality_preference": 0.5,
            "support_needed": 0.5,
            "expertise_level": 0.5
        })
        
        requirements.append(f"""
        ADAPT TO {persona_data["current_persona"].upper()}:
        - Match formality preference: {interaction_style.get("formality_preference", 0.5):.2f}
        - Support level needed: {interaction_style.get("support_needed", 0.5):.2f}
        - Expertise level: {interaction_style.get("expertise_level", 0.5):.2f}
        - Consider rapport indicators: {', '.join(persona_data.get("rapport_indicators", [])[:3] or ["neutral"])}
        - Adaptation needed: {persona_data.get("adaptation_needed", False)}""")
        
        # Build the complete prompt
        prompt = f"""Enhance this question by applying ALL of the following modifications simultaneously:
        
User response: {state["user_message"]}

Original question: {question}

CURRENT STATE ANALYSIS:
{chr(10).join(requirements)}

GLOBAL RULES:
- Make exactly ONE version incorporating ALL modifications
- Keep it in Czech
- Maintain professional research context
- Preserve the core question intent
- Make all changes feel natural and integrated
- Avoid making the question too long or complex
- Ensure the question flows naturally

Return only the enhanced question in Czech."""

        return prompt

    def enhance_question(self, state: Dict, question: str) -> str:
        """Enhance question based on all humanity aspects with detailed logging."""
        # Build the enhancement prompt
        prompt = self._build_enhancement_prompt(question, state)
        
        # Get the enhanced question
        result = self.model.invoke([SystemMessage(content=prompt)])
        enhanced_question = result.content.strip()
        
        # Log the enhancement process
        print("\n=== HUMANITY ENHANCEMENT PROCESS ===")
        print(f"Original Question: {question}")
        print("\nKey Metrics Used:")
        print(f"- Emotional Weight: {state['humanity']['emotional']['emotional_weight']:.2f}")
        print(f"- Cognitive Load: {state['humanity']['cognitive']['current_load']:.2f}")
        print(f"- Engagement Level: {state['humanity']['engagement']['engagement_level']:.2f}")
        print(f"- Formality Level: {state['humanity']['formality']['current_level']}")
        print(f"- Current Persona: {state['humanity']['persona']['current_persona']}")
        print(f"\nEnhanced Question: {enhanced_question}")
        print("\nEnhancements Applied:")
        for req in self._build_enhancement_prompt(question, state).split("CURRENT STATE ANALYSIS:")[1].split("GLOBAL RULES:")[0].strip().split("\n"):
            if req.strip():
                print(req)
        
        return enhanced_question

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