from typing import Dict, List, Set, Optional
from datetime import datetime
from state import State
import re

class EngagementTracker:
    def __init__(self):
        self.dismissive_patterns = [
            r"nevím",
            r"to je jedno",
            r"už jsem říkal/a",
            r"nemám co dodat"
        ]
        
        self.engagement_indicators = [
            r"zajímavé",
            r"když se zamyslím",
            r"například",
            r"konkrétně"
        ]

    def analyze_engagement(self, response: str, state: State) -> Dict:
        # Track short answers
        is_short = len(response.split()) < 5
        
        # Check for repetitive phrases
        repetitive = self._check_repetition(response, state)
        
        # Count dismissive responses
        dismissive = any(re.search(pattern, response.lower()) 
                        for pattern in self.dismissive_patterns)
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement(
            response, is_short, repetitive, dismissive
        )
        
        return {
            "engagement_score": engagement_score,
            "is_short_answer": is_short,
            "repetitive_phrases": repetitive,
            "is_dismissive": dismissive
        }

    def _check_repetition(self, response: str, state: State) -> bool:
        # Get recent responses
        recent_responses = [
            exchange["user"] 
            for exchange in state.conversation_history[-3:]
        ]
        
        # Check for similar phrases
        for prev_response in recent_responses:
            if self._calculate_similarity(response, prev_response) > 0.7:
                return True
        return False

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        # Simple similarity check - could be improved with more sophisticated methods
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def _calculate_engagement(self, response: str, is_short: bool, 
                            repetitive: bool, dismissive: bool) -> float:
        score = 1.0  # Start with full engagement
        
        if is_short:
            score -= 0.3
        if repetitive:
            score -= 0.2
        if dismissive:
            score -= 0.4
            
        # Check for positive engagement indicators
        for indicator in self.engagement_indicators:
            if re.search(indicator, response.lower()):
                score += 0.1
                
        return max(0.0, min(1.0, score))

    def generate_engagement_prompt(self, state: State) -> Optional[str]:
        """Generate a re-engagement prompt if needed."""
        if state.humanity.engagement.engagement_level < 0.5:
            if state.humanity.engagement.short_answers_count > 2:
                return "Můžete mi to prosím více přiblížit?"
            elif state.humanity.engagement.dismissive_responses > 2:
                return "Je v tom tématu něco, co vám připadá důležité?"
        return None

    def update_state(self, state: State, response: str) -> State:
        analysis = self.analyze_engagement(response, state)
        
        # Update engagement state
        state.humanity.engagement.engagement_level = analysis["engagement_score"]
        if analysis["is_short_answer"]:
            state.humanity.engagement.short_answers_count += 1
        if analysis["is_dismissive"]:
            state.humanity.engagement.dismissive_responses += 1
            
        # Track repetitive phrases
        if analysis["repetitive_phrases"]:
            if response not in state.humanity.engagement.repetitive_phrases:
                state.humanity.engagement.repetitive_phrases[response] = 1
            else:
                state.humanity.engagement.repetitive_phrases[response] += 1
        
        state.humanity.engagement.last_engagement_check = datetime.now()
        
        return state 