from typing import Dict, List, Optional
from state import State, TransitionState
from datetime import datetime
import random

class AdaptiveTransitionModule:
    def __init__(self):
        self.transition_templates = {
            "gentle": [
                "Zdá se, že jsme toto téma probrali podrobně. Můžeme přejít k další oblasti?",
                "Než budeme pokračovat dál, chtěl/a byste k tomuto tématu ještě něco dodat?",
                "Myslím, že máme dobrou představu o této situaci. Souhlasil/a byste s přechodem k dalšímu tématu?"
            ],
            "direct": [
                "Ještě něco důležitého, co bychom měli u tohoto tématu zmínit?",
                "Máte k tomuto ještě nějakou důležitou myšlenku?",
                "Můžeme přejít k dalšímu tématu?"
            ],
            "reflective": [
                "Sdílel/a jste několik zajímavých postřehů. Než půjdeme dál, napadá vás ještě něco podstatného?",
                "Probrali jsme několik důležitých aspektů. Je tu ještě něco, co byste rád/a zmínil/a?",
                "Než změníme téma, chtěl/a byste se ještě k něčemu vrátit?"
            ]
        }

    def calculate_transition_confidence(self, state: State) -> float:
        current_topic = state.topics[state.current_topic_id]
        
        # Calculate average factor coverage
        coverage_scores = list(current_topic.covered_factors.values())
        avg_coverage = sum(coverage_scores) / len(coverage_scores)
        
        # Factor in number of questions asked
        questions_weight = min(state.questions_in_topic / 5.0, 1.0)
        
        # Consider engagement level
        engagement_weight = state.humanity.engagement.engagement_level
        
        # Combine factors
        confidence = (avg_coverage * 0.5 + 
                     questions_weight * 0.3 + 
                     engagement_weight * 0.2)
        
        return min(1.0, confidence)

    def select_transition_style(self, state: State) -> str:
        emotional_weight = state.humanity.emotional.emotional_weight
        disclosure_level = state.humanity.self_disclosure.disclosure_level
        
        if emotional_weight > 0.7 or disclosure_level > 0.7:
            return "reflective"
        elif state.humanity.engagement.engagement_level < 0.5:
            return "direct"
        return "gentle"

    def generate_transition(self, state: State) -> str:
        confidence = self.calculate_transition_confidence(state)
        state.humanity.transition.confidence_to_transition = confidence
        
        if confidence < 0.6:
            return ""  # Not ready to transition
            
        style = self.select_transition_style(state)
        transition = random.choice(self.transition_templates[style])
        
        state.humanity.transition.last_transition_type = style
        return transition

    def update_state(self, state: State) -> State:
        # Update covered aspects
        current_topic = state.topics[state.current_topic_id]
        for factor, coverage in current_topic.covered_factors.items():
            if coverage > 0.7:
                state.humanity.transition.covered_aspects.add(factor)
        
        # Update topic exhaustion
        exhaustion = len(state.humanity.transition.covered_aspects) / len(current_topic.factors)
        state.humanity.transition.topic_exhaustion = exhaustion
        
        return state 