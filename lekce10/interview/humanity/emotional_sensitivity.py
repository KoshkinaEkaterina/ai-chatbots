from datetime import datetime, timedelta
from typing import Dict, List
from state import State, EmotionalState
import re

class EmotionalSensitivityModule:
    def __init__(self):
        self.emotional_keywords = {
            "stress": ["stres", "náročné", "těžké", "vyčerpávající"],
            "anxiety": ["úzkost", "strach", "obavy", "nervozita"],
            "frustration": ["frustrace", "štve", "naštvaný", "rozčilený"],
            "sadness": ["smutný", "zklamaný", "nešťastný"],
            "overwhelm": ["přetížený", "nezvládám", "moc", "příliš"]
        }

    def analyze_emotional_content(self, response: str) -> EmotionalState:
        emotional_state = EmotionalState()
        
        # Analyze emotional weight
        emotional_state.emotional_weight = self._calculate_emotional_weight(response)
        
        # Check for trauma indicators
        emotional_state.trauma_indicators = self._detect_trauma_indicators(response)
        
        # Identify key emotions
        emotional_state.key_emotions = self._identify_emotions(response)
        
        # Detect emotional cues
        emotional_state.emotional_cues = self._detect_cues(response)
        
        # Determine if support is needed
        emotional_state.requires_support = (
            emotional_state.emotional_weight > 0.7 or
            emotional_state.trauma_indicators or
            len(emotional_state.key_emotions) > 2
        )
        
        return emotional_state

    def _calculate_emotional_weight(self, text: str) -> float:
        # Implementation of emotional weight calculation
        weight = 0.0
        # Add your emotional weight calculation logic here
        return min(1.0, weight)

    def _detect_trauma_indicators(self, text: str) -> bool:
        trauma_patterns = [
            r"nikdy nezapomenu",
            r"pořád se mi vrací",
            r"nemůžu na to zapomenout",
            r"dodnes mě to trápí"
        ]
        return any(re.search(pattern, text.lower()) for pattern in trauma_patterns)

    def _identify_emotions(self, text: str) -> List[str]:
        found_emotions = []
        for emotion, keywords in self.emotional_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                found_emotions.append(emotion)
        return found_emotions

    def _detect_cues(self, text: str) -> List[str]:
        cues = []
        # Add cue detection logic here
        return cues

    def modify_response_style(self, state: State) -> Dict[str, any]:
        modifiers = {
            "tone_softness": 1.0,
            "sentence_length": "normal",
            "probe_depth": "normal"
        }
        
        if state.humanity.emotional.emotional_weight > 0.7:
            modifiers["tone_softness"] = 1.5
            modifiers["sentence_length"] = "short"
            modifiers["probe_depth"] = "shallow"
            
        return modifiers

    def update_state(self, state: State, response: str) -> State:
        emotional_state = self.analyze_emotional_content(response)
        state.humanity.emotional = emotional_state
        return state 