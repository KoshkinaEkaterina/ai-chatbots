from typing import Dict, List, Optional
from datetime import datetime
from state import State, CognitiveLoad
import re

class CognitiveLoadModule:
    def __init__(self):
        self.fatigue_indicators = [
            r"nevím",
            r"je toho moc",
            r"už si nepamatuji",
            r"těžko říct"
        ]
        
        self.complexity_markers = {
            "conjunctions": ["a", "ale", "protože", "když", "aby"],
            "subordinate_clauses": [r"který", r"že", r"kde", r"jak", r"proč"]
        }

    def estimate_load(self, response: str, state: State) -> Dict:
        # Calculate response length
        words = len(response.split())
        
        # Count complex structures
        complexity = self._calculate_complexity(response)
        
        # Count fatigue indicators
        fatigue = sum(1 for pattern in self.fatigue_indicators 
                     if re.search(pattern, response.lower()))
        
        # Track response time if available
        response_time = self._get_response_time(state)
        
        # Determine load level
        load_level = self._determine_load_level(
            words, complexity, fatigue, response_time
        )
        
        return {
            "load_level": load_level,
            "word_count": words,
            "complexity_score": complexity,
            "fatigue_indicators": fatigue,
            "response_time": response_time
        }

    def _calculate_complexity(self, text: str) -> float:
        complexity = 0.0
        text_lower = text.lower()
        
        # Count complex sentence structures
        for conj in self.complexity_markers["conjunctions"]:
            complexity += text_lower.count(f" {conj} ") * 0.1
            
        for clause in self.complexity_markers["subordinate_clauses"]:
            complexity += len(re.findall(clause, text_lower)) * 0.2
            
        return min(1.0, complexity)

    def _get_response_time(self, state: State) -> Optional[float]:
        # If timing data is available in state
        return state.humanity.cognitive.response_times[-1] if state.humanity.cognitive.response_times else None

    def _determine_load_level(self, words: int, complexity: float, 
                            fatigue: int, response_time: Optional[float]) -> CognitiveLoad:
        # Combine factors to determine load
        load_score = 0.0
        
        # Word count factor
        if words < 10:
            load_score += 0.2  # Very short might indicate high load
        elif words > 100:
            load_score += 0.4  # Very long might indicate low load
            
        # Complexity factor
        load_score += complexity * 0.3
        
        # Fatigue factor
        load_score += min(1.0, fatigue * 0.2)
        
        # Response time factor (if available)
        if response_time:
            if response_time > 10.0:  # More than 10 seconds
                load_score += 0.2
        
        # Determine level
        if load_score > 0.7:
            return CognitiveLoad.HIGH
        elif load_score > 0.3:
            return CognitiveLoad.MODERATE
        return CognitiveLoad.LOW

    def update_state(self, state: State, response: str) -> State:
        load_analysis = self.estimate_load(response, state)
        
        # Update cognitive state
        state.humanity.cognitive.current_load = load_analysis["load_level"]
        state.humanity.cognitive.response_lengths.append(load_analysis["word_count"])
        if load_analysis["response_time"]:
            state.humanity.cognitive.response_times.append(load_analysis["response_time"])
        state.humanity.cognitive.clause_complexity.append(load_analysis["complexity_score"])
        state.humanity.cognitive.fatigue_indicators += load_analysis["fatigue_indicators"]
        
        return state 