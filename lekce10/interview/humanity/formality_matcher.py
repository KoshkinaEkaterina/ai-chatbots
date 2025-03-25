from typing import Dict, Tuple, List
from state import State, FormalityLevel, FormalityState
import re

class FormalityMatcher:
    def __init__(self):
        self.formal_markers = [
            r"prosím",
            r"děkuji",
            r"mohl byste",
            r"mohla byste"
        ]
        
        self.informal_markers = [
            r"hele",
            r"prostě",
            r"jako",
            r"víš co"
        ]

    def detect_formality(self, response: str) -> Tuple[FormalityLevel, Dict[str, float]]:
        scores = {
            "formal": self._count_patterns(response, self.formal_markers),
            "informal": self._count_patterns(response, self.informal_markers),
            "ty_vy": self._analyze_ty_vy(response)
        }
        
        level = self._determine_formality_level(scores)
        return level, scores

    def _count_patterns(self, text: str, patterns: List[str]) -> int:
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text.lower()))
        return count

    def _analyze_ty_vy(self, text: str) -> float:
        ty_count = len(re.findall(r'\bty\b|\btebe\b|\btobě\b', text.lower()))
        vy_count = len(re.findall(r'\bvy\b|\bvás\b|\bvám\b', text.lower()))
        
        if ty_count + vy_count == 0:
            return 0.5
        return ty_count / (ty_count + vy_count)

    def _determine_formality_level(self, scores: Dict[str, float]) -> FormalityLevel:
        if scores["formal"] > scores["informal"] and scores["ty_vy"] < 0.3:
            return FormalityLevel.FORMAL
        elif scores["informal"] > scores["formal"] and scores["ty_vy"] > 0.7:
            return FormalityLevel.INFORMAL
        return FormalityLevel.SEMI_FORMAL

    def update_state(self, state: State, response: str) -> State:
        level, scores = self.detect_formality(response)
        
        state.humanity.formality.current_level = level
        state.humanity.formality.formality_scores = scores
        state.humanity.formality.ty_vy_ratio = scores["ty_vy"]
        
        return state 