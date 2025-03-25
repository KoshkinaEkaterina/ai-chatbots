from datetime import datetime, timedelta
from typing import Dict, List
from state import State, SelfDisclosureState
import re

class SelfDisclosureModule:
    def __init__(self):
        self.personal_indicators = [
            r"já jsem",
            r"pro mě",
            r"cítil/a jsem",
            r"myslel/a jsem",
            r"zhroutil/a jsem se",
            r"nedokázal/a jsem"
        ]
        
        self.emotional_weight_phrases = {
            "high": [
                r"nikdy nezapomenu",
                r"změnilo mi to život",
                r"nejhorší zkušenost"
            ],
            "medium": [
                r"bylo to těžké",
                r"trápilo mě to",
                r"nevěděl/a jsem si rady"
            ]
        }

    def detect_self_disclosure(self, response: str) -> Dict:
        disclosure = {
            "level": self._calculate_disclosure_level(response),
            "personal_content": self._extract_personal_content(response),
            "requires_acknowledgment": False
        }
        
        disclosure["requires_acknowledgment"] = (
            disclosure["level"] > 0.7 or
            len(disclosure["personal_content"]) > 0
        )
        
        return disclosure

    def _calculate_disclosure_level(self, text: str) -> float:
        level = 0.0
        
        # Check for personal indicators
        for pattern in self.personal_indicators:
            if re.search(pattern, text.lower()):
                level += 0.2
                
        # Check for emotional weight
        for weight, phrases in self.emotional_weight_phrases.items():
            for phrase in phrases:
                if re.search(phrase, text.lower()):
                    level += 0.3 if weight == "high" else 0.1
                    
        return min(1.0, level)

    def _extract_personal_content(self, text: str) -> List[Dict]:
        content = []
        sentences = text.split('.')
        
        for sentence in sentences:
            if any(re.search(pattern, sentence.lower()) for pattern in self.personal_indicators):
                content.append({
                    "text": sentence.strip(),
                    "timestamp": datetime.now(),
                    "type": "personal_share"
                })
                
        return content

    def generate_acknowledgment(self, disclosure_level: float) -> str:
        if disclosure_level > 0.8:
            return "Děkuji za vaši otevřenost a důvěru."
        elif disclosure_level > 0.5:
            return "Děkuji, že jste to sdílel/a."
        return ""

    def update_state(self, state: State, response: str) -> State:
        disclosure = self.detect_self_disclosure(response)
        
        state.humanity.self_disclosure.disclosure_level = disclosure["level"]
        if disclosure["personal_content"]:
            state.humanity.self_disclosure.personal_shares.extend(
                disclosure["personal_content"]
            )
            
        if disclosure["requires_acknowledgment"]:
            state.humanity.self_disclosure.last_acknowledgment = datetime.now()
            
        return state 