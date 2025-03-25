from typing import Dict, List, Set
from state import State, LanguageStyleState
import re
from datetime import datetime

class LanguageStyleModule:
    def __init__(self):
        self.metaphor_patterns = [
            r"jako by",
            r"připomíná to",
            r"je to jako",
            r"podobné jako"
        ]
        
        self.idiom_patterns = {
            "common_idioms": [
                r"mít plné ruce práce",
                r"být v koncích",
                r"držet palce",
                r"mít těžkou hlavu"
            ],
            "teaching_specific": [
                r"otevřít oči",
                r"najít společnou řeč",
                r"být za vodou",
                r"jít příkladem"
            ]
        }

    def extract_style_elements(self, text: str) -> Dict:
        elements = {
            "metaphors": self._find_metaphors(text),
            "idioms": self._find_idioms(text),
            "emotional_expressions": self._find_emotional_expressions(text)
        }
        return elements

    def _find_metaphors(self, text: str) -> List[Dict]:
        metaphors = []
        for pattern in self.metaphor_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                # Get the full metaphorical expression
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                
                metaphors.append({
                    "text": context,
                    "pattern": pattern,
                    "timestamp": datetime.now()
                })
        return metaphors

    def _find_idioms(self, text: str) -> Set[str]:
        found_idioms = set()
        for category, patterns in self.idiom_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    found_idioms.add(pattern)
        return found_idioms

    def _find_emotional_expressions(self, text: str) -> Dict[str, str]:
        expressions = {}
        # Add emotional expression detection logic
        return expressions

    def mirror_style(self, state: State) -> Dict[str, str]:
        """Generate style suggestions for response generation."""
        style_suggestions = {
            "use_metaphors": [],
            "use_idioms": [],
            "emotional_tone": "neutral"
        }
        
        # If user used metaphors recently, suggest using similar ones
        if state.humanity.language_style.metaphors:
            recent_metaphor = state.humanity.language_style.metaphors[-1]
            style_suggestions["use_metaphors"].append(recent_metaphor)
        
        # If user used idioms, suggest incorporating them
        if state.humanity.language_style.idioms:
            style_suggestions["use_idioms"].extend(
                list(state.humanity.language_style.idioms)[-2:]
            )
        
        return style_suggestions

    def update_state(self, state: State, response: str) -> State:
        # Extract style elements
        elements = self.extract_style_elements(response)
        
        # Update metaphors
        state.humanity.language_style.metaphors.extend(elements["metaphors"])
        
        # Update idioms
        state.humanity.language_style.idioms.update(elements["idioms"])
        
        # Update emotional expressions
        state.humanity.language_style.emotional_expressions.update(
            elements["emotional_expressions"]
        )
        
        # Track style markers
        for style_type, items in elements.items():
            if style_type not in state.humanity.language_style.style_markers:
                state.humanity.language_style.style_markers[style_type] = 0
            state.humanity.language_style.style_markers[style_type] += len(items)
        
        return state 