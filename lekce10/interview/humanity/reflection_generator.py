from typing import Dict, List
from state import State
import random

class ReflectionGenerator:
    def __init__(self):
        self.reflection_templates = {
            "summary": [
                "Pokud tomu dobře rozumím, {summary}...",
                "Takže když to shrnu, {summary}...",
                "Z toho, co říkáte, vyplývá, že {summary}..."
            ],
            "verification": [
                "Je to tak správně?",
                "Vystihuje to vaši zkušenost?",
                "Rozumím tomu správně?"
            ]
        }

    def generate_reflection(self, state: State, last_responses: List[str]) -> str:
        # Extract key points from recent responses
        key_points = self._extract_key_points(last_responses)
        
        # Create summary
        summary = self._create_summary(key_points)
        
        # Format reflection
        reflection = random.choice(self.reflection_templates["summary"]).format(
            summary=summary
        )
        
        # Add verification
        reflection += " " + random.choice(self.reflection_templates["verification"])
        
        return reflection

    def _extract_key_points(self, responses: List[str]) -> List[str]:
        # Add logic to extract main points from responses
        return []

    def _create_summary(self, points: List[str]) -> str:
        # Create coherent summary from points
        return " ".join(points) 