from typing import Dict, List, Set
from state import State
import re
import random

class QuestionQualityControl:
    def __init__(self):
        self.suggestive_patterns = [
            r"třeba",
            r"možná",
            r"není to tak, že",
            r"nebylo to proto, že",
            r"nemyslíte si, že"
        ]
        
        self.leading_phrases = [
            r"určitě jste",
            r"asi jste",
            r"předpokládám, že",
            r"pravděpodobně"
        ]

    def check_question_quality(self, question: str) -> Dict:
        return {
            "is_suggestive": self._check_suggestive(question),
            "is_leading": self._check_leading(question),
            "complexity": self._assess_complexity(question)
        }

    def _check_suggestive(self, question: str) -> bool:
        return any(re.search(pattern, question.lower()) for pattern in self.suggestive_patterns)

    def _check_leading(self, question: str) -> bool:
        return any(re.search(pattern, question.lower()) for pattern in self.leading_phrases)

    def rephrase_question(self, question: str, state: State) -> str:
        """Rephrase suggestive or leading questions to be more neutral."""
        if self._check_suggestive(question) or self._check_leading(question):
            # Replace suggestive patterns with open-ended alternatives
            question = re.sub(r"třeba|možná", "jak", question)
            question = re.sub(r"není to tak, že|nebylo to proto, že", "co si o tom myslíte", question)
            question = re.sub(r"nemyslíte si, že", "jaký je váš pohled na", question)
            
        return question

    def create_open_ended_question(self, topic: str) -> str:
        """Generate open-ended alternative questions."""
        templates = [
            f"Jak byste popsal/a {topic}?",
            f"Co vás k tomu napadá?",
            f"Můžete mi říct více o {topic}?",
            f"Jak jste tuto situaci vnímal/a?"
        ]
        return random.choice(templates) 