from typing import List, Dict
from dataclasses import dataclass

@dataclass
class TestResponse:
    text: str
    emotional_weight: float
    key_emotions: List[str]
    complexity: float

class TestBot:
    """Simulates teacher responses based on predefined scenario."""
    
    def __init__(self):
        self.current_step = 0
        self.responses = [
            TestResponse(
                text="One person tried to kill himself in front of others, it was a really terrible experience.",
                emotional_weight=0.9,
                key_emotions=["shock", "distress", "fear"],
                complexity=0.8
            ),
            TestResponse(
                text="The student really didn't like himself, he was really ashamed of his marks and he wasn't really liked in the classroom. He was a loner.",
                emotional_weight=0.8,
                key_emotions=["shame", "isolation", "self-loathing", "sadness"],
                complexity=0.75
            ),
            TestResponse(
                text="I already replied to this question.",
                emotional_weight=0.8,
                key_emotions=["shame", "isolation", "self-loathing", "sadness"],
                complexity=0.75
            )
        ]

    def get_response(self, question: str) -> TestResponse:
        """Get next response in the sequence."""
        if self.current_step < len(self.responses):
            response = self.responses[self.current_step]
            self.current_step += 1
            return response
        return TestResponse(
            text="I don't want to talk about this anymore.",
            emotional_weight=0.7,
            key_emotions=["frustration", "fatigue"],
            complexity=0.5
        ) 