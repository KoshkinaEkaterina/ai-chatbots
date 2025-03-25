from typing import Dict, List
from interview_bot import InterviewBot
from test_bot import TestBot, TestResponse
import json

class TestBotV3:
    """Runs tests between TestBot and InterviewBot."""
    
    def __init__(self):
        self.interview_bot = InterviewBot()
        self.test_bot = TestBot()
        self.conversation_history = []
        
    def run_test(self) -> List[Dict]:
        """Run the complete test conversation."""
        # Initialize interview
        response = self.interview_bot.chat()
        self.conversation_history.append({
            "bot": response["response"],
            "stats": self._format_stats(response)
        })
        
        # Run conversation
        while True:
            # Get test bot response
            test_response = self.test_bot.get_response(response["response"])
            
            # Process through interview bot
            response = self.interview_bot.chat(test_response.text)
            
            # Store exchange
            self.conversation_history.append({
                "user": test_response.text,
                "user_emotional_state": {
                    "weight": test_response.emotional_weight,
                    "emotions": test_response.key_emotions,
                    "complexity": test_response.complexity
                },
                "bot": response["response"],
                "stats": self._format_stats(response)
            })
            
            # Check for end conditions
            if "don't want to talk" in test_response.text.lower():
                break
                
        return self.conversation_history
    
    def _format_stats(self, response: Dict) -> Dict:
        """Format response stats for logging."""
        stats = {}
        
        if "topic_stats" in response:
            topic = response["topic_stats"]
            stats["topic"] = topic["topic_question"]
            stats["questions_asked"] = topic["questions_asked"]
            stats["factor_coverage"] = {
                factor: {
                    "coverage": f"{details['coverage']*100:.1f}%",
                    "questions": details["questions"]
                }
                for factor, details in topic["factor_coverage"].items()
            }
            
        if "humanity_analysis" in response:
            humanity = response["humanity_analysis"]
            stats["emotional_state"] = {
                "weight": humanity["emotional"]["emotional_weight"],
                "emotions": humanity["emotional"]["key_emotions"],
                "complexity": humanity["emotional"]["emotional_complexity"]
            }
            
        return stats

if __name__ == "__main__":
    # Run test
    tester = TestBotV3()
    conversation = tester.run_test()
    
    # Print results
    print("\n=== TEST CONVERSATION ===\n")
    for turn in conversation:
        print("\n---\n")
        if "user" in turn:
            print(f"USER: {turn['user']}")
            print("\nUser Emotional State:")
            state = turn["user_emotional_state"]
            print(f"Weight: {state['weight']:.2f}")
            print(f"Emotions: {', '.join(state['emotions'])}")
            print(f"Complexity: {state['complexity']:.2f}")
            
        print(f"\nBOT: {turn['bot']}")
        print("\nStats:")
        print(json.dumps(turn["stats"], indent=2)) 