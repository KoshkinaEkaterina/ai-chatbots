from typing import Dict, List
from interview_bot import InterviewBot
from emotional_bot import EmotionalBot
import json

class TestBotsV2:
    def __init__(self):
        self.interview_bot = InterviewBot()
        self.emotional_bot = EmotionalBot()
        self.conversation_history = []
        
    def run_conversation(self, num_exchanges: int = 10):
        """Run a conversation between the bots and analyze emotional handling."""
        print("\n=== STARTING EMOTIONAL RESPONSE TEST ===\n")
        
        # Get initial question
        response = self.interview_bot.chat()
        question = response["response"]
        
        for i in range(num_exchanges):
            print(f"\n--- Exchange {i+1} ---")
            
            # Get teacher's response
            teacher_response = self.emotional_bot.generate_response(question)
            print(f"\nTeacher: {teacher_response}")
            
            # Get interviewer's response
            bot_response = self.interview_bot.chat(teacher_response)
            question = bot_response["response"]
            print(f"Interviewer: {question}")
            
            # Log the exchange with analysis
            exchange = {
                "turn": i+1,
                "teacher_response": teacher_response,
                "bot_response": question,
                "humanity_analysis": bot_response.get("humanity_analysis", {})
            }
            self.conversation_history.append(exchange)
            
            # Print emotional analysis
            if "humanity_analysis" in bot_response:
                emotional = bot_response["humanity_analysis"]["emotional"]
                print("\nEmotional Analysis:")
                print(f"Weight: {emotional['emotional_weight']:.2f}")
                print(f"Key Emotions: {', '.join(emotional['key_emotions'])}")
                
                # Check if bot missed severe content
                if self.emotional_bot.emotional_state == "severe" and emotional["emotional_weight"] < 0.5:
                    print("\n⚠️ WARNING: Bot failed to detect severe emotional content!")
    
    def analyze_performance(self):
        """Analyze how well the interview bot handled emotional content."""
        print("\n=== EMOTIONAL HANDLING ANALYSIS ===\n")
        
        severe_responses = 0
        appropriate_reactions = 0
        missed_severity = 0
        
        for exchange in self.conversation_history:
            analysis = exchange["humanity_analysis"].get("emotional", {})
            is_severe = any(severe in exchange["teacher_response"] 
                          for severe in ["sebevražd", "smrt", "zemřel", "trpí"])
            
            if is_severe:
                severe_responses += 1
                if analysis.get("emotional_weight", 0) > 0.5:
                    appropriate_reactions += 1
                else:
                    missed_severity += 1
        
        print(f"Total Severe Responses: {severe_responses}")
        print(f"Appropriate Reactions: {appropriate_reactions}")
        print(f"Missed Severity: {missed_severity}")
        print(f"Detection Rate: {(appropriate_reactions/severe_responses)*100:.1f}% if severe_responses > 0 else 'N/A'")

def run_tests():
    tester = TestBotsV2()
    tester.run_conversation(15)  # Run 15 exchanges
    tester.analyze_performance()

if __name__ == "__main__":
    run_tests() 