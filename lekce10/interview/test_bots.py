from interview_bot import InterviewBot
from teacher_bot_v1 import TeacherBot  # Use V1 version
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_conversation():
    interviewer = InterviewBot()
    teacher = TeacherBot()
    
    # Start conversation with initial greeting
    response = interviewer.chat("Ahoj")  # Initial null message starts conversation
    print("\nInterviewer:", response["response"])
    
    while True:
        # Get teacher's response
        teacher_response = teacher.respond(response["response"])  # Use respond() instead of chat()
        print("\nTeacher:", teacher_response)
        
        # Get interviewer's next question
        response = interviewer.chat(teacher_response)  # Pass raw response string
        print("\nInterviewer:", response["response"])
        
        # Optional: add a way to end conversation
        if "Děkuji za rozhovor" in response["response"]:
            break

if __name__ == "__main__":
    print("Starting conversation between InterviewBot and TeacherBotV1...")
    print("=" * 80)
    run_conversation() 