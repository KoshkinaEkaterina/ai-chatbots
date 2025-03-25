import argparse
import logging
from interview_bot import InterviewBot
from teacher_bot import TeacherBot
import time
from random import uniform

def run_manual_mode():
    """Run interview with manual user input."""
    bot = InterviewBot()
    
    print("\n=== Starting Manual Interview ===\n")
    
    # Initialize conversation
    result = bot.chat()
    print(f"Tazatel: {result['question']}")
    
    while True:
        # Get user input
        print("\nVaše odpověď (nebo 'konec' pro ukončení):")
        response = input("> ").strip()
        
        if response.lower() == 'konec':
            print("\nRozhovor ukončen.")
            break
        
        # Process response
        result = bot.chat(response)
        
        # Show emotional analysis if significant
        if result.get('emotional_analysis', {}).get('emotional_weight', 0) > 0.5:
            emotions = result['emotional_analysis'].get('key_emotions', [])
            if emotions:
                print("\n[Emoční kontext:", ", ".join(emotions), "]")
        
        # Show coverage for current topic
        print("\nPokrytí témat:")
        for factor, score in result['covered_factors'].items():
            print(f"- {factor}: {score:.2f}")
        
        # Print next question immediately
        print("\n" + "="*50 + "\n")
        print(f"Tazatel: {result['question']}")
        
        if result['is_complete']:
            print("\n=== Rozhovor dokončen ===")
            break

def run_teacher_mode():
    """Run interview with automated teacher responses."""
    interviewer = InterviewBot()
    teacher = TeacherBot()
    
    print("\n=== Starting Automated Interview ===\n")
    
    # Initialize conversation
    result = interviewer.chat()
    print(f"Tazatel: {result['question']}")
    
    while True:
        # Simulate teacher waiting
        print("\nUčitel přemýšlí...")
        time.sleep(uniform(1.5, 3.0))
        
        # Get teacher's response
        teacher_response = teacher.respond(result['question'])
        print(f"Učitel: {teacher_response}")
        
        # Process through interviewer
        result = interviewer.chat(teacher_response)
        
        # Show emotional analysis if significant
        if result.get('emotional_analysis', {}).get('emotional_weight', 0) > 0.5:
            emotions = result['emotional_analysis'].get('key_emotions', [])
            if emotions:
                print("\n[Emoční kontext:", ", ".join(emotions), "]")
        
        # Show coverage for current topic
        print("\nPokrytí témat:")
        for factor, score in result['covered_factors'].items():
            print(f"- {factor}: {score:.2f}")
        
        # Print next question immediately
        print("\n" + "="*50 + "\n")
        print(f"Tazatel: {result['question']}")
        
        if result['is_complete']:
            print("\n=== Rozhovor dokončen ===")
            break
        
        # Add small delay between exchanges
        time.sleep(uniform(1.0, 2.0))

def main():
    parser = argparse.ArgumentParser(description='Run interview simulation')
    parser.add_argument('--mode', choices=['manual', 'teacher'], default='manual',
                      help='Run in manual mode or with automated teacher responses')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'manual':
            run_manual_mode()
        else:
            run_teacher_mode()
    except KeyboardInterrupt:
        print("\n\nRozhovor přerušen uživatelem.")
    except Exception as e:
        logging.error(f"Error during interview: {str(e)}")
        raise

if __name__ == "__main__":
    main() 