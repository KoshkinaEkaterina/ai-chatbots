from simple_bot import SimpleBot

def main():
    bot = SimpleBot()
    print("\n=== Starting Simple Chat ===\n")
    
    # Get initial greeting
    result = bot.chat()
    print(f"Bot: {result['response']}\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Process through bot
        result = bot.chat(user_input)
        print(f"\nBot: {result['response']}\n")
        
        if result["is_complete"]:
            break

if __name__ == "__main__":
    main() 