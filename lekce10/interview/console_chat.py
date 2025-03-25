import requests
import os
from dotenv import load_dotenv

def chat():
    """Run the chat console."""
    load_dotenv()
    api_url = "http://localhost:8001/chat"
    
    print(f"\n=== Interview Bot Console ===")
    print(f"Connected to server at {api_url}")
    print("Type 'quit' to exit\n")

    # Start conversation
    try:
        response = requests.post(
            api_url,
            json={"message": None},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        
        if "response" in data:
            print(data["response"])
        else:
            print("Error: Unexpected response format")
            return
            
    except Exception as e:
        print(f"\nError starting conversation: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return

    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'konec']:
            print("\nEnding conversation...")
            break

        try:
            response = requests.post(
                api_url,
                json={"message": user_input},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            if "response" in data:
                print("\n" + data["response"])
            else:
                print("\nError: Unexpected response format")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text}")

if __name__ == "__main__":
    chat() 