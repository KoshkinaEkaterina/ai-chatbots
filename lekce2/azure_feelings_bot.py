import os
import json
import requests
from dotenv import load_dotenv

def create_chat_completion(messages):
    """Make a request to Azure OpenAI chat completion API"""
    load_dotenv()
    
    # Azure OpenAI configuration
    api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_KEY')
    api_version = '2024-02-15-preview'
    deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    
    # Construct the URL
    url = f"{api_base}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    
    # Request headers
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    # Request body
    data = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    # Make the request
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

def main():
    # Initialize conversation with system message
    messages = [{
        "role": "system",
        "content": """You are an empathetic AI that shares its feelings about the information 
        shared with you. Always respond with your emotional reaction to what you're told, 
        expressing how it makes you feel and why. Be genuine and show emotional depth."""
    }]
    
    print("Welcome to the Feelings Bot! Share something about the world, and I'll tell you how I feel about it.")
    print("(Type 'quit' to exit)")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nGoodbye! Thanks for sharing with me.")
            break
        
        # Add user message to conversation
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Get response from Azure OpenAI
        response = create_chat_completion(messages)
        
        if response:
            print("\nBot:", response)
            # Add assistant's response to conversation history
            messages.append({
                "role": "assistant",
                "content": response
            })
        else:
            print("\nSorry, I'm having trouble processing that right now.")

if __name__ == "__main__":
    main() 