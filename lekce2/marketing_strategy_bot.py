import os
import json
import requests
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime

class MarketingStrategyBot:
    def __init__(self):
        load_dotenv()
        self.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_key = os.getenv('AZURE_OPENAI_KEY')
        self.api_version = '2024-02-15-preview'
        self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        
        # Initialize conversation memory
        self.conversation_history = []
        self.task_memory = {}
        
        # Define core marketing steps
        self.marketing_steps = [
            "Market Research and Competition Analysis",
            "Target Customer Profiling and Value Proposition",
            "Marketing Strategy and Channel Selection",
            "Budget Planning and ROI Projections",
            "Implementation Roadmap"
        ]

    def create_chat_completion(self, messages: List[Dict], max_tokens: int = 2000) -> str:
        """Make a request to Azure OpenAI chat completion API"""
        url = f"{self.api_base}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        data = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return None

    def gather_initial_information(self) -> Dict:
        """Gather essential information from the user"""
        questions = [
            "Please describe your product or service in detail, including its unique features and benefits.",
            "What is your target market's biggest pain point that your product/service solves?",
            "What are your top 3 competitors and what makes your offering different from theirs?"
        ]
        
        print("\n" + "="*80)
        print("INITIAL BUSINESS ANALYSIS")
        print("="*80)
        print("\nTo create an effective marketing strategy, I need to understand your business better.")
        print("Please answer these three critical questions:\n")
        
        responses = {}
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            response = input("> ").strip()
            responses[f"Q{i}"] = response
            print("\n" + "-"*40)  # Separator between questions
            
        return responses

    def execute_task_step(self, step: str, context: Dict) -> str:
        """Execute a single step of the marketing strategy"""
        system_message = {
            "role": "system",
            "content": f"""You are an expert marketing strategist focusing on: {step}.
            Based on the client's information, provide an in-depth analysis and specific, actionable recommendations.
            Format your response with clear sections:
            1. Current Situation Analysis
            2. Key Findings
            3. Specific Recommendations
            4. Implementation Steps
            
            Use bullet points and numbering for clarity. Be specific and detailed in your recommendations."""
        }
        
        context_summary = "\n".join([f"{k}: {v}" for k, v in context.items()])
        
        messages = [
            system_message,
            {"role": "user", "content": f"Analyze this business information and provide detailed recommendations:\n\n{context_summary}"}
        ]
        
        return self.create_chat_completion(messages)

    def generate_final_summary(self, context: Dict, steps_results: Dict[str, str]) -> str:
        """Generate a comprehensive final marketing strategy"""
        summary_prompt = {
            "role": "system",
            "content": """You are a senior marketing consultant creating a comprehensive marketing strategy document.
            The document must be at least 3000 words and include:
            
            1. Executive Summary
            2. Detailed Business Analysis
            3. Strategic Recommendations
            4. Implementation Plan
            5. Timeline and Milestones
            6. Budget Considerations
            7. Success Metrics and KPIs
            8. Risk Analysis and Mitigation
            
            Format the document professionally with clear sections, subsections, and bullet points.
            Be extremely specific with actionable recommendations and measurable goals."""
        }
        
        context_summary = "\n".join([f"{k}: {v}" for k, v in context.items()])
        steps_summary = "\n\n".join([f"{step}:\n{result}" for step, result in steps_results.items()])
        
        messages = [
            summary_prompt,
            {"role": "user", "content": f"""Create a comprehensive marketing strategy based on:
            
            BUSINESS INFORMATION:
            {context_summary}
            
            DETAILED ANALYSIS:
            {steps_summary}"""}
        ]
        
        return self.create_chat_completion(messages, max_tokens=4000)

    def process_marketing_strategy(self) -> str:
        """Process a complete marketing strategy request"""
        context = self.gather_initial_information()
        
        print("\n" + "="*80)
        print("MARKETING STRATEGY DEVELOPMENT")
        print("="*80)
        print("\nDeveloping your marketing strategy based on the provided information.")
        print("Each step will be analyzed in detail.\n")
        
        steps_results = {}
        
        for step in self.marketing_steps:
            print(f"\n{'='*80}")
            print(f"STEP: {step}")
            print(f"{'='*80}\n")
            
            step_result = self.execute_task_step(step, context)
            steps_results[step] = step_result
            
            # Display the detailed analysis
            print("ANALYSIS AND RECOMMENDATIONS:")
            print("-"*40)
            print(step_result)
            print("\n" + "-"*40)
            print("✓ Step completed\n")
        
        print("\n" + "="*80)
        print("GENERATING FINAL MARKETING STRATEGY")
        print("="*80)
        final_strategy = self.generate_final_summary(context, steps_results)
        
        # Store in memory
        self.task_memory[datetime.now().isoformat()] = {
            "context": context,
            "steps_results": steps_results,
            "final_strategy": final_strategy
        }
        
        return final_strategy

    def run(self):
        """Run the interactive marketing strategy bot"""
        print("\n" + "="*80)
        print("MARKETING STRATEGY ASSISTANT")
        print("="*80)
        print("\nI'll help you create a comprehensive marketing strategy for your business.")
        
        while True:
            print("\nWhat would you like to do?")
            print("1. Create new marketing strategy")
            print("2. View previous strategies")
            print("3. Exit")
            
            choice = input("\nChoice > ").strip()
            
            if choice == "1":
                final_strategy = self.process_marketing_strategy()
                print("\n" + "="*80)
                print("FINAL MARKETING STRATEGY")
                print("="*80)
                print(final_strategy)
                
            elif choice == "2":
                if not self.task_memory:
                    print("\nNo previous strategies found.")
                else:
                    print("\nPrevious Strategies:")
                    for timestamp, task in self.task_memory.items():
                        print(f"\nDate: {timestamp}")
                        print("\nBusiness Context:")
                        for q, a in task['context'].items():
                            print(f"{q}: {a}")
                        print("\nWould you like to see the full strategy? (yes/no)")
                        if input("> ").lower().startswith('y'):
                            print("\nFull Strategy:")
                            print("="*80)
                            print(task['final_strategy'])
                            
            elif choice == "3" or choice.lower() == 'quit':
                print("\nThank you for using the Marketing Strategy Assistant. Goodbye!")
                break

if __name__ == "__main__":
    bot = MarketingStrategyBot()
    bot.run() 