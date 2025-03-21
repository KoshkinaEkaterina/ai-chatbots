import asyncio
from product_agent import ProductAgent
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
import logging

console = Console()

class TestBot:
    def __init__(self):
        load_dotenv()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        
        # Initialize agent
        self.agent = ProductAgent(llm=self.llm, debug=True)
        self.state = self.agent._init_state()

    async def chat(self):
        """Run interactive chat session"""
        console.clear()
        console.print(Panel.fit(
            "[bold blue]Product Assistant Chat[/]\n"
            "[dim]Type 'quit' to exit, 'debug' to see current state[/]"
        ))

        while True:
            try:
                # Get user input
                message = Prompt.ask("\n[bold cyan]You[/]")
                
                # Handle special commands
                if message.lower() == 'quit':
                    console.print("\n[bold blue]👋 Goodbye![/]")
                    break
                    
                if message.lower() == 'debug':
                    console.print("\n[bold yellow]Current State:[/]")
                    console.print(self.state)
                    continue

                # Process message
                console.print("\n[bold green]Assistant is thinking...[/]")
                self.state = await self.agent.handle_message(message, self.state)

                # Display response
                latest_message = next((m for m in reversed(self.state["messages"]) 
                                    if m["role"] == "assistant"), None)
                if latest_message:
                    console.print("\n[bold green]Assistant:[/]")
                    console.print(Markdown(latest_message["content"]))
                    
                products = self.state.get("scored_products", [])
                console.print("\n[bold blue]Filtered Search Results:[/]")
                if products:
                    for product in products:
                        console.print(f"- {product['name']}")
                        console.print(f"  Price: ${product['price']}")
                        console.print(f"  Category: {product['category']}")
                else:
                    console.print("[red]No products found![/]")

            except Exception as e:
                self.logger.exception("Error in chat loop")
                console.print(f"\n[bold red]Error: {str(e)}[/]")
                console.print("\n[yellow]Try another message or type 'quit' to exit[/]")

def main():
    bot = TestBot()
    asyncio.run(bot.chat())

if __name__ == "__main__":
    main() 