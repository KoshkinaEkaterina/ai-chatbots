import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import pinecone
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.style import Style
from rich.text import Text
from product_recommendation import ProductRecommender
import logging
from rich.logging import RichHandler

# Setup colorful console
console = Console(color_system="auto")

class SearchTester:
    def __init__(self):
        load_dotenv()
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
        self.recommender = ProductRecommender(self.llm)
        self.logger = logging.getLogger(__name__)
        
        self.pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX"))

    def print_metadata_structure(self):
        """Print example metadata structure from index"""
        try:
            results = self.index.query(
                vector=[0]*1536,
                top_k=1,
                include_metadata=True
            )
            
            if results.matches:
                console.print(Panel(
                    json.dumps(results.matches[0].metadata, indent=2),
                    title="[bold cyan]INDEX STRUCTURE[/bold cyan]",
                    border_style="cyan"
                ))
            else:
                console.print("[bold red]No results found in index![/bold red]")
                
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")

    async def test_search(self, query: str):
        try:
            console.print(f"\n[bold yellow]Testing query:[/bold yellow] {query}")
            
            state = {
                "last_query": query,
                "messages": [],
                "current_criteria": {},
                "scored_products": []
            }
            
            # Analyze query
            criteria = await self.recommender.analyze_query(state)
            state["current_criteria"] = criteria
            
            console.print(Panel(
                json.dumps(criteria, indent=2),
                title="[bold green]Extracted Criteria[/bold green]",
                border_style="green"
            ))
            
            # Search products
            state = self.recommender.search_and_score_products(state)
            products = state.get("scored_products", [])
            
            # Create results table
            table = Table(title=f"[bold]Found {len(products)} Products[/bold]")
            table.add_column("Name", style="cyan")
            table.add_column("Price", justify="right", style="green")
            table.add_column("Category", style="magenta")
            table.add_column("Score", justify="right", style="yellow")
            
            for product in products[:5]:
                table.add_row(
                    product['name'],
                    f"${product['price']:.2f}",
                    " > ".join(product['category']),
                    f"{product['score']:.3f}"
                )
            
            console.print(table)
            
            return state
            
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
            raise

async def main():
    tester = SearchTester()
    
    # Print index structure
    tester.print_metadata_structure()
    
    # Test various queries
    test_queries = [
        "I need a toy under $200",
        "Looking for a laptop with at least 16GB RAM",
        "Show me office chairs under $500",
        "I want a gaming monitor bigger than 27 inches",
        "Need a desk that's at least 150cm wide",
        "Show me wireless keyboards under $100",
        "I need a lightweight backpack for travel",
        "Looking for noise-canceling headphones"
    ]
    
    for query in test_queries:
        try:
            await tester.test_search(query)
            console.print("\n[bold blue]" + "="*80 + "[/bold blue]\n")
        except Exception as e:
            console.print(f"[bold red]Failed query '{query}': {str(e)}[/bold red]")
            continue

if __name__ == "__main__":
    # Setup colorful logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
    asyncio.run(main()) 