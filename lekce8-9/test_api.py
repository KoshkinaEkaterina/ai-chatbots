import asyncio
import aiohttp
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from typing import Dict, List
import os
from dotenv import load_dotenv

load_dotenv()
console = Console()

class APITester:
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.conversation_id = None
        
    async def send_message(self, message: str) -> Dict:
        """Send a message to the chat endpoint"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/chat"
            payload = {
                "message": message,
                "conversation_id": self.conversation_id
            }
            
            console.print(f"\n[bold blue]Sending message:[/] {message}")
            
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    console.print(f"[bold red]Error:[/] {response.status}")
                    return None
                
                data = await response.json()
                self.conversation_id = data.get("conversation_id")
                return data

    def print_products(self, products: List[Dict]):
        """Print products in a nice table"""
        if not products:
            console.print("[yellow]No products found[/]")
            return
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Name")
        table.add_column("Price")
        table.add_column("Category")
        table.add_column("Score")
        
        for product in products:
            category = " > ".join(product["category"]) if isinstance(product["category"], list) else product["category"]
            table.add_row(
                product["name"],
                f"${product['price']:.2f}",
                category,
                f"{product['confidence']*100:.1f}%"
            )
        
        console.print("\n[bold green]Found Products:[/]")
        console.print(table)

    def print_criteria(self, criteria: Dict):
        """Print search criteria"""
        if not criteria:
            return
        
        console.print("\n[bold cyan]Search Criteria:[/]")
        
        if price_range := criteria.get("price_range"):
            price_str = ""
            if price_range.get("min"):
                price_str += f"Min: ${price_range['min']}"
            if price_range.get("max"):
                if price_str:
                    price_str += " - "
                price_str += f"Max: ${price_range['max']}"
            if price_str:
                console.print(f"Price Range: {price_str}")
        
        if category := criteria.get("category"):
            console.print(f"Category: {category}")
        
        if features := criteria.get("features"):
            console.print(f"Features: {', '.join(features)}")
        
        if requirements := criteria.get("requirements"):
            console.print(f"Requirements: {', '.join(requirements)}")

    async def test_queries(self):
        """Test a series of queries"""
        test_queries = [
            "I need a toy under $20",
            "Show me electronics between $100 and $500",
            "I want kitchen appliances under $200",
            "Find me some headphones under $100",
            "I need a laptop for gaming under $1500"
        ]
        
        for query in test_queries:
            console.print("\n" + "="*80)
            response = await self.send_message(query)
            
            if response:
                # Print assistant's message
                console.print("\n[bold green]Assistant:[/]")
                console.print(Markdown(response["message"]))
                
                # Print criteria and products
                self.print_criteria(response.get("criteria"))
                self.print_products(response.get("products", []))
            
            await asyncio.sleep(1)  # Small delay between requests

    async def interactive_chat(self):
        """Interactive chat with the API"""
        console.print("[bold blue]Chat with Product Agent API[/]")
        console.print("Type 'quit' to exit")
        
        while True:
            try:
                message = input("\nYou: ").strip()
                if message.lower() in ['quit', 'exit', 'bye']:
                    break
                
                response = await self.send_message(message)
                if response:
                    # Print assistant's message
                    console.print("\n[bold green]Assistant:[/]")
                    console.print(Markdown(response["message"]))
                    
                    # Print criteria and products
                    self.print_criteria(response.get("criteria"))
                    self.print_products(response.get("products", []))
                
            except Exception as e:
                console.print(f"[bold red]Error:[/] {str(e)}")

    async def test_chair_search(self):
        """Test chair search functionality with various queries"""
        chair_test_queries = [
            {
                "query": "I need an office chair under $300",
                "expected_category": "chair",
                "expected_price_max": 300.0,
                "description": "Basic chair search with price limit"
            },
            {
                "query": "Looking for a gaming chair between $200 and $400",
                "expected_category": "chair",
                "expected_price_min": 200.0,
                "expected_price_max": 400.0,
                "description": "Gaming chair with price range"
            },
            {
                "query": "Need a comfortable desk chair with armrests",
                "expected_category": "chair",
                "features": ["armrests", "comfortable"],
                "description": "Chair search with specific features"
            },
            {
                "query": "Kids chair for study desk under $150",
                "expected_category": "chair",
                "expected_price_max": 150.0,
                "description": "Kids chair with price limit"
            }
        ]

        console.print("\n[bold blue]Running Chair Search Tests[/]")
        console.print("=" * 80)

        for test in chair_test_queries:
            console.print(f"\n[bold cyan]Test Case:[/] {test['description']}")
            console.print(f"Query: {test['query']}")
            
            response = await self.send_message(test['query'])
            
            if response:
                # Print extracted criteria
                if criteria := response.get("criteria"):
                    console.print("\n[bold green]Extracted Criteria:[/]")
                    if price_range := criteria.get("price_range"):
                        console.print(f"Price Range: {price_range}")
                    if category := criteria.get("category"):
                        console.print(f"Category: {category}")
                    if features := criteria.get("features"):
                        console.print(f"Features: {features}")
                    
                    # Validate criteria
                    validation_errors = []
                    if test.get('expected_category') and category != test['expected_category']:
                        validation_errors.append(f"Expected category '{test['expected_category']}', got '{category}'")
                    if test.get('expected_price_max') and (not price_range or price_range.get('max') != test['expected_price_max']):
                        validation_errors.append(f"Expected max price {test['expected_price_max']}, got {price_range.get('max') if price_range else 'None'}")
                    if test.get('expected_price_min') and (not price_range or price_range.get('min') != test['expected_price_min']):
                        validation_errors.append(f"Expected min price {test['expected_price_min']}, got {price_range.get('min') if price_range else 'None'}")
                    
                    if validation_errors:
                        console.print("\n[bold red]Validation Errors:[/]")
                        for error in validation_errors:
                            console.print(f"- {error}")
                
                # Print found products
                if products := response.get("products"):
                    console.print("\n[bold green]Found Products:[/]")
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Name")
                    table.add_column("Price")
                    table.add_column("Category")
                    table.add_column("Score")
                    
                    for product in products[:5]:  # Show top 5 products
                        category = " > ".join(product["category"]) if isinstance(product["category"], list) else product["category"]
                        table.add_row(
                            product["name"],
                            f"${product['price']:.2f}",
                            category,
                            f"{product['confidence']*100:.1f}%"
                        )
                    
                    console.print(table)
                    
                    # Validate products
                    if not any("chair" in (p.get("category", "").lower() if isinstance(p.get("category"), str) 
                              else " ".join(p.get("category", [])).lower()) for p in products):
                        console.print("[bold red]Warning:[/] No chair products found in results!")
                else:
                    console.print("[bold red]No products found![/]")
            
            console.print("\n" + "=" * 80)

async def main():
    tester = APITester()
    
    console.print("\n[bold]Choose test mode:[/]")
    console.print("1. Run test queries")
    console.print("2. Interactive chat")
    console.print("3. Run chair search tests")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        await tester.test_queries()
    elif choice == "2":
        await tester.interactive_chat()
    elif choice == "3":
        await tester.test_chair_search()
    else:
        console.print("[bold red]Invalid choice![/]")

if __name__ == "__main__":
    asyncio.run(main()) 