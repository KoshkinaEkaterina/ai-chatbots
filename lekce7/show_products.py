from dotenv import load_dotenv
import os
from pinecone import Pinecone
from rich.console import Console
from rich.table import Table

def show_products():
    # Load environment variables
    load_dotenv()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX")
    
    # Get index
    index = pc.Index(index_name)
    
    # List first 20 IDs
    try:
        # First, get stats to see if we have any vectors
        stats = index.describe_index_stats()
        console = Console()
        console.print(f"\n[bold]Index Stats:[/bold]")
        console.print(f"Total vectors: {stats.total_vector_count}")
        
        # Get list of IDs (first 20)
        results = index.query(
            vector=[1] * 1536,  # Unit vector instead of zero vector
            top_k=20,
            include_metadata=True
        )
        
        if not results.matches:
            console.print("\n[bold red]No products found in index![/bold red]")
            console.print("Make sure you've run process_products.py first to populate the index.")
            return
            
        # Create pretty table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim")
        table.add_column("Name")
        table.add_column("Category")
        table.add_column("Price", justify="right")
        table.add_column("Dimensions (cm)", justify="right")
        
        # Add products to table
        for match in results.matches:
            metadata = match.metadata
            dimensions = []
            if metadata.get("length"): dimensions.append(f"L:{metadata['length']:.1f}")
            if metadata.get("width"): dimensions.append(f"W:{metadata['width']:.1f}")
            if metadata.get("height"): dimensions.append(f"H:{metadata['height']:.1f}")
            dims = " x ".join(dimensions) if dimensions else "N/A"
            
            category = " > ".join(filter(None, [
                metadata.get("category_l1", ""),
                metadata.get("category_l2", ""),
                metadata.get("category_l3", ""),
                metadata.get("category_l4", "")
            ]))
            
            table.add_row(
                match.id,
                metadata.get("product_name", "N/A")[:100],  # Truncate long names
                category[:50],  # Truncate long categories
                f"${metadata.get('price', 0):.2f}",
                dims
            )
        
        # Print table
        console.print("\n[bold]First 20 Products in Index[/bold]\n")
        console.print(table)
        
    except Exception as e:
        console = Console()
        console.print(f"\n[bold red]Error accessing index: {str(e)}[/bold red]")
        console.print("Make sure your index exists and you have the correct credentials.")

if __name__ == "__main__":
    show_products() 