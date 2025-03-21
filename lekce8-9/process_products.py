from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import AzureOpenAIEmbeddings
from tqdm import tqdm
import re
from datasets import load_dataset
import time

class ProductProcessor:
    def __init__(self):
        load_dotenv()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX")
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

    def parse_dimensions(self, spec_text: str) -> Dict[str, float]:
        """Parse dimensions from product specifications"""
        dimensions = {"length": None, "width": None, "height": None}
        
        if not spec_text or pd.isna(spec_text):
            return dimensions
            
        # Clean up the text
        spec_text = re.sub(r'\s+', '', spec_text)  # Remove all whitespace
        
        # Look for ProductDimensions or specific dimension patterns
        dim_match = re.search(r'ProductDimensions:?(\d+\.?\d*)x(\d+\.?\d*)x(\d+\.?\d*)(inches?|in)', spec_text, re.IGNORECASE)
        
        if dim_match:
            # Convert inches to cm
            dimensions["length"] = float(dim_match.group(1)) * 2.54
            dimensions["width"] = float(dim_match.group(2)) * 2.54
            dimensions["height"] = float(dim_match.group(3)) * 2.54
            
        return dimensions

    def parse_weights(self, spec_text: str) -> Dict[str, float]:
        """Parse item and shipping weights"""
        weights = {"item_weight": None, "shipping_weight": None}
        
        if not spec_text or pd.isna(spec_text):
            return weights
            
        # Clean up the text
        spec_text = re.sub(r'\s+', '', spec_text)
        
        # Item weight pattern
        item_match = re.search(r'ItemWeight:?(\d+\.?\d*)(pounds?|ounces?|oz)', spec_text, re.IGNORECASE)
        if item_match:
            value = float(item_match.group(1))
            unit = item_match.group(2).lower()
            # Convert to kg
            if 'pound' in unit:
                weights["item_weight"] = value * 0.45359237
            elif 'ounce' in unit or unit == 'oz':
                weights["item_weight"] = value * 0.0283495
        
        # Shipping weight pattern
        ship_match = re.search(r'ShippingWeight:?(\d+\.?\d*)(pounds?|ounces?|oz)', spec_text, re.IGNORECASE)
        if ship_match:
            value = float(ship_match.group(1))
            unit = ship_match.group(2).lower()
            # Convert to kg
            if 'pound' in unit:
                weights["shipping_weight"] = value * 0.45359237
            elif 'ounce' in unit or unit == 'oz':
                weights["shipping_weight"] = value * 0.0283495
        
        return weights

    def parse_price(self, price_text: str) -> float:
        """Parse price string to float"""
        if pd.isna(price_text):
            return 0.0
        
        # Remove currency symbols and convert to float
        price_str = re.sub(r'[^\d.]', '', str(price_text))
        try:
            return float(price_str)
        except ValueError:
            return 0.0

    def parse_categories(self, category_text: str) -> List[str]:
        """Parse category text into list of levels"""
        if pd.isna(category_text):
            return ["", "", "", ""]
        
        # Split on common separators and clean
        categories = re.split(r'\s*[>|/]\s*', category_text)
        categories = [c.strip() for c in categories if c.strip()]
        
        # Ensure exactly 4 levels
        categories.extend([""] * (4 - len(categories)))
        return categories[:4]

    def process_products(self, batch_size: int = 100, limit: int = 10**10):
        """Process products from HuggingFace dataset and upload to Pinecone"""
        print("Loading products from HuggingFace dataset...")
        dataset = load_dataset("ckandemir/amazon-products", split='train')
        
        print("\nDataset info:")
        print(f"Total products available: {len(dataset)}")
        
        # Map the dataset fields to our expected fields
        dataset = dataset.rename_columns({
            'Product Name': 'title',
            'Category': 'category',
            'Description': 'description',
            'Selling Price': 'price',
            'Product Specification': 'features'
        })
        
        # Get products up to limit
        limit = min(limit, len(dataset))
        print(f"Processing {limit} products...")
        
   
        try:
            self.pc.delete_index(self.index_name)
            print(f"\nCreating new index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric='cosine',
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
            print("Waiting for index to initialize...")
            time.sleep(20)  # Give it more time to initialize
            index = self.pc.Index(self.index_name)
        except Exception as e:
            print(f"Error with index setup: {str(e)}")
            return
        
        successful_count = 0
        error_count = 0
        
        def clean_metadata(metadata: dict) -> dict:
            """Clean metadata by removing None values and ensuring correct types"""
            cleaned = {}
            for key, value in metadata.items():
                if value is not None:  # Only include non-None values
                    if isinstance(value, (int, float)):
                        cleaned[key] = float(value)  # Convert all numbers to float
                    elif isinstance(value, str):
                        cleaned[key] = value[:500]  # Limit string length
                    elif isinstance(value, list):
                        cleaned[key] = [str(x)[:100] for x in value if x is not None]  # Clean lists
            return cleaned

        try:
            for i in tqdm(range(0, limit, batch_size)):
                batch = dataset.select(range(i, min(i + batch_size, limit)))
                vector_batch = []
                
                for idx, product in enumerate(batch):
                    try:
                        # Skip if essential data is missing
                        if not product['title'] or not product['description']:
                            error_count += 1
                            continue
                        
                        # Parse product data
                        categories = self.parse_categories(product['category'])
                        dimensions = self.parse_dimensions(product['features'])
                        weights = self.parse_weights(product['features'])
                        price = self.parse_price(product['price'])
                        
                        # Create search text and get embedding
                        search_text = f"{product['title']}\n{product['description']}"
                        embedding = self.embeddings.embed_query(search_text)
                        
                        # Prepare metadata
                        metadata = {
                            "product_name": product['title'],
                            "description": product['description'],
                            "price": price if price > 0 else None,
                            "category_l1": categories[0],
                            "category_l2": categories[1],
                            "category_l3": categories[2],
                            "category_l4": categories[3]
                        }
                        
                        # Add dimensions only if they exist
                        if dimensions["length"]: metadata["length"] = dimensions["length"]
                        if dimensions["width"]: metadata["width"] = dimensions["width"]
                        if dimensions["height"]: metadata["height"] = dimensions["height"]
                        
                        # Add weights only if they exist
                        if weights["item_weight"]: metadata["item_weight"] = weights["item_weight"]
                        if weights["shipping_weight"]: metadata["shipping_weight"] = weights["shipping_weight"]
                        
                        # Clean metadata and add to batch
                        vector_batch.append({
                            "id": f"prod_{i+idx}",
                            "values": embedding,
                            "metadata": clean_metadata(metadata)
                        })
                        
                        successful_count += 1
                        
                    except Exception as e:
                        error_count += 1
                        print(f"\nError processing product {i+idx}: {str(e)}")
                        continue
                
                # Batch upsert to Pinecone with retry
                if vector_batch:
                    for attempt in range(3):
                        try:
                            index.upsert(vectors=vector_batch)
                            print(f"\nUploaded batch of {len(vector_batch)} products")
                            break
                        except Exception as e:
                            if attempt == 2:
                                print(f"\nFailed to upload batch after 3 attempts: {str(e)}")
                                print("Sample metadata:", vector_batch[0]["metadata"])
                            else:
                                print(f"\nRetrying batch upload after error: {str(e)}")
                                time.sleep(5)
            
            # Final stats
            print("\nProcessing complete!")
            print(f"Successfully processed: {successful_count} products")
            print(f"Errors encountered: {error_count} products")
            print(f"Total products attempted: {limit}")
            
            stats = index.describe_index_stats()
            print(f"\nFinal index stats:")
            print(f"Total vectors in index: {stats.total_vector_count}")
            
        except Exception as e:
            print(f"\nFatal error during processing: {str(e)}")

def main():
    processor = ProductProcessor()
    processor.process_products()

if __name__ == "__main__":
    main() 