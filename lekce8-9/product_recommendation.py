from typing import Dict, List, Optional
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import pinecone
import json
import logging
from product_classes import QueryAnalysis, ProductNode, ProductCriteria, PriceRange, Intent, DimensionConstraint, WeightConstraint, ProductComparison, ComparisonResponse
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ProductRecommender:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        # Setup embeddings and vector store
        load_dotenv()
        self.embeddings = AzureOpenAIEmbeddings()
        self.pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX")

    async def analyze_query(self, state: Dict) -> Dict:
        """Extract search terms and criteria from any query"""
        try:
            query = state.get("last_query", "")
            self.logger.info(f"Analyzing query: '{query}'")

            prompt = f"""Extract search terms and criteria from this query: "{query}"

Return a JSON object with:
1. search_terms: Words describing what they're looking for (product type, description, etc.)
2. constraints: Any specific requirements like:
   - price limits
   - size/dimensions
   - weight
   - color
   - material
   - any other specific features

Example format:
{{
    "search_terms": ["desk", "office", "wooden"],
    "constraints": {{
        "price_max": 500,
        "dimensions": {{"width": 120, "unit": "cm"}},
        "material": "wood",
        "color": "brown"
    }}
}}"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                content = response.content.strip()
                if "```" in content:
                    content = content.split("```")[1].replace("json", "").strip()
                
                criteria = json.loads(content)
                self.logger.info(f"Extracted search criteria: {criteria}")
                return criteria
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse criteria: {response.content}")
                return {
                    "search_terms": [query],
                    "constraints": {}
                }
                
        except Exception as e:
            self.logger.exception("Error analyzing query")
            return {
                "search_terms": [query],
                "constraints": {}
            }

    def search_and_score_products(self, state: Dict) -> Dict:
        """Search products using extracted terms and criteria"""
        try:
            criteria = state.get("current_criteria", {})
            self.logger.info(f"Searching with criteria: {criteria}")
            
            # Build filter conditions
            filter_parts = []
            
            # Handle numeric constraints
            if constraints := criteria.get("constraints", {}):
                # Price constraints
                if "price_max" in constraints:
                    filter_parts.append({"price": {"$lte": float(constraints["price_max"])}})
                if "price_min" in constraints:
                    filter_parts.append({"price": {"$gte": float(constraints["price_min"])}})
                
                # Dimension constraints
                if dims := constraints.get("dimensions"):
                    for dim, value in dims.items():
                        if dim != "unit":  # Skip unit specification
                            filter_parts.append({dim.lower(): {"$lte": float(value)}})
                
                # Weight constraints
                if weight := constraints.get("weight"):
                    filter_parts.append({"item_weight": {"$lte": float(weight)}})

            # Get search terms
            search_terms = criteria.get("search_terms", [])
            query = " ".join(search_terms) if search_terms else state.get("last_query", "")
            
            self.logger.info(f"Using search terms: {query}")
            query_embedding = self.embeddings.embed_query(query)

            # Combine filters
            filter_conditions = {"$and": filter_parts} if len(filter_parts) > 1 else filter_parts[0] if filter_parts else None
            
            self.logger.info(f"Using filters: {json.dumps(filter_conditions, indent=2)}")
            
            # Execute search
            results = self.pc.Index(self.index_name).query(
                vector=query_embedding,
                filter=filter_conditions,
                top_k=10,
                include_metadata=True
            )
            
            self.logger.info(f"Found {len(results.matches)} matches")
            
            # Process results
            products = []
            for match in results.matches:
                try:
                    metadata = match.metadata
                    self.logger.debug(f"Processing match: {metadata}")
                    
                    # Extract all available metadata
                    product = {
                        "id": match.id,
                        "name": metadata.get("product_name", "Unknown"),
                        "price": float(metadata.get("price", 0)),
                        "description": metadata.get("description", ""),
                        "score": match.score,
                        "confidence": match.score
                    }
                    
                    # Add any available dimensions
                    dimensions = {}
                    for dim in ["length", "width", "height", "depth"]:
                        if dim in metadata:
                            dimensions[dim] = float(metadata[dim])
                    if dimensions:
                        product["dimensions"] = dimensions
                    
                    # Add weight if available
                    if "item_weight" in metadata:
                        product["weight"] = float(metadata["item_weight"])
                    
                    # Add all available categories
                    categories = []
                    for i in range(1, 5):
                        if cat := metadata.get(f"category_l{i}"):
                            categories.append(cat)
                    product["category"] = categories
                    
                    # Add any other metadata that might be useful
                    for key, value in metadata.items():
                        if key not in ["product_name", "price", "description", "length", "width", "height", "depth", "item_weight"]:
                            if key.startswith("category_l"):
                                continue
                            product[key] = value
                    
                    products.append(product)
                    
                except Exception as e:
                    self.logger.error(f"Error processing product {match.id}: {e}")
                    continue
            
            state["scored_products"] = products
            self.logger.info(f"Successfully processed {len(products)} products")
            
            return state
            
        except Exception as e:
            self.logger.exception("Product search failed")
            state["scored_products"] = []
            return state

    async def generate_recommendations(self, state: Dict) -> Dict:
        """Generate product recommendations with explanations"""
        try:
            products = state.get("scored_products", [])
            if not products:
                return {
                    "message": "I couldn't find any products matching your criteria. Could you try rephrasing your search?",
                    "products": [],
                    "criteria": state.get("current_criteria", {})
                }
            
            # Deduplicate products by name
            unique_products = {}
            for p in products:
                if p["name"] not in unique_products:
                    unique_products[p["name"]] = p
            
            top_products = list(unique_products.values())[:5]
            
            # Format the response as markdown
            markdown_response = "# Product Recommendations\n\n"
            markdown_response += f"I found {len(top_products)} products matching your criteria:\n\n"
            
            for product in top_products:
                markdown_response += f"## {product['name']}\n"
                markdown_response += f"- Price: ${product['price']:.2f}\n"
                if product.get('category'):
                    markdown_response += f"- Category: {' > '.join(product['category'])}\n"
                if product.get('description'):
                    markdown_response += f"- Description: {product['description']}\n"
                markdown_response += "\n"
            
            return {
                "message": markdown_response,
                "products": top_products,
                "criteria": state.get("current_criteria", {})
            }
            
        except Exception as e:
            self.logger.exception("Recommendation generation failed")
            return {
                "message": "I encountered an error analyzing these products. Please try again.",
                "products": products[:5] if products else [],
                "criteria": state.get("current_criteria", {})
            }