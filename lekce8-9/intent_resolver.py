from typing import Dict, List, Optional, Callable
from pydantic import BaseModel, Field
import json
import logging
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

class IntentResolver:
    """Handles resolution of specific intents with appropriate responses"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.logger = logging.getLogger(__name__)


    
    async def handle_comparison(self, state: Dict, search_func: Optional[Callable] = None) -> Dict:
        """Handle comparison intent - analyze and compare products with detailed feature breakdown"""
        self.logger.info("=== Starting COMPARISON handler ===")
        self.logger.debug(f"Initial state: {state}")
        products = state.get("scored_products", [])
        
        self.logger.debug(f"Found {len(products)} products to compare")
        
        if not products:
            self.logger.warning("No products found for comparison")
            state["messages"].append({
                "role": "assistant",
                "content": "I don't have any products to compare yet. Could you tell me what kind of products you're interested in?"
            })
            return state

        try:
            # Extract key features from all products for comparison
            all_features = set()
            for product in products:
                self.logger.debug(f"Extracting features from product: {product['name']}")
                # Get all possible feature keys from product data
                all_features.update(product.get("features", {}).keys())
                if "dimensions" in product:
                    all_features.update(product["dimensions"].keys())
                all_features.update(["price", "weight", "material", "brand"])

            self.logger.debug(f"Extracted features: {all_features}")

            # Select top 3 most diverse products based on feature differences
            scored_differences = []
            for i, prod1 in enumerate(products):
                for j, prod2 in enumerate(products[i+1:], i+1):
                    self.logger.debug(f"Comparing {prod1['name']} with {prod2['name']}")
                    diff_score = 0
                    # Compare features between products
                    for feature in all_features:
                        val1 = prod1.get(feature) or prod1.get("features", {}).get(feature)
                        val2 = prod2.get(feature) or prod2.get("features", {}).get(feature)
                        if val1 != val2:
                            diff_score += 1
                    # Consider price range differences
                    price_diff = abs(prod1["price"] - prod2["price"])
                    diff_score += price_diff / 100  # Weight price differences
                    scored_differences.append((diff_score, i, j))

            # Select 3 products with maximum differences
            selected_indices = set()
            for _, i, j in sorted(scored_differences, reverse=True):
                if len(selected_indices) < 3:
                    selected_indices.add(i)
                    selected_indices.add(j)
            
            top_products = [products[i] for i in list(selected_indices)[:3]]

            # Create detailed comparison prompt
            prompt = f"""Analyze these 3 products and create a detailed comparison:

Products:
{json.dumps([{
    'name': p['name'],
    'price': p['price'],
    'features': p.get('features', {}),
    'category': p['category'],
    'description': p['description'],
    'dimensions': p.get('dimensions', {}),
    'weight': p.get('weight'),
    'material': p.get('material'),
    'brand': p.get('brand'),
    'score': p['score']
} for p in top_products], indent=2)}

Create a detailed breakdown that:
1. For EACH product:
   - List its unique strengths
   - Identify its ideal use case
   - Highlight where it outperforms others
   - Note any limitations

2. Direct comparisons:
   - Feature-by-feature analysis
   - Price-to-feature value comparison
   - Quality and durability assessment
   - Performance in specific scenarios

3. Clear recommendations:
   - Best overall value
   - Best premium option
   - Best budget choice
   - Specific user scenarios for each

Format as markdown with clear sections. Use tables for feature comparisons where relevant."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Format final response with structured comparison
            markdown_response = f"""# Detailed Product Comparison Analysis

{response.content}

## Feature Comparison Table

| Feature | {" | ".join([p['name'] for p in top_products])} |
|---------|{"|".join(["------" for _ in top_products])}|
| Price | {" | ".join([f"${p['price']:.2f}" for p in top_products])} |
"""
            # Add common features to comparison table
            for feature in sorted(all_features):
                if feature not in ['price', 'name', 'description', 'category', 'score']:
                    markdown_response += f"| {feature.title()} | "
                    for p in top_products:
                        val = p.get(feature) or p.get("features", {}).get(feature) or "N/A"
                        markdown_response += f"{val} | "
                    markdown_response += "\n"

            state["messages"].append({
                "role": "assistant",
                "content": markdown_response
            })

        except Exception as e:
            self.logger.exception("Error in comparison handler")
            state["messages"].append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error while comparing the products."
            })
        
        self.logger.debug(f"State after comparison: {state}")
        return state

    async def handle_requirements(self, state: Dict) -> Dict:
        """Handle requirements gathering and product compatibility questions"""
        self.logger.info("=== Starting REQUIREMENTS handler ===")
        self.logger.debug(f"Initial state: {state}")
        criteria = state.get("current_criteria", {})
        self.logger.debug(f"Current criteria: {criteria}")
        category = criteria.get("category")
        last_query = state.get("last_query", "")
        current_product = state.get("current_product")
        
        try:
            if current_product:
                self.logger.debug(f"Handling requirements for specific product: {current_product['name']}")
                # If we have a specific product, handle compatibility/requirements for it
                prompt = f"""Given this product and user query about requirements/compatibility:
                
Product: {json.dumps(current_product, indent=2)}
Query: {last_query}

Provide a detailed response about:
1. System requirements
2. Compatibility with other products/systems
3. Installation requirements
4. Usage requirements
5. Any limitations or restrictions

Response should be clear and structured."""
            else:
                self.logger.debug("Handling general requirements gathering")
                # Otherwise, help gather requirements for product search
                prompt = f"""Help gather product requirements from the user.

Current Context:
- Last Query: {last_query}
- Category: {category if category else 'Not specified'}
- Current Criteria: {json.dumps(criteria, indent=2)}

Create a helpful response that:
1. Acknowledges any requirements already mentioned
2. Asks specific questions about:
   - Primary use case/purpose
   - Must-have features
   - Budget constraints
   - Size/space limitations
   - Quality/durability needs
   - Brand preferences
   - Specific deal-breakers
3. Explains why each requirement is important
4. Provides examples of good responses

Format the response in a friendly, conversational way using markdown."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            markdown_response = response.content
            if not current_product:
                markdown_response = f"""# Let's Find Your Perfect Product

{response.content}

---
*Feel free to answer any or all of these questions - the more you share, the better I can help find the right product for you!*"""

            state["messages"].append({
                "role": "assistant",
                "content": markdown_response
            })

        except Exception as e:
            self.logger.exception("Error in requirements handler")
            state["messages"].append({
                "role": "assistant",
                "content": """I'd love to help you find the right product! Could you tell me:
1. What will you primarily use it for?
2. Do you have any specific features in mind?
3. What's your budget range?"""
            })
        
        self.logger.debug(f"State after requirements: {state}")
        return state

    async def handle_purchase(self, state: Dict, search_func: Optional[Callable] = None) -> Dict:
        """Handle purchase intent"""
        self.logger.info("=== Starting PURCHASE handler ===")
        self.logger.debug(f"Initial state: {state}")
        products = state.get("scored_products", [])
        criteria = state.get("current_criteria", {})
        self.logger.debug(f"Found {len(products)} products matching criteria: {criteria}")

        if not products:
            if not criteria:
                state["messages"].append({
                    "role": "assistant",
                    "content": "What kind of product are you looking to purchase? Please tell me your requirements."
                })
                return state
            
            if search_func:
                state = search_func(state)
                products = state.get("scored_products", [])

        if not products:
            state["messages"].append({
                "role": "assistant",
                "content": "I couldn't find any products matching your criteria. Could you adjust your requirements?"
            })
            return state

        # Generate recommendations for found products
        prompt = f"""Recommend from these specific products:

Products Available:
{json.dumps([{
    'name': p['name'],
    'price': p['price'],
    'category': p['category'],
    'description': p['description']
} for p in products[:5]], indent=2)}

User's Criteria:
{json.dumps(criteria, indent=2)}

Provide recommendations focusing on:
1. Best matches for their needs
2. Value for money
3. Key features and benefits

Format as markdown. Only recommend from these products."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            state["messages"].append({
                "role": "assistant",
                "content": response.content
            })
            self.logger.debug("Generated purchase recommendations")
        except Exception as e:
            self.logger.exception("Error in purchase handler")
            state["messages"].append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error while generating recommendations."
            })
        
        return state

    async def handle_upgrade(self, state: Dict, search_func: Optional[Callable] = None) -> Dict:
        """Handle upgrade intent"""
        self.logger.info("=== Starting UPGRADE handler ===")
        self.logger.debug(f"Initial state: {state}")
        products = state.get("scored_products", [])
        criteria = state.get("current_criteria", {})

        if not products and search_func:
            state = search_func(state)
            products = state.get("scored_products", [])

        if not products:
            state["messages"].append({
                "role": "assistant",
                "content": "Could you tell me what product you're looking to upgrade from? "
                "This will help me find better alternatives."
            })
            return state

        # Generate upgrade recommendations
        prompt = f"""Recommend upgrades from these products:

Available Products:
{json.dumps([{
    'name': p['name'],
    'price': p['price'],
    'category': p['category'],
    'description': p['description']
} for p in products[:5]], indent=2)}

Focus on:
1. Improved features and capabilities
2. Value for the upgrade cost
3. Key benefits over basic models

Format as markdown. Only recommend from these products."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            state["messages"].append({
                "role": "assistant",
                "content": response.content
            })
            self.logger.debug("Generated upgrade recommendations")
        except Exception as e:
            self.logger.exception("Error in upgrade handler")
            state["messages"].append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error while suggesting upgrades."
            })
        
        return state

    # Add other handlers as needed... 

    async def handle_first_use(self, state: Dict) -> Dict:
        """Handle first-time usage instructions for specific product or product category"""
        self.logger.info("=== Starting FIRST_USE handler ===")
        self.logger.debug(f"Initial state: {state}")
        try:
            products = state.get("scored_products", [])
            current_product = state.get("current_product")
            last_query = state.get("last_query", "")
            
            self.logger.debug(f"Products available: {len(products)}, Current product: {current_product and current_product['name']}")
            
            if current_product:
                # Handle specific product first-use instructions
                prompt = f"""Create first-time usage instructions for this specific product:
                
Product: {json.dumps(current_product, indent=2)}
User Query: {last_query}

Include:
1. Unboxing and setup steps
2. Initial configuration process
3. Basic usage guide
4. Common first-time issues and solutions
5. Tips for beginners
6. Safety precautions
7. Maintenance recommendations

Make it clear and easy to follow."""

            elif products:
                # Get the first product and extract category info
                first_product = products[0]
                category = first_product.get("category", "")
                
                # Create category-based guidance using available products
                prompt = f"""Create first-time usage guidance for {category} products, based on these examples:

Products:
{json.dumps([{
    'name': p['name'],
    'category': p['category'],
    'description': p['description'],
    'features': p.get('features', {}),
} for p in products[:3]], indent=2)}

User Query: {last_query}

Provide:
1. General setup guidelines for this type of product
2. Key considerations before first use
3. Common features and how to use them
4. Basic troubleshooting tips
5. Maintenance best practices
6. Safety guidelines
7. What to expect during first use
8. Tips for getting the most value

Include specific examples from the available products when relevant."""

            else:
                state["messages"].append({
                    "role": "assistant",
                    "content": "I'd be happy to provide first-use instructions! Could you tell me which product or type of product you're interested in?"
                })
                return state

            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Format response with clear sections
            markdown_response = f"""# First-Time Usage Guide
{response.content}

---
*If you have any specific questions about setup or usage, feel free to ask!*"""

            state["messages"].append({
                "role": "assistant",
                "content": markdown_response
            })

        except Exception as e:
            self.logger.exception("Error handling first use")
            state["messages"].append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error while creating the usage guide. Could you please try again or rephrase your question?"
            })
            
        self.logger.debug(f"State after first use: {state}")
        return state

    async def handle_comparison(self, state: Dict, search_products_func) -> Dict:
        """Compare products based on user criteria"""
        self.logger.info("=== Starting COMPARISON handler ===")
        try:
            current_product = state["current_product"]
            search_context = state["search_context"]
            
            self.logger.debug(f"Comparing product: {current_product['name']}")
            
            # Find similar products using the passed search function
            similar_products = await search_products_func(search_context)
            
            prompt = f"""Compare these products based on user's criteria:

Current Product: {json.dumps(current_product, indent=2)}
Similar Products: {json.dumps([p.dict() for p in similar_products[:3]], indent=2)}
User Criteria: {json.dumps(search_context, indent=2)}

Provide:
1. Feature comparison
2. Price comparison
3. Pros and cons
4. Best use cases
5. Value for money analysis

Make it objective and detailed."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            self.logger.debug("Generated comparison response")
            
            state["messages"].append({
                "role": "assistant",
                "content": response.content
            })
            return state

        except Exception as e:
            self.logger.exception("Error handling comparison")
            return state

   
    async def handle_support(self, state: Dict) -> Dict:
        """Handle product support questions"""
        self.logger.info("=== Starting SUPPORT handler ===")
        self.logger.debug(f"Initial state: {state}")
        try:
            product = state["current_product"]
            query = state["last_query"]
            
            self.logger.debug(f"Support query for product: {product['name']}")
            
            prompt = f"""Address this support question for the product:

Product: {json.dumps(product, indent=2)}
Question: {query}

Provide:
1. Direct answer to the question
2. Troubleshooting steps if needed
3. Maintenance tips
4. Common solutions
5. When to seek professional help

Make it practical and easy to follow."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            self.logger.debug("Generated support response")
            
            state["messages"].append({
                "role": "assistant",
                "content": response.content
            })
            return state

        except Exception as e:
            self.logger.exception("Error handling support")
            return state

    async def handle_replacement(self, state: Dict, search_products_func) -> Dict:
        """Handle product replacement inquiries"""
        self.logger.info("=== Starting REPLACEMENT handler ===")
        self.logger.debug(f"Initial state: {state}")
        try:
            current_product = state["current_product"]
            search_context = state["search_context"]
            
            self.logger.debug(f"Finding replacements for: {current_product['name']}")
            
            prompt = f"""Help user replace their current product:

Current Product: {json.dumps(current_product, indent=2)}
Search Context: {json.dumps(search_context, indent=2)}

Provide guidance on:
1. Direct replacements available
2. Upgraded alternatives
3. Key differences from current productx
4. Installation/migration considerations
5. Price comparison
6. Compatibility checks

Focus on making the transition smooth."""

            # Find replacement options
            similar_products = await search_products_func(search_context)
            
            if similar_products:
                product_info = "\n\nRecommended replacements:\n" + \
                    "\n".join([f"- {p.name}: ${p.price} - {p.description[:100]}..." 
                              for p in similar_products[:3]])
            else:
                product_info = "\n\nNo direct replacements found."

            response = self.llm.invoke([HumanMessage(content=prompt + product_info)])
            
            self.logger.debug("Generated replacement recommendations")
            state["messages"].append({
                "role": "assistant",
                "content": response.content
            })
            return state

        except Exception as e:
            self.logger.exception("Error handling replacement")
            return state


    async def handle_warranty(self, state: Dict) -> Dict:
        """Handle warranty and return questions"""
        self.logger.info("=== Starting WARRANTY handler ===")
        self.logger.debug(f"Initial state: {state}")
        try:
            product = state.get("current_product")
            query = state.get("last_query")
            
            self.logger.debug(f"Handling warranty query: {query}")
            self.logger.debug(f"Current product: {product}")
            
            prompt = f"""Address warranty and return questions for:

Product: {json.dumps(product, indent=2)}
Question: {query}

Cover:
1. Warranty terms and duration
2. What's covered/not covered
3. Return policy details
4. Claim process
5. Required documentation
6. Contact information
7. Common warranty issues

Be specific about terms and conditions."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            self.logger.debug("Generated warranty response")
            
            state["messages"].append({
                "role": "assistant",
                "content": response.content
            })
            return state

        except Exception as e:
            self.logger.exception("Error in WARRANTY handler")
            state["messages"].append({
                "role": "assistant",
                "content": "I apologize, but I encountered an error processing your warranty question. Could you try asking in a different way?"
            })
            return state

   
   