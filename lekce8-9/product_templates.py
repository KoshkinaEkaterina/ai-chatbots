from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class ProductUsageIntent(Enum):
    """Different ways a user might want to use or learn about a product"""
    
    # Basic Product Information
    FEATURES = "what are the main features"
    SPECS = "technical specifications"
    COMPARISON = "compare with similar products"
    
    # Usage & Setup
    SETUP = "how to set up or install"
    FIRST_USE = "how to use for the first time"
    MAINTENANCE = "how to maintain or clean"
    STORAGE = "how to store properly"
    TROUBLESHOOT = "how to fix common issues"
    
    # Safety & Requirements
    SAFETY = "safety information"
    REQUIREMENTS = "requirements and compatibility"
    AGE_RESTRICTIONS = "age and user restrictions"
    
    # Advanced & Care
    ADVANCED_USE = "advanced usage tips"
    CUSTOMIZATION = "customization options"
    ACCESSORIES = "compatible accessories"
    CARE = "care and maintenance"
    
    # Support & Disposal
    WARRANTY = "warranty information"
    SUPPORT = "customer support"
    DISPOSAL = "disposal and recycling"

    GAMING = "gaming"
    WORK = "work"
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    ENTERTAINMENT = "entertainment"

class ProductAdviceGenerator:
    """Generates contextual product advice based on intent"""
    
    def __init__(self):
        self.templates = {
            # Basic Product Information
            ProductUsageIntent.FEATURES: (
                "Key Features - {product}:\n\n"
                "Product Category: {category}\n"
                "Size: {dimensions}\n"
                "Price Point: {price}\n\n"
                "Description:\n{desc}\n\n"
                "Refer to product manual for complete feature list."
            ),
            
            ProductUsageIntent.SPECS: (
                "Technical Specifications - {product}:\n\n"
                "Physical Dimensions: {dimensions}\n"
                "Category: {category}\n"
                "Price: {price}\n\n"
                "Additional Specifications:\n{desc}\n\n"
                "For detailed specifications, consult product documentation."
            ),
            
            ProductUsageIntent.COMPARISON: (
                "Product Comparison Guide - {product}:\n\n"
                "Current Product:\n"
                "- Category: {category}\n"
                "- Size: {dimensions}\n"
                "- Price: {price}\n"
                "- Features: {desc}\n\n"
                "Compare these specifications with other products in the same category."
            ),
            
            # Usage & Setup
            ProductUsageIntent.SETUP: (
                "Setup Guide - {product}:\n\n"
                "1. Required Space: {dimensions}\n"
                "2. Unpack and verify all components\n"
                "3. Follow included assembly/setup instructions\n"
                "4. Test all functions before regular use\n"
                "5. Review {category} specific guidelines\n\n"
                "Additional Notes:\n{desc}"
            ),
            
            ProductUsageIntent.FIRST_USE: (
                "First Time Use - {product}:\n\n"
                "1. Confirm proper setup/assembly\n"
                "2. Review safety guidelines\n"
                "3. Start with basic functions\n"
                "4. Test within product limits ({dimensions})\n"
                "5. Keep manual accessible\n\n"
                "Important Information:\n{desc}"
            ),
            
            ProductUsageIntent.MAINTENANCE: (
                "Maintenance Guide - {product}:\n\n"
                "Regular Maintenance:\n"
                "1. Clean according to {category} guidelines\n"
                "2. Inspect all components\n"
                "3. Check for wear (especially at {dimensions} points)\n"
                "4. Address any issues immediately\n"
                "5. Document maintenance performed\n\n"
                "Product-Specific Care:\n{desc}"
            ),
            
            ProductUsageIntent.STORAGE: (
                "Storage Instructions - {product}:\n\n"
                "1. Required Storage Space: {dimensions}\n"
                "2. Clean before storage\n"
                "3. Protect from environmental factors\n"
                "4. Follow {category} storage guidelines\n"
                "5. Regular checks during storage\n\n"
                "Special Considerations:\n{desc}"
            ),
            
            ProductUsageIntent.TROUBLESHOOT: (
                "Troubleshooting Guide - {product}:\n\n"
                "Common Checks:\n"
                "1. Verify proper assembly\n"
                "2. Check all connections\n"
                "3. Confirm proper space ({dimensions})\n"
                "4. Review {category} requirements\n"
                "5. Inspect for damage\n\n"
                "Product-Specific Notes:\n{desc}"
            ),
            
            # Safety & Requirements
            ProductUsageIntent.SAFETY: (
                "Safety Information - {product}:\n\n"
                "Key Safety Points:\n"
                "1. Required clearance: {dimensions}\n"
                "2. Follow {category} safety guidelines\n"
                "3. Regular safety checks\n"
                "4. Proper use limitations\n"
                "5. Emergency procedures\n\n"
                "Important Safety Notes:\n{desc}"
            ),
            
            ProductUsageIntent.REQUIREMENTS: (
                "Requirements & Compatibility - {product}:\n\n"
                "Physical Requirements:\n"
                "- Space needed: {dimensions}\n"
                "- Category standards: {category}\n\n"
                "Additional Requirements:\n{desc}\n\n"
                "Price Consideration: {price}"
            ),
            
            ProductUsageIntent.AGE_RESTRICTIONS: (
                "Usage Restrictions - {product}:\n\n"
                "Product Information:\n"
                "- Category: {category}\n"
                "- Size: {dimensions}\n"
                "- Price: {price}\n\n"
                "Restrictions & Requirements:\n{desc}"
            ),
            
            # Advanced & Care
            ProductUsageIntent.ADVANCED_USE: (
                "Advanced Usage Guide - {product}:\n\n"
                "Product Specifications:\n"
                "- Dimensions: {dimensions}\n"
                "- Category: {category}\n\n"
                "Advanced Features & Tips:\n{desc}"
            ),
            
            ProductUsageIntent.CUSTOMIZATION: (
                "Customization Options - {product}:\n\n"
                "Base Product:\n"
                "- Size: {dimensions}\n"
                "- Category: {category}\n"
                "- Base Price: {price}\n\n"
                "Customization Information:\n{desc}"
            ),
            
            ProductUsageIntent.ACCESSORIES: (
                "Accessories Guide - {product}:\n\n"
                "Main Product Details:\n"
                "- Dimensions: {dimensions}\n"
                "- Category: {category}\n"
                "- Price: {price}\n\n"
                "Compatible Accessories:\n{desc}"
            ),
            
            # Support & Disposal
            ProductUsageIntent.WARRANTY: (
                "Warranty Information - {product}:\n\n"
                "Product Details:\n"
                "- Category: {category}\n"
                "- Purchase Price: {price}\n\n"
                "Warranty Terms:\n{desc}"
            ),
            
            ProductUsageIntent.SUPPORT: (
                "Support Information - {product}:\n\n"
                "Product Identification:\n"
                "- Category: {category}\n"
                "- Size: {dimensions}\n"
                "- Price: {price}\n\n"
                "Support Details:\n{desc}"
            ),
            
            ProductUsageIntent.DISPOSAL: (
                "Disposal Guidelines - {product}:\n\n"
                "Product Information:\n"
                "- Category: {category}\n"
                "- Size: {dimensions}\n\n"
                "Disposal Instructions:\n{desc}"
            )
        }
        self.logger = logging.getLogger(__name__)
    
    def get_advice(self, intent: ProductUsageIntent, product_data: Dict) -> str:
        """Generate advice based on product data and intent"""
        try:
            # Format dimensions for display
            dims = product_data.get("dimensions", {})
            dimensions = f"{dims.get('length', 0)}x{dims.get('width', 0)}x{dims.get('height', 0)}cm"
            
            # Prepare template data
            template_data = {
                "product": product_data.get("name", "Product"),
                "category": product_data.get("category", ["Product"])[-1],
                "dimensions": dimensions,
                "desc": product_data.get("description", "No additional details available"),
                "price": f"${product_data.get('price', 0):.2f}"
            }
            
            # Get template for intent or use generic
            template = self.templates.get(intent, (
                "Information about {product} ({category}):\n\n"
                "Please refer to the product documentation for {intent}.\n"
                "Product details: {desc}\n"
                "Dimensions: {dimensions}\n"
                "Price: {price}"
            ))
            
            return template.format(**template_data, intent=intent.value)
            
        except Exception as e:
            return (f"Information about this product's {intent.value} "
                   f"should be available in the product documentation.")

    def generate_advice(self, context) -> str:
        """Generate product advice based on purchase context"""
        try:
            self.logger.debug(f"Generating advice for context: {context}")
            
            # Build response based on context
            lines = []
            
            # Add primary need
            lines.append(f"Based on your need for {context.primary_need}, here's my advice:")
            lines.append("")
            
            # Add usage-based recommendations
            if context.use_frequency == "frequent":
                lines.append("Since you'll be using this frequently:")
                lines.append("- Look for high durability materials")
                lines.append("- Consider premium options for better longevity")
                lines.append("- Prioritize ergonomic features")
            
            # Add environment-specific advice
            if context.environment and context.environment != "Not specified":
                lines.append(f"\nFor {context.environment} use:")
                if "home" in context.environment.lower():
                    lines.append("- Consider the available space")
                    lines.append("- Think about aesthetic fit with your setup")
                if "office" in context.environment.lower():
                    lines.append("- Focus on professional features")
                    lines.append("- Consider adjustability options")
            
            # Add constraint-based recommendations
            if context.constraints:
                lines.append("\nBased on your constraints:")
                for constraint, value in context.constraints.items():
                    lines.append(f"- {constraint.title()}: {value}")
            
            # Add preference-based suggestions
            if context.preferences:
                lines.append("\nPrioritizing your preferences:")
                for pref, weight in context.preferences.items():
                    if weight > 0.7:
                        lines.append(f"- High priority: {pref}")
                    elif weight > 0.3:
                        lines.append(f"- Medium priority: {pref}")
            
            # Add budget advice if specified
            if hasattr(context, 'budget_flexibility'):
                if context.budget_flexibility != "Not specified":
                    lines.append(f"\nRegarding budget ({context.budget_flexibility}):")
                    if "flexible" in context.budget_flexibility.lower():
                        lines.append("- Consider premium features for long-term value")
                    else:
                        lines.append("- Focus on essential features within budget")
            
            # Add general recommendations
            lines.append("\nGeneral recommendations:")
            lines.append("- Compare multiple options before deciding")
            lines.append("- Read recent user reviews")
            lines.append("- Check warranty terms")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.exception("Error generating advice")
            return f"I apologize, but I encountered an error generating advice: {str(e)}"

    def format_product_comparison(self, products: List[Dict]) -> str:
        """Format product comparison table"""
        if not products:
            return "No products to compare"
        
        try:
            lines = []
            
            # Header
            lines.append("Product Comparison:")
            lines.append("\n| Feature | " + " | ".join(p["name"] for p in products) + " |")
            lines.append("|" + "-|" * (len(products) + 1))
            
            # Common features to compare
            features = ["Price", "Rating", "Dimensions", "Weight"]
            
            for feature in features:
                row = f"| {feature} | "
                for product in products:
                    if feature == "Price":
                        row += f"${product.get('price', 'N/A')} | "
                    elif feature == "Rating":
                        row += f"{product.get('rating', 'N/A')}⭐ | "
                    elif feature == "Dimensions":
                        dims = product.get('dimensions', {})
                        row += f"{dims.get('length', 'N/A')}x{dims.get('width', 'N/A')}x{dims.get('height', 'N/A')} cm | "
                    elif feature == "Weight":
                        row += f"{product.get('weight', 'N/A')} kg | "
                lines.append(row)
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.exception("Error formatting product comparison")
            return f"Error creating comparison: {str(e)}"

# Example usage:
"""
generator = ProductAdviceGenerator()

product = {
    "name": "Some Product",
    "category": ["Category", "Subcategory"],
    "dimensions": {"length": 70, "width": 65, "height": 140},
    "price": 299.99,
    "description": "Product description here"
}

advice = generator.get_advice(ProductUsageIntent.SETUP, product)
""" 