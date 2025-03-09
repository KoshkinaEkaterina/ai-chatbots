from typing import Dict, Any
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
import json
from .parameter_models import ExtractedParameter, ParameterExtractionResult

class ParameterExtractor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
    def create_extraction_prompt(self, message: str) -> str:
        return f"""Analyze this message and extract financial parameters.
        Message: {message}
        
        Return a valid JSON array containing all parameters mentioned. Each parameter must have:
        - key: The parameter name from the list below
        - value: The numerical value (convert all units)
        - original_text: The exact text that contained this parameter
        
        Parameter keys:
        - retirement_age: Age at retirement
        - current_age: Current age
        - return_rate: Investment return rate (convert % to decimal)
        - inflation_rate: Inflation rate (convert % to decimal)
        - income_replacement: Income replacement ratio (convert % to decimal)
        - savings: Current savings (convert K/M to full numbers)
        - income: Annual income (convert K/M to full numbers)
        
        Conversion rules:
        - Percentages: "9%" -> 0.09
        - Thousands: "100k" -> 100000
        - Millions: "1M" -> 1000000
        
        Example response:
        [
            {{
                "key": "savings",
                "value": 100000,
                "original_text": "current savings to 100k"
            }},
            {{
                "key": "retirement_age",
                "value": 80,
                "original_text": "age of retirement to 80"
            }}
        ]
        
        Respond with ONLY the JSON array, no other text."""

    def extract_parameters(self, message: str) -> ParameterExtractionResult:
        """Extract parameters from message using LLM"""
        prompt = self.create_extraction_prompt(message)
        response = self.llm.invoke([SystemMessage(content=prompt)])
        
        try:
            # Clean the response to ensure valid JSON
            cleaned_response = response.content.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json prefix
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ``` suffix
            cleaned_response = cleaned_response.strip()
            
            # Parse LLM response into structured format
            extracted = json.loads(cleaned_response)
            
            if not isinstance(extracted, list):
                print("\nError: Expected JSON array of parameters")
                return ParameterExtractionResult(
                    parameters=[],
                    raw_message=message,
                    unmatched_text=message
                )

            # Convert to Pydantic models with validation
            parameters = []
            seen_keys = {}  # Track latest value for each key
            
            for param in extracted:
                try:
                    extracted_param = ExtractedParameter(**param)
                    # Keep only the latest value for each parameter
                    seen_keys[extracted_param.key] = extracted_param
                except Exception as e:
                    print(f"\nWarning: Skipping invalid parameter: {param}")
                    print(f"Reason: {str(e)}")
                    continue
            
            # Use only the latest value for each parameter
            parameters = list(seen_keys.values())
            
            # Print analysis
            if parameters:
                print("\nParameter Analysis:")
                print(f"Found {len(parameters)} parameters in your message:")
                for param in parameters:
                    if param.key.endswith("_rate"):
                        print(f"- {param.key}: {param.value*100}% (from '{param.original_text}')")
                    elif param.key in ["savings", "income"]:
                        print(f"- {param.key}: ${param.value:,.2f} (from '{param.original_text}')")
                    else:
                        print(f"- {param.key}: {param.value} (from '{param.original_text}')")
            
            return ParameterExtractionResult(
                parameters=parameters,
                raw_message=message
            )
            
        except json.JSONDecodeError as e:
            print(f"\nError: Could not parse LLM response as JSON")
            print(f"Raw response: {cleaned_response}")
            print(f"Error details: {str(e)}")
            return ParameterExtractionResult(
                parameters=[],
                raw_message=message,
                unmatched_text=message
            )
        except Exception as e:
            print(f"\nUnexpected error while parsing parameters: {str(e)}")
            return ParameterExtractionResult(
                parameters=[],
                raw_message=message,
                unmatched_text=message
            )

    def __call__(self, state: Dict) -> Dict:
        message = state["messages"][-1].content
        current_params = state.get("parameters", {})
        
        # Extract parameters
        result = self.extract_parameters(message)
        
        # Update state with new parameters
        new_params = {
            param.key: param.value 
            for param in result.parameters
        }
        
        if new_params:
            print("\nUpdating your financial parameters with these changes")
        else:
            print("\nNo valid parameters found in your message")
            
        return {
            **state,
            "parameters": {**current_params, **new_params}
        } 