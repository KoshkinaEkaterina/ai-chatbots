from langsmith import wrappers, Client
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from typing import Dict, List
import re

# Load environment variables
load_dotenv()

# Initialize LangSmith client
client = Client()

# Initialize Azure OpenAI with correct env var names
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.7
)

# Create inputs and reference outputs
examples = [
    (
        "Which country is Mount Kilimanjaro located in?",
        "Mount Kilimanjaro is located in Tanzania.",
    ),
    (
        "What is Earth's lowest point?",
        "Earth's lowest point is The Dead Sea, located between Israel and Jordan, at 430.5 meters (1,412 feet) below sea level.",
    ),
    (
        "Who wrote 'Romeo and Juliet'?",
        "William Shakespeare wrote 'Romeo and Juliet', believed to be written between 1591 and 1595.",
    ),
    (
        "What is the capital of Japan?",
        "Tokyo is the capital city of Japan.",
    ),
    (
        "What is the chemical formula for water?",
        "The chemical formula for water is H2O, consisting of two hydrogen atoms and one oxygen atom.",
    ),
    (
        "Who painted the Mona Lisa?",
        "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519.",
    ),
    (
        "What is the largest planet in our solar system?",
        "Jupiter is the largest planet in our solar system.",
    ),
    (
        "What is the speed of light?",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    ),
    (
        "Who invented the telephone?",
        "Alexander Graham Bell invented the first practical telephone and patented it in 1876.",
    ),
    (
        "What is the main ingredient in guacamole?",
        "The main ingredient in guacamole is avocado.",
    ),
    (
        "What is the longest river in the world?",
        "The Nile River is the longest river in the world, stretching approximately 6,650 kilometers (4,132 miles).",
    ),
    (
        "What year did World War II end?",
        "World War II ended in 1945, with Germany surrendering in May and Japan surrendering in August following the atomic bombings.",
    ),
    (
        "What is the hardest natural substance on Earth?",
        "Diamond is the hardest naturally occurring substance on Earth, ranking 10 on the Mohs scale of mineral hardness.",
    ),
    (
        "Who was the first woman to win a Nobel Prize?",
        "Marie Curie was the first woman to win a Nobel Prize, winning the Physics Prize in 1903 and the Chemistry Prize in 1911.",
    ),
    (
        "What is the boiling point of water at sea level?",
        "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level under normal atmospheric pressure.",
    )
]

inputs = [{"question": input_prompt} for input_prompt, _ in examples]
outputs = [{"answer": output_answer} for _, output_answer in examples]

# Create dataset in LangSmith
dataset = client.create_dataset(
    dataset_name="Sample dataset 2",
    description="A sample dataset in LangSmith."
)

# Add examples to the dataset
client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)

# Define target function using Azure OpenAI
def target(inputs: dict) -> dict:
    response = llm.invoke([
        {"role": "system", "content": "Answer the following question accurately"},
        {"role": "user", "content": inputs["question"]}
    ])
    return {"response": response.content}

# Define output schema for the LLM judge
class Grade(BaseModel):
    score: bool = Field(description="Boolean that indicates whether the response is accurate relative to the reference answer")

# Define LLM judge
def accuracy(outputs: dict, reference_outputs: dict) -> bool:
    instructions = """Evaluate Student Answer against Ground Truth for conceptual similarity and classify true or false: 
    - False: No conceptual match and similarity
    - True: Most or full conceptual match and similarity
    - Key criteria: Concept should match, not exact wording.
    """
    
    response = llm.invoke([
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"""Ground Truth answer: {reference_outputs["answer"]}; 
        Student's Answer: {outputs["response"]}"""}
    ])
    
    # Parse response to get boolean score
    return "true" in response.content.lower()

# Add new evaluator classes
class Evaluators:
    @staticmethod
    def factual_accuracy(outputs: dict, reference_outputs: dict) -> Dict:
        """Evaluates if specific facts and numbers match the reference"""
        instructions = """Compare the factual elements (dates, numbers, names, locations) between the answers.
        Return a score between 0 and 1, where:
        1.0 = All facts match exactly
        0.5 = Core facts match but some details differ
        0.0 = Critical facts are incorrect
        
        Also return specific mismatches found."""
        
        response = llm.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"""Reference: {reference_outputs["answer"]}
            Response: {outputs["response"]}"""}
        ])
        
        return {
            "score": float(re.search(r"(\d+\.?\d*)", response.content).group(1)),
            "explanation": response.content
        }
    
    @staticmethod
    def completeness(outputs: dict, reference_outputs: dict) -> Dict:
        """Evaluates how complete the answer is compared to reference"""
        instructions = """Analyze the completeness of the response compared to the reference.
        Score based on:
        - Main point coverage (0-0.4)
        - Supporting details (0-0.3)
        - Context/clarifications (0-0.3)
        
        Return score and missing elements."""
        
        response = llm.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"""Reference: {reference_outputs["answer"]}
            Response: {outputs["response"]}"""}
        ])
        
        return {
            "score": float(re.search(r"(\d+\.?\d*)", response.content).group(1)),
            "missing_elements": response.content
        }
    
    @staticmethod
    def clarity(outputs: dict) -> Dict:
        """Evaluates response clarity and readability"""
        instructions = """Rate the clarity of the response on:
        1. Grammar/syntax (0-0.3)
        2. Clear structure (0-0.3)
        3. Appropriate detail level (0-0.2)
        4. No ambiguity (0-0.2)
        
        Return score and improvement suggestions."""
        
        response = llm.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Response to evaluate: {outputs['response']}"}
        ])
        
        return {
            "score": float(re.search(r"(\d+\.?\d*)", response.content).group(1)),
            "suggestions": response.content
        }

# Update evaluation run with new evaluators
experiment_results = client.evaluate(
    target,
    data="Sample dataset 2",
    evaluators=[
        accuracy,  # Original boolean accuracy
        Evaluators.factual_accuracy,  # New detailed fact checker
        Evaluators.completeness,  # New completeness checker
        Evaluators.clarity  # New clarity analyzer
    ],
    experiment_prefix="azure-eval-in-langsmith",
    max_concurrency=2,
)

if __name__ == "__main__":
    print("Dataset created and evaluation started!")
    print(f"Dataset ID: {dataset.id}")
    print(f"Number of examples: {len(examples)}")
    print("\nEvaluations being run:")
    print("1. Basic Accuracy (true/false)")
    print("2. Factual Accuracy (0-1 with explanation)")
    print("3. Completeness (0-1 with missing elements)")
    print("4. Clarity (0-1 with suggestions)")
    print("\nResults will be available in LangSmith.")