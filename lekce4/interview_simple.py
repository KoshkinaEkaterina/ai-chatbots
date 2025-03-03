from typing import TypedDict, List, Dict, Optional, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
import os
import csv
from dotenv import load_dotenv
import time
from random import uniform, random, choice
from dataclasses import dataclass
from classes import Topic, FactorInsight, State
from uuid import uuid4
from pydantic import BaseModel, Field
import json  # Add to imports at top
import argparse  # Add to imports at top
from pathlib import Path
import logging

# Configure logging - set to INFO level
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")
)
    
# Get the directory containing this file
current_dir = Path(__file__).parent
topics_path = current_dir / "topics.csv"

def load_topics(file_path: str) -> Dict[str, Topic]:
    """Load interview topics and factors from CSV."""
    topics = {}
    current_topic = None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['type'] == 'topic':
                current_topic = Topic(row['id'], row['content'])
                topics[row['id']] = current_topic
            elif row['type'] == 'factor':
                if current_topic and row['id'] == current_topic.id:
                    # Use the description as the factor name to match what the LLM returns
                    factor_name = row['content']  # Changed from row['factor']
                    current_topic.factors[factor_name] = row['factor']  # Swapped these
                    current_topic.covered_factors[factor_name] = 0.0
    return topics

def introduce_interview(state: State) -> State:
    """Generate the interview introduction in Czech."""
    intro_prompt = """You are a professional interviewer starting a conversation in Czech. 
    Introduce yourself and explain that you'll be conducting an interview about classroom experiences 
    and student behavior. Be welcoming and friendly, but maintain a professional tone.
    
    Write a brief introduction in Czech followed by the first topic question."""
    
    response = model.invoke([SystemMessage(content=intro_prompt)])
    current_topic = state["topics"][state["current_topic_id"]]
    
    return {
        **state,
        "current_question": f"{response.content}\n\n{current_topic.question}",
        "introduction_done": True
    }

# Add these Pydantic models at the top of the file
class Finding(BaseModel):
    detail: str = Field(..., description="Specific information found in the response")
    quote: str = Field(..., description="Exact quote from the text supporting this finding")
    relevance: float = Field(..., ge=0.0, le=1.0, description="How relevant this finding is")

class FactorAnalysis(BaseModel):
    factor: str = Field(..., description="Name of the factor being analyzed")
    score: float = Field(..., ge=0.0, le=1.0, description="Overall coverage score for this factor")
    found_info: List[Finding] = Field(..., description="List of specific findings for this factor")
    summary: str = Field(..., description="Overall summary of what was found")
    missing: str = Field(..., description="What information is still needed")

class AnalysisResponse(BaseModel):
    analysis: List[FactorAnalysis]

def analyze_response(response: str, topic: Topic) -> Dict[str, float]:
    """Analyze response with handling for off-topic or chaotic answers."""
    # First, check if the response is completely off-topic
    relevance_prompt = f"""Analyze if this response is relevant to the topic:
    TOPIC: {topic.question}
    RESPONSE: {response}
    
    Return JSON:
    {{
        "is_relevant": bool,
        "reason": "brief explanation"
    }}"""
    
    relevance_check = model.invoke([
        SystemMessage(content="Determine if the response is on-topic."),
        SystemMessage(content=relevance_prompt)
    ])
    
    try:
        relevance = json.loads(relevance_check.content.strip())
        if not relevance["is_relevant"]:
            print(f"\nUPOZORNĚNÍ: Odpověď je mimo téma - {relevance['reason']}")
            return {factor: 0.0 for factor in topic.factors.keys()}
    except:
        pass  # If relevance check fails, continue with normal analysis

    prompt = f"""Analyze this Czech teacher's response and extract specific information.
    
    RESPONSE: {response}
    
    Analyze EXACTLY these factors (use these exact names):
    {chr(10).join(f'- {factor}' for factor in topic.factors.keys())}
    
    Return a structured analysis following this exact schema:
    {{
        "analysis": [
            {{
                "factor": "EXACT_FACTOR_NAME_FROM_LIST_ABOVE",
                "score": 0.8,
                "found_info": [
                    {{
                        "detail": "specific information found",
                        "quote": "exact quote from text",
                        "relevance": 0.9
                    }}
                ],
                "summary": "overall summary of findings",
                "missing": "what information is still needed"
            }}
        ]
    }}"""
    
    result = model.invoke([
        SystemMessage(content="You are a precise JSON generator. Use EXACT factor names as provided."),
        SystemMessage(content=prompt)
    ])
    
    try:
        # Clean up the response to ensure it's valid JSON
        response_text = result.content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Use model_validate_json instead of parse_raw for Pydantic v2
        analysis = AnalysisResponse.model_validate_json(response_text)
        coverage = {}
        
        for factor_analysis in analysis.analysis:
            factor = factor_analysis.factor
            if factor not in topic.factors:
                print(f"Warning: Unknown factor {factor}")
                continue
            
            coverage[factor] = factor_analysis.score
            topic.covered_factors[factor] = max(
                topic.covered_factors.get(factor, 0.0),
                factor_analysis.score
            )
            
            # Store insights for each finding
            for finding in factor_analysis.found_info:
                insight = FactorInsight(
                    answer_id=str(uuid4()),
                    content=finding.detail,
                    source_answer=response,
                    relevance_score=finding.relevance,
                    evidence=factor_analysis.summary,
                    quote=finding.quote
                )
                topic.add_insight(factor, insight)
        
        return coverage
        
    except Exception as e:
        print(f"Error parsing analysis: {e}")
        print("Raw response:")
        print(response_text)
        return {factor: 0.0 for factor in topic.factors.keys()}

def print_topic_status(topic: Topic):
    """Print detailed status of topic coverage and insights."""
    print(f"\n{'='*100}")
    print(f"DETAILNÍ ANALÝZA ODPOVĚDI PRO TÉMA: {topic.question}")
    print(f"{'='*100}")
    
    for factor, description in topic.factors.items():
        print(f"\n{'='*50}")
        print(f"FAKTOR: {factor}")
        print(f"POPIS: {description}")
        print(f"{'='*50}")
        
        if factor in topic.factor_insights and topic.factor_insights[factor]:
            print("\nNALEZENÉ INFORMACE:")
            for insight in topic.factor_insights[factor]:
                print(f"\n• DETAIL: {insight['key_info']}")
                if 'evidence' in insight:
                    print(f"  DŮKAZ: {insight['evidence']}")
                if 'quote' in insight:
                    print(f"  CITACE: \"{insight['quote']}\"")
                print(f"  RELEVANCE: {insight.get('score', 0.0):.2f}")
            
            print(f"\nCELKOVÉ POKRYTÍ: {topic.covered_factors.get(factor, 0.0):.2f}")
        else:
            print("\nŽÁDNÉ INFORMACE NEBYLY NALEZENY")
            print(f"POKRYTÍ: {topic.covered_factors.get(factor, 0.0):.2f}")
        
        print(f"\n{'-'*50}")
    
    print(f"\n{'='*100}")

def analyze_emotional_content(response: str) -> dict:
    """Analyze the emotional weight and trauma level of a response."""
    prompt = f"""Analyze the emotional content of this teacher's response:
    RESPONSE: {response}
    
    Return JSON:
    {{
        "emotional_weight": float,  # 0.0-1.0, how emotionally heavy is the content
        "trauma_indicators": bool,  # whether the response indicates traumatic experience
        "emotional_cues": [str],  # list of emotional indicators in the text (pauses, voice changes, etc.)
        "key_emotions": [str],  # main emotions expressed
        "requires_support": bool  # whether the response needs emotional acknowledgment
    }}"""
    
    result = model.invoke([SystemMessage(content=prompt)])
    try:
        return json.loads(result.content.strip())
    except:
        return {
            "emotional_weight": 0.0,
            "trauma_indicators": False,
            "emotional_cues": [],
            "key_emotions": [],
            "requires_support": False
        }

def generate_question(state: State) -> State:
    """Generate naturally flowing, empathetic follow-up questions."""
    current_topic = state["topics"][state["current_topic_id"]]
    
    # Get recent conversation history to track our empathetic responses
    recent_history = state.get("conversation_history", [])[-3:]
    previous_responses = [
        exchange.get("interviewer_response", "")
        for exchange in recent_history
    ]
    
    last_response = state.get("user_message")
    if last_response:
        emotional_analysis = analyze_emotional_content(last_response)
        
        if emotional_analysis["emotional_weight"] > 0.6 or emotional_analysis["trauma_indicators"]:
            support_prompt = (
                f"""The teacher just shared a deeply emotional experience in an ongoing conversation.

                Recent conversation history:
                {chr(10).join(f'Q: {ex["question"]}, A: {ex["answer"]}' for ex in recent_history)}
                
                Latest response: "{last_response}"
                
                Emotional context:
                - Weight: {emotional_analysis['emotional_weight']}
                - Emotions: {', '.join(emotional_analysis['key_emotions'])}
                - Previous empathetic responses used: {previous_responses}
                
                Generate a natural, flowing response in Czech that:
                1. Shows you're listening by referencing specific details they shared
                2. Asks ONE clear follow-up question
                
                IMPORTANT GUIDELINES:
                - Only offer to stop/change topic if trauma indicators are very high
                - Don't use generic empathy phrases
                - Stay focused on what they're actually telling you
                - Let them guide the emotional depth
                
                BAD EXAMPLES (too generic/repetitive):
                - "Samozřejmě, pokud byste o tom raději nemluvila, plně to respektuji."
                - "Chápu, že je to těžké téma."
                - "Děkuji za vaši otevřenost."
                
                GOOD EXAMPLES (specific to their story):
                - "Ten moment s tím nožem musel být opravdu intenzivní... Jak jste dokázala zachovat klid?"
                - "Zmínila jste, že vás to dodnes pronásleduje. Co vám pomáhá se s tím vyrovnat?"
                
                Current emotional weight: {emotional_analysis['emotional_weight']}"""
            )
            
            response = model.invoke([SystemMessage(content=support_prompt)])
            
            return {
                **state,
                "current_question": response.content,
                "emotional_context": emotional_analysis
            }
    
    # If not emotionally heavy, proceed with normal question generation
    # Ensure all required state keys exist
    if not isinstance(state, dict):
        state = state.copy()
    
    # Initialize required state keys with defaults
    preserved_state = {
        "current_question": state.get("current_question"),
        "user_message": state.get("user_message"),
        "conversation_history": state.get("conversation_history", []),
        "topics": state.get("topics", load_topics(topics_path)),
        "current_topic_id": state.get("current_topic_id", "T1"),
        "introduction_done": state.get("introduction_done", False),
        "interview_complete": state.get("interview_complete", False)
    }
    state = preserved_state
    
    if not state.get("introduction_done"):
        return introduce_interview(state)
    
    current_topic = state["topics"][state["current_topic_id"]]
    uncovered_factors = [
        (factor, desc) for factor, desc in current_topic.factors.items()
        if current_topic.covered_factors[factor] < 0.7
    ]
    
    # Format conversation history
    history_text = "\n\n".join([
        f"Q: {exchange['question']}\nA: {exchange['answer']}"
        for exchange in state.get("conversation_history", [])[-3:]
    ])
    
    prompt = f"""You are conducting an interview in Czech about classroom experiences.
    Current topic: {current_topic.question}
    
    Recent conversation:
    {history_text}
    
    IMPORTANT GUIDELINES:
    1. If the teacher shares traumatic or emotionally heavy experiences:
       - Acknowledge the emotional weight first
       - Show empathy and understanding
       - Give space for processing
       - Only gently proceed with follow-up if appropriate
       
    2. When asking follow-up questions:
       - Stay with difficult topics rather than rushing forward
       - Show you're listening and care about their experience
       - Validate their feelings and experiences
       
    3. Question formulation:
       - Keep it gentle and supportive when needed
       - Allow space for "no" or "I'd rather not discuss this further"
       - Focus on understanding their experience
    
    Uncovered aspects to explore:
    {chr(10).join(f'- {desc}' for _, desc in uncovered_factors)}
    
    Generate an appropriate follow-up response or question in Czech that puts being human first and gathering information second."""
    
    response = model.invoke([SystemMessage(content=prompt)])
    
    return {
        **state,
        "current_question": response.content
    }

def print_topic_summary(topic: Topic, file_path: str = "interview_analysis.txt"):
    """Create a detailed summary of all insights gathered for each factor in the topic."""
    # First print to console
    print(f"\n{'='*100}")
    print(f"SOUHRNNÁ ANALÝZA TÉMATU: {topic.question}")
    print(f"{'='*100}\n")
    
    for factor, description in topic.factors.items():
        print(f"\nFAKTOR: {factor}")
        print(f"POPIS: {description}")
        print(f"CELKOVÉ POKRYTÍ: {topic.covered_factors.get(factor, 0.0):.2f}")
        
        if factor in topic.factor_insights and topic.factor_insights[factor]:
            print("\nVŠECHNA ZJIŠTĚNÍ:")
            for insight in topic.factor_insights[factor]:
                print(f"\n• INFORMACE: {insight['key_info']}")
                print(f"  CITACE: \"{insight['quote']}\"")
                print(f"  RELEVANCE: {insight.get('score', 0.0):.2f}")
            
            # Generate an overall summary
            summary_prompt = f"""Create a concise summary of these insights about {factor}:
            {chr(10).join(f'- {i["key_info"]}' for i in topic.factor_insights[factor])}
            
            Return a 2-3 sentence summary in Czech."""
            
            summary = model.invoke([SystemMessage(content=summary_prompt)])
            print(f"\nSOUHRN FAKTORU:\n{summary.content}")
        else:
            print("\nŽÁDNÉ INFORMACE NEBYLY ZÍSKÁNY")
        
        print(f"\n{'-'*50}")
    
    # Then write to file
    with open(file_path, "a", encoding='utf-8') as f:
        f.write(f"\n{'='*100}\n")
        f.write(f"SOUHRNNÁ ANALÝZA TÉMATU: {topic.question}\n")
        f.write(f"{'='*100}\n\n")
        
        for factor, description in topic.factors.items():
            f.write(f"\nFAKTOR: {factor}\n")
            f.write(f"POPIS: {description}\n")
            f.write(f"CELKOVÉ POKRYTÍ: {topic.covered_factors.get(factor, 0.0):.2f}\n")
            
            if factor in topic.factor_insights and topic.factor_insights[factor]:
                f.write("\nVŠECHNA ZJIŠTĚNÍ:\n")
                for insight in topic.factor_insights[factor]:
                    f.write(f"\n• INFORMACE: {insight['key_info']}\n")
                    f.write(f"  CITACE: \"{insight['quote']}\"\n")
                    f.write(f"  RELEVANCE: {insight.get('score', 0.0):.2f}\n")
                
                # Generate an overall summary
                summary_prompt = f"""Create a concise summary of these insights about {factor}:
                {chr(10).join(f'- {i["key_info"]}' for i in topic.factor_insights[factor])}
                
                Return a 2-3 sentence summary in Czech."""
                
                summary = model.invoke([SystemMessage(content=summary_prompt)])
                f.write(f"\nSOUHRN FAKTORU:\n{summary.content}\n")
            else:
                f.write("\nŽÁDNÉ INFORMACE NEBYLY ZÍSKÁNY\n")
            
            f.write(f"\n{'-'*50}\n")

def process_response(state: State) -> State:
    """Process the user's response and update factor coverage."""
    # Ensure all required state keys exist
    if not isinstance(state, dict):
        state = state.copy()
    
    # Initialize required state keys with defaults
    preserved_state = {
        "current_question": state.get("current_question"),
        "user_message": state.get("user_message"),
        "conversation_history": state.get("conversation_history", []),
        "topics": state.get("topics", load_topics(topics_path)),
        "current_topic_id": state.get("current_topic_id", "T1"),
        "introduction_done": state.get("introduction_done", False),
        "interview_complete": state.get("interview_complete", False)
    }
    state = preserved_state
    
    if not state.get("user_message"):
        return state
    
    # Print the teacher's response immediately
    print(f"\nUčitel: {state['user_message']}")
    
    current_topic = state["topics"][state["current_topic_id"]]
    
    # Analyze factor coverage
    coverage = analyze_response(state["user_message"], current_topic)
    for factor, score in coverage.items():
        current_topic.covered_factors[factor] = max(
            current_topic.covered_factors[factor],
            score
        )
    
    # Update conversation history
    history = state.get("conversation_history", [])
    history.append({
        "question": state["current_question"],
        "answer": state["user_message"],
        "coverage": coverage
    })
    
    # Check if we should move to next topic
    if all(score >= 0.7 for score in current_topic.covered_factors.values()):
        print("\nTÉMA DOKONČENO - Generuji souhrnnou analýzu...\n")
        print_topic_summary(current_topic)
        
        # Find next topic
        topic_ids = list(state["topics"].keys())
        current_index = topic_ids.index(state["current_topic_id"])
        
        # Check if this was the last topic
        if current_index >= len(topic_ids) - 1:
            print("\nVŠECHNA TÉMATA DOKONČENA!")
            return {
                **state,
                "interview_complete": True,
                "conversation_history": history,
                "user_message": None
            }
        else:
            state["current_topic_id"] = topic_ids[current_index + 1]
    
    return {
        **state,
        "conversation_history": history,
        "user_message": None
    }

def simulate_teacher_response(question: str) -> str:
    """Simulate a Czech teacher with contextually appropriate responses."""
    
    # First check if the question is about difficult situations
    difficult_situation_keywords = [
        "problém", "náročn", "konflikt", "incident", "chování", "řešit", "situac",
        "zasáhnout", "krize", "těžk", "šikan", "násilí", "agres"
    ]
    
    is_about_difficulties = any(keyword in question.lower() for keyword in difficult_situation_keywords)
    
    # If the question is about difficult situations, 80% chance to share a serious experience
    if is_about_difficulties and random() < 0.8:
        prompt = f"""Generate a very brief teacher's response (max 20 words) in Czech about a difficult classroom situation.
        Question asked: {question}
        
        IMPORTANT:
        - Keep it under 20 words
        - Make it feel natural and spontaneous
        - Focus on one specific moment or detail
        - Include emotional impact briefly
        
        Response should be concise but meaningful."""
        
        response = model.invoke([SystemMessage(content=prompt)])
        return response.content
    
    # 30% chance of giving a non-standard response (these can be scripted as they're throw-offs)
    if random() < 0.3:
        response_types = [
            # Questioning the interview process
            {
                "type": "meta",
                "responses": [
                    "Můžete mi vysvětlit, proč se ptáte zrovna na tohle?",
                    "Nejsem si jistá, jestli je vhodné o tomhle mluvit. Jaký je účel těchto otázek?",
                    "Než odpovím, chtěla bych vědět, jak s těmito informacemi naložíte.",
                    "Tohle je docela osobní téma. Můžete mi říct více o tom, proč vás to zajímá?",
                ]
            },
            # Confusion about question
            {
                "type": "confusion",
                "responses": [
                    "Promiňte, ale není mi úplně jasné, na co se ptáte. Můžete to formulovat jinak?",
                    "Ta otázka je dost složitá. Můžete ji nějak zjednodušit?",
                    "Nevím, jestli správně chápu, co chcete vědět...",
                    "Tohle je hodně komplexní téma. Můžeme to rozebrat po částech?",
                ]
            },
            # Process questions
            {
                "type": "process",
                "responses": [
                    "Jak dlouho tento rozhovor ještě potrvá?",
                    "Kolik takových rozhovorů už jste dělali?",
                    "Kdo všechno bude mít přístup k těmto informacím?",
                    "Můžeme si udělat krátkou přestávku? Je toho na mě hodně.",
                ]
            }
        ]
        
        response_type = choice(response_types)
        return choice(response_type["responses"])
    
    # For normal responses, generate contextually appropriate content
    prompt = f"""Generate a very brief teacher's response (max 20 words) in Czech to this question: {question}
    
    Make it:
    - Maximum 20 words
    - Natural and specific
    - Focus on one concrete example
    - Use conversational Czech"""
    
    response = model.invoke([SystemMessage(content=prompt)])
    return response.content

def generate_response_content() -> str:
    """Generate additional content for normal responses."""
    # Add some basic response content
    contents = [
        "děti byly trochu neklidné během matematiky, ale nakonec jsme to zvládli.",
        "museli jsme řešit menší konflikt mezi dvěma žáky, ale vyřešili jsme to diskuzí.",
        "jedna studentka měla problémy se začleněním do kolektivu, ale třída ji nakonec přijala.",
        "došlo k nedorozumění mezi žáky, ale společně jsme našli řešení."
    ]
    return choice(contents)

# Build graph
builder = StateGraph(State)

# Add nodes
builder.add_node("process_response", process_response)
builder.add_node("generate_question", generate_question)

# Create flow: START -> process_response -> generate_question -> END
builder.add_edge(START, "process_response")
builder.add_edge("process_response", "generate_question")
builder.add_edge("generate_question", END)

graph = builder.compile()

# Default state for LangGraph Studio
def get_default_state():
    """Initialize default state with all required keys."""
    return {
        "current_question": None,
        "user_message": None,
        "conversation_history": [],
        "topics": load_topics(topics_path),
        "current_topic_id": "T1",
        "introduction_done": False,
        "interview_complete": False
    }

def log_detailed_analysis(topic: Topic, response: str, analysis_data: Dict, file_path: str = "interview_analysis.txt"):
    """Write detailed analysis to a file."""
    with open(file_path, "a", encoding='utf-8') as f:
        f.write(f"\n{'='*100}\n")
        f.write(f"TÉMA: {topic.question}\n")
        f.write(f"ODPOVĚĎ: {response}\n\n")
        
        for factor, data in analysis_data.items():
            f.write(f"\nFAKTOR: {factor}\n")
            f.write(f"POPIS: {topic.factors[factor]}\n")
            f.write(f"SKÓRE: {topic.covered_factors[factor]:.2f}\n")
            
            if factor in topic.factor_insights:
                for insight in topic.factor_insights[factor]:
                    f.write(f"\n• DETAIL: {insight['key_info']}\n")
                    f.write(f"  CITACE: \"{insight['quote']}\"\n")
                    f.write(f"  DŮKAZ: {insight['evidence']}\n")
            
            f.write(f"\n{'-'*50}\n")

def print_brief_status(old_state: State, answer: str, next_question: str):
    """Print status with emotional awareness."""
    current_topic = old_state["topics"][old_state["current_topic_id"]]
    
    # Check for emotional content
    emotional_analysis = analyze_emotional_content(answer)
    
    # If the response was emotionally significant, acknowledge before analysis
    if emotional_analysis["emotional_weight"] > 0.6:
        print("\n" + "-"*50)
        print("EMOČNÍ KONTEXT:")
        print("Učitel sdílel velmi citlivou zkušenost. Dejme prostor pro zpracování...")
        print("-"*50 + "\n")
    
    # Only print analysis and next question (removed answer printing)
    print("\nANALÝZA:")
    covered = {f: s for f, s in current_topic.covered_factors.items() if s > 0}
    if covered:
        for factor, score in covered.items():
            print(f"✓ {factor}: {score:.2f}")
    else:
        print("❌ Odpověď neposkytla žádné relevantní informace k tématu.")
    
    print("\nDALŠÍ OTÁZKA:")
    print(next_question)
    print("\nZDŮVODNĚNÍ:")
    if not covered:
        print("Předchozí odpověď byla mimo téma. Zkusíme otázku položit jinak.")
    else:
        uncovered = [f for f, s in current_topic.covered_factors.items() if s < 0.7]
        if uncovered:
            print(f"Potřebujeme více informací o: {', '.join(uncovered)}")
        else:
            print("Přecházíme k dalšímu tématu.")

def manual_interview_loop(state: State):
    """Run the interview in manual mode where the user provides responses."""
    
    # Get first question
    state = generate_question(state)
    print(f"\nTazatel: {state['current_question']}")
    
    while True:
        # Get user input
        print("\nVaše odpověď (nebo 'konec' pro ukončení): ")
        answer = input().strip()
        
        if answer.lower() == 'konec':
            print("\nRozhovor ukončen uživatelem.")
            break
        
        # Process the answer
        state["user_message"] = answer
        old_state = state.copy()
        state = graph.invoke(state)
        
        # Show analysis and next question
        print_brief_status(old_state, answer, state["current_question"])
        
        # Log detailed analysis to file
        current_topic = state["topics"][state["current_topic_id"]]
        log_detailed_analysis(current_topic, answer, current_topic.covered_factors)
        
        if state.get("interview_complete"):
            print("\nRozhovor dokončen!")
            break

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Run the interview system.')
    parser.add_argument('--manual', action='store_true', 
                       help='Run in manual mode where you provide the responses')
    args = parser.parse_args()
    
    state = get_default_state()
    print("\nStarting interview...\n")
    
    if args.manual:
        manual_interview_loop(state)
    else:
        # Original automated interview code
        state = generate_question(state)
        print(f"\nTazatel: {state['current_question']}")
        
        while True:
            print("\nUčitel přemýšlí...", flush=True)
            answer = simulate_teacher_response(state["current_question"])
            
            state["user_message"] = answer
            old_state = state.copy()
            state = graph.invoke(state)
            
            print_brief_status(old_state, answer, state["current_question"])
            
            current_topic = state["topics"][state["current_topic_id"]]
            log_detailed_analysis(current_topic, answer, current_topic.covered_factors)
            
            if state.get("interview_complete"):
                print("\nRozhovor dokončen!")
                break
        