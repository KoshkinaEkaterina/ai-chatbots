from typing import TypedDict, List, Dict, Sequence
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as PineconeStore
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
import numpy as np
import re
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image

# Initialize environment and Pinecone
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

# Initialize vectorstore
vectorstore = PineconeStore.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX"),
    embedding=OpenAIEmbeddings()
)

# Debug check of the vector store
print("\nChecking Vector Store Content...")
try:
    test_results = vectorstore.similarity_search(
        "what is this document about?",
        k=3
    )
    print("\nFound documents in vector store:")
    for doc in test_results:
        print(f"\nSource: {doc.metadata.get('source', 'Unknown')}")
        print(f"Preview: {doc.page_content[:200]}...")
except Exception as e:
    print(f"\nError accessing vector store: {e}")
    print("\nMake sure you've run process_pdfs.py first to load the essays!")

# Prompt template
CONTEXT_PROMPT = PromptTemplate(
    template="""
👜 You are an expert assistant in luxury handbag investments. You help users understand market trends, brand performance,
 resale value, and investment strategies related to luxury bags.

Use the following information extracted from research reports to respond to the user's question. Maintain a helpful,
 professional, and friendly tone.

🔁 **Context from previous conversation:**
{chat_history}

📚 **Relevant passages:**
{context}

👤 **User question:**
{question}

---

🎯 **Instructions:**
1. Ask clarifying questions if the user's input is vague.
2. Compare brands or trends using bullet points.
3. Mention Hermès, Chanel, Louis Vuitton if relevant and include resale stats.
4. If data is insufficient, let the user know and suggest alternative sources.
5. Always cite sources and invite follow-up questions.

🎁 **Output format example:**

📈 Top brands for resale value: Hermès, Chanel, Louis Vuitton  
• Hermès Birkin bags: **up to 103%** value retention  
• Chanel Classic Flap: **87%** resale value  
• Louis Vuitton Neverfull: strong demand and **92%** retention  

🔗 Sources: Christie's 2023, KPMG Report  
🤔 Would you like to see a graph or a comparison over time?

---
Now respond based on the context above:
""",
    input_variables=["context", "chat_history", "question"]
)

# Define state schema with enhanced context
class ChatState(TypedDict):
    messages: Sequence[BaseMessage]
    context: Dict  # Store relevant context from previous exchanges

# Create chat chain with enhanced context handling
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4", temperature=0.7),
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    ),
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        input_key="question"
    ),
    return_source_documents=True,
    verbose=True,
    combine_docs_chain_kwargs={
        "prompt": CONTEXT_PROMPT
    }
)

# Confidence threshold values
HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.55


def calculate_confidence(scores: List[float]) -> float:
    return np.mean(scores) if scores else 0.0

def bolden_brands_and_numbers(text: str) -> str:
    brands = ["Hermès", "Chanel", "Louis Vuitton"]
    for brand in brands:
        text = re.sub(rf"\b{brand}\b", f"**{brand}**", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d{1,3}(?:[.,]\d+)?\s*%)", r"**\1**", text)
    return text


# Detection of a vague query
def is_vague(question: str) -> bool:
    short = len(question.strip().split()) < 4  # too short
    vague_keywords = ["bags", "investment", "luxury", "brand", "value", "trend"]
    exact_match = any(question.lower().strip() == word for word in vague_keywords)
    return short or exact_match


def clean_text(text):
    return text.encode("latin-1", "ignore").decode("latin-1")


def agent(state: ChatState) -> ChatState:
    messages = state["messages"]
    chat_history = state["context"].get("chat_history", [])
    current_message = messages[-1].content

    # If the query is too vague, the bot will ask a follow-up question instead of answering
    if any(keyword in current_message.lower() for keyword in ["graph", "visualization", "pdf", "export"]):
        pass  # allow route decision to handle it
    elif is_vague(current_message):
        clarification_response = (
            "🤔 Your query is a bit too general. Could you clarify what exactly you're interested in?\n\n"
            "For example:\n"
            "- Which brand has the best investment value?\n"
            "- How does the performance of Hermès and Chanel handbags differ?\n"
            "- What are the trends in 2024?\n"
        )

        return {
            "messages": list(messages) + [AIMessage(content=clarification_response)],
            "context": state["context"]
        }

    # Search for documents and retrieve their relevance scores
    retrieved_docs = vectorstore.similarity_search_with_score(current_message, k=10)

    # Retrieve relevance scores and sort documents by trustworthiness
    scored_docs = sorted(retrieved_docs, key=lambda x: x[1], reverse=True)
    scores = [score for _, score in scored_docs]

    # Calculate the overall trustworthiness of the answer
    confidence = calculate_confidence(scores)

    # Decision logic based on the trustworthiness of the results
    if confidence > HIGH_CONFIDENCE:
        response_type = "direct"
    elif confidence > LOW_CONFIDENCE:
        response_type = "cautious"
    else:
        response_type = "ask_clarification"

    if response_type == "direct":
        # We'll use the best matching documents
        relevant_context = ""
        sources_used = []

        for doc, _ in scored_docs[:3]:
            preview = doc.page_content[:300].strip().replace("\n", " ")
            filename = Path(doc.metadata.get("source", "Unknown source")).stem
            relevant_context += f"• {preview}...\n"
            sources_used.append(filename)

            unique_sources = list(set(sources_used))  # Remove duplicates

            final_response = (
                f"📌 Most relevant information (confidence: {confidence:.2f}):\n{relevant_context}\n"
                f"🔗 Sources used: {', '.join(unique_sources)}\n"
                f"📊 Would you like to see a graph? Type 'graph' or 'visualization'."
            )

    elif response_type == "cautious":
##        final_response = "🔍 I found some relevant information, but some details might be missing. " \
##                         "Would you like more sources?"
        fallback_prompt = PromptTemplate.from_template("""
        You are a helpful assistant who answers questions about luxury handbag investments, even if there's no direct source material.
        Use your general knowledge and prior conversation to respond clearly and kindly.
        Conversation so far:
        {chat_history}
        User question:
        {question}
        """)
        fallback_chain = fallback_prompt | ChatOpenAI(model="gpt-4", temperature=0.7)
        response = fallback_chain.invoke({
            "chat_history": "\n".join([f"User: {m['human']}\nAI: {m['ai']}" for m in chat_history]),
            "question": current_message
        })
        final_response = (
            f"{response.content}\n\n"
            "📌 *Note: This answer was generated based on general knowledge (confidence: {confidence:.2f}). "
            "It was not directly supported by specific documents from the archive.*"
        )

    else:
        final_response = "🤔 I couldn't find sufficiently precise information. Could you clarify your question?"

    return {
        "messages": list(messages) + [AIMessage(content=final_response)],
        "context": {
            "last_documents": scored_docs[:3], 
            "chat_history": chat_history + [{"human": current_message, "ai": final_response}],
            "confidence": confidence
        }
    }

# Graph agent
def graph_agent(state: ChatState) -> ChatState:
    """Plots a graph comparing brand appreciation and saves it as a file"""

    # Sample data
    brands = ["Hermès", "Chanel", "Louis Vuitton"]
    values = [103, 87, 92]

    # Graph
    plt.figure(figsize=(6, 4))
    plt.bar(brands, values, color=["gold", "black", "brown"])
    plt.title("Comparison of Luxury Handbag Appreciation")
    plt.ylabel("Appreciation (%)")
    plt.tight_layout()

    file_path = "graph.png"
    plt.savefig(file_path)
    plt.close()

    final_response = (
        f"🖼️ The graph has been saved as **{file_path}**.\n"
        "Open this file to view the brand comparison. 📊"
    )

    return {
        "messages": list(state["messages"]) + [AIMessage(content=final_response)],
        "context": state["context"]
    }


def pdf_export_agent(state: ChatState) -> ChatState:
    """Exports the conversation and the graph to a PDF (without emojis)"""

    img_path = "graph.png"
    pdf_path = "output.pdf"

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(0, 10, clean_text("Conversation with AI about Investments in Luxury Handbags"), ln=True)

        pdf.set_font("Arial", size=12)
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                pdf.set_text_color(0, 0, 200)
                pdf.multi_cell(0, 10, clean_text(f"You: {msg.content}"))
            elif isinstance(msg, AIMessage):
                pdf.set_text_color(0, 100, 0)
                pdf.multi_cell(0, 10, clean_text(f"AI: {msg.content}"))
            pdf.ln(1)

        # New page for graph
        pdf.add_page()
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, clean_text("Brand Comparison Chart"), ln=True)

        image = Image.open(img_path)
        width, height = image.size
        width_mm = min(180, width * 0.264583)
        height_mm = width_mm * height / width
        pdf.image(img_path, x=10, y=30, w=width_mm, h=height_mm)

        pdf.output(pdf_path)

        message = f"The PDF has been created: **{pdf_path}** (emoji-free, fully compatible)."

    except Exception as e:
        message = f"❌ Failed to create PDF: {str(e)}"

    return {
        "messages": list(state["messages"]) + [AIMessage(content=message)],
        "context": state["context"]
    }


def create_graph():
    """Create the chat graph with context handling and visualization agent"""

    workflow = StateGraph(ChatState)

    # Adding nodes
    workflow.add_node("agent", agent)
    workflow.add_node("visualizer", graph_agent)
    workflow.add_node("pdf_export", pdf_export_agent)

    # Entry point
    workflow.set_entry_point("agent")

    # Decision logic – where to proceed after 'agent'
    def route(state: ChatState) -> str:
        # Manual override
        if "force_next" in state:
            return state["force_next"]

        # Fallback behavior
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                last_user_msg = message.content.lower()
                break
        else:
            return "agent"

        # Avoid overly generic matching
        if any(word in last_user_msg for word in ["graph", "visualization", "chart", "plot", "bar graph"]):
            return "visualizer"
        elif "pdf" in last_user_msg or "export" in last_user_msg:
            return "pdf_export"
        else:
            return END

    # Conditional routing
    workflow.add_conditional_edges("agent", route)
    workflow.add_edge("visualizer", END)
    workflow.add_edge("pdf_export", END)

    return workflow.compile()


def get_initial_state():
    """Create initial state with welcome message and empty context"""
    welcome_message = """Welcome! I'm an AI assistant specialized in luxury handbag investments.
I maintain context throughout our conversation, so you can ask follow-up questions or request more details about
previously discussed topics.

I can help you explore:
- The investment potential of luxury bags
- Resale value trends of brands like Hermès, Chanel, and Louis Vuitton
- Factors that influence appreciation, such as rarity, craftsmanship, and market demand
- Tips on selecting handbags with the best long-term value

What would you like to know about investing in luxury handbags?"""

    return {
        "messages": [
            SystemMessage(content="""You are an expert on investing in luxury handbags. Always reference market trends,
             brand history, and financial insights.
Maintain conversation context and provide relevant data or examples. If a topic isn't covered in available research,
 say so."""),
            AIMessage(content=welcome_message)
        ],
        "context": {
            "last_documents": [],
            "chat_history": []
        }
    }

# Create the runnable graph
graph = create_graph()

if __name__ == "__main__":
    # Initialize chat
    state = get_initial_state()
    print("\nAI:", state["messages"][-1].content)

    # Chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nAI: Goodbye! Thanks for chatting about investments.")
            break

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Get response with context
        state = graph.invoke(state)

        # Print response
        print("\nAI:", state["messages"][-1].content)