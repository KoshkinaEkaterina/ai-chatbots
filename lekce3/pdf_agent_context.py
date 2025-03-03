from typing import TypedDict, List, Dict, Sequence
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as PineconeStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory  # From base langchain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain  # Full path
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph

# Initialize environment and Pinecone
env_path = Path(__file__).parent / '.env'  # Changed to look in lekce3 directory first
load_dotenv(env_path)

# Verify environment variables are loaded
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX")

if not all([api_key, env, index_name]):
    raise ValueError("Missing required environment variables. Check your .env file.")

pc = Pinecone(
    api_key=api_key,
    environment=env
)

# Initialize vectorstore
vectorstore = PineconeStore.from_existing_index(
    index_name=index_name,
    embedding=OpenAIEmbeddings()
)

# After initializing vectorstore, add this debug section
print("\nChecking Vector Store Content...")
try:
    # Try to do a simple search to verify content
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

# Enhanced prompt template for better context handling
CONTEXT_PROMPT = PromptTemplate(
    template="""You are an expert on Paul Graham's essays. Use the following pieces of context to answer the question.
If you don't find the answer in the essays provided, say so clearly.

Previous conversation context:
{chat_history}

Relevant passages from Paul Graham's essays:
{context}

Given this context and the conversation history, please answer:
{question}

Guidelines:
1. If the question refers to something mentioned in the previous conversation, address that connection explicitly
2. Quote relevant passages from the essays when appropriate
3. If expanding on a previously mentioned topic, explain how this new information relates
4. Maintain the context of the ongoing conversation
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

def agent(state: ChatState) -> ChatState:
    """Process the latest message and generate a response with context awareness"""
    
    # Get chat history and last message
    messages = state["messages"]
    
    # Format chat history for context
    chat_history = []
    for i in range(0, len(messages)-1, 2):
        if i+1 < len(messages):
            chat_history.append({
                "human": messages[i].content,
                "ai": messages[i+1].content
            })
    
    current_message = messages[-1].content
    
    # Generate response with context
    response = chain.invoke({
        "question": current_message,
        "chat_history": chat_history
    })
    
    # Log retrieved documents for context
    if response.get("source_documents"):
        print("\nRelevant passages found:")
        for doc in response["source_documents"]:
            print(f"\nFrom essay: {doc.metadata.get('essay_title', 'Unknown')}")
            print(f"Context: {doc.page_content[:200]}...")
    
    # Update state with new message and context
    return {
        "messages": list(messages) + [AIMessage(content=response["answer"])],
        "context": {
            "last_documents": response.get("source_documents", []),
            "chat_history": chat_history
        }
    }

def create_graph():
    """Create the chat graph with context handling"""
    workflow = StateGraph(ChatState)
    workflow.add_node("agent", agent)
    workflow.set_entry_point("agent")
    return workflow.compile()

def get_initial_state():
    """Create initial state with welcome message and empty context"""
    welcome_message = """Welcome! I'm an AI assistant specialized in Paul Graham's essays. 
I maintain context throughout our conversation, so you can ask follow-up questions or request more details about previously discussed topics.

I can help you explore his thoughts on:
- Startups and entrepreneurship
- Technology and programming
- Business and wealth creation
- Society and life philosophy

What would you like to know about Paul Graham's ideas?"""
    
    return {
        "messages": [
            SystemMessage(content="""You are an expert on Paul Graham's essays. Always draw from his writings,
maintain conversation context, and provide relevant quotes. If a topic isn't covered in his essays, say so."""),
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
            print("\nAI: Goodbye! Thanks for chatting about Paul Graham's essays.")
            break
        
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        
        # Get response with context
        state = graph.invoke(state)
        
        # Print response
        print("\nAI:", state["messages"][-1].content) 