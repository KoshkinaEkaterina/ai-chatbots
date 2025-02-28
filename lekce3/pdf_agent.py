from typing import TypedDict, List, Dict, Sequence
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate

from langgraph.graph import StateGraph

# Initialize environment and Pinecone
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

# Initialize vectorstore
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX"),
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

# Create chat chain with better prompting
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}  # Retrieve more documents for context
    ),
    return_source_documents=True,
    verbose=True,  # This will help us debug
    combine_docs_chain_kwargs={
        "prompt": PromptTemplate(
            template="""You are an expert on Paul Graham's essays. Use the following pieces of context from his essays to answer the question. 
            If you don't find the answer in the essays provided, say so clearly.

            Context from essays: {context}

            Chat History: {chat_history}
            Human: {question}
            Assistant: """,
            input_variables=["context", "chat_history", "question"]
        )
    }
)

# Define state schema
class ChatState(TypedDict):
    messages: Sequence[BaseMessage]

# Define agent function
def agent(state: ChatState) -> ChatState:
    """Process the latest message and generate a response"""
    
    # Get chat history and last message
    messages = state["messages"]
    chat_history = [(msg.content, messages[i+1].content) 
                   for i, msg in enumerate(messages[:-1:2])]
    current_message = messages[-1].content
    
    # Generate response using invoke instead of direct call
    response = chain.invoke({
        "question": current_message,
        "chat_history": chat_history
    })
    
    # Log retrieved documents (for debugging)
    if response.get("source_documents"):
        print("\nRetrieved documents:")
        for doc in response["source_documents"]:
            print(f"- {doc.metadata.get('source', 'Unknown source')}")
            print(f"  Preview: {doc.page_content[:100]}...")
    
    # Return updated state
    return {
        "messages": list(messages) + [AIMessage(content=response["answer"])]
    }

# Create graph
def create_graph():
    """Create the chat graph"""
    # Create graph with state schema
    workflow = StateGraph(ChatState)
    
    # Add agent node
    workflow.add_node("agent", agent)
    
    # Add conditional edges
    workflow.set_entry_point("agent")
    
    # Compile graph
    return workflow.compile()

# Initialize chat
def get_initial_state():
    """Create initial state with welcome message"""
    welcome_message = """Welcome! I'm an AI assistant specialized in Paul Graham's essays. 
I have access to his writings and can help you explore his thoughts on various topics like:
- Startups and entrepreneurship
- Technology and programming
- Business and wealth creation
- Society and life philosophy

What would you like to know about Paul Graham's ideas?"""
    
    return {
        "messages": [
            SystemMessage(content="""You are an expert on Paul Graham's essays. Always draw from his writings,
quote relevant passages, and maintain his nuanced perspective. If a topic isn't covered in his essays, say so."""),
            AIMessage(content=welcome_message)
        ]
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
        
        # Get response
        state = graph.invoke(state)
        
        # Print response
        print("\nAI:", state["messages"][-1].content) 