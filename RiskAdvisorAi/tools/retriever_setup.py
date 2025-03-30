from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv(".env", usecwd=True)
load_dotenv(dotenv_path)
import os

my_openai_api_key = os.getenv("OPENAI_API_KEY")
my_pinecone_api_key = os.getenv("PINECONE_API_KEY")
my_pinecone_environemnt = os.getenv("PINECONE_ENVIRONMENT")
my_pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

pinecone_client = pinecone.Pinecone(api_key=my_pinecone_api_key)

def build_vectorstore_from_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = loader.load_and_split()
    print(f"✅ Loaded {len(docs)} documents from PDF")

    embeddings = OpenAIEmbeddings(openai_api_key=my_openai_api_key)

    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=my_pinecone_index_name,
        pinecone_api_key=my_pinecone_api_key
    )

    print("✅ Documents uploaded to Pinecone.")
    return vectorstore.as_retriever()