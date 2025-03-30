import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as PineconeStore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import time
import argparse


def find_env_file():
    """Find the .env file in the ai-chatbots root"""
    current_dir = Path(__file__).parent
    possible_locations = [
        current_dir / '.env',             # current directory (lekce3) first
        current_dir.parent / '.env',      # ai-chatbots directory second
        Path.cwd() / '.env'              # current working directory last
    ]

    for env_path in possible_locations:
        if env_path.exists():
            print(f"Found .env file at: {env_path}")
            return env_path

    raise FileNotFoundError("No .env file found")


def reset_index(pc, index_name):
    """Delete and recreate the index"""
    print(f"\nResetting index: {index_name}")

    try:
        # Delete if exists
        if index_name in pc.list_indexes().names():
            print("Deleting existing index...")
            pc.delete_index(index_name)

            # Wait for deletion to complete
            print("Waiting for deletion to complete...")
            while index_name in pc.list_indexes().names():
                time.sleep(1)

        print("Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

        print("Index reset complete!")

    except Exception as e:
        print(f"Error during index reset: {e}")
        raise


def process_pdfs(reset=False):
    try:
        # Set up paths relative to the script
        script_dir = Path(__file__).parent
        pdf_dir = script_dir / "pdfs"
        pdf_dir.mkdir(exist_ok=True)

        print(f"\nLooking for PDFs in: {pdf_dir}")

        # Load PDFs from directory
        loader = DirectoryLoader(
            pdf_dir,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()

        if not documents:
            raise ValueError(f"No PDF documents found in {pdf_dir}")

        print(f"\nFound {len(documents)} PDF documents")

        # Split documents into chunks with significant overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,           # Smaller chunks (was 2000)
            chunk_overlap=100,        # Smaller overlap (was 500)
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # Better separation at paragraphs
            add_start_index=True,     # Add start index for ordering
        )
        splits = text_splitter.split_documents(documents)

        # Add essay title and section metadata
        for split in splits:
            if 'source' in split.metadata:
                filename = Path(split.metadata['source']).stem
                split.metadata['essay_title'] = filename
                # Add the first 100 chars as a preview
                split.metadata['preview'] = split.page_content[:100].replace('\n', ' ')
                # Add chunk number for context
                split.metadata['chunk_number'] = split.metadata.get('start_index', 0)

        print(f"\nCreated {len(splits)} chunks from the documents")

        # Initialize embeddings
        embeddings = OpenAIEmbeddings()

        # Initialize Pinecone
        pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )

        # Get index name
        index_name = os.getenv("PINECONE_INDEX")

        # Optionally reset the index
        if reset:
            reset_index(pc, index_name)

        # Create vectorstore
        vectorstore = PineconeStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=index_name
        )

        print("\nProcessing complete!")
        print(f"Successfully processed and uploaded {len(documents)} PDFs")
        print(f"Total chunks created and stored: {len(splits)}")

        # Test query to verify
        print("\nTesting retrieval...")
        results = vectorstore.similarity_search(
            "What is this document about?",
            k=2
        )
        for doc in results:
            print(f"\nSource: {doc.metadata.get('essay_title', 'Unknown')}")
            print(f"Preview: {doc.page_content[:200]}...")

        return vectorstore

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise


def main():
    print("\n" + "="*80)
    print("PDF PROCESSING SCRIPT")
    print("="*80)

    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Process PDFs and update Pinecone index')
    parser.add_argument('--reset', action='store_true', help='Reset the index before processing')
    args = parser.parse_args()

    try:
        # First try to load the environment
        env_path = find_env_file()
        load_dotenv(env_path)
        print("\n✅ Kontrola environment proměnných:")
        print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
        print("PINECONE_API_KEY:", os.getenv("PINECONE_API_KEY"))
        print("PINECONE_ENV:", os.getenv("PINECONE_ENV"))
        print("PINECONE_INDEX:", os.getenv("PINECONE_INDEX"))

        # Verify environment variables
        required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENV", "PINECONE_INDEX"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            print("\nError: Missing required environment variables:")
            for var in missing_vars:
                print(f"- {var}")
            print("\nPlease check your .env file")
            return

        # If environment is good, process PDFs with reset flag
        process_pdfs(reset=args.reset)

    except Exception as e:
        print(f"\nFatal error: {str(e)}")
    finally:
        print("\nScript execution completed")


if __name__ == "__main__":
    main()
