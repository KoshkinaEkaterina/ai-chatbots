from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from tools.retriever_setup import build_vectorstore_from_pdf

retriever = build_vectorstore_from_pdf("data/fake_docs/Risk_Procedures.pdf")

llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

async def handle_doc_query(message: str) -> dict:
    result = qa_chain(message)

    sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
    return {
        "text": result["result"],
        "source": "Docs",
        "raw": {
            "matched_sources": sources
        }
    }