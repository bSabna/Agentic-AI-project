from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

def build_if_needed():
    import shutil
    shutil.rmtree("rag/chroma_db", ignore_errors=True)
    from rag.build_vectorstore import build_vectorstore
    build_vectorstore()

def _get_vectorstore():
    build_if_needed()
    embedding = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory="rag/chroma_db",
        embedding_function=embedding
    )

def get_retriever():
    return _get_vectorstore().as_retriever(search_kwargs={"k": 3})

def retrieve_context(query: str) -> str:
    """Plain text context — used by agents for LLM prompt injection."""
    docs = get_retriever().invoke(query)
    return "\n".join([doc.page_content for doc in docs])

def retrieve_with_scores(query: str, k: int = 3) -> list[dict]:
    """
    Returns a list of dicts with full metadata + similarity score.
    Used by app.py to render the 📚 Evidence Retrieved card.

    Each dict:
        {
            "source":     "cms_guidelines.txt",
            "section":    4,
            "score":      0.93,
            "text":       "Patients with..."
        }
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    evidence = []
    for doc, score in results:
        evidence.append({
            "source":  doc.metadata.get("source",  "Unknown"),
            "section": doc.metadata.get("section", "—"),
            "score":   round(score, 2),
            "text":    doc.page_content.strip()
        })
    return evidence