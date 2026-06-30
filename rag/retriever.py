# retriever.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import shutil

_vectorstore = None  # cache across calls within the same process

def build_if_needed():
    global _vectorstore
    if _vectorstore is not None:
        return  # already built this session, skip

    chroma_path = "rag/chroma_db"
    shutil.rmtree(chroma_path, ignore_errors=True)
    from rag.build_vectorstore import build_vectorstore
    build_vectorstore()

def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        build_if_needed()
        embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = Chroma(
            persist_directory="rag/chroma_db",
            embedding_function=embedding
        )
    return _vectorstore

def get_retriever():
    return _get_vectorstore().as_retriever(search_kwargs={"k": 3})

def retrieve_context(query: str) -> str:
    docs = get_retriever().invoke(query)
    return "\n".join([doc.page_content for doc in docs])

def retrieve_with_scores(query: str, k: int = 3) -> list[dict]:
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