from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import tempfile
import os

_vectorstore = None
_chroma_path = None

def _get_chroma_path():
    global _chroma_path
    if _chroma_path is None:
        _chroma_path = os.path.join(tempfile.gettempdir(), "chroma_db_session")
    return _chroma_path

def _build_vectorstore_at(path):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    docs_path = "data/healthcare_docs"
    documents = []
    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_path, filename), "r") as f:
                content = f.read()
                documents.append(
                    Document(page_content=content, metadata={"source": filename})
                )

    if not documents:
        raise RuntimeError("No documents found in data/healthcare_docs/")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    source_counters = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counters[src] = source_counters.get(src, 0) + 1
        chunk.metadata["section"] = source_counters[src]

    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=path
    )
    print(f"Vector store built with {len(chunks)} chunks at {path}")
    return vectorstore

def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        path = _get_chroma_path()
        import shutil
        shutil.rmtree(path, ignore_errors=True)
        _vectorstore = _build_vectorstore_at(path)
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