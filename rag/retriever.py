from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

_vectorstore = None  # cached for the session

def _build_vectorstore():
    docs_path = "data/healthcare_docs"
    documents = []

    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_path, filename), "r") as f:
                content = f.read()
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": filename}
                    )
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

    # No persist_directory = pure in-memory, no filesystem issues
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
    )

    print(f"✅ Vector store built in-memory with {len(chunks)} chunks!")
    return vectorstore


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    return _vectorstore


def get_retriever():
    return _get_vectorstore().as_retriever(search_kwargs={"k": 3})


def retrieve_context(query: str) -> str:
    docs = get_retriever().invoke(query)
    return "\n".join([doc.page_content for doc in docs])


def retrieve_with_scores(query: str, k: int = 3) -> list[dict]:
    vectorstore = _get_vectorstore()
    # Fetch more than needed so dedup still leaves enough unique results
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k * 2)

    evidence = []
    seen_texts = set()

    for doc, score in results:
        # Deduplicate by actual text content
        text = doc.page_content.strip()
        if text in seen_texts:
            continue
        seen_texts.add(text)

        evidence.append({
            "source":  doc.metadata.get("source",  "Unknown"),
            "section": doc.metadata.get("section", "—"),
            "score":   round(score, 2),
            "text":    text
        })

        if len(evidence) == k:  # stop once we have k unique results
            break

    return evidence