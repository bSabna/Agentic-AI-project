from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
#from langchain.schema import Document
from langchain_core.documents import Document
import os

def build_vectorstore():
    docs_path = "data/healthcare_docs"
    documents = []

    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_path, filename), "r") as f:
                content = f.read()
                # Store the source filename as metadata
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": filename}
                    )
                )

    if not documents:
        print("No documents found in data/healthcare_docs/")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Add section number (chunk index per source) to metadata
    source_counters = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counters[src] = source_counters.get(src, 0) + 1
        chunk.metadata["section"] = source_counters[src]

    embedding = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="rag/chroma_db"
    )

    print(f"Vector store built with {len(chunks)} chunks!")

if __name__ == "__main__":
    build_vectorstore()