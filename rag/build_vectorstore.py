from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

def build_vectorstore():
    docs_path = "data/healthcare_docs"
    documents = []

    # Load all .txt files from healthcare_docs folder
    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_path, filename), "r") as f:
                content = f.read()
                documents.append(content)

    if not documents:
        print(" No documents found in data/healthcare_docs/")
        return

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.create_documents(documents)

    # Create embeddings + save to ChromaDB
    embedding = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="rag/chroma_db"
    )

    print(f" Vector store built with {len(chunks)} chunks!")

if __name__ == "__main__":
    build_vectorstore()