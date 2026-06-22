from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def build_if_needed():
    if not os.path.exists("rag/chroma_db"):
        from rag.build_vectorstore import build_vectorstore
        build_vectorstore()

def get_retriever():
    build_if_needed()
    embedding = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="rag/chroma_db",
        embedding_function=embedding
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def retrieve_context(query: str) -> str:
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])