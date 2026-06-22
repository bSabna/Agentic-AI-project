from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

def get_retriever():
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