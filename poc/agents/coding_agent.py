from langchain_groq import ChatGroq
from rag.retriever import retrieve_context
import os

try:
    import streamlit as st
    api_key = st.secrets["GROQ_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

def run_coding_agent(claim: dict) -> dict:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key
    )

    context = retrieve_context(
        f"ICD-10 code {claim['icd_code']} {claim['diagnosis']}"
    )

    prompt = f"""You are a Medical Coding Validation Agent for Cotiviti.

KNOWLEDGE BASE CONTEXT:
{context}

CLAIM TO ANALYZE:
- Diagnosis Description: {claim['diagnosis']}
- ICD-10 Code Submitted: {claim['icd_code']}
- Procedure: {claim['procedure']}
- Provider Type: {claim['provider_type']}

TASK:
1. Verify if the ICD-10 code matches the diagnosis description
2. Check if the procedure is appropriate for this diagnosis
3. Flag any coding errors or mismatches
4. Give a coding accuracy score from 1-10 (1=many errors, 10=perfect)

Respond in this exact format:
CODE_MATCH: [Valid/Invalid/Partially Valid]
PROCEDURE_MATCH: [Appropriate/Inappropriate/Needs Review]
CODING_ACCURACY_SCORE: X/10
REASONING: Your explanation here
"""

    response = llm.invoke(prompt)

    return {
        "agent": "Coding Validation Agent",
        "output": response.content
    }