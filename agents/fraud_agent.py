from langchain_groq import ChatGroq
from rag.retriever import retrieve_context
from dotenv import load_dotenv
import os

load_dotenv()

def run_fraud_agent(claim: dict) -> dict:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    context = retrieve_context("fraud indicators waste abuse billing")

    prompt = f"""You are a Fraud, Waste & Abuse Detection Agent for Cotiviti.

KNOWLEDGE BASE CONTEXT:
{context}

CLAIM TO ANALYZE:
- Claim ID: {claim['claim_id']}
- Patient Age: {claim['age']}
- Diagnosis: {claim['diagnosis']}
- ICD Code: {claim['icd_code']}
- Procedure: {claim['procedure']}
- Billed Amount: ${claim['billed_amount']}
- Provider Type: {claim['provider_type']}
- Days Since Last Claim: {claim['days_since_last_claim']}
- Number of Claims This Month: {claim['claims_this_month']}

TASK:
1. Check for fraud indicators from the knowledge base
2. Identify any red flags in this claim
3. Give a fraud risk score from 1-10 (1=clean, 10=highly suspicious)

Respond in this exact format:
FRAUD_INDICATORS_FOUND: [Yes/No]
RED_FLAGS: List any red flags or "None found"
FRAUD_RISK_SCORE: X/10
REASONING: Your explanation here
"""

    response = llm.invoke(prompt)

    return {
        "agent": "Fraud Detection Agent",
        "output": response.content
    }