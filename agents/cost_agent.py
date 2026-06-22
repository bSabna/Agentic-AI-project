from langchain_groq import ChatGroq
from rag.retriever import retrieve_context
from dotenv import load_dotenv
import os

load_dotenv()

def run_cost_agent(claim: dict) -> dict:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Get relevant context from RAG
    context = retrieve_context(
        f"expected cost for {claim['diagnosis']} {claim['procedure']}"
    )

    prompt = f"""You are a Healthcare Cost Analysis Agent for Cotiviti.

KNOWLEDGE BASE CONTEXT:
{context}

CLAIM TO ANALYZE:
- Patient Age: {claim['age']}
- Diagnosis: {claim['diagnosis']}
- ICD Code: {claim['icd_code']}
- Procedure: {claim['procedure']}
- Billed Amount: ${claim['billed_amount']}
- Provider Type: {claim['provider_type']}

TASK:
1. Based on the knowledge base, what is the expected cost range for this condition?
2. Is the billed amount within normal range, slightly high, or significantly anomalous?
3. Give a cost risk score from 1-10 (1=normal, 10=severely anomalous)

Respond in this exact format:
EXPECTED_RANGE: $X - $Y
COST_ASSESSMENT: [Normal/Slightly High/Anomalous/Severely Anomalous]
COST_RISK_SCORE: X/10
REASONING: Your explanation here
"""

    response = llm.invoke(prompt)
    
    return {
        "agent": "Cost Analysis Agent",
        "output": response.content
    }