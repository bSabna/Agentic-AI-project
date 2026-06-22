from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def run_decision_agent(claim: dict, cost_result: dict, 
                        fraud_result: dict, coding_result: dict) -> dict:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""You are the Final Decision Agent for Cotiviti's Claims Review System.

ORIGINAL CLAIM:
- Claim ID: {claim['claim_id']}
- Patient Age: {claim['age']}
- Diagnosis: {claim['diagnosis']}
- Billed Amount: ${claim['billed_amount']}

COST ANALYSIS AGENT FINDINGS:
{cost_result['output']}

FRAUD DETECTION AGENT FINDINGS:
{fraud_result['output']}

CODING VALIDATION AGENT FINDINGS:
{coding_result['output']}

TASK:
Based on all three agent findings above, make a final decision on this claim.

Respond in this exact format:
FINAL_DECISION: [APPROVE / FLAG FOR REVIEW / REJECT]
OVERALL_RISK_SCORE: X/10
PRIORITY: [Low/Medium/High/Critical]
RECOMMENDED_ACTION: What should happen next
SUMMARY: 2-3 sentence summary of why this decision was made
"""

    response = llm.invoke(prompt)

    return {
        "agent": "Decision Agent",
        "output": response.content
    }