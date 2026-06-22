from agents.cost_agent import run_cost_agent
from agents.fraud_agent import run_fraud_agent
from agents.coding_agent import run_coding_agent
from agents.decision_agent import run_decision_agent

def run_orchestrator(claim: dict) -> dict:
    print(f"\n Starting Agentic AI Pipeline for Claim {claim['claim_id']}")
    
    print(" Running Cost Analysis Agent...")
    cost_result = run_cost_agent(claim)
    
    print(" Running Fraud Detection Agent...")
    fraud_result = run_fraud_agent(claim)
    
    print(" Running Coding Validation Agent...")
    coding_result = run_coding_agent(claim)
    
    print("  Running Decision Agent...")
    decision_result = run_decision_agent(
        claim, cost_result, fraud_result, coding_result
    )

    return {
        "claim_id": claim['claim_id'],
        "cost_analysis": cost_result,
        "fraud_detection": fraud_result,
        "coding_validation": coding_result,
        "final_decision": decision_result
    }