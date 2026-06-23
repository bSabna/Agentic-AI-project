import json
import os
from datetime import datetime

LOG_FILE = "audit_logs/governance_log.json"

def ensure_log_dir():
    os.makedirs("audit_logs", exist_ok=True)

def extract_confidence(agent_output: str) -> float:
    """Extract score from agent output and convert to confidence %"""
    import re
    patterns = [
        r'COST_RISK_SCORE:\s*(\d+(?:\.\d+)?)/10',
        r'FRAUD_RISK_SCORE:\s*(\d+(?:\.\d+)?)/10',
        r'CODING_ACCURACY_SCORE:\s*(\d+(?:\.\d+)?)/10',
        r'OVERALL_RISK_SCORE:\s*(\d+(?:\.\d+)?)/10'
    ]
    for pattern in patterns:
        match = re.search(pattern, agent_output)
        if match:
            score = float(match.group(1))
            return round((10 - score) / 10 * 100, 1)
    return 75.0

def log_agent_decision(claim_id: str, agent_name: str, 
                        output: str, override: str = None):
    ensure_log_dir()
    
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "claim_id": claim_id,
        "agent": agent_name,
        "confidence_score": extract_confidence(output),
        "decision_output": output,
        "human_override": override,
        "flagged_for_bias": check_bias(output)
    }
    
    logs.append(entry)
    
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)
    
    return entry

def check_bias(output: str) -> bool:
    """Simple bias check - flag if decision seems inconsistent"""
    high_risk_keywords = ["severely", "critical", "reject", "fraud"]
    low_evidence_keywords = ["unclear", "insufficient", "unable to determine"]
    
    has_high_risk = any(k in output.lower() for k in high_risk_keywords)
    has_low_evidence = any(k in output.lower() for k in low_evidence_keywords)
    
    return has_high_risk and has_low_evidence

def get_all_logs():
    ensure_log_dir()
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        try:
            return json.load(f)
        except:
            return []