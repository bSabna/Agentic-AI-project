import streamlit as st
from agents.orchestrator import run_orchestrator

st.set_page_config(
    page_title="Cotiviti Agentic AI",
    page_icon="🏥",
    layout="wide"
)

st.title("Cotiviti Healthcare Claims Agentic AI")
st.subheader("Autonomous Claims Review System")
st.markdown("---")

# Input Form
st.header(" Submit a Healthcare Claim")

col1, col2 = st.columns(2)

with col1:
    claim_id = st.text_input("Claim ID", value="CLM-2024-001")
    age = st.number_input("Patient Age", min_value=1, max_value=120, value=45)
    diagnosis = st.text_input("Diagnosis", value="Type 2 Diabetes")
    icd_code = st.text_input("ICD-10 Code", value="E11.9")

with col2:
    procedure = st.text_input("Procedure", value="Outpatient consultation and blood work")
    billed_amount = st.number_input("Billed Amount ($)", min_value=1, value=1500)
    provider_type = st.selectbox("Provider Type", 
        ["Primary Care", "Specialist", "Hospital", "Emergency", "Lab"])
    days_since_last_claim = st.number_input(
        "Days Since Last Claim", min_value=0, value=45)
    claims_this_month = st.number_input(
        "Number of Claims This Month", min_value=0, value=1)

st.markdown("---")

# Run Agent Button
if st.button(" Run Agentic AI Analysis", type="primary"):
    claim = {
        "claim_id": claim_id,
        "age": age,
        "diagnosis": diagnosis,
        "icd_code": icd_code,
        "procedure": procedure,
        "billed_amount": billed_amount,
        "provider_type": provider_type,
        "days_since_last_claim": days_since_last_claim,
        "claims_this_month": claims_this_month
    }

    st.header(" Agent Pipeline Running...")

    # Cost Agent
    with st.spinner("Cost Analysis Agent thinking..."):
        from agents.cost_agent import run_cost_agent
        cost_result = run_cost_agent(claim)

    with st.expander(" Cost Analysis Agent Results", expanded=True):
        st.success(" Cost Analysis Complete")
        st.text(cost_result["output"])

    # Fraud Agent
    with st.spinner(" Fraud Detection Agent thinking..."):
        from agents.fraud_agent import run_fraud_agent
        fraud_result = run_fraud_agent(claim)

    with st.expander(" Fraud Detection Agent Results", expanded=True):
        st.success(" Fraud Detection Complete")
        st.text(fraud_result["output"])

    # Coding Agent
    with st.spinner("Coding Validation Agent thinking..."):
        from agents.coding_agent import run_coding_agent
        coding_result = run_coding_agent(claim)

    with st.expander(" Coding Validation Agent Results", expanded=True):
        st.success("Coding Validation Complete")
        st.text(coding_result["output"])

    # Decision Agent
    with st.spinner(" Decision Agent making final verdict..."):
        from agents.decision_agent import run_decision_agent
        decision_result = run_decision_agent(
            claim, cost_result, fraud_result, coding_result)

    st.markdown("---")
    st.header(" FINAL AUDIT DECISION")

    # Color code the decision
    output = decision_result["output"]
    if "APPROVE" in output:
        st.success(" DECISION: APPROVED")
    elif "FLAG" in output:
        st.warning(" DECISION: FLAGGED FOR REVIEW")
    elif "REJECT" in output:
        st.error(" DECISION: REJECTED")

    st.text(output)

    st.markdown("---")
    st.info(" This analysis was performed by Cotiviti's Agentic AI Pipeline: "
            "Cost Agent → Fraud Agent → Coding Agent → Decision Agent")