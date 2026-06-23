import streamlit as st
from agents.orchestrator import run_orchestrator
from agents.governance_logger import log_agent_decision, get_all_logs
from datetime import datetime
import json

st.set_page_config(
    page_title="Healthcare Claims Agentic AI",
    page_icon="🏥",
    layout="wide"
)

st.title("Healthcare Claims Agentic AI")
st.subheader("Autonomous Claims Review System with AI Governance")
st.markdown("---")

# Navigation tabs
tab1, tab2 = st.tabs(["Claims Analysis", " AI Governance Dashboard"])


# CLAIMS ANALYSIS

with tab1:
    st.header(" Submit a Healthcare Claim")

    col1, col2 = st.columns(2)

    with col1:
        claim_id = st.text_input("Claim ID", 
            placeholder="e.g. CLM-2024-001")
        age = st.number_input("Patient Age", 
            min_value=1, max_value=120, value=25)
        diagnosis = st.text_input("Diagnosis", 
            placeholder="e.g. Type 2 Diabetes")
        icd_code = st.text_input("ICD-10 Code", 
            placeholder="e.g. E11.9")

    with col2:
        procedure = st.text_input("Procedure", 
            placeholder="e.g. Outpatient consultation and blood work")
        billed_amount = st.number_input("Billed Amount ($)", 
            min_value=0, value=0)
        provider_type = st.selectbox("Provider Type",
            ["Select Provider Type", "Primary Care", 
             "Specialist", "Hospital", "Emergency", "Lab"])
        days_since_last_claim = st.number_input(
            "Days Since Last Claim", min_value=0, value=0)
        claims_this_month = st.number_input(
            "Number of Claims This Month", min_value=0, value=0)

    st.markdown("---")

    if st.button(" Run Agentic AI Analysis", type="primary"):
        if not claim_id or not diagnosis or not icd_code or not procedure:
            st.error("Please fill in all fields before running analysis!")
        else:
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
            with st.spinner(" Cost Analysis Agent thinking..."):
                from agents.cost_agent import run_cost_agent
                cost_result = run_cost_agent(claim)
                cost_log = log_agent_decision(
                    claim_id, "Cost Analysis Agent", 
                    cost_result["output"])

            with st.expander(" Cost Analysis Agent", expanded=True):
                st.success(" Complete")
                st.text(cost_result["output"])
                st.caption(f"Confidence Score: "
                          f"{cost_log['confidence_score']}% | "
                          f"{cost_log['timestamp']}")

            # Fraud Agent
            with st.spinner(" Fraud Detection Agent thinking..."):
                from agents.fraud_agent import run_fraud_agent
                fraud_result = run_fraud_agent(claim)
                fraud_log = log_agent_decision(
                    claim_id, "Fraud Detection Agent",
                    fraud_result["output"])

            with st.expander(" Fraud Detection Agent", expanded=True):
                st.success("Complete")
                st.text(fraud_result["output"])
                st.caption(f" Confidence Score: "
                          f"{fraud_log['confidence_score']}% | "
                          f" {fraud_log['timestamp']}")

            # Coding Agent
            with st.spinner("Coding Validation Agent thinking..."):
                from agents.coding_agent import run_coding_agent
                coding_result = run_coding_agent(claim)
                coding_log = log_agent_decision(
                    claim_id, "Coding Validation Agent",
                    coding_result["output"])

            with st.expander("Coding Validation Agent", expanded=True):
                st.success(" Complete")
                st.text(coding_result["output"])
                st.caption(f" Confidence Score: "
                          f"{coding_log['confidence_score']}% | "
                          f"{coding_log['timestamp']}")

            # Decision Agent
            with st.spinner(" Decision Agent making final verdict..."):
                from agents.decision_agent import run_decision_agent
                decision_result = run_decision_agent(
                    claim, cost_result, fraud_result, coding_result)
                decision_log = log_agent_decision(
                    claim_id, "Decision Agent",
                    decision_result["output"])

            st.markdown("---")
            st.header(" FINAL AUDIT DECISION")

            output = decision_result["output"]
            if "APPROVE" in output:
                st.success("DECISION: APPROVED")
            elif "FLAG" in output:
                st.warning("⚠️ DECISION: FLAGGED FOR REVIEW")
            elif "REJECT" in output:
                st.error("DECISION: REJECTED")

            st.text(output)
            st.caption(f"Confidence Score: "
                      f"{decision_log['confidence_score']}% | "
                      f"Bias Flag: {decision_log['flagged_for_bias']}")

            # ── HUMAN OVERRIDE ──
            st.markdown("---")
            st.subheader(" Human Override (AI Governance)")
            st.caption("Reviewer can override the AI decision below")

            override_col1, override_col2 = st.columns(2)
            with override_col1:
                override_decision = st.selectbox(
                    "Override Decision",
                    ["No Override", "APPROVE", "FLAG FOR REVIEW", "REJECT"])
            with override_col2:
                override_reason = st.text_input(
                    "Reason for Override", 
                    placeholder="e.g. Additional documentation provided")

            if st.button("Submit Override"):
                if override_decision != "No Override":
                    log_agent_decision(
                        claim_id, "Human Override",
                        f"Override: {override_decision} | "
                        f"Reason: {override_reason}",
                        override=override_decision)
                    st.success(f"Override logged: {override_decision}")
                    st.info("This override is recorded in the "
                           "AI Governance audit trail ")

            st.markdown("---")
            st.info("All agent decisions are logged with timestamps, "
                   "confidence scores, and bias flags in the "
                   "AI Governance Dashboard ")


#  AI GOVERNANCE DASHBOARD

with tab2:
    st.header("AI Governance Dashboard")
    st.caption("Full audit trail of all agent decisions")

    logs = get_all_logs()

    if not logs:
        st.info("No decisions logged yet. Run a claims analysis first!")
    else:
        # Summary metrics
        total = len(logs)
        overrides = sum(1 for l in logs if l.get("human_override"))
        bias_flags = sum(1 for l in logs if l.get("flagged_for_bias"))
        avg_confidence = round(
            sum(l.get("confidence_score", 0) for l in logs) / total, 1)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Decisions Logged", total)
        m2.metric("Human Overrides", overrides)
        m3.metric("Bias Flags", bias_flags)
        m4.metric("Avg Confidence Score", f"{avg_confidence}%")

        st.markdown("---")
        st.subheader("Full Audit Trail")

        for log in reversed(logs):
            with st.expander(
                f" {log['timestamp']} | "
                f"{log['agent']} | "
                f"Claim: {log['claim_id']}"):
                st.write(f"**Confidence Score:** "
                        f"{log['confidence_score']}%")
                st.write(f"**Bias Flagged:** {log['flagged_for_bias']}")
                if log.get('human_override'):
                    st.warning(f"**Human Override:** "
                              f"{log['human_override']}")
                st.text(log['decision_output'])

        # Download audit log
        st.markdown("---")
        st.download_button(
            label=" Download Full Audit Log (JSON)",
            data=json.dumps(logs, indent=2),
            file_name=f"audit_log_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )