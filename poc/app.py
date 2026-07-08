import streamlit as st
from agents.orchestrator import run_orchestrator
from agents.governance_logger import log_agent_decision, get_all_logs
from datetime import datetime
import json
import re

st.set_page_config(
    page_title="Healthcare Claims Agentic AI",
    page_icon="🏥",
    layout="wide"
)

st.title("Healthcare Claims Agentic AI")
st.subheader("Autonomous Claims Review System with AI Governance")
st.markdown("---")


# ── RAG Evidence Card ─────────────────────────────────────────────────────────
def render_rag_evidence(query: str):
    from rag.retriever import retrieve_with_scores
    results = retrieve_with_scores(query, k=3)
    if not results:
        return
    with st.expander("📚 Evidence Retrieved", expanded=False):
        for i, ev in enumerate(results):
            score = ev["score"]
            score_color = "🟢" if score >= 0.80 else "🟡" if score >= 0.60 else "🔴"
            source_label = ev["source"].replace(".txt", "").replace("_", " ").title()
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown("**Source**")
                st.markdown(f"📄 {source_label}")
            with c2:
                st.markdown("**Section**")
                st.markdown(f"§ {ev['section']}")
            with c3:
                st.markdown("**Similarity Score**")
                st.markdown(f"{score_color} {score}")
            st.markdown("**Retrieved Text**")
            preview = ev["text"][:300] + ("..." if len(ev["text"]) > 300 else "")
            st.markdown(
                f"<div style='background:#f8f9fa;border-left:3px solid #6c757d;"
                f"padding:8px 12px;border-radius:4px;font-size:0.875rem;"
                f"color:#343a40;'>\"{preview}\"</div>",
                unsafe_allow_html=True
            )
            if len(ev["text"]) > 300:
                with st.expander("Show full text"):
                    st.markdown(f'"{ev["text"]}"')
            if i < len(results) - 1:
                st.markdown("---")


# ── Cost Agent ────────────────────────────────────────────────────────────────
def render_cost_agent(output, log, claim):
    _expected  = re.search(
        r'EXPECTED_RANGE:\s*(.+?)(?=\s+COST_ASSESSMENT:|\s+COST_RISK_SCORE:|\s+REASONING:|$)',
        output, re.DOTALL)
    _assess    = re.search(
        r'COST_ASSESSMENT:\s*(.+?)(?=\s+COST_RISK_SCORE:|\s+REASONING:|$)',
        output, re.DOTALL)
    _risk      = re.search(
        r'COST_RISK_SCORE:\s*(\d+(?:\.\d+)?)/10',
        output)
    _reasoning = re.search(
        r'REASONING:\s*(.+)$',
        output, re.DOTALL)

    _expected_val  = _expected.group(1).strip()  if _expected  else "N/A"
    _assess_val    = _assess.group(1).strip()    if _assess    else "N/A"
    _risk_val      = _risk.group(1).strip()      if _risk      else "N/A"
    _reasoning_val = _reasoning.group(1).strip() if _reasoning else output

    _risk_int    = int(float(_risk_val)) if _risk_val != "N/A" else 0
    _risk_icon   = "🔴" if _risk_int >= 7 else "🟡" if _risk_int >= 4 else "🟢"
    _assess_icon = "⚠️" if "anomal" in _assess_val.lower() else "✅"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Status**"); st.markdown("✅ Complete")
    with c2:
        st.markdown("**Confidence**"); st.markdown(f"📊 {log['confidence_score']}%")

    st.markdown("")

    table_data = {
        "Item": [
            "💰 Expected Range",
            "📊 Risk Score",
            "🔎 Assessment",
        ],
        "Value": [
            _expected_val,
            f"{_risk_icon} {_risk_val}/10" if _risk_val != "N/A" else "N/A",
            f"{_assess_icon} {_assess_val}",
        ],
    }
    st.table(table_data)

    with st.expander("🧠 Detailed AI Reasoning"):
        st.write(output)

    render_rag_evidence(f"{claim['diagnosis']} {claim['procedure']} cost")
    st.caption(f"Confidence Score: {log['confidence_score']}% | {log['timestamp']}")


# ── Fraud Agent ───────────────────────────────────────────────────────────────
def render_fraud_agent(output, log, claim):
    _indicators = re.search(
        r'FRAUD_INDICATORS_FOUND:\s*(.+?)(?=\s+RED_FLAGS:|\s+FRAUD_RISK_SCORE:|\s+REASONING:|$)',
        output, re.DOTALL)
    _red_flags  = re.search(
        r'RED_FLAGS:\s*(.+?)(?=\s+FRAUD_RISK_SCORE:|\s+REASONING:|$)',
        output, re.DOTALL)
    _risk       = re.search(
        r'FRAUD_RISK_SCORE:\s*(\d+(?:\.\d+)?)/10',
        output)
    _reasoning  = re.search(
        r'REASONING:\s*(.+)$',
        output, re.DOTALL)

    _indicators_val = _indicators.group(1).strip() if _indicators else "N/A"
    _red_flags_val  = _red_flags.group(1).strip()  if _red_flags  else "None found"
    _risk_val       = _risk.group(1).strip()       if _risk       else "N/A"
    _reasoning_val  = _reasoning.group(1).strip()  if _reasoning  else output

    _risk_int  = int(float(_risk_val)) if _risk_val != "N/A" else 0
    _risk_icon = "🔴" if _risk_int >= 7 else "🟡" if _risk_int >= 4 else "🟢"
    _flag_icon = "🚨" if "yes" in _indicators_val.lower() else "✅"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Status**"); st.markdown("✅ Complete")
    with c2:
        st.markdown("**Confidence**"); st.markdown(f"📊 {log['confidence_score']}%")

    st.markdown("")

    table_data = {
        "Item": [
            "🚩 Indicators Found",
            "📊 Risk Score",
            "🔍 Red Flags",
        ],
        "Value": [
            f"{_flag_icon} {_indicators_val}",
            f"{_risk_icon} {_risk_val}/10" if _risk_val != "N/A" else "N/A",
            _red_flags_val,
        ],
    }
    st.table(table_data)

    with st.expander("🧠 Detailed AI Reasoning"):
        st.write(output)

    render_rag_evidence(f"fraud detection {claim['procedure']} {claim['icd_code']}")
    st.caption(f"Confidence Score: {log['confidence_score']}% | {log['timestamp']}")


# ── Coding Agent ──────────────────────────────────────────────────────────────
def render_coding_agent(output, log, claim):
    
    _code_match = re.search(
        r'CODE_MATCH:\s*(.+?)(?=\s+PROCEDURE_MATCH:|\s+CODING_ACCURACY_SCORE:|\s+REASONING:|$)',
        output, re.DOTALL)
    _proc_match = re.search(
        r'PROCEDURE_MATCH:\s*(.+?)(?=\s+CODING_ACCURACY_SCORE:|\s+REASONING:|$)',
        output, re.DOTALL)
    _score      = re.search(
        r'CODING_ACCURACY_SCORE:\s*(\d+(?:\.\d+)?)/10',
        output)
    _reasoning  = re.search(
        r'REASONING:\s*(.+)$',
        output, re.DOTALL)

    _code_match_val = _code_match.group(1).strip() if _code_match else "N/A"
    _proc_match_val = _proc_match.group(1).strip() if _proc_match else "N/A"
    _score_val      = _score.group(1).strip()      if _score      else "N/A"
    _reasoning_val  = _reasoning.group(1).strip()  if _reasoning  else output

    _valid_icon = "✅" if "valid" in _code_match_val.lower() and "invalid" not in _code_match_val.lower() else "❌"
    _match_icon = "✅" if "appropriate" in _proc_match_val.lower() else "⚠️"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Status**"); st.markdown("✅ Complete")
    with c2:
        st.markdown("**Confidence**"); st.markdown(f"📊 {log['confidence_score']}%")

    st.markdown("")

    table_data = {
        "Item": [
            "🧾 Code Match",
            "⚙️ Procedure Match",
            "📊 Accuracy Score",
        ],
        "Value": [
            f"{_valid_icon} {_code_match_val}",
            f"{_match_icon} {_proc_match_val}",
            f"{_score_val}/10" if _score_val != "N/A" else "N/A",
        ],
    }
    st.table(table_data)

    with st.expander("🧠 Detailed AI Reasoning"):
        st.write(output)

    render_rag_evidence(f"ICD-10 {claim['icd_code']} {claim['diagnosis']} coding guidelines")
    st.caption(f"Confidence Score: {log['confidence_score']}% | {log['timestamp']}")


# ── Decision Agent ────────────────────────────────────────────────────────────
def render_decision_agent(output, log, claim):
    _decision  = re.search(
        r'FINAL_DECISION:\s*(.+?)(?=\s+OVERALL_RISK_SCORE:|\s+PRIORITY:|\s+RECOMMENDED_ACTION:|\s+SUMMARY:|$)',
        output, re.DOTALL)
    _risk      = re.search(
        r'OVERALL_RISK_SCORE:\s*(\d+(?:\.\d+)?)/10',
        output)
    _priority  = re.search(
        r'PRIORITY:\s*(.+?)(?=\s+RECOMMENDED_ACTION:|\s+SUMMARY:|$)',
        output, re.DOTALL)
    _action    = re.search(
        r'RECOMMENDED_ACTION:\s*(.+?)(?=\s+SUMMARY:|$)',
        output, re.DOTALL)
    _summary   = re.search(
        r'SUMMARY:\s*(.+)$',
        output, re.DOTALL)

    _decision_val = _decision.group(1).strip() if _decision else "N/A"
    _risk_val     = _risk.group(1).strip()     if _risk     else "N/A"
    _priority_val = _priority.group(1).strip() if _priority else "N/A"
    _action_val   = _action.group(1).strip()   if _action   else "N/A"
    _summary_val  = _summary.group(1).strip()  if _summary  else "N/A"

    _decision_icon = (
        "✅" if "approve" in _decision_val.lower() else
        "❌" if "reject"  in _decision_val.lower() else "⚠️"
    )
    _priority_icon = (
        "🔴" if "high"   in _priority_val.lower() or "critical" in _priority_val.lower() else
        "🟡" if "medium" in _priority_val.lower() else "🟢"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Status**"); st.markdown("✅ Complete")
    with c2:
        st.markdown("**Confidence**"); st.markdown(f"📊 {log['confidence_score']}%")

    st.markdown("")

    table_data = {
        "Item": [
            "✅ Decision",
            "📊 Overall Risk",
            "🚦 Priority",
            "📄 Recommended Action",
            "📝 Summary",
        ],
        "Value": [
            f"{_decision_icon} {_decision_val}",
            f"{_risk_val}/10" if _risk_val != "N/A" else "N/A",
            f"{_priority_icon} {_priority_val}",
            _action_val,
            _summary_val,
        ],
    }
    st.table(table_data)

    with st.expander("🧠 Detailed AI Reasoning"):
        st.write(output)

    render_rag_evidence(
        f"{claim['diagnosis']} {claim['procedure']} {claim['icd_code']} claim decision"
    )
    st.caption(
        f"Confidence Score: {log['confidence_score']}% | "
        f"Bias Flag: {log['flagged_for_bias']}"
    )


# ── Navigation tabs ───────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Claims Analysis", "AI Governance Dashboard"])


# ── CLAIMS ANALYSIS ───────────────────────────────────────────────────────────
with tab1:
    st.header("Submit a Healthcare Claim")

    col1, col2 = st.columns(2)
    with col1:
        claim_id  = st.text_input("Claim ID",    placeholder="e.g. CLM-2024-001")
        age       = st.number_input("Patient Age", min_value=1, max_value=120, value=25)
        diagnosis = st.text_input("Diagnosis",   placeholder="e.g. Type 2 Diabetes")
        icd_code  = st.text_input("ICD-10 Code", placeholder="e.g. E11.9")
    with col2:
        procedure             = st.text_input("Procedure",
                                    placeholder="e.g. Outpatient consultation and blood work")
        billed_amount         = st.number_input("Billed Amount ($)", min_value=0, value=0)
        provider_type         = st.selectbox("Provider Type",
                                    ["Select Provider Type", "Primary Care",
                                     "Specialist", "Hospital", "Emergency", "Lab"])
        days_since_last_claim = st.number_input("Days Since Last Claim", min_value=0, value=0)
        claims_this_month     = st.number_input("Number of Claims This Month", min_value=0, value=0)

    st.markdown("---")

    if st.button("Run Agentic AI Analysis", type="primary"):
        if not claim_id or not diagnosis or not icd_code or not procedure:
            st.error("Please fill in all fields before running analysis!")
        else:
            claim = {
                "claim_id":              claim_id,
                "age":                   age,
                "diagnosis":             diagnosis,
                "icd_code":              icd_code,
                "procedure":             procedure,
                "billed_amount":         billed_amount,
                "provider_type":         provider_type,
                "days_since_last_claim": days_since_last_claim,
                "claims_this_month":     claims_this_month
            }

            st.header("Agent Pipeline Running...")

            # Cost Agent
            with st.spinner("💰 Cost Analysis Agent thinking..."):
                from agents.cost_agent import run_cost_agent
                cost_result = run_cost_agent(claim)
                cost_log = log_agent_decision(
                    claim_id, "Cost Analysis Agent", cost_result["output"])
            with st.expander("💰 Cost Analysis Agent", expanded=True):
                render_cost_agent(cost_result["output"], cost_log, claim)

            # Fraud Agent
            with st.spinner("🔍 Fraud Detection Agent thinking..."):
                from agents.fraud_agent import run_fraud_agent
                fraud_result = run_fraud_agent(claim)
                fraud_log = log_agent_decision(
                    claim_id, "Fraud Detection Agent", fraud_result["output"])
            with st.expander("🔍 Fraud Detection Agent", expanded=True):
                render_fraud_agent(fraud_result["output"], fraud_log, claim)

            # Coding Agent
            with st.spinner("🧾 Coding Validation Agent thinking..."):
                from agents.coding_agent import run_coding_agent
                coding_result = run_coding_agent(claim)
                coding_log = log_agent_decision(
                    claim_id, "Coding Validation Agent", coding_result["output"])
            with st.expander("🧾 Coding Validation Agent", expanded=True):
                render_coding_agent(coding_result["output"], coding_log, claim)

            # Decision Agent
            with st.spinner("⚖️ Decision Agent making final verdict..."):
                from agents.decision_agent import run_decision_agent
                decision_result = run_decision_agent(
                    claim, cost_result, fraud_result, coding_result)
                decision_log = log_agent_decision(
                    claim_id, "Decision Agent", decision_result["output"])

            st.markdown("---")
            st.header("⚖️ FINAL AUDIT DECISION")
            output = decision_result["output"]
            if "APPROVE" in output:
                st.success("✅ DECISION: APPROVED")
            elif "FLAG" in output:
                st.warning("⚠️ DECISION: FLAGGED FOR REVIEW")
            elif "REJECT" in output:
                st.error("❌ DECISION: REJECTED")

            with st.expander("⚖️ Decision Agent", expanded=True):
                render_decision_agent(decision_result["output"], decision_log, claim)

            # Human Override
            st.markdown("---")
            st.subheader("👤 Human Override (AI Governance)")
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
                        f"Override: {override_decision} | Reason: {override_reason}",
                        override=override_decision)
                    st.success(f"✅ Override logged: {override_decision}")
                    st.info("This override is recorded in the AI Governance audit trail 📋")

            st.markdown("---")
            st.info("All agent decisions are logged with timestamps, confidence scores, "
                   "and bias flags in the AI Governance Dashboard 📊")


# ── AI GOVERNANCE DASHBOARD ───────────────────────────────────────────────────
with tab2:
    st.header("🛡️ AI Governance Dashboard")
    st.caption("Full audit trail of all agent decisions")

    logs = get_all_logs()

    if not logs:
        st.info("No decisions logged yet. Run a claims analysis first!")
    else:
        total          = len(logs)
        overrides      = sum(1 for l in logs if l.get("human_override"))
        bias_flags     = sum(1 for l in logs if l.get("flagged_for_bias"))
        avg_confidence = round(
            sum(l.get("confidence_score", 0) for l in logs) / total, 1)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Decisions Logged", total)
        m2.metric("Human Overrides",        overrides)
        m3.metric("Bias Flags",             bias_flags)
        m4.metric("Avg Confidence Score",   f"{avg_confidence}%")

        st.markdown("---")
        st.subheader("📋 Full Audit Trail")

        for log in reversed(logs):
            with st.expander(
                f"📌 {log['timestamp']} | {log['agent']} | Claim: {log['claim_id']}"):
                st.write(f"**Confidence Score:** {log['confidence_score']}%")
                st.write(f"**Bias Flagged:** {log['flagged_for_bias']}")
                if log.get("human_override"):
                    st.warning(f"**Human Override:** {log['human_override']}")
                st.text(log["decision_output"])

        st.markdown("---")
        st.download_button(
            label="⬇️ Download Full Audit Log (JSON)",
            data=json.dumps(logs, indent=2),
            file_name=f"audit_log_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )