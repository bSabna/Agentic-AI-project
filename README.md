#  Healthcare Agentic AI System

An autonomous multi-agent AI pipeline for healthcare claims review,
built to mirror real-world payment integrity workflows.

##  Agent Pipeline
1. **Cost Analysis Agent** — Detects cost anomalies using RAG
2. **Fraud Detection Agent** — Flags FWA patterns
3. **Coding Validation Agent** — Validates ICD-10 codes
4. **Decision Agent** — Makes final audit verdict

##  Tech Stack
- Python, LangChain, Groq (Llama 3.3 70B)
- ChromaDB (Vector Database)
- RAG over CMS Guidelines & ICD-10 codes
- Streamlit UI

## Run Locally
pip install -r requirements.txt
streamlit run app.py