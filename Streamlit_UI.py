import streamlit as st
import requests
import os
from dotenv import load_dotenv


# 1. Load environment variables FIRST before calling getenv
load_dotenv()

# Set OpenAI API key (will fetch from .env or your environment)
api_key = os.getenv("OPENAI_API_KEY")
lang_key = os.getenv("LANGSMITH_API_KEY")

# Page Configuration
st.set_page_config(
    page_title="Your AI Assistant for Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Header
st.title("🌾 Your AI Assistant for Agriculture")
st.markdown("Intelligent retrieval from TNAU web portals and uploaded reference documents powered by LangChain, NeMo Guardrails & Deepeval.")
st.divider()

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/agriculture.png", width=70)
    st.header("⚙️ Data Ingestion")
    
    default_url = "https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html"
    url_input = st.text_input("🔗 Target Web URL", value=default_url)
    
    uploaded_files = st.file_uploader(
        "📂 Upload Reference PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    st.markdown("---")
    
    BACKEND_URL = "http://127.0.0.1:8000"

    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "chunks_count" not in st.session_state:
        st.session_state.chunks_count = 0
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = {}

    if st.button("🚀 Load & Index Hybrid Data", use_container_width=True):
        if not api_key:
            st.error("OpenAI API Key is missing in your .env file!")
        else:
            with st.spinner("Scraping web data & parsing PDFs into FAISS..."):
                try:
                    files_payload = []
                    if uploaded_files:
                        for f in uploaded_files:
                            files_payload.append(("files", (f.name, f.getvalue(), "application/pdf")))
                    
                    data_payload = {
                        "url": url_input,
                        "openai_api_key": api_key,
                        "langsmith_api_key": lang_key if lang_key else ""
                    }

                    res = requests.post(f"{BACKEND_URL}/ingest", data=data_payload, files=files_payload if files_payload else None)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.indexed = True
                        st.session_state.chunks_count = data['total_chunks_indexed']
                        st.success("Indexed successfully!")
                    else:
                        st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error("Connection failed. Ensure FastAPI backend is running.")

    # Status Indicator
    st.markdown("### System Status")
    if st.session_state.indexed:
        st.success("🟢 Vector DB Online")
    else:
        st.warning("🟡 Waiting for Ingestion")

# Main Content Area - Tabs Layout
tab1, tab2 = st.tabs(["💬 Query Assistant", "📊 Evaluation & Metrics"])

with tab1:
    if st.session_state.indexed:
        # Metrics Row using native Streamlit containers and metrics
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.metric(label="Total Chunks Indexed", value=st.session_state.chunks_count)
        with col2:
            with st.container(border=True):
                st.metric(label="Active Data Sources", value=f"Web URL + {len(uploaded_files) if uploaded_files else 0} PDFs")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Ask a Question")
        
        user_query = st.text_input("Type your question regarding agricultural schemes:", placeholder="e.g., What are the subsidy schemes available for paddy cultivation?")
        
        if st.button("🔍 Search", use_container_width=True):
            if user_query:
                with st.spinner("Retrieving strict context from FAISS & evaluating response..."):
                    try:
                        payload = {
                            "question": user_query,
                            "openai_api_key": api_key
                        }
                        res = requests.post(f"{BACKEND_URL}/query", json=payload)
                        if res.status_code == 200:
                            result = res.json()
                            answer = result.get("answer")
                            st.session_state.last_evaluation = result.get("evaluation", {})

                            st.markdown("### 💡 Answer:")
                            with st.container(border=True):
                                st.write(answer)
                        else:
                            st.error(f"Error: {res.json().get('detail', 'Query failed')}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            else:
                st.warning("Please enter a question first.")
    else:
        st.info("👈 Please upload your reference PDFs (optional), set your target URL, and click **Load & Index Hybrid Data** in the sidebar to get started.")

with tab2:
    st.subheader("Deepeval Response Quality Metrics")
    if st.session_state.last_evaluation:
        eval_data = st.session_state.last_evaluation
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            with st.container(border=True):
                score = eval_data.get("answer_relevance_score", 0)
                passed = eval_data.get("answer_relevance_passed", False)
                st.metric("Answer Relevance Score", f"{score:.2f}", "Passed ✅" if passed else "Failed ❌")
        with col_m2:
            with st.container(border=True):
                score_f = eval_data.get("faithfulness_score", 0)
                passed_f = eval_data.get("faithfulness_passed", False)
                st.metric("Faithfulness Score", f"{score_f:.2f}", "Passed ✅" if passed_f else "Failed ❌")
            
        st.markdown("#### Raw Evaluation Payload")
        with st.container(border=True):
            st.json(eval_data)
    else:
        st.info("No queries evaluated yet. Ask a question in the **Query Assistant** tab to view metrics here.")
