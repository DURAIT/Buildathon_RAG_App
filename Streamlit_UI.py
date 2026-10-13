import streamlit as st
import requests
import os
from dotenv import load_dotenv


# 1. Load environment variables FIRST before calling getenv
load_dotenv()

# Set OpenAI API key (will fetch from .env or your environment)
api_key = os.getenv("OPENAI_API_KEY")
lang_key = os.getenv("LANGSMITH_API_KEY")

st.set_page_config(page_title="Your AI Assistant for Agriculture", layout="centered")

st.title("🌾 Your AI Assistant for Agriculture")
st.markdown("Ask questions about Agri schemes from the TNAU portal.")

st.sidebar.header("Configuration")
default_url = "https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html"
url_input = st.sidebar.text_input("Target URL", value=default_url)

BACKEND_URL = "http://127.0.0.1:8000"

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if st.sidebar.button("Load & Index URL"):
    if not api_key:
        st.error("OpenAI API Key is missing! Please check your .env file or environment variables.")
    else:
        with st.spinner("Scraping URL (with Playwright fallback if needed) and indexing into FAISS..."):
            try:
                payload = {
                    "url": url_input,
                    "openai_api_key": api_key,
                    "langsmith_api_key": lang_key if lang_key else None
                }
                res = requests.post(f"{BACKEND_URL}/ingest", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.indexed = True
                    st.success(f"Successfully indexed! Total chunks indexed: **{data['total_chunks_indexed']}**")
                else:
                    st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI backend at {BACKEND_URL}. Ensure backend.py is running.")

st.divider()

if st.session_state.indexed:
    st.subheader("Ask a Question")
    user_query = st.text_input("Type your question regarding Agri schemes:")
    
    if st.button("Search"):
        if user_query:
            if not api_key:
                st.error("OpenAI API Key is missing from the environment.")
            else:
                with st.spinner("Retrieving from FAISS Vector Store..."):
                    try:
                        payload = {
                            "question": user_query,
                            "openai_api_key": api_key
                        }
                        res = requests.post(f"{BACKEND_URL}/query", json=payload)
                        if res.status_code == 200:
                            answer = res.json().get("answer")
                            st.markdown("### Answer:")
                            st.write(answer)
                        else:
                            st.error(f"Error: {res.json().get('detail', 'Query failed')}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Click **Load & Index URL** in the sidebar to get started.")
