import os
import sys
import threading
import time
import requests
import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# LangChain & Community imports
from langchain_community.document_loaders import WebBaseLoader, AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import uvicorn

# 1. Load environment variables FIRST before calling getenv
load_dotenv()

# Set OpenAI API key (will fetch from .env or your environment)
api_key = os.getenv("OPENAI_API_KEY")
lang_key = os.getenv("LANGSMITH_API_KEY")

# ==========================================
# PART 1: FASTAPI BACKEND APP
# ==========================================
app = FastAPI(title="TNAU Paddy Schemes RAG API")

vector_store = None
retrieval_chain = None

class IngestRequest(BaseModel):
    url: str
    openai_api_key: str
    langsmith_api_key: Optional[str] = None
    project_name: str = "Agri-Schemes-RAG"

class QueryRequest(BaseModel):
    question: str
    openai_api_key: str

@app.post("/ingest")
async def ingest_data(request: IngestRequest):
    global vector_store, retrieval_chain
    
    if request.langsmith_api_key:
      os.environ["LANGSMITH_TRACING"] = "false"
      os.environ["LANGSMITH_API_KEY"] = lang_key
      os.environ["LANGSMITH_PROJECT"] = "Agri-Schemes-RAG"
      os.environ["OPENAI_API_KEY"] = api_key
    docs = []
    try:
        loader = WebBaseLoader(requests.request.url)
        docs = loader.load()
    except Exception as e:
        print(f"WebBaseLoader failed: {e}. Falling back to Playwright...")

    if not docs or not docs[0].page_content.strip():
        try:
            loader = AsyncChromiumLoader(urls=[request.url])
            html_docs = loader.load()
            html2text = Html2TextTransformer()
            docs = html2text.transform_documents(html_docs)
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Both loaders failed. Playwright error: {str(ex)}")

    if not docs or not docs[0].page_content.strip():
        raise HTTPException(status_code=400, detail="Could not extract any content from the provided URL.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    total_chunks = len(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=request.openai_api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question strictly using the provided context below. Do not use outside knowledge.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    return {
        "status": "success",
        "message": "Data successfully scraped, indexed, and vector store created.",
        "total_chunks_indexed": total_chunks
    }

@app.post("/query")
async def query_rag(request: QueryRequest):
    global retrieval_chain
    if not retrieval_chain:
        raise HTTPException(status_code=400, detail="Vector database is not initialized. Please ingest data first.")
    
    response = retrieval_chain.invoke({"input": request.question})
    return {"answer": response["answer"]}


# Function to run FastAPI server in a background thread when Streamlit starts
def start_fastapi_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

if "server_started" not in st.session_state:
    server_thread = threading.Thread(target=start_fastapi_server, daemon=True)
    server_thread.start()
    st.session_state.server_started = True
    time.sleep(2)  # Give uvicorn a brief second to boot up


# ==========================================
# PART 2: STREAMLIT FRONTEND UI
# ==========================================
st.set_page_config(page_title="TNAU Paddy Schemes RAG Assistant", layout="centered")

st.title("🌾 TNAU Paddy Schemes RAG Assistant")
st.markdown("Ask questions about paddy schemes based strictly on indexed data from the TNAU portal.")

st.sidebar.header("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
langsmith_api_key = st.sidebar.text_input("LangSmith API Key (Optional)", type="password")
default_url = "https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html"
url_input = st.sidebar.text_input("Target URL", value=default_url)

BACKEND_URL = "http://127.0.0.1:8000"

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if st.sidebar.button("Load & Index URL"):
    if not openai_api_key:
        st.error("Please provide your OpenAI API Key.")
    else:
        with st.spinner("Scraping URL (with Playwright fallback if needed) and indexing into FAISS..."):
            try:
                payload = {
                    "url": url_input,
                    "openai_api_key": openai_api_key,
                    "langsmith_api_key": langsmith_api_key if langsmith_api_key else None
                }
                res = requests.post(f"{BACKEND_URL}/ingest", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.indexed = True
                    st.success(f"Successfully indexed! Total chunks indexed: **{data['total_chunks_indexed']}**")
                else:
                    st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI backend: {e}")

st.divider()

if st.session_state.indexed:
    st.subheader("Ask a Question")
    user_query = st.text_input("Type your question regarding paddy schemes:")
    
    if st.button("Search Vector DB"):
        if user_query:
            with st.spinner("Retrieving from FAISS Vector Store..."):
                try:
                    payload = {
                        "question": user_query,
                        "openai_api_key": openai_api_key
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
    st.info("Please enter your OpenAI API key and click **Load & Index URL** in the sidebar to get started.")