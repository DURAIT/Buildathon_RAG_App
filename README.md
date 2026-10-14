# 🌾 Hybrid Agri Schemes RAG Assistant

A production-grade Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **Streamlit**, **LangChain**, **FAISS**, **Playwright**, **NeMo Guardrails**, and **Deepeval**. 

This assistant extracts and indexes data from both targeted web links and uploaded multi-PDF documents into a unified vector store, ensuring all answers are retrieved strictly from local data without outside web searches.

---

## 🚀 Key Features

* **Hybrid Ingestion Pipeline**: Automatically extracts content from web portals (using Playwright's `AsyncChromiumLoader` and `Html2TextTransformer`) and uploaded reference PDF documents.
* **Unified Vector Database**: Combines all scraped and parsed documents into a single FAISS vector index using open-source HuggingFace embeddings (`all-MiniLM-L6-v2`).
* **Strict Retrieval & Guardrails**: Enforces local vector database retrieval only, integrated with NeMo Guardrails for safety and domain-boundary enforcement.
* **On-the-Fly Quality Evaluation**: Evaluates every user query response using Deepeval metrics (`FaithfulnessMetric` and `AnswerRelevanceMetric`).
* **Monitoring**: Integrated with LangSmith tracing.
* **Polished UI Dashboard**: A clean, theme-adaptive Streamlit interface featuring metric cards, tabbed views, and real-time response evaluations.

---

## 📂 Project Structure

```text
Buildathon_RAG_App/
│
├── FastAPI_backend.py      # FastAPI backend handling hybrid ingestion, FAISS, guardrails, & deepeval
├── Streamlit_UI.py         # Polished Streamlit frontend UI for file uploads and querying
├── requirements.txt        # Required Python packages
└── .env                    # Environment file for secret keys
