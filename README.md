# 🌾 Agri-Schemes RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system tailored for agricultural query processing. The application combines web scraping of agricultural portals (such as TNAU) and PDF parsing into a hybrid FAISS vector index, powered by **FastAPI** on the backend and **Streamlit** on the frontend. It includes evaluation hooks via **DeepEval** and support for guardrails and observational tracing via **LangSmith**.

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Architecture](#-architecture)
3. [Key Features](#-key-features)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Prerequisites & Installation](#-prerequisites--installation)
7. [Environment Variables](#-environment-variables)
8. [Running the Application](#-running-the-application)
9. [API Endpoints](#-api-endpoints)
10. [Evaluation & Metrics](#-evaluation--metrics)

---

## 🌾 Overview

The system allows users to ingest dynamic dynamic web sources (e.g., TNAU Agritech Portal schemes) combined with custom local PDF documentation. The backend processes, chunks, and embeds the content using HuggingFace models, indexing them into a local FAISS vector database. Users can submit queries through an interactive Streamlit UI, receiving context-bound answers from GPT-4o-mini alongside quality evaluation scores (Relevance & Faithfulness).

---

## 🏗 Architecture

```
                                  +-----------------------+
                                  | Streamlit Frontend UI |
                                  +-----------+-----------+
                                              |
                                      HTTP / JSON Payload
                                              |
                                              v
                                  +-----------------------+
                                  |    FastAPI Backend    |
                                  +-----------+-----------+
                                              |
                       +----------------------+----------------------+
                       |                                             |
                       v                                             v
        +------------------------------+             +-------------------------------+
        | Ingestion Pipeline           |             | Retrieval & Evaluation        |
        | - WebBaseLoader / Playwright |             | - FAISS Retriever (Top K=5)   |
        | - PyPDFLoader                |             | - LangChain RAG Chain         |
        | - Character Text Splitter    |             | - GPT-4o-mini Integration     |
        | - HuggingFace Embeddings     |             | - DeepEval Metrics            |
        +------------------------------+             +-------------------------------+
```

---

## ✨ Key Features

* **Hybrid Ingestion Pipeline**: Ingest data directly from dynamic/static URLs as well as uploaded PDF files simultaneously.
* **Fallback Web Scraping**: Integrates `WebBaseLoader` with a secondary `AsyncChromiumLoader` (Playwright) fallback for JavaScript-rendered web pages.
* **Strict Contextual Answers**: Prompt-engineered RAG pipeline ensuring responses stay grounded in the retrieved FAISS context to minimize hallucinations.
* **Response Evaluation**: Automatically evaluates live responses for **Faithfulness** and **Answer Relevance** using DeepEval.
* **Observability & Tracing**: Native support for LangSmith tracing to debug chain performance and token usage.
* **Interactive UI**: Multi-tab Streamlit frontend featuring interactive data ingestion controls, system health indicators, and metric visualizers.

---

## 🛠 Tech Stack

| Category | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Backend Framework** | FastAPI, Uvicorn |
| **LLM & Orchestration**| LangChain, OpenAI (`gpt-4o-mini`) |
| **Vector Store & Embeddings** | FAISS, HuggingFace (`all-MiniLM-L6-v2`) |
| **Document Loaders** | `WebBaseLoader`, `AsyncChromiumLoader`, `PyPDFLoader` |
| **Evaluation & Guardrails** | DeepEval, NeMo Guardrails |
| **Observability** | LangSmith |

---

## 📁 Project Structure

```bash
.
├── FastAPI_backend.py     # FastAPI application (Ingestion, RAG Chain, Evaluation)
├── Streamlit_UI.py        # Streamlit user interface
├── .env                   # Environment variables config file
└── README.md              # Project documentation
```

---

## ⚙️ Prerequisites & Installation

### 1. Python Environment Setup
Ensure Python 3.10 or higher is installed. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Required Dependencies
Install all required libraries using `pip`:

```bash
pip install fastapi uvicorn streamlit requests python-dotenv \
            langchain langchain-community langchain-openai langchain-text-splitters \
            faiss-cpu sentence-transformers pypdf html2text playwright \
            deepeval nemoguardrails
```

If using `AsyncChromiumLoader` (Playwright) for JavaScript-rendered websites, initialize the headless browser drivers:

```bash
playwright install
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory of your project with the following keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here # Optional
```

---

## 🚀 Running the Application

To run the application, launch both the FastAPI backend server and the Streamlit frontend UI.

### Step 1: Start the FastAPI Backend
Launch the backend server using Uvicorn:

```bash
python FastAPI_backend.py
```
*The server runs locally at `http://127.0.0.1:8000`.*

### Step 2: Start the Streamlit Frontend
In a new terminal window (with the virtual environment activated), start the Streamlit UI:

```bash
streamlit run Streamlit_UI.py
```
*The UI will open in your default browser at `http://localhost:8501`.*

---

## 📡 API Endpoints

### 1. Ingest Data (`POST /ingest`)
Parses web URLs and PDF attachments, splits text into overlapping chunks, generates embeddings, and populates the local FAISS vector store.

* **Content-Type**: `multipart/form-data`
* **Form Data**:
  * `url` *(string, required)*: Web URL to scrape.
  * `openai_api_key` *(string, required)*: OpenAI key.
  * `langsmith_api_key` *(string, optional)*: Key for LangSmith tracing.
  * `files` *(binary array, optional)*: Multiple PDF files to upload.

### 2. Query RAG (`POST /query`)
Executes vector search against FAISS, routes retrieved context to GPT-4o-mini, and evaluates output quality.

* **Content-Type**: `application/json`
* **Request Body**:
```json
{
  "question": "What are the subsidy schemes available for paddy cultivation?",
  "openai_api_key": "your_openai_api_key"
}
```
* **Response Sample**:
```json
{
  "answer": "The available subsidy schemes include...",
  "evaluation": {
    "answer_relevance_score": 0.92,
    "answer_relevance_passed": true,
    "faithfulness_score": 1.0,
    "faithfulness_passed": true
  }
}
```

---

## 📊 Evaluation & Metrics

The backend leverages **DeepEval** to compute real-time metrics during query processing:

1. **Answer Relevance**: Measures how pertinent the generated answer is to the user's prompt (Threshold: `0.5`).
2. **Faithfulness**: Measures whether the generated output is strictly supported by the retrieved document chunks in FAISS, preventing factual hallucinations (Threshold: `0.5`).

Evaluation metrics can be monitored via the **Evaluation & Metrics** tab in the Streamlit UI.
