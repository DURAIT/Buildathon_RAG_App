# 🌾 Agri-Schemes RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system tailored for agricultural query processing. The application combines web scraping of agricultural portals (such as TNAU) and PDF parsing into a persistent **FAISS** vector index, powered by **FastAPI** on the backend and **Streamlit** on the frontend. It includes automated metric evaluation via **DeepEval** and support for observability tracing via **LangSmith**.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Architecture](#-architecture)
3. [Key Features & Improvements](#-key-features--improvements)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Prerequisites & Installation](#-prerequisites--installation)
7. [Environment Variables](#-environment-variables)
8. [Running the Application](#-running-the-application)
9. [API Endpoints](#-api-endpoints)
10. [Evaluation & Metrics](#-evaluation--metrics)

---

## 🌾 Overview

The system allows users to ingest dynamic web sources (e.g., TNAU Agritech Portal schemes) combined with uploaded local PDF documentation. The backend processes, chunks, and embeds the content using HuggingFace models, persisting the index locally via FAISS. Users submit queries through an interactive Streamlit UI, receiving context-bound answers from `gpt-4o-mini` along with explicit document/page source citations and automated real-time quality evaluation metrics (**Answer Relevancy** & **Faithfulness**).

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
        | Ingestion Pipeline           |             | Retrieval, Citations & Eval   |
        | - WebBaseLoader / Playwright |             | - Persistent FAISS Index      |
        | - PyPDFLoader + Metadata     |             | - LangChain RAG Chain         |
        | - Recursive Text Splitter    |             | - Server-Side OpenAI API Key  |
        | - HuggingFace Embeddings     |             | - Source Citation Extraction  |
        | - Local Disk Index Storage   |             | - DeepEval Test Cases         |
        +------------------------------+             +-------------------------------+
```

---

## ✨ Key Features & Improvements

* **Persistent Vector Database**: Saves vector indices to local disk storage (`faiss_index`) upon ingestion and automatically reloads them on backend startup, eliminating redundant document re-embedding across restarts.
* **Metadata & Citation Engine**: Tags ingested chunks with source metadata (`source` URL or PDF filename, and `page` numbers) and returns cited references with every query response.
* **Server-Side API Key Management**: Security enhancement ensuring sensitive keys (`OPENAI_API_KEY`) are managed strictly on the backend rather than sent in client HTTP request payloads.
* **Fixed DeepEval Metrics**: Integrated `AnswerRelevancyMetric` and `FaithfulnessMetric` for live single-query evaluation and benchmark testing over pre-configured evaluation test sets (`/evaluate-fixed-test-set`).
* **Hybrid Ingestion with Fallback**: Combines `WebBaseLoader` with an `AsyncChromiumLoader` (Playwright) fallback for JavaScript-heavy web portals, alongside multi-file PDF processing via `PyPDFLoader`.
* **LangSmith Observability**: Built-in support for execution tracing, monitoring chain latency, and tracking token usage.

---

## 🛠 Tech Stack

| Category | Technology |
| --- | --- |
| **Frontend** | Streamlit |
| **Backend Framework** | FastAPI, Uvicorn |
| **LLM & Orchestration** | LangChain, OpenAI (`gpt-4o-mini`) |
| **Vector Store & Embeddings** | FAISS (`faiss-cpu`), HuggingFace (`all-MiniLM-L6-v2`) |
| **Document Loaders** | `WebBaseLoader`, `AsyncChromiumLoader`, `PyPDFLoader` |
| **Evaluation Framework** | DeepEval (`AnswerRelevancyMetric`, `FaithfulnessMetric`) |
| **Observability** | LangSmith |

---

## 📁 Project Structure

```
.
├── FastAPI_backend.py     # FastAPI server (Ingestion, Persistent FAISS, RAG Query, Evaluation)
├── Streamlit_UI.py        # Interactive Streamlit UI (Ingestion Controls, Query & Source Viewer)
├── faiss_index/           # Auto-created local storage directory for persistent FAISS index
├── .env                   # Local environment variable configuration
└── README.md              # Project documentation
```

---

## ⚙️ Prerequisites & Installation

### 1. Python Environment Setup

Ensure Python 3.10+ is installed. Create and activate a virtual environment:

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install all core packages using `pip`:

```bash
pip install fastapi uvicorn streamlit requests python-dotenv \
            langchain langchain-community langchain-openai langchain-text-splitters \
            faiss-cpu sentence-transformers pypdf html2text playwright \
            deepeval
```

For dynamic web scraping fallback, install the headless Chromium drivers for Playwright:

```bash
playwright install
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
```

*Note: The OpenAI API key is read securely server-side from environment variables and does not need to be passed by the client UI.*

---

## 🚀 Running the Application

### Step 1: Launch the FastAPI Backend

Start the FastAPI application via Uvicorn:

```bash
python FastAPI_backend.py
```

* Backend running at: `http://127.0.0.1:8000`
* Interactive API Documentation (Swagger): `http://127.0.0.1:8000/docs`

*On startup, the backend automatically checks for and loads any pre-existing local FAISS index saved in `./faiss_index`.*

### Step 2: Launch the Streamlit Frontend

In a separate terminal (with your virtual environment active), start Streamlit:

```bash
streamlit run Streamlit_UI.py
```

* Interface accessible at: `http://localhost:8501`

---

## 📡 API Endpoints

### 1. Ingest Data (`POST /ingest`)
Scrapes web pages, parses PDF files, applies metadata tags (source name, page number), generates embeddings, and saves the vector database to disk.

* **Content-Type**: `multipart/form-data`
* **Parameters**:
  * `url` *(string, required)*: Web address to parse.
  * `langsmith_api_key` *(string, optional)*: API key for observability tracing.
  * `files` *(binary upload, optional)*: One or more PDF files.

### 2. Query RAG (`POST /query`)
Performs similarity retrieval, generates grounded answers, returns cited document sources with page numbers, and evaluates output quality.

* **Content-Type**: `application/json`
* **Request Payload**:
  ```json
  {
    "question": "What are the subsidy schemes available for paddy cultivation?"
  }
  ```
* **Sample Response**:
  ```json
  {
    "answer": "Subsidies for paddy cultivation include financial assistance for seeds, micro-irrigation, and machinery...",
    "sources": [
      {
        "source": "Schemes.html",
        "page": "Web Page"
      },
      {
        "source": "paddy_guidelines.pdf",
        "page": 4
      }
    ],
    "evaluation": {
      "answer_relevance_score": 0.95,
      "answer_relevance_passed": true,
      "faithfulness_score": 1.0,
      "faithfulness_passed": true
    }
  }
  ```

### 3. Fixed Test Set Evaluation (`POST /evaluate-fixed-test-set`)
Runs the RAG chain against a predefined suite of agricultural benchmark questions and reports DeepEval metrics.

* **Content-Type**: `application/json`
* **Response**: List of test cases showing `answer_relevancy_score` and `faithfulness_score` per evaluation case.

---

## 📊 Evaluation & Metrics

The system uses **DeepEval** to compute evaluation metrics:

1. **Answer Relevancy (`AnswerRelevancyMetric`)**: Evaluates how directly and completely the model's response addresses the original user prompt (Threshold: `0.5`).
2. **Faithfulness (`FaithfulnessMetric`)**: Measures if the answer is derived strictly from retrieved context chunks without factual hallucinations (Threshold: `0.5`).

Metrics are displayed directly in the **Evaluation & Metrics** tab of the Streamlit UI after executing a query.
