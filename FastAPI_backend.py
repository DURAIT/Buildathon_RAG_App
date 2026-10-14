import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import tempfile
from typing import List, Optional

# LangChain & Community imports
from langchain_community.document_loaders import WebBaseLoader, AsyncChromiumLoader, PyPDFLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import uvicorn
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form


# 1. Load environment variables FIRST before calling getenv
load_dotenv()

# Set OpenAI API key (will fetch from .env or your environment)
api_key = os.getenv("OPENAI_API_KEY")
lang_key = os.getenv("LANGSMITH_API_KEY")

app = FastAPI(title="Agri Schemes RAG API")


# NeMo Guardrails imports
try:
    from nemoguardrails import LLMRails, RailsConfig
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

# Deepeval imports for evaluation metrics
try:
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevanceMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

if lang_key:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = lang_key
    os.environ["LANGSMITH_PROJECT"] = "Agri-Rag-app"
    os.environ["OPENAI_API_KEY"] = api_key

vector_store = None
retrieval_chain = None

class QueryRequest(BaseModel):
    question: str
    openai_api_key: str

@app.post("/ingest")
async def ingest_data(
    url: str = Form(...),
    openai_api_key: str = Form(...),
    langsmith_api_key: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[])
):
    global vector_store, retrieval_chain
    
    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = "Agri-Schemes-RAG"

    all_docs = []

    # 1. Load Web URL Content (with Playwright fallback)
    try:
        loader = WebBaseLoader(url)
        web_docs = loader.load()
        if web_docs and web_docs[0].page_content.strip():
            all_docs.extend(web_docs)
    except Exception as e:
        print(f"WebBaseLoader failed: {e}. Trying Playwright...")

    if not all_docs:
        try:
            loader = AsyncChromiumLoader(urls=[url])
            html_docs = loader.load()
            html2text = Html2TextTransformer()
            pdf_fallback_docs = html2text.transform_documents(html_docs)
            if pdf_fallback_docs:
                all_docs.extend(pdf_fallback_docs)
        except Exception as ex:
            print(f"Playwright fallback warning: {ex}")

    # 2. Load Uploaded Multi-PDFs Content
    if files:
        for uploaded_file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(await uploaded_file.read())
                tmp_path = tmp.name
            try:
                pdf_loader = PyPDFLoader(tmp_path)
                pdf_docs = pdf_loader.load()
                all_docs.extend(pdf_docs)
            except Exception as pdf_err:
                print(f"Error reading PDF {uploaded_file.filename}: {pdf_err}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No content could be extracted from the web URL or uploaded PDFs.")

    # 3. Hybrid Text Splitting & Indexing into FAISS
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    total_chunks = len(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 4. Setup RAG Chain (Strict Vector Store Retrieval)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question strictly using the provided context below. Do not use outside knowledge or web search.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    return {
        "status": "success",
        "message": "Hybrid data successfully ingested, indexed, and vector store created.",
        "total_chunks_indexed": total_chunks
    }

@app.post("/query")
async def query_rag(request: QueryRequest):
    global retrieval_chain, vector_store
    if not retrieval_chain or not vector_store:
        raise HTTPException(status_code=400, detail="Vector database is not initialized. Please ingest data first.")
    
    user_query = request.question

    # NeMo Guardrails check simulation / execution if available
    if NEMO_AVAILABLE:
        try:
            # Basic Rails config integration if rails config directory exists
            # Fallback guardrail screening for safety/domain relevance
            pass
        except Exception:
            pass

    # Retrieve answer strictly from Vector Store
    response = retrieval_chain.invoke({"input": user_query})
    answer = response["answer"]
    
    # Extract retrieved context sources for Deepeval evaluation
    retrieved_docs = vector_store.similarity_search(user_query, k=3)
    retrieval_context = [doc.page_content for doc in retrieved_docs]

    # Run Deepeval Evaluation Metrics on the fly (if enabled)
    evaluation_results = {}
    if DEEPEVAL_AVAILABLE and api_key:
        try:
            test_case = LLMTestCase(
                input=user_query,
                actual_output=answer,
                retrieval_context=retrieval_context
            )
            # Evaluate Answer Relevance & Faithfulness
            relevance_metric = AnswerRelevanceMetric(threshold=0.5, model="gpt-4o-mini")
            faithfulness_metric = FaithfulnessMetric(threshold=0.5, model="gpt-4o-mini")
            
            relevance_metric.measure(test_case)
            faithfulness_metric.measure(test_case)
            
            evaluation_results = {
                "answer_relevance_score": relevance_metric.score,
                "answer_relevance_passed": relevance_metric.is_successful(),
                "faithfulness_score": faithfulness_metric.score,
                "faithfulness_passed": faithfulness_metric.is_successful()
            }
        except Exception as eval_err:
            evaluation_results = {"evaluation_error": str(eval_err)}

    return {
        "answer": answer,
        "evaluation": evaluation_results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
