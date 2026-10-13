import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

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
import os
from dotenv import load_dotenv


# 1. Load environment variables FIRST before calling getenv
load_dotenv()

# Set OpenAI API key (will fetch from .env or your environment)
api_key = os.getenv("OPENAI_API_KEY")
lang_key = os.getenv("LANGSMITH_API_KEY")

# Turn LangSmith tracking ON
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = lang_key   # <-- paste your key
os.environ["LANGSMITH_PROJECT"] = "KG-Rag-app"          # runs are grouped under this name
os.environ["OPENAI_API_KEY"] = api_key        # <-- paste your key

app = FastAPI(title="Agri Schemes RAG API")

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
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = request.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = request.project_name

    docs = []
    try:
        loader = WebBaseLoader(request.url)
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
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)