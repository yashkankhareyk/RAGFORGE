"""
api.py  (v2 — memory + metrics endpoints)
==========================================
New endpoints vs v1:
  POST /query          — now accepts session_id, returns latency breakdown + token cost
  DELETE /session/{id} — clears conversation memory for a session
  GET  /metrics        — full observability dashboard data
  GET  /metrics/live   — lightweight ping for real-time updates
  GET  /health         — unchanged
  GET  /stats          — unchanged

Run: uvicorn src.api:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import init_pipeline, RAGPipeline
from src.tracker import tracker
from src.memory import get_session_count, get_all_sessions

import logging
logger = logging.getLogger(__name__)

pipeline: RAGPipeline = None
chunk_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, chunk_count
    logger.info("Starting RAGForge v2...")

    # Auto-ingest if ChromaDB does not exist yet
    # This handles the case where PDFs were uploaded after the Docker build
    chroma_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
    data_dir    = "./data"

    if not Path(chroma_path).exists():
        logger.info("ChromaDB not found — running ingestion now...")
        try:
            from src.ingestion import load_documents, chunk_documents, build_vectorstore
            docs   = load_documents(data_dir)
            chunks = chunk_documents(docs)
            build_vectorstore(chunks)
            logger.info(f"Ingestion complete. {len(chunks)} chunks indexed.")
        except Exception as e:
            logger.error(f"Auto-ingestion failed: {e}")
            logger.error("Make sure PDFs are in the data/ folder.")

    try:
        pipeline, chunks = init_pipeline()
        chunk_count = len(chunks)
        logger.info(f"Ready. Chunks: {chunk_count}")
    except Exception as e:
        logger.error(f"Pipeline init failed: {e}")
        logger.error("Run: python src/ingestion.py --data_dir ./data")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="RAGForge API v2",
    description="Production RAG with memory, hybrid retrieval, reranking, and observability.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — needed for the HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:         str  = Field(..., min_length=2, max_length=1000)
    session_id:       str  = Field(default="default", description="Browser session ID for memory")
    include_contexts: bool = Field(default=False)


class QueryResponse(BaseModel):
    question:            str
    answer:              str
    sources:             list[str]
    latency_ms:          float
    retrieval_ms:        float
    rerank_ms:           float
    llm_ms:              float
    prompt_tokens:       int
    completion_tokens:   int
    estimated_cost_usd:  float
    contexts:            list[str] | None = None


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared:    bool


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Query with full memory context and tracking."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready. Check server logs.")
    try:
        result = pipeline.query(
            question=req.question,
            session_id=req.session_id,
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(500, str(e))

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        sources=list(set(result["sources"])),
        latency_ms=result["latency_ms"],
        retrieval_ms=result["retrieval_ms"],
        rerank_ms=result["rerank_ms"],
        llm_ms=result["llm_ms"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        estimated_cost_usd=result["estimated_cost_usd"],
        contexts=result["contexts"] if req.include_contexts else None,
    )


@app.delete("/session/{session_id}", response_model=ClearSessionResponse)
async def clear_session_endpoint(session_id: str):
    """Clear conversation memory for a session (New Chat button)."""
    if pipeline:
        pipeline.clear_session(session_id)
    return ClearSessionResponse(session_id=session_id, cleared=True)


@app.get("/metrics")
async def metrics():
    """
    Full observability data for the dashboard.
    Returns aggregated stats + last 20 queries with full breakdown.
    """
    summary = tracker.get_summary()
    summary["active_sessions"] = get_session_count()
    summary["session_list"]    = get_all_sessions()
    return summary


@app.get("/metrics/live")
async def metrics_live():
    """Lightweight endpoint for real-time dashboard polling."""
    s = tracker.get_summary()
    return {
        "total_queries":    s["total_queries"],
        "avg_total_ms":     s["avg_total_ms"],
        "p95_total_ms":     s["p95_total_ms"],
        "total_cost_usd":   s["total_cost_usd"],
        "active_sessions":  get_session_count(),
    }


@app.get("/health")
async def health():
    return {
        "status":       "ok" if pipeline else "degraded",
        "pipeline_ok":  pipeline is not None,
        "chunk_count":  chunk_count,
        "model":        os.getenv("GROQ_MODEL") or os.getenv("OPENROUTER_MODEL", "unknown"),
        "total_queries": tracker.total_queries,
    }


@app.get("/stats")
async def stats():
    return {
        "chunk_count":   chunk_count,
        "embed_model":   "BAAI/bge-base-en-v1.5",
        "rerank_model":  "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "llm_provider":  os.getenv("LLM_PROVIDER", "groq"),
        "llm_model":     os.getenv("GROQ_MODEL") or os.getenv("OPENROUTER_MODEL", "unknown"),
        "memory_window": 6,
        "chroma_path":   os.getenv("CHROMA_PERSIST_PATH", "./chroma_db"),
    }


@app.get("/")
async def root():
    return {"message": "RAGForge v2", "docs": "/docs", "metrics": "/metrics"}