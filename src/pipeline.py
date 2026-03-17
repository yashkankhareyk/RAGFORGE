"""
pipeline.py  (v2 — with memory + tracking)
===========================================
Changes from v1:
  1. query() now accepts session_id and uses conversational memory
  2. QueryTimer tracks latency per stage and token cost
  3. response_metadata extracted from LangChain response for token counts
  4. clear_session() exposed for "New Chat" button
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion import load_vectorstore
from src.retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.memory import (
    build_conversational_rag_chain,
    trim_history,
    clear_session,
    get_session_count,
)
from src.tracker import QueryTimer, tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Prompt used for dense/hybrid pipelines (no history) — kept for A/B eval
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say: "I don't have enough information."

Context:
{context}

Question: {question}
Answer:"""
)


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    if provider == "groq":
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
            temperature=0.0, max_tokens=1024,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "openrouter":
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct"),
            temperature=0.0, max_tokens=1024,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "http://localhost", "X-Title": "RAGForge"},
        )
    raise ValueError(f"Unknown LLM_PROVIDER='{provider}'")


def _load_chunks_from_vectorstore(vectorstore, limit=5000) -> List[Document]:
    result = vectorstore._collection.get(
        limit=limit, include=["documents", "metadatas"]
    )
    return [
        Document(page_content=c, metadata=m or {})
        for c, m in zip(result["documents"], result["metadatas"])
    ]


class RAGPipeline:
    """
    Production RAG pipeline with:
      - Hybrid retrieval (BM25 + ChromaDB + RRF)
      - Cross-encoder reranking
      - Conversational memory (window buffer, last 6 messages)
      - Per-stage latency tracking
      - Token usage + cost estimation
    """

    def __init__(self, retriever, reranker, llm):
        self.retriever = retriever
        self.reranker  = reranker
        self.llm       = llm
        self.model_name = (
            getattr(llm, 'model', None) or
            getattr(llm, 'model_name', 'unknown')
        )
        # Conversational chain — wraps LLM with history
        self.conv_chain = build_conversational_rag_chain(None, llm)

    def query(
        self,
        question: str,
        session_id: str = "default",
        verbose: bool = False,
    ) -> dict:
        """
        Full pipeline query with memory and tracking.

        Args:
            question:   User question
            session_id: Browser session ID for memory isolation
            verbose:    Log retrieved chunks

        Returns dict with:
            answer, contexts, sources, latency_ms,
            retrieval_ms, rerank_ms, llm_ms,
            prompt_tokens, completion_tokens, estimated_cost_usd
        """
        qt = QueryTimer(
            session_id=session_id,
            model=self.model_name,
            pipeline="hybrid+rerank",
        )

        # ── Stage 1: Hybrid retrieval ────────────────────────────────────────
        with qt.stage("retrieval"):
            candidates = self.retriever.retrieve(question)

        # ── Stage 2: Cross-encoder reranking ─────────────────────────────────
        with qt.stage("rerank"):
            top_docs = self.reranker.rerank(question, candidates)

        if verbose:
            for i, d in enumerate(top_docs, 1):
                logger.info(f"  [{i}] {d.metadata.get('source','?')}: {d.page_content[:100]}")

        context = "\n\n---\n\n".join(d.page_content for d in top_docs)

        # ── Stage 3: LLM with memory ─────────────────────────────────────────
        with qt.stage("llm"):
            response = self.conv_chain.invoke(
                {"question": question, "context": context},
                config={"configurable": {"session_id": session_id}},
            )

        # Trim memory window after each turn
        trim_history(session_id)

        # Extract token metadata from LangChain response
        response_metadata = getattr(response, "response_metadata", {}) or {}

        # Build tracking entry and record it
        entry = qt.finish(response_metadata, question)
        tracker.record(entry)

        return {
            "answer":              response.content,
            "contexts":            [d.page_content for d in top_docs],
            "sources":             [d.metadata.get("source", "unknown") for d in top_docs],
            "latency_ms":          entry["total_ms"],
            "retrieval_ms":        entry["retrieval_ms"],
            "rerank_ms":           entry["rerank_ms"],
            "llm_ms":              entry["llm_ms"],
            "prompt_tokens":       entry["prompt_tokens"],
            "completion_tokens":   entry["completion_tokens"],
            "estimated_cost_usd":  entry["estimated_cost_usd"],
        }

    def clear_session(self, session_id: str):
        """Clears conversation history for a session."""
        clear_session(session_id)


def init_pipeline(retriever_k=20, reranker_top_n=5):
    logger.info("Initialising RAG pipeline v2...")
    vectorstore = load_vectorstore()
    chunks      = _load_chunks_from_vectorstore(vectorstore)
    retriever   = HybridRetriever(vectorstore=vectorstore, chunks=chunks, k=retriever_k)
    reranker    = CrossEncoderReranker(top_n=reranker_top_n)
    llm         = get_llm()
    pipeline    = RAGPipeline(retriever=retriever, reranker=reranker, llm=llm)
    logger.info("Pipeline v2 ready ✓")
    return pipeline, chunks