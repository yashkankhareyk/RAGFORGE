"""
tracker.py
==========
Zero-cost observability — tracks every query's latency breakdown,
token usage, and estimated cost entirely in-process.

No external service needed. No API key. No Langfuse.
All data lives in memory and is exposed via /metrics endpoint.

Token cost estimation:
  OpenRouter returns usage.prompt_tokens and usage.completion_tokens
  in the LangChain response's response_metadata.
  We map model name to per-token cost from OpenRouter's published rates.
  Free models ($0) are tracked but show $0.00000 cost.

Metrics tracked per query:
  - retrieval_ms:   BM25 + ChromaDB + RRF fusion time
  - rerank_ms:      Cross-encoder scoring time
  - llm_ms:         LLM generation time
  - total_ms:       End-to-end
  - prompt_tokens:  Input tokens
  - completion_tokens: Output tokens
  - estimated_cost_usd: Estimated $ cost
  - session_id:     Which session (truncated)
  - pipeline:       Which config was used
  - timestamp:      ISO timestamp
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)

# Rolling window — keep last 200 queries
MAX_HISTORY = 200

# OpenRouter cost per 1M tokens (input, output) in USD
# Free models = 0. Update if you switch models.
MODEL_COSTS = {
    "meta-llama/llama-3-8b-instruct":          (0.0, 0.0),     # free
    "meta-llama/llama-3.1-8b-instruct":        (0.0, 0.0),     # free
    "mistralai/mistral-7b-instruct":           (0.0, 0.0),     # free
    "google/gemma-2-9b-it":                    (0.0, 0.0),     # free
    "microsoft/phi-3-mini-128k-instruct":      (0.0, 0.0),     # free
    "llama3-8b-8192":                          (0.0, 0.0),     # groq free
    "llama3-70b-8192":                         (0.0, 0.0),     # groq free
    "mixtral-8x7b-32768":                      (0.0, 0.0),     # groq free
    # Paid models (if you ever switch)
    "openai/gpt-4o-mini":                      (0.15, 0.60),
    "openai/gpt-4o":                           (5.00, 15.00),
    "anthropic/claude-3-haiku":                (0.25, 1.25),
}


class QueryTracker:
    """
    Tracks all query metrics in a rolling deque.
    Thread-safe for single-worker FastAPI (default uvicorn).
    """

    def __init__(self):
        self.history: deque = deque(maxlen=MAX_HISTORY)
        self.total_queries = 0
        self.total_cost_usd = 0.0
        self.total_tokens = 0

    def record(self, entry: dict):
        """Records a completed query entry."""
        self.history.append(entry)
        self.total_queries += 1
        self.total_cost_usd += entry.get("estimated_cost_usd", 0.0)
        self.total_tokens += entry.get("prompt_tokens", 0) + entry.get("completion_tokens", 0)

    def get_summary(self) -> dict:
        """Returns aggregated stats for the dashboard."""
        if not self.history:
            return self._empty_summary()

        entries = list(self.history)
        n = len(entries)

        def avg(key):
            vals = [e[key] for e in entries if e.get(key) is not None]
            return round(sum(vals) / len(vals), 1) if vals else 0.0

        def p95(key):
            vals = sorted([e[key] for e in entries if e.get(key) is not None])
            if not vals:
                return 0.0
            idx = int(len(vals) * 0.95)
            return round(vals[min(idx, len(vals)-1)], 1)

        return {
            "total_queries":       self.total_queries,
            "recent_n":            n,
            "avg_total_ms":        avg("total_ms"),
            "p95_total_ms":        p95("total_ms"),
            "avg_retrieval_ms":    avg("retrieval_ms"),
            "avg_rerank_ms":       avg("rerank_ms"),
            "avg_llm_ms":          avg("llm_ms"),
            "total_prompt_tokens": sum(e.get("prompt_tokens", 0) for e in entries),
            "total_completion_tokens": sum(e.get("completion_tokens", 0) for e in entries),
            "total_cost_usd":      round(self.total_cost_usd, 6),
            "avg_cost_per_query":  round(self.total_cost_usd / max(self.total_queries, 1), 6),
            "recent_queries":      list(reversed(entries))[:20],  # last 20 for table
        }

    def _empty_summary(self) -> dict:
        return {
            "total_queries": 0, "recent_n": 0,
            "avg_total_ms": 0, "p95_total_ms": 0,
            "avg_retrieval_ms": 0, "avg_rerank_ms": 0, "avg_llm_ms": 0,
            "total_prompt_tokens": 0, "total_completion_tokens": 0,
            "total_cost_usd": 0.0, "avg_cost_per_query": 0.0,
            "recent_queries": [],
        }


# Singleton — imported by pipeline.py and api.py
tracker = QueryTracker()


class QueryTimer:
    """
    Context-manager-style timer for tracking pipeline stages.

    Usage:
        qt = QueryTimer(session_id="abc", model="llama3-8b-8192")
        with qt.stage("retrieval"):
            docs = retriever.retrieve(query)
        with qt.stage("rerank"):
            top = reranker.rerank(query, docs)
        with qt.stage("llm"):
            response = chain.invoke(...)
        entry = qt.finish(response_metadata, question)
        tracker.record(entry)
    """

    def __init__(self, session_id: str = "unknown", model: str = "unknown", pipeline: str = "full"):
        self.session_id = session_id
        self.model = model
        self.pipeline = pipeline
        self._stages: Dict[str, float] = {}
        self._current_stage: str = None
        self._stage_start: float = None
        self._start = time.perf_counter()

    def stage(self, name: str):
        """Returns a context manager for timing a stage."""
        return _StageTimer(self, name)

    def _record_stage(self, name: str, elapsed_ms: float):
        self._stages[name] = round(elapsed_ms, 1)

    def finish(self, response_metadata: dict, question: str) -> dict:
        """Builds the final entry dict from all recorded stages."""
        total_ms = round((time.perf_counter() - self._start) * 1000, 1)

        # Extract token usage from LangChain response_metadata
        usage = response_metadata.get("token_usage", {}) or \
                response_metadata.get("usage", {}) or {}
        prompt_tokens     = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

        # Estimate cost
        in_rate, out_rate = MODEL_COSTS.get(self.model, (0.0, 0.0))
        cost = (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000

        return {
            "timestamp":          datetime.now().isoformat(),
            "session_id":         self.session_id[:8] + "..." if len(self.session_id) > 8 else self.session_id,
            "pipeline":           self.pipeline,
            "model":              self.model,
            "question_preview":   question[:60] + ("..." if len(question) > 60 else ""),
            "retrieval_ms":       self._stages.get("retrieval", 0.0),
            "rerank_ms":          self._stages.get("rerank", 0.0),
            "llm_ms":             self._stages.get("llm", 0.0),
            "total_ms":           total_ms,
            "prompt_tokens":      prompt_tokens,
            "completion_tokens":  completion_tokens,
            "estimated_cost_usd": round(cost, 8),
        }


class _StageTimer:
    """Internal context manager used by QueryTimer.stage()."""
    def __init__(self, qt: QueryTimer, name: str):
        self.qt = qt
        self.name = name
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = (time.perf_counter() - self._start) * 1000
        self.qt._record_stage(self.name, elapsed)