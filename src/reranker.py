"""
reranker.py
===========
Cross-Encoder reranking: takes top-20 candidates from HybridRetriever
and scores each (query, document) pair jointly, returning the best top-N.

Why cross-encoder over bi-encoder?
  Bi-encoders (like BGE used in ChromaDB) embed query and document separately
  — fast but less precise. Cross-encoders process query+document together,
  giving a richer relevance signal at the cost of extra latency (~100-250ms
  on CPU for 20 docs).

  In practice: cross-encoder reranking is the single highest-impact quality
  improvement in a RAG pipeline after hybrid retrieval.

Model used:
  cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking (ideal for RAG Q&A)
  - MiniLM architecture: fast on CPU (~200ms for 20 docs)
  - Downloads once (~80MB), cached in ~/.cache/huggingface/
  - Completely free, runs locally
"""

import logging
import time
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# CrossEncoderReranker
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Reranks a list of candidate Documents using a cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker(top_n=5)
        top_docs = reranker.rerank("your query", candidate_docs)
    """

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        top_n: int = 5,
    ):
        """
        Args:
            model_name: HuggingFace cross-encoder model name
            top_n:      Number of documents to return after reranking
        """
        self.top_n = top_n
        logger.info(f"Loading cross-encoder: {model_name}")
        # CrossEncoder loads on first instantiation — subsequent calls use cache
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info("Cross-encoder loaded successfully")

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Scores each (query, document) pair and returns top_n by score.

        Args:
            query:     The user's search query
            documents: Candidate documents from HybridRetriever (up to 20)

        Returns:
            Top-N documents re-ranked by cross-encoder relevance score
        """
        if not documents:
            logger.warning("Reranker received empty document list")
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs in one batch — much faster than one-by-one
        t0 = time.time()
        scores = self.model.predict(pairs)  # returns numpy array of floats
        elapsed_ms = (time.time() - t0) * 1000

        logger.debug(
            f"Reranked {len(documents)} docs in {elapsed_ms:.0f}ms | "
            f"Top score: {max(scores):.3f}"
        )

        # Zip docs with their scores, sort descending, return top_n
        doc_score_pairs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_docs = [doc for doc, _ in doc_score_pairs[: self.top_n]]
        return top_docs

    def rerank_with_scores(
        self, query: str, documents: List[Document]
    ) -> List[tuple]:
        """
        Same as rerank() but also returns the scores.
        Useful for debugging and analysis.

        Returns:
            List of (Document, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        doc_score_pairs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return list(doc_score_pairs[: self.top_n])
