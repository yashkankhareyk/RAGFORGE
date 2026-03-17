"""
retriever.py
============
Implements:
  1. BM25Retriever  — keyword-based sparse retrieval (rank-bm25)
  2. HybridRetriever — fuses dense (ChromaDB) + sparse (BM25) via Reciprocal Rank Fusion

Why hybrid?
  - Dense retrieval catches semantic similarity but misses exact keywords.
  - BM25 catches exact terms but misses paraphrases.
  - RRF combines both without needing score normalisation.

Interview talking point:
  "BM25 improved context_recall by X% because keyword-heavy queries
   (names, IDs, jargon) were being missed by dense retrieval alone."
"""

import logging
from collections import defaultdict
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BM25 Sparse Retriever
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Lightweight BM25 retriever built on top of rank-bm25.
    Must be initialised with the same chunk list used in ChromaDB so
    retrieved Document objects match exactly.
    """

    def __init__(self, chunks: List[Document], k: int = 20):
        """
        Args:
            chunks: List of LangChain Document objects (same as ingested into Chroma)
            k:      Number of top results to return
        """
        self.chunks = chunks
        self.k = k

        # Tokenise by whitespace and lowercase — simple but effective
        corpus = [self._tokenise(doc.page_content) for doc in chunks]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25Retriever built. Corpus size: {len(corpus)} chunks")

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Lower-case whitespace tokenisation."""
        return text.lower().split()

    def retrieve(self, query: str) -> List[Document]:
        """Returns top-k Documents ranked by BM25 score."""
        tokens = self._tokenise(query)
        scores = self.bm25.get_scores(tokens)

        # Sort indices by score descending, take top k
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:self.k]

        return [self.chunks[i] for i in top_indices]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = 60,
    top_n: int = 20,
) -> List[Document]:
    """
    Merges multiple ranked document lists using Reciprocal Rank Fusion (RRF).

    Formula: score(doc) = Σ  1 / (k + rank)
             where rank starts at 1.

    Why k=60?
        Prevents the top-ranked document from dominating too much.
        Standard default from the original RRF paper (Cormack et al. 2009).
        Lower k → more weight to rank-1 documents.

    Args:
        result_lists: List of ranked Document lists (e.g. [dense_results, sparse_results])
        k:            Smoothing constant (default 60)
        top_n:        Number of final results to return

    Returns:
        Fused and re-ranked list of Documents
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            # Use first 200 chars as a stable key (avoids issues with metadata)
            doc_id = doc.page_content[:200].strip()
            scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

    # Sort by fused score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in ranked[:top_n]]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HybridRetriever
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines ChromaDB dense retrieval + BM25 sparse retrieval via RRF.

    Usage:
        retriever = HybridRetriever(vectorstore, chunks, k=20)
        docs = retriever.retrieve("your query here")
        # Returns top-20 candidates (before reranking)
    """

    def __init__(
        self,
        vectorstore: Chroma,
        chunks: List[Document],
        k: int = 20,
    ):
        """
        Args:
            vectorstore: Loaded Chroma vectorstore instance
            chunks:      Same chunk list used to build the vectorstore
            k:           Number of candidates to fetch from each retriever
        """
        self.k = k
        self.dense_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        self.sparse_retriever = BM25Retriever(chunks, k=k)
        logger.info(f"HybridRetriever ready (dense + sparse, k={k})")

    def retrieve(self, query: str) -> List[Document]:
        """
        Runs dense + sparse retrieval and fuses results with RRF.

        Returns:
            Up to 20 re-ranked Document objects
        """
        # Dense retrieval (semantic similarity via ChromaDB)
        dense_docs = self.dense_retriever.invoke(query)

        # Sparse retrieval (BM25 keyword matching)
        sparse_docs = self.sparse_retriever.retrieve(query)

        logger.debug(
            f"Query='{query[:60]}' | "
            f"Dense={len(dense_docs)} | Sparse={len(sparse_docs)}"
        )

        # Fuse with RRF → top 20 candidates for the reranker
        fused = reciprocal_rank_fusion(
            [dense_docs, sparse_docs],
            k=60,
            top_n=20,
        )
        return fused
