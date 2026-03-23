"""
evaluate.py  —  RAGForge
========================
Verified: ragas==0.4.3

CRITICAL FACTS about ragas 0.4.x:
  1. ragas.metrics.collections metrics do NOT work with ragas.evaluate()
  2. They expose .ascore(**kwargs) directly — called per sample
  3. ascore() internally calls agenerate() which REQUIRES AsyncOpenAI, never OpenAI
  4. Each metric has different kwargs:
       Faithfulness              -> user_input, response, retrieved_contexts
       AnswerRelevancy           -> user_input, response
       ContextPrecisionWithRef   -> user_input, reference, retrieved_contexts
       ContextRecall             -> user_input, retrieved_contexts, reference

Run:
    python src/evaluate.py --mode single
    python src/evaluate.py --mode compare
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import List
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── RAGAS 0.4.x ──────────────────────────────────────────────────────────────
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
)
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings

# MUST be AsyncOpenAI — sync OpenAI will always raise TypeError in agenerate()
from openai import AsyncOpenAI

# ── Local modules ─────────────────────────────────────────────────────────────
from src.ingestion import load_vectorstore
from src.retriever import HybridRetriever, VectorlessRetriever
from src.reranker import CrossEncoderReranker
from src.pipeline import (
    RAGPipeline,
    init_pipeline,
    _load_chunks_from_vectorstore,
    get_llm,
    RAG_PROMPT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = "./tests/golden_dataset.json"
RESULTS_PATH        = "./tests/eval_results.json"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RAGAS LLM  (AsyncOpenAI → llm_factory)
# ─────────────────────────────────────────────────────────────────────────────

def get_ragas_llm():
    """
    Builds a RAGAS InstructorLLM using AsyncOpenAI.
    Both Groq and OpenRouter expose an OpenAI-compatible async API.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()

    if provider == "groq":
        api_key  = os.getenv("GROQ_API_KEY")
        model    = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        base_url = "https://api.groq.com/openai/v1"
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")

    elif provider == "openrouter":
        api_key  = os.getenv("OPENROUTER_API_KEY")
        model    = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
        base_url = "https://openrouter.ai/api/v1"
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. Valid: groq | openrouter"
        )

    logger.info(f"RAGAS evaluator: {provider} / {model}")

    # AsyncOpenAI is mandatory — metrics use agenerate() internally
    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return llm_factory(model=model, client=async_client)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RAGAS Embeddings  (local HuggingFace, free)
# ─────────────────────────────────────────────────────────────────────────────

def get_ragas_embeddings():
    logger.info("Loading embeddings: BAAI/bge-base-en-v1.5")
    return RagasHFEmbeddings(
        model="BAAI/bge-base-en-v1.5",
        device="cpu",
        normalize_embeddings=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Golden Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_golden_dataset(path: str = GOLDEN_DATASET_PATH) -> List[dict]:
    """
    Loads QA pairs from JSON.
    Format: [{"question": "...", "ground_truth": "..."}, ...]
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {path}\n"
            "Fill tests/golden_dataset.json with real QA pairs."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("golden_dataset.json is empty.")
    if "Replace this" in data[0].get("question", ""):
        raise ValueError(
            "golden_dataset.json still has placeholder text.\n"
            "Replace it with real questions from your documents."
        )

    logger.info(f"Loaded {len(data)} QA pairs")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Collect Pipeline Outputs
# ─────────────────────────────────────────────────────────────────────────────

def collect_pipeline_outputs(pipeline, golden_data: List[dict]) -> List[dict]:
    """Runs every question through the pipeline and collects results."""
    results = []
    for i, item in enumerate(golden_data, 1):
        q  = item["question"]
        gt = item["ground_truth"]
        logger.info(f"  [{i}/{len(golden_data)}] {q[:70]}...")
        try:
            out = pipeline.query(q)
            results.append({
                "question":     q,
                "answer":       out["answer"],
                "contexts":     out["contexts"],   # List[str]
                "ground_truth": gt,
            })
        except Exception as e:
            logger.warning(f"  Skipped '{q[:50]}': {e}")

    logger.info(f"Collected {len(results)} results")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Score with RAGAS  (async, per-sample, direct .ascore() calls)
# ─────────────────────────────────────────────────────────────────────────────

async def _score_all_async(
    results: List[dict],
    ragas_llm,
    ragas_emb,
) -> List[dict]:
    """
    Scores every sample using the four metrics.
    Each metric is called with its own specific kwargs — not a Sample object.
    """
    faith_metric  = Faithfulness(llm=ragas_llm)
    rel_metric    = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
    prec_metric   = ContextPrecisionWithReference(llm=ragas_llm)
    recall_metric = ContextRecall(llm=ragas_llm)

    scored = []
    for i, r in enumerate(results, 1):
        q        = r["question"]
        answer   = r["answer"]
        contexts = r["contexts"]        # List[str]
        ref      = r["ground_truth"]

        logger.info(f"  Scoring [{i}/{len(results)}] ...")

        f_res  = await faith_metric.ascore(
            user_input=q,
            response=answer,
            retrieved_contexts=contexts,
        )
        ar_res = await rel_metric.ascore(
            user_input=q,
            response=answer,
        )
        cp_res = await prec_metric.ascore(
            user_input=q,
            reference=ref,
            retrieved_contexts=contexts,
        )
        cr_res = await recall_metric.ascore(
            user_input=q,
            retrieved_contexts=contexts,
            reference=ref,
        )

        scored.append({
            "question":          q,
            "faithfulness":      float(f_res.value),
            "answer_relevancy":  float(ar_res.value),
            "context_precision": float(cp_res.value),
            "context_recall":    float(cr_res.value),
        })

        logger.info(
            f"    faith={scored[-1]['faithfulness']:.3f} | "
            f"relevancy={scored[-1]['answer_relevancy']:.3f} | "
            f"precision={scored[-1]['context_precision']:.3f} | "
            f"recall={scored[-1]['context_recall']:.3f}"
        )

    return scored


def run_ragas_evaluation(
    results: List[dict],
    ragas_llm,
    ragas_emb,
    config_name: str = "pipeline",
) -> dict:
    """Runs async scoring and returns averaged scores."""
    logger.info(
        f"Running RAGAS evaluation: {config_name} ({len(results)} samples)..."
    )
    scored = asyncio.run(_score_all_async(results, ragas_llm, ragas_emb))

    def avg(key):
        vals = [s[key] for s in scored if s.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    scores = {
        "config":            config_name,
        "faithfulness":      avg("faithfulness"),
        "answer_relevancy":  avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall":    avg("context_recall"),
        "num_samples":       len(scored),
        "timestamp":         datetime.now().isoformat(),
        "per_sample":        scored,
    }

    logger.info(
        f"  Results → "
        f"faithfulness={scores['faithfulness']} | "
        f"answer_relevancy={scores['answer_relevancy']} | "
        f"context_precision={scores['context_precision']} | "
        f"context_recall={scores['context_recall']}"
    )
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 6.  A/B/C Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(
    golden_data: List[dict],
    ragas_llm,
    ragas_emb,
) -> List[dict]:
    """
    Evaluates 4 configs — now includes vectorless (BM25-only) as baseline:

      A — Vectorless   (pure BM25, no embeddings, no vector DB)   ← NEW
      B — Dense-only   (ChromaDB top-5, no BM25, no reranking)
      C — Hybrid       (BM25 + Dense + RRF, top-5, no reranking)
      D — Hybrid+Rerank (full production pipeline)

    The 4-config comparison tells a complete story:
      "Pure BM25 is fast but misses semantics.
       Dense catches semantics but misses keywords.
       Hybrid fixes both. Reranking polishes the top results.
       Each layer adds measurable value — here are the RAGAS numbers."
    """
    vectorstore = load_vectorstore()
    chunks      = _load_chunks_from_vectorstore(vectorstore)
    llm         = get_llm()
    all_scores  = []

    # ── A: Vectorless — pure BM25, zero vector DB ─────────────────────────────
    logger.info("\n=== Config A: Vectorless (Pure BM25 — no embeddings) ===")

    class _VectorlessPipeline:
        def __init__(self):
            self.retriever = VectorlessRetriever(chunks, k=5)
            self.chain     = RAG_PROMPT | llm
        def query(self, question):
            docs = self.retriever.retrieve(question)
            ctx  = "\n\n---\n\n".join(d.page_content for d in docs)
            resp = self.chain.invoke({"context": ctx, "question": question})
            return {"answer": resp.content,
                    "contexts": [d.page_content for d in docs]}

    out_a = collect_pipeline_outputs(_VectorlessPipeline(), golden_data)
    all_scores.append(run_ragas_evaluation(out_a, ragas_llm, ragas_emb, "Vectorless-BM25"))

    # ── B: Dense-only ─────────────────────────────────────────────────────────
    logger.info("\n=== Config B: Dense-Only (ChromaDB) ===")

    class _DensePipeline:
        def __init__(self):
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            self.chain     = RAG_PROMPT | llm
        def query(self, question):
            docs = self.retriever.invoke(question)
            ctx  = "\n\n---\n\n".join(d.page_content for d in docs)
            resp = self.chain.invoke({"context": ctx, "question": question})
            return {"answer": resp.content,
                    "contexts": [d.page_content for d in docs]}

    out_b = collect_pipeline_outputs(_DensePipeline(), golden_data)
    all_scores.append(run_ragas_evaluation(out_b, ragas_llm, ragas_emb, "Dense-Only"))

    # ── C: Hybrid, no reranking ────────────────────────────────────────────────
    logger.info("\n=== Config C: Hybrid (BM25 + Dense + RRF, no reranking) ===")

    class _HybridPipeline:
        def __init__(self):
            self.retriever = HybridRetriever(vectorstore, chunks, k=10)
            self.chain     = RAG_PROMPT | llm
        def query(self, question):
            docs = self.retriever.retrieve(question)[:5]
            ctx  = "\n\n---\n\n".join(d.page_content for d in docs)
            resp = self.chain.invoke({"context": ctx, "question": question})
            return {"answer": resp.content,
                    "contexts": [d.page_content for d in docs]}

    out_c = collect_pipeline_outputs(_HybridPipeline(), golden_data)
    all_scores.append(run_ragas_evaluation(out_c, ragas_llm, ragas_emb, "Hybrid-NoRerank"))

    # ── D: Full pipeline — Hybrid + Cross-Encoder Reranking ───────────────────
    logger.info("\n=== Config D: Hybrid + Cross-Encoder Reranking (FULL) ===")
    full = RAGPipeline(
        retriever=HybridRetriever(vectorstore, chunks, k=20),
        reranker=CrossEncoderReranker(top_n=5),
        llm=llm,
    )
    out_d = collect_pipeline_outputs(full, golden_data)
    all_scores.append(run_ragas_evaluation(out_d, ragas_llm, ragas_emb, "Hybrid+Rerank"))

    return all_scores


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Scorecard Printer
# ─────────────────────────────────────────────────────────────────────────────

def print_scorecard(results: List[dict]):
    """
    Prints three views:
      1. Full 4-config scorecard table
      2. Embedding vs Vectorless head-to-head (the core comparison)
      3. Full pipeline delta vs vectorless baseline
    """
    metrics   = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    col_width = 20

    # ── 1. Full scorecard ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RAGAS EVALUATION SCORECARD — ALL CONFIGS")
    print("=" * 80)
    header = f"{'Metric':<26}" + "".join(f"{r['config']:<{col_width}}" for r in results)
    print(header)
    print("-" * 80)
    for metric in metrics:
        row = f"{metric:<26}"
        vals = [r.get(metric, 0.0) for r in results]
        best = max(vals)
        for v in vals:
            cell = f"{v:.4f}"
            if abs(v - best) < 0.0001:
                cell = cell + " ★"
            row += f"{cell:<{col_width}}"
        print(row)
    print("-" * 80)
    print("  ★ = best score for that metric")
    print("\nScores 0-1. Higher is better.\n")

    # ── 2. Embedding vs Vectorless head-to-head ──────────────────────────────
    vectorless = next((r for r in results if "Vectorless" in r["config"]), None)
    dense      = next((r for r in results if "Dense" in r["config"]), None)

    if vectorless and dense:
        print("=" * 60)
        print("  EMBEDDING vs VECTORLESS — HEAD TO HEAD")
        print("=" * 60)
        print(f"  {'Metric':<28} {'Vectorless (BM25)':<22} {'Dense (Embeddings)':<22} {'Winner'}")
        print("  " + "-" * 56)
        for metric in metrics:
            v_score = vectorless.get(metric, 0.0)
            d_score = dense.get(metric, 0.0)
            delta   = d_score - v_score
            if delta > 0.005:
                winner = "Embeddings ▲ +" + f"{delta:.4f}"
            elif delta < -0.005:
                winner = "Vectorless ▲ +" + f"{abs(delta):.4f}"
            else:
                winner = "Tie"
            print(f"  {metric:<28} {v_score:<22.4f} {d_score:<22.4f} {winner}")
        print()

        # Overall winner
        embed_wins = sum(
            1 for m in metrics
            if dense.get(m, 0.0) > vectorless.get(m, 0.0) + 0.005
        )
        vless_wins = sum(
            1 for m in metrics
            if vectorless.get(m, 0.0) > dense.get(m, 0.0) + 0.005
        )
        print(f"  Embeddings win: {embed_wins}/4 metrics")
        print(f"  Vectorless win: {vless_wins}/4 metrics")
        if embed_wins > vless_wins:
            print("  → Verdict: Embeddings justify the infrastructure cost on this dataset.")
        elif vless_wins > embed_wins:
            print("  → Verdict: BM25 competitive here — keyword-heavy domain favours sparse retrieval.")
        else:
            print("  → Verdict: Mixed results — domain has both keyword and semantic queries.")
        print()

    # ── 3. Full pipeline delta vs vectorless baseline ────────────────────────
    if len(results) >= 2:
        baseline = results[0]
        full     = results[-1]
        print(f"Delta: {full['config']} vs {baseline['config']} (baseline)")
        for metric in metrics:
            delta = full.get(metric, 0.0) - baseline.get(metric, 0.0)
            arrow = "▲" if delta >= 0 else "▼"
            print(f"  {metric:<34} {arrow} {delta:+.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["single", "compare"],
        default="single",
        help="single: full pipeline only | compare: A/B/C comparison",
    )
    args = parser.parse_args()

    ragas_llm   = get_ragas_llm()
    ragas_emb   = get_ragas_embeddings()
    golden_data = load_golden_dataset()

    if args.mode == "compare":
        print("\nRunning A/B/C/D comparison (~7-12 min for 30 QA pairs)...")
        results = run_comparison(golden_data, ragas_llm, ragas_emb)
    else:
        print("\nEvaluating full pipeline (Hybrid + Reranking)...")
        pipeline, _ = init_pipeline()
        outputs     = collect_pipeline_outputs(pipeline, golden_data)
        results     = [run_ragas_evaluation(
            outputs, ragas_llm, ragas_emb, "Hybrid+Rerank"
        )]

    print_scorecard(results)

    Path("./tests").mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {RESULTS_PATH}\n")