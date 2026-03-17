"""
ingestion.py
============
Handles: PDF loading → semantic chunking → HuggingFace embeddings → ChromaDB persistence.

Run standalone:
    python src/ingestion.py --data_dir ./data --reset

Imports used by pipeline.py:
    from src.ingestion import build_vectorstore, load_vectorstore, get_embeddings
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── LangChain imports (all up-to-date for 2025) ──────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL   = "BAAI/bge-base-en-v1.5"   # Free, local, beats ada-002 on MTEB
CHROMA_PATH   = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
COLLECTION    = "rag_collection"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFaceEmbeddings instance backed by BAAI/bge-base-en-v1.5.
    Model is downloaded once and cached in ~/.cache/huggingface/hub.
    """
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},          # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},  # required for cosine similarity
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Document Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(data_dir: str) -> list:
    """
    Loads all PDFs and .txt files from data_dir.
    Returns a flat list of LangChain Document objects.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents = []

    # Load PDFs
    pdf_files = list(data_path.glob("**/*.pdf"))
    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF file(s)")
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"  Loaded: {pdf_path.name} ({len(docs)} pages)")
            except Exception as e:
                logger.warning(f"  Failed to load {pdf_path.name}: {e}")

    # Load plain text files
    txt_files = list(data_path.glob("**/*.txt"))
    if txt_files:
        logger.info(f"Found {len(txt_files)} .txt file(s)")
        for txt_path in txt_files:
            try:
                loader = TextLoader(str(txt_path), encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"  Loaded: {txt_path.name}")
            except Exception as e:
                logger.warning(f"  Failed to load {txt_path.name}: {e}")

    if not documents:
        raise ValueError(
            f"No documents found in {data_dir}. "
            "Add PDF or .txt files to the data/ folder."
        )

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(documents: list) -> list:
    """
    Splits documents into chunks using RecursiveCharacterTextSplitter.

    Why these settings:
    - chunk_size=512: Sweet spot for bge embeddings (trained on ~512 token passages)
    - chunk_overlap=64: Prevents context from being cut at chunk boundaries
    - separators: Prioritises paragraph > sentence > word breaks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    # Filter out very short chunks that add noise
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
    logger.info(f"Created {len(chunks)} chunks (filtered short ones)")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ChromaDB Build & Load
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorstore(chunks: list, reset: bool = False) -> Chroma:
    """
    Embeds chunks and persists them to ChromaDB on disk.

    Args:
        chunks: List of LangChain Document objects
        reset:  If True, deletes existing collection before ingesting

    Returns:
        Chroma vectorstore instance
    """
    embeddings = get_embeddings()

    if reset and Path(CHROMA_PATH).exists():
        import shutil
        shutil.rmtree(CHROMA_PATH)
        logger.info(f"Deleted existing ChromaDB at {CHROMA_PATH}")

    logger.info(f"Building ChromaDB at {CHROMA_PATH} ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION,
    )
    logger.info(f"ChromaDB built. Total vectors: {vectorstore._collection.count()}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Loads an existing ChromaDB from disk (no re-embedding).
    Call this after build_vectorstore() has been run at least once.
    """
    if not Path(CHROMA_PATH).exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_PATH}. "
            "Run ingestion.py first: python src/ingestion.py --data_dir ./data"
        )
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )
    count = vectorstore._collection.count()
    logger.info(f"Loaded ChromaDB from {CHROMA_PATH}. Vectors: {count}")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--data_dir", default="./data", help="Folder containing PDFs/txts")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild ChromaDB")
    args = parser.parse_args()

    print("\n=== RAG Ingestion Pipeline ===\n")
    docs   = load_documents(args.data_dir)
    chunks = chunk_documents(docs)
    vs     = build_vectorstore(chunks, reset=args.reset)

    # Quick smoke test
    print("\n--- Smoke Test (top 2 results for 'test') ---")
    results = vs.similarity_search("test", k=2)
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", "unknown")
        preview = r.page_content[:150].replace("\n", " ")
        print(f"  [{i}] source={source}\n      {preview}...\n")

    print("✓ Ingestion complete. Run pipeline.py next.\n")
