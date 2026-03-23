---
title: RAGForge
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# RAGForge — Production RAG Pipeline

Hybrid retrieval (BM25 + ChromaDB) + Cross-Encoder Reranking + Conversational Memory.

## API Endpoints

- `POST /query` — Ask a question
- `GET /metrics` — Latency + token usage dashboard
- `GET /health` — Health check
- `GET /docs` — Interactive API docs

## Stack

LangChain · ChromaDB · RAGAS · FastAPI · Groq · HuggingFace Embeddings
