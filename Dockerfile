# Dockerfile — RAGForge Backend
# Deploy to: Hugging Face Spaces (free, 2vCPU / 16GB RAM)
# Port: 7860 (HF Spaces default — do NOT change this)

FROM python:3.11-slim

# System deps for sentence-transformers + chromadb
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker layer cache — reinstall only if reqs change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/

# Pre-build ChromaDB at image build time so first request isn't slow
# This runs ingestion during docker build — vectors baked into the image
RUN python src/ingestion.py --data_dir ./data || echo "Ingestion skipped — add PDFs to data/ first"

# HF Spaces requires port 7860
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
