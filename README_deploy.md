# RAGForge — Deployment Guide

## Architecture

- **Backend:** Hugging Face Spaces (Docker) — free forever
- **Frontend:** GitHub Pages — free forever
- **Total cost:** $0

---

## Step 1 — Prepare your repo

```bash
# From your RAGForge project root
git init
git add .
git commit -m "initial commit"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/ragforge
git push -u origin main
```

---

## Step 2 — Deploy Backend to Hugging Face Spaces

### 2a. Create the Space

1. Go to huggingface.co → New Space
2. Space name: `ragforge`
3. SDK: **Docker**
4. Hardware: **CPU basic** (free)
5. Visibility: Public

### 2b. Add your API key as a Secret

- Space Settings → Variables and Secrets → New Secret
- Name: `GROQ_API_KEY` → paste your key  
  OR
- Name: `OPENROUTER_API_KEY` → paste your key
- Name: `LLM_PROVIDER` → `groq` or `openrouter`

### 2c. Push your code to the Space

```bash
# Add HF remote
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/ragforge

# Push
git push space main
```

HF Spaces will detect the Dockerfile, build it, and start the server.
Build takes ~5-8 minutes on first push.

Your API will be live at:
`https://YOUR_HF_USERNAME-ragforge.hf.space`

---

## Step 3 — Deploy Frontend to GitHub Pages

### 3a. Update API_BASE in both HTML files

Open `frontend/index.html` and `frontend/dashboard.html`.
Change this line in both files:

```js
const API = "http://localhost:8000";
```

To:

```js
const API = "https://YOUR_HF_USERNAME-ragforge.hf.space";
```

### 3b. Move frontend to /docs folder

```bash
mkdir docs
cp frontend/index.html docs/
cp frontend/dashboard.html docs/
cp tests/eval_results.json docs/   # dashboard reads this
git add docs/
git commit -m "add frontend to docs for GitHub Pages"
git push origin main
```

### 3c. Enable GitHub Pages

- GitHub repo → Settings → Pages
- Source: Deploy from branch → `main` → `/docs`
- Save

Your frontend will be live at:
`https://YOUR_USERNAME.github.io/ragforge`

---

## Step 4 — Fix CORS (already done in api.py)

The CORS middleware is already added in api.py:

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

No changes needed.

---

## Step 5 — Verify everything works

1. Open your GitHub Pages URL
2. Go to Chat — ask a question
3. Go to Metrics — check latency dashboard
4. Place eval_results.json in docs/ for RAGAS scorecard

### Data panel (new)

Visitors can now browse the contents of the `data/` folder from the left sidebar. Click the **Data** menu item to open a documents panel that lists files in `data/` and shows a short preview for text files (and an Open link/viewer for PDFs). To make documents available to visitors, add them to the project's `data/` folder before deploying.

---

## Troubleshooting

**Build fails on HF Spaces:**

- Check Space logs (Spaces → Logs tab)
- Most common issue: missing PDF in data/ → add at least one PDF

**Frontend can't reach backend:**

- Make sure API_BASE is updated in both HTML files
- Check HF Space is running (green dot in Space header)
- Check browser console for CORS errors

**Slow first response:**

- Normal — cross-encoder model loads on first request (~10s)
- Subsequent requests are fast

---

## Resume line

```
RAGForge — Production RAG Pipeline | Live Demo: https://YOUR_USERNAME.github.io/ragforge
LangChain · ChromaDB · RAGAS · FastAPI · Docker · HF Spaces
```
