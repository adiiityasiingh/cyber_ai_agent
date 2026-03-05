# Cyber Ireland Intelligence Agent

An autonomous RAG system that ingests the **Cyber Ireland 2022 Cybersecurity Sector Report** and answers complex, multi-step queries using a Gemini-powered ReAct agent — with full citation, table parsing, and verified arithmetic.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Frontend  (frontend/index.html)             │
│          Served via Nginx / any static host              │
└─────────────────────────┬────────────────────────────────┘
                          │ POST /query
┌─────────────────────────▼────────────────────────────────┐
│              Flask Backend  (app.py)                     │
│              CORS-enabled, Gunicorn in prod              │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│         ReAct Agent Loop  (agent/orchestrator.py)        │
│  Gemini 2.0 Flash  ←──────────►  Tool Dispatcher        │
│                            (agent/tools.py)              │
│  Tools: search_document · search_tables                  │
│         get_page · calculate · calculate_cagr            │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│         SQLite Knowledge Base  (data/knowledge.db)       │
│     Text chunks + Table chunks + TF-IDF index            │
│     Built by ETL Pipeline  (etl/ingest.py)               │
└──────────────────────────────────────────────────────────┘
```

---

## Local Development

### Prerequisites
- Python 3.11+
- Google AI Studio API key → https://aistudio.google.com/app/apikey

### 1. Install dependencies

**Windows:**
```powershell
py -m pip install -r requirements.txt
```

**Mac / Linux:**
```bash
pip install -r requirements.txt
```

### 2. Place the PDF

Download the Cyber Ireland 2022 report and save it as:
```
data/cyberireland_2022.pdf
```

### 3. Run the ETL pipeline

**Windows:**
```powershell
py etl/ingest.py
```
Expected output:
```
[ETL] Total pages: 52
[ETL] Extracted 187 chunks total
[ETL] Stored 187 chunks — 43 tables
```

### 4. Set your API key

**Windows:**
```powershell
set GOOGLE_API_KEY=AIza_your_key_here
```

### 5. Start the backend

**Windows:**
```powershell
py app.py
```

**Mac / Linux:**
```bash
python app.py
```

Backend runs at `http://localhost:8000`

### 6. Open the frontend

Open `frontend/index.html` directly in your browser, **or** serve it:

```bash
# Python quick server
cd frontend
python -m http.server 3000
# → open http://localhost:3000
```

> **Note:** The frontend talks to `http://localhost:8000` by default. If your backend is elsewhere, edit the `API_BASE` constant at the top of `frontend/index.html`.

### 7. Run evaluation tests

```powershell
py run_tests.py      # Windows
python run_tests.py  # Mac/Linux
```

---

## API Reference

### `POST /query`
```json
// Request
{ "query": "What is the total number of jobs reported?" }

// Response
{
  "session_id":   "a3f91b2c",
  "query":        "...",
  "answer":       "...",
  "iterations":   4,
  "duration_sec": 21.3,
  "model":        "gemini-2.0-flash",
  "trace": [
    {
      "iteration":     1,
      "finish_reason": "tool_use",
      "latency_ms":    1842,
      "thoughts":      ["..."],
      "tool_calls":    [{"tool": "search_document", "inputs": {...}}],
      "tool_results":  [{"result_preview": "..."}]
    }
  ]
}
```

### `https://cyber-ai-agent.onrender.com/health` — liveness check
### `https://cyber-ai-agent.onrender.com/status` — knowledge base statistics

---

## Evaluation Results

| Test | Query | Result |
|---|---|---|
| Test 1: Verification | Total jobs reported + citation | Returns exact integer with page number |
| Test 2: Synthesis | Pure-Play South-West vs National Average | Extracts table data, computes difference |
| Test 3: Forecasting | CAGR 2022→2030 | Uses `calculate_cagr` tool: **11.74%** |

See `logs/evaluation_report_sample.json` for full agent traces.

---

## Architecture Justification

| Choice | Rationale |
|---|---|
| **pdfplumber** for ETL | Most accurate table extraction for dense sector reports; bbox-based cell detection handles merged cells |
| **TF-IDF + SQLite** | Zero external services, fully offline after ingestion; sufficient precision for a 50-page domain document |
| **Gemini 2.0 Flash** | Free via AI Studio, excellent function-calling support, fast latency |
| **Native ReAct loop** (no LangChain) | Full transparency into every reasoning step; simpler to debug and extend |
| **Dedicated math tools** | LLMs are unreliable at arithmetic; `calculate` and `calculate_cagr` are deterministic |
| **Markdown table serialisation** | Allows Gemini to read and reason over tables natively in its context window |

---

## Limitations & Scaling Path

| Limitation | Production Fix |
|---|---|
| TF-IDF recall drops on paraphrased queries | Swap to ChromaDB + `all-MiniLM-L6-v2` embeddings |
| SQLite not concurrent | Migrate to PostgreSQL + pgvector |
| No query caching | Add Redis cache keyed by query hash |
| Single PDF | Add `doc_id` namespace; build multi-document retrieval router |
| pdfplumber misses rotated/merged table cells | Add camelot-py fallback + Gemini Vision for complex layouts |
| No auth | Add API key middleware or OAuth |

---

## Project Structure

```
cyber-rag-agent/
├── app.py                     # Flask backend (CORS-enabled)
├── run_tests.py               # Evaluation test suite
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── .env.example
├── .gitignore
├── README.md
│
├── frontend/
│   └── index.html             # Self-contained UI (no build step)
│
├── agent/
│   ├── orchestrator.py        # Gemini ReAct loop
│   └── tools.py               # 5 deterministic tools
│
├── etl/
│   └── ingest.py              # PDF extraction + TF-IDF + SQLite
│
├── data/
│   ├── cyberireland_2022.pdf  # Place PDF here (not committed)
│   └── knowledge.db           # Generated by ETL
│
└── logs/
    ├── evaluation_report_sample.json
    └── trace_*.json            # Live traces
```