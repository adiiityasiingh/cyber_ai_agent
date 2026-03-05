"""
Flask Backend  –  Production-ready with CORS
=============================================
Endpoints:
  POST /query   – Run the agent on a user question
  GET  /health  – Liveness check
  GET  /status  – DB stats
"""

import json
import os
import sqlite3
import sys
from pathlib import Path
from functools import wraps

from flask import Flask, jsonify, request, Response

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.orchestrator import run_agent
from etl.ingest import DB_PATH

app = Flask(__name__)

# ── CORS ───────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = ALLOWED_ORIGINS
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

app.after_request(add_cors)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return Response(status=204,
                        headers={"Access-Control-Allow-Origin":  ALLOWED_ORIGINS,
                                 "Access-Control-Allow-Headers": "Content-Type",
                                 "Access-Control-Allow-Methods": "GET, POST, OPTIONS"})


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/query")
def query():
    body       = request.get_json(silent=True) or {}
    user_query = (body.get("query") or body.get("question") or "").strip()

    if not user_query:
        return jsonify({"error": "Missing 'query' field in request body"}), 400

    if not DB_PATH.exists():
        return jsonify({"error": "Knowledge base not initialised. Run etl/ingest.py first."}), 503

    try:
        result = run_agent(user_query)
        return jsonify(result), 200
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Unexpected error: {exc}"}), 500


@app.get("/")
def root():
    db_ready = DB_PATH.exists()
    return jsonify({
        "service":   "Cyber Ireland Intelligence Agent",
        "status":    "online",
        "version":   "1.0.0",
        "model":     "gemini-2.0-flash",
        "kb_status": "ready" if db_ready else "not_initialised",
        "endpoints": {
            "query":  "POST /query",
            "health": "GET  /health",
            "status": "GET  /status"
        },
        "github": "https://github.com/adiiityasiingh/cyber_ai_agent",
        "live_app": "https://cyber-rag-agent.onrender.com/"
    }), 200


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "cyber-rag-agent"}), 200


@app.get("/status")
def status():
    if not DB_PATH.exists():
        return jsonify({"status": "not_initialised"}), 200

    conn     = sqlite3.connect(str(DB_PATH))
    total    = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    tables   = conn.execute("SELECT COUNT(*) FROM chunks WHERE type='table'").fetchone()[0]
    pages    = conn.execute("SELECT COUNT(DISTINCT page) FROM chunks").fetchone()[0]
    ingested = conn.execute("SELECT value FROM meta WHERE key='ingested_at'").fetchone()
    conn.close()

    return jsonify({
        "status":       "ready",
        "total_chunks": total,
        "table_chunks": tables,
        "text_chunks":  total - tables,
        "unique_pages": pages,
        "ingested_at":  ingested[0] if ingested else "unknown",
    }), 200


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[Server] Starting on http://0.0.0.0:{port}")
    print(f"[Server] Google API Key: {'SET' if os.environ.get('GOOGLE_API_KEY') else 'NOT SET – export GOOGLE_API_KEY'}")
    app.run(host="0.0.0.0", port=port, debug=False)