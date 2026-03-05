"""
Agent Tools
===========
Discrete, deterministic tool functions the LLM agent can call.
Each tool has a strict JSON schema so the LLM can invoke them reliably.

Tools:
  1. search_document   – TF-IDF retrieval over ingested chunks
  2. get_page          – Fetch all chunks from a specific page
  3. calculate         – Safe math evaluator (fixes LLM arithmetic unreliability)
  4. calculate_cagr    – Dedicated CAGR formula tool
  5. search_tables     – Targeted table-only retrieval
"""

import math
import json
import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "knowledge.db"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS  (sent to Claude as tools list)
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "search_document",
        "description": (
            "Search the Cyber Ireland 2022 report for text passages relevant to a query. "
            "Returns the top matching chunks with page numbers and exact content. "
            "Use this to find specific facts, statistics, or narrative content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query – use specific keywords from the domain (e.g. 'total jobs employed', 'Pure-Play firms South-West')."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 6, max 12).",
                    "default": 6
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_tables",
        "description": (
            "Search specifically within TABLE content of the Cyber Ireland 2022 report. "
            "Use this when you need numeric data from regional tables, breakdowns by company type, "
            "or structured comparative data. Returns table chunks with page numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing the table data you need (e.g. 'South-West Pure-Play concentration region')."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of table results to return (default 6).",
                    "default": 6
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_page",
        "description": (
            "Retrieve ALL content (text and tables) from a specific page number of the report. "
            "Use this when you know the page number and want to verify context or get complete page data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "page_number": {
                    "type": "integer",
                    "description": "The 1-indexed page number to retrieve."
                }
            },
            "required": ["page_number"]
        }
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a safe mathematical expression. "
            "Use this for ALL arithmetic – do NOT attempt mental math. "
            "Supports: +, -, *, /, **, sqrt(), log(), round(), abs(). "
            "Example: '(2030 - 2022)' or '(7200 / 6500) ** (1/8) - 1'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python-compatible math expression string. No imports needed."
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "calculate_cagr",
        "description": (
            "Calculate Compound Annual Growth Rate (CAGR). "
            "Formula: CAGR = (end_value / start_value)^(1/years) - 1. "
            "Returns the CAGR as a percentage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_value": {
                    "type": "number",
                    "description": "The baseline / starting value (e.g. jobs in 2022)."
                },
                "end_value": {
                    "type": "number",
                    "description": "The target / ending value (e.g. jobs target in 2030)."
                },
                "years": {
                    "type": "number",
                    "description": "Number of years between start and end."
                }
            },
            "required": ["start_value", "end_value", "years"]
        }
    }
]


# ══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _tfidf_retrieve(query: str, top_k: int, filter_type=None) -> list[dict]:
    """Internal TF-IDF retrieval (duplicated here to keep tools self-contained)."""
    STOPWORDS = set("""
    a an the and or but in on at to for of with is are was were be been being
    have has had do does did will would could should may might shall can
    it its this that these those i we you he she they us them our your his her their
    """.split())
    
    def tokenize(text):
        return [t for t in re.findall(r"[a-z0-9]+", text.lower())
                if len(t) > 1 and t not in STOPWORDS]
    
    from collections import Counter
    
    conn = sqlite3.connect(str(DB_PATH))
    where = f"WHERE type='{filter_type}'" if filter_type else ""
    rows = conn.execute(
        f"SELECT id, page, type, content, tfidf_json FROM chunks {where}"
    ).fetchall()
    conn.close()
    
    q_tokens = tokenize(query)
    q_tf = Counter(q_tokens)
    q_total = sum(q_tf.values()) or 1
    q_vec = {t: c/q_total for t, c in q_tf.items()}
    
    def cosine(a, b):
        common = set(a) & set(b)
        if not common: return 0.0
        dot = sum(a[t]*b[t] for t in common)
        ma = math.sqrt(sum(v*v for v in a.values()))
        mb = math.sqrt(sum(v*v for v in b.values()))
        return dot/(ma*mb) if ma and mb else 0.0
    
    scored = []
    for row_id, page, rtype, content, tfidf_json in rows:
        doc_vec = json.loads(tfidf_json)
        score = cosine(q_vec, doc_vec)
        scored.append({"id": row_id, "page": page, "type": rtype,
                        "content": content, "score": score})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def tool_search_document(query: str, top_k: int = 6) -> dict:
    top_k = min(int(top_k), 12)
    results = _tfidf_retrieve(query, top_k)
    return {
        "query":   query,
        "results": [
            {"page": r["page"], "type": r["type"],
             "content": r["content"], "relevance_score": round(r["score"], 4)}
            for r in results
        ]
    }


def tool_search_tables(query: str, top_k: int = 6) -> dict:
    top_k = min(int(top_k), 12)
    results = _tfidf_retrieve(query, top_k, filter_type="table")
    return {
        "query":   query,
        "results": [
            {"page": r["page"], "type": r["type"],
             "content": r["content"], "relevance_score": round(r["score"], 4)}
            for r in results
        ]
    }


def tool_get_page(page_number: int) -> dict:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT page, type, content FROM chunks WHERE page=? ORDER BY type DESC",
        (int(page_number),)
    ).fetchall()
    conn.close()
    
    if not rows:
        return {"page": page_number, "found": False, "content": []}
    
    return {
        "page":    page_number,
        "found":   True,
        "content": [{"type": r[1], "text": r[2]} for r in rows]
    }


def tool_calculate(expression: str) -> dict:
    """Safe math evaluator – only allows numeric operations."""
    # Whitelist: digits, operators, parens, spaces, and safe function names
    safe_names = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "log2": math.log2, "exp": math.exp, "abs": abs,
        "round": round, "pow": pow, "pi": math.pi, "e": math.e,
    }
    # Strip anything that isn't safe
    cleaned = expression.strip()
    
    # Guard: only allow alphanumeric, operators, parens, dots, commas, spaces
    if re.search(r"[^0-9a-zA-Z_+\-*/()., ^%]", cleaned):
        return {"error": f"Unsafe expression: {cleaned}"}
    
    # Replace ^ with ** for Python
    cleaned = cleaned.replace("^", "**")
    
    try:
        result = eval(cleaned, {"__builtins__": {}}, safe_names)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc), "expression": expression}


def tool_calculate_cagr(start_value: float, end_value: float, years: float) -> dict:
    """Calculate CAGR with full workings."""
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return {"error": "All values must be positive numbers."}
    
    cagr = (end_value / start_value) ** (1 / years) - 1
    cagr_pct = round(cagr * 100, 4)
    
    return {
        "start_value":  start_value,
        "end_value":    end_value,
        "years":        years,
        "cagr_decimal": round(cagr, 6),
        "cagr_percent": cagr_pct,
        "formula":      f"({end_value} / {start_value})^(1/{years}) - 1 = {cagr_pct}%",
        "interpretation": (
            f"To grow from {start_value:,.0f} to {end_value:,.0f} "
            f"over {years:.0f} years requires a CAGR of {cagr_pct}%."
        )
    }


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

TOOL_MAP = {
    "search_document":  tool_search_document,
    "search_tables":    tool_search_tables,
    "get_page":         tool_get_page,
    "calculate":        tool_calculate,
    "calculate_cagr":   tool_calculate_cagr,
}


def dispatch_tool(name: str, inputs: dict) -> str:
    """Call the named tool and return its result as a JSON string."""
    if name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = TOOL_MAP[name](**inputs)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": name, "inputs": inputs})