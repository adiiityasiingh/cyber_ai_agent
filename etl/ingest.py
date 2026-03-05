"""
ETL Pipeline: Cyber Ireland 2022 PDF Ingestion
==============================================
Extracts text + tables from PDF, stores in SQLite with TF-IDF index.
Handles complex table structures via pdfplumber's table extraction.
"""

import pdfplumber
import sqlite3
import json
import re
import math
import os
import sys
from pathlib import Path
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
DB_PATH    = DATA_DIR / "knowledge.db"
PDF_PATH   = DATA_DIR / "cyberireland_2022.pdf"

DATA_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PDF EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract every page as a structured chunk.
    Returns list of dicts: {page, type, content, raw_text}
    
    Strategy:
      • pdfplumber for accurate table detection (bbox-based cell merging)
      • Full page text captured alongside tables for context
      • Tables serialised to Markdown so the LLM can read them naturally
    """
    chunks = []
    print(f"[ETL] Opening PDF: {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"[ETL] Total pages: {total}")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            # ── raw text ──────────────────────────────────────────────────────
            raw_text = page.extract_text() or ""
            
            # ── tables ───────────────────────────────────────────────────────
            tables = page.extract_tables()
            table_md_blocks = []
            
            for t_idx, table in enumerate(tables):
                if not table:
                    continue
                md = _table_to_markdown(table)
                table_md_blocks.append(md)
                chunks.append({
                    "page":     page_num,
                    "type":     "table",
                    "content":  md,
                    "raw_text": md,
                    "table_idx": t_idx,
                })
            
            # ── text chunk (always add, even if also has tables) ──────────────
            if raw_text.strip():
                # Remove table noise from the text chunk when tables were found
                clean_text = raw_text.strip()
                chunks.append({
                    "page":      page_num,
                    "type":      "text",
                    "content":   clean_text,
                    "raw_text":  clean_text,
                    "table_idx": -1,
                })
            
            if page_num % 10 == 0:
                print(f"[ETL]   Processed page {page_num}/{total}")
    
    print(f"[ETL] Extracted {len(chunks)} chunks total")
    return chunks


def _table_to_markdown(table: list[list]) -> str:
    """Convert pdfplumber table (list of rows) to Markdown table string."""
    if not table:
        return ""
    
    # Clean cells
    cleaned = []
    for row in table:
        cleaned_row = []
        for cell in row:
            val = str(cell).strip() if cell is not None else ""
            val = val.replace("\n", " ").replace("|", "\\|")
            cleaned_row.append(val)
        cleaned.append(cleaned_row)
    
    if not cleaned:
        return ""
    
    # Determine column count
    col_count = max(len(r) for r in cleaned)
    
    # Pad all rows to same width
    padded = [r + [""] * (col_count - len(r)) for r in cleaned]
    
    # Build markdown
    lines = []
    header = padded[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * col_count) + "|")
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TF-IDF INDEX  (no external vector DB required)
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser, lowercase."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    # Remove very short tokens
    return [t for t in tokens if len(t) > 1]


STOPWORDS = set("""
a an the and or but in on at to for of with is are was were be been being
have has had do does did will would could should may might shall can
it its this that these those i we you he she they us them our your his her their
""".split())


def build_tfidf_index(chunks: list[dict]) -> tuple[list[dict], dict]:
    """
    Build a simple TF-IDF index over all chunks.
    Returns: (tfidf_docs, idf)
      - tfidf_docs: list of dicts with TF-IDF scores per chunk
      - idf: dict of IDF scores per term
    """
    N = len(chunks)
    
    # Term frequency per doc
    tf_docs = []
    for chunk in chunks:
        tokens = [t for t in _tokenize(chunk["content"]) if t not in STOPWORDS]
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        tf_docs.append({t: c/total for t, c in tf.items()})
    
    # Document frequency
    df = Counter()
    for tf in tf_docs:
        for term in tf:
            df[term] += 1
    
    # IDF
    idf = {term: math.log((N + 1) / (count + 1)) + 1 for term, count in df.items()}
    
    # TF-IDF
    tfidf_docs = []
    for tf in tf_docs:
        tfidf = {term: tf_val * idf.get(term, 1) for term, tf_val in tf.items()}
        tfidf_docs.append(tfidf)
    
    return tfidf_docs, idf


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SQLITE STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            page        INTEGER,
            type        TEXT,
            content     TEXT,
            raw_text    TEXT,
            table_idx   INTEGER,
            tfidf_json  TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_page ON chunks(page)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON chunks(type)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    return conn


def store_chunks(conn: sqlite3.Connection, chunks: list[dict], tfidf_docs: list[dict]):
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM meta")
    
    rows = []
    for i, (chunk, tfidf) in enumerate(zip(chunks, tfidf_docs)):
        rows.append((
            chunk["page"],
            chunk["type"],
            chunk["content"],
            chunk["raw_text"],
            chunk.get("table_idx", -1),
            json.dumps(tfidf),
        ))
    
    conn.executemany(
        "INSERT INTO chunks (page, type, content, raw_text, table_idx, tfidf_json) VALUES (?,?,?,?,?,?)",
        rows
    )
    
    conn.execute("INSERT INTO meta VALUES ('chunk_count', ?)", (str(len(chunks)),))
    conn.execute("INSERT INTO meta VALUES ('ingested_at', datetime('now'))")
    conn.commit()
    print(f"[ETL] Stored {len(chunks)} chunks in {DB_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  RETRIEVAL UTILITY  (used by agent at query-time)
# ══════════════════════════════════════════════════════════════════════════════

def cosine_sim(vec_a: dict, vec_b: dict) -> float:
    """Sparse cosine similarity between two TF-IDF dicts."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot    = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a  = math.sqrt(sum(v*v for v in vec_a.values()))
    mag_b  = math.sqrt(sum(v*v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve(query: str, db_path: Path, top_k: int = 8,
             filter_type: str | None = None) -> list[dict]:
    """
    TF-IDF retrieval: returns top_k chunks most similar to query.
    filter_type: 'table' | 'text' | None (both)
    """
    conn = sqlite3.connect(str(db_path))
    
    where = ""
    if filter_type:
        where = f"WHERE type = '{filter_type}'"
    
    rows = conn.execute(
        f"SELECT id, page, type, content, tfidf_json FROM chunks {where}"
    ).fetchall()
    conn.close()
    
    # Build query vector (no IDF weighting at query time for simplicity)
    q_tokens = [t for t in _tokenize(query) if t not in STOPWORDS]
    q_tf = Counter(q_tokens)
    q_total = sum(q_tf.values()) or 1
    q_vec = {t: c/q_total for t, c in q_tf.items()}
    
    scored = []
    for row_id, page, rtype, content, tfidf_json in rows:
        doc_vec = json.loads(tfidf_json)
        score = cosine_sim(q_vec, doc_vec)
        scored.append({
            "id":      row_id,
            "page":    page,
            "type":    rtype,
            "content": content,
            "score":   score,
        })
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_etl(pdf_path: Path = PDF_PATH):
    if not pdf_path.exists():
        print(f"[ETL] ERROR: PDF not found at {pdf_path}")
        print(f"[ETL] Please place the Cyber Ireland 2022 PDF at: {pdf_path}")
        sys.exit(1)
    
    print("[ETL] ═══ Starting ETL Pipeline ═══")
    
    # 1. Extract
    chunks = extract_pdf(pdf_path)
    
    # 2. Build TF-IDF index
    print("[ETL] Building TF-IDF index …")
    tfidf_docs, idf = build_tfidf_index(chunks)
    
    # 3. Store
    conn = init_db(DB_PATH)
    store_chunks(conn, chunks, tfidf_docs)
    conn.close()
    
    print("[ETL] ═══ ETL Pipeline Complete ═══")
    print(f"[ETL] Database: {DB_PATH}")
    
    # Quick sanity
    conn2 = sqlite3.connect(str(DB_PATH))
    n = conn2.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    tables = conn2.execute("SELECT COUNT(*) FROM chunks WHERE type='table'").fetchone()[0]
    conn2.close()
    print(f"[ETL] Chunks: {n} total / {tables} tables")


if __name__ == "__main__":
    pdf_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else PDF_PATH
    run_etl(pdf_arg)