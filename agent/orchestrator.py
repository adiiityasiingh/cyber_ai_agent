"""
Agentic Orchestrator
====================
Implements a ReAct (Reasoning + Acting) loop using Google Gemini
via the Google AI Studio REST API (generativelanguage.googleapis.com).

Flow:
  1. User query arrives
  2. Gemini reasons about what tools to call
  3. Tools execute deterministically
  4. Results feed back into Gemini's context
  5. Loop until Gemini produces a final answer
  6. Full trace logged as JSON
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

from agent.tools import TOOLS, dispatch_tool

# ── Config ─────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MODEL          = "gemini-2.5-flash"
MAX_TOKENS     = 4096
MAX_ITERATIONS = 12
LOGS_DIR       = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a precise, expert analyst of the Cyber Ireland 2022 Cybersecurity Sector Report.

Your role is to answer questions about this document with ABSOLUTE FACTUAL ACCURACY.
You have access to tools that let you search the document, retrieve tables, and perform calculations.

CRITICAL RULES:
1. NEVER hallucinate figures. If you are not certain, use search_document or get_page to verify.
2. ALWAYS use the calculate or calculate_cagr tool for any arithmetic - do NOT compute math mentally.
3. For questions about specific numbers (jobs, percentages), always retrieve the source chunk and cite the page.
4. For table-based comparisons (regional data, firm types), always use search_tables first.
5. When you find a relevant passage, include the exact page number in your citation.
6. Structure your final answer with: (a) Direct Answer, (b) Source Citation with page number, (c) Working/Calculation if applicable.

Available tools:
- search_document: Full-text TF-IDF search over all document chunks
- search_tables: Search only within table content
- get_page: Retrieve complete page content by page number
- calculate: Safe arithmetic evaluator
- calculate_cagr: Dedicated CAGR calculator with full workings shown

Begin by planning your approach, then systematically use tools to gather evidence before concluding."""


# ══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMA CONVERSION  (Anthropic format -> Gemini format)
# ══════════════════════════════════════════════════════════════════════════════

def _anthropic_tools_to_gemini(tools):
    declarations = []
    for t in tools:
        declarations.append({
            "name":        t["name"],
            "description": t["description"],
            "parameters":  t["input_schema"],
        })
    return [{"functionDeclarations": declarations}]


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI API CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def call_gemini(contents, gemini_tools):
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set.\n"
            "Get your key at: https://aistudio.google.com/app/apikey\n"
            "Windows:   set GOOGLE_API_KEY=your-key-here\n"
            "Mac/Linux: export GOOGLE_API_KEY=your-key-here"
        )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={GOOGLE_API_KEY}"
    )

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents":         contents,
        "tools":            gemini_tools,
        "generationConfig": {
            "maxOutputTokens": MAX_TOKENS,
            "temperature":     0.1,
        },
    }

    resp = requests.post(url, json=payload, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {resp.status_code}: {resp.text[:600]}"
        )

    return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_gemini_response(response):
    finish_reason  = "OTHER"
    function_calls = []
    text_parts     = []

    candidates = response.get("candidates", [])
    if not candidates:
        return finish_reason, function_calls, text_parts

    candidate     = candidates[0]
    finish_reason = candidate.get("finishReason", "OTHER")
    parts         = candidate.get("content", {}).get("parts", [])

    for part in parts:
        if "text" in part:
            text_parts.append(part["text"])
        if "functionCall" in part:
            fc = part["functionCall"]
            function_calls.append({
                "name":    fc.get("name", ""),
                "args":    fc.get("args", {}),
                "call_id": fc.get("name", "") + "_" + str(int(time.time() * 1000)),
            })

    return finish_reason, function_calls, text_parts


# ══════════════════════════════════════════════════════════════════════════════
# REACT AGENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(query):
    session_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    trace    = []
    contents = [{"role": "user", "parts": [{"text": query}]}]

    gemini_tools = _anthropic_tools_to_gemini(TOOLS)

    print(f"\n[Agent:{session_id}] ══════════════════════════════════════")
    print(f"[Agent:{session_id}] Query: {query}")
    print(f"[Agent:{session_id}] Model: {MODEL}")
    print(f"[Agent:{session_id}] ══════════════════════════════════════")

    final_answer = ""
    iteration    = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n[Agent:{session_id}] -- Iteration {iteration} --")

        t0                                         = time.time()
        response                                   = call_gemini(contents, gemini_tools)
        call_ms                                    = round((time.time() - t0) * 1000)
        finish_reason, function_calls, text_parts  = _parse_gemini_response(response)

        step = {
            "iteration":     iteration,
            "finish_reason": finish_reason,
            "latency_ms":    call_ms,
            "thoughts":      [],
            "tool_calls":    [],
            "tool_results":  [],
            "final_text":    None,
        }

        for txt in text_parts:
            step["thoughts"].append(txt[:500] + ("..." if len(txt) > 500 else ""))
            print(f"[Agent:{session_id}] THINK: {txt[:200]}")

        for fc in function_calls:
            print(f"[Agent:{session_id}] TOOL: {fc['name']}({json.dumps(fc['args'])[:120]})")
            step["tool_calls"].append({
                "tool":    fc["name"],
                "inputs":  fc["args"],
                "call_id": fc["call_id"],
            })

        # Add model turn
        model_parts = []
        for txt in text_parts:
            model_parts.append({"text": txt})
        for fc in function_calls:
            model_parts.append({"functionCall": {"name": fc["name"], "args": fc["args"]}})
        if model_parts:
            contents.append({"role": "model", "parts": model_parts})

        # Execute tools
        if function_calls:
            function_response_parts = []

            for fc in function_calls:
                t_start = time.time()
                result  = dispatch_tool(fc["name"], fc["args"])
                t_ms    = round((time.time() - t_start) * 1000)

                result_preview = result[:300] + ("..." if len(result) > 300 else "")
                print(f"[Agent:{session_id}] RESULT ({t_ms}ms): {result_preview}")

                step["tool_results"].append({
                    "tool":           fc["name"],
                    "call_id":        fc["call_id"],
                    "result_preview": result_preview,
                    "latency_ms":     t_ms,
                })

                try:
                    result_obj = json.loads(result)
                except Exception:
                    result_obj = {"raw": result}

                function_response_parts.append({
                    "functionResponse": {
                        "name":     fc["name"],
                        "response": result_obj,
                    }
                })

            contents.append({"role": "user", "parts": function_response_parts})

        elif finish_reason == "STOP":
            final_answer = "\n".join(text_parts).strip()
            step["final_text"] = final_answer
            trace.append(step)
            break

        trace.append(step)

    duration = round(time.time() - start_time, 2)

    print(f"\n[Agent:{session_id}] DONE in {duration}s")
    print(f"[Agent:{session_id}] Answer: {final_answer[:300]}")

    result = {
        "session_id":   session_id,
        "query":        query,
        "answer":       final_answer,
        "trace":        trace,
        "iterations":   iteration,
        "duration_sec": duration,
        "model":        MODEL,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }

    log_path = LOGS_DIR / f"trace_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[Agent:{session_id}] Trace saved -> {log_path}")

    return result