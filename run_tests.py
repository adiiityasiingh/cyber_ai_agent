"""
Evaluation Test Runner
=======================
Executes the three mandatory test scenarios against the live agent
and saves detailed JSON traces to the /logs directory.

Usage:
  python run_tests.py

Requires:
  - GOOGLE_API_KEY set in environment
  - ETL pipeline already run (data/knowledge.db exists)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.orchestrator import run_agent

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# TEST SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

TEST_QUERIES = [
    {
        "id":          "test_1_verification",
        "label":       "Test 1: Verification Challenge",
        "query":       "What is the total number of jobs reported, and where exactly is this stated?",
        "description": (
            "Must return exact integer, page number, and verifiable citation. "
            "Hallucinations or near-matches fail."
        ),
        "expected_behaviour": [
            "Retrieves specific job count as an integer",
            "Cites exact page number",
            "Includes verbatim or near-verbatim passage from document",
        ],
    },
    {
        "id":          "test_2_synthesis",
        "label":       "Test 2: Data Synthesis Challenge",
        "query":       "Compare the concentration of 'Pure-Play' cybersecurity firms in the South-West against the National Average.",
        "description": (
            "Must navigate regional tables, extract comparative metrics, "
            "and synthesise a mathematically accurate response."
        ),
        "expected_behaviour": [
            "Uses search_tables tool to find regional breakdown",
            "Extracts South-West Pure-Play percentage",
            "Extracts National Average Pure-Play percentage",
            "Produces accurate comparison with numbers",
        ],
    },
    {
        "id":          "test_3_forecasting",
        "label":       "Test 3: Forecasting Challenge",
        "query":       (
            "Based on our 2022 baseline and the stated 2030 job target, "
            "what is the required compound annual growth rate (CAGR) to hit that goal?"
        ),
        "description": (
            "Must find baseline and target, recognise need for CAGR calculation, "
            "and use the calculate_cagr tool."
        ),
        "expected_behaviour": [
            "Searches document for 2022 job baseline",
            "Searches document for 2030 job target",
            "Calls calculate_cagr tool with correct values",
            "Reports CAGR as a percentage with workings",
        ],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    print("\n" + "═" * 60)
    print("  CYBER IRELAND RAG AGENT – EVALUATION TEST SUITE")
    print("═" * 60)
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n❌ ERROR: GOOGLE_API_KEY is not set.")
        sys.exit(1)
    
    all_results = []
    suite_start = time.time()
    
    for i, test in enumerate(TEST_QUERIES, start=1):
        print(f"\n{'─'*60}")
        print(f"  {test['label']}")
        print(f"{'─'*60}")
        print(f"  Query:  {test['query']}")
        print(f"  Expect: {test['description']}")
        print()
        
        try:
            result = run_agent(test["query"])
            result["test_meta"] = {
                "test_id":    test["id"],
                "test_label": test["label"],
                "expected":   test["expected_behaviour"],
            }
            all_results.append({"status": "success", **result})
            
            print(f"\n  ✅ ANSWER:\n  {result['answer'][:600]}")
            print(f"  ⏱  {result['duration_sec']}s | {result['iterations']} iterations")
            
            # Check which tools were used
            tools_used = set()
            for step in result.get("trace", []):
                for tc in step.get("tool_calls", []):
                    tools_used.add(tc["tool"])
            print(f"  🔧 Tools used: {', '.join(sorted(tools_used)) or 'none'}")
            
        except Exception as exc:
            print(f"\n  ❌ FAILED: {exc}")
            all_results.append({
                "status": "error",
                "test_meta": test,
                "error": str(exc),
            })
        
        # Small pause between tests to respect rate limits
        if i < len(TEST_QUERIES):
            time.sleep(2)
    
    suite_duration = round(time.time() - suite_start, 2)
    
    # ── Save combined report ──────────────────────────────────────────────────
    report = {
        "suite":          "Cyber Ireland RAG Agent – Evaluation",
        "run_at":         datetime.utcnow().isoformat() + "Z",
        "total_duration": suite_duration,
        "tests":          all_results,
    }
    
    report_path = LOGS_DIR / f"evaluation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    
    print(f"\n{'═'*60}")
    print(f"  SUITE COMPLETE in {suite_duration}s")
    print(f"  Passed: {sum(1 for r in all_results if r['status']=='success')}/{len(all_results)}")
    print(f"  Report: {report_path}")
    print("═" * 60)
    
    return report


if __name__ == "__main__":
    run_all_tests()