#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

from app.main import app


STRATEGIES = {
    "naive": {"rag": "/api/v1/rag/naive", "chat": "/api/v1/chat"},
    "reranked": {"rag": "/api/v1/rag/reranked", "chat": "/api/v1/chat/reranked"},
    "hyde": {"rag": "/api/v1/rag/hyde", "chat": "/api/v1/chat/hyde"},
    "hyde_reranked": {"rag": "/api/v1/rag/hyde_reranked", "chat": "/api/v1/chat/hyde_reranked"},
}


@dataclass
class CaseResult:
    case_id: str
    strategy: str
    guardrail_blocked: bool
    retrieval_hit: bool
    retrieval_top1_guide: str | None
    runtime_ms: float
    answer_present: bool
    has_sources_line: bool
    has_safety_word: bool


def _has_sources_line(text: str) -> bool:
    lowered = text.lower()
    return "\nsources:" in lowered or lowered.startswith("sources:")


def _has_safety_word(text: str) -> bool:
    lowered = text.lower()
    for kw in ("safety", "unplug", "turn off", "shut off", "disconnect power"):
        if kw in lowered:
            return True
    return False


def run_case(client: TestClient, case: dict[str, Any], strategy: str) -> CaseResult:
    endpoints = STRATEGIES[strategy]
    payload = {
        "query": case["query"],
        "appliance_category": case.get("appliance_category"),
        "top_k": 5,
    }

    start = time.perf_counter()
    rag = client.post(endpoints["rag"], json=payload).json()
    chat = client.post(endpoints["chat"], json=payload).json()
    runtime_ms = (time.perf_counter() - start) * 1000.0

    guardrail_blocked = bool(chat.get("guardrail_blocked") or rag.get("guardrail_blocked"))
    expected_guide = case.get("expected_guide_title")
    results = rag.get("results") or []
    top1_guide = results[0].get("guide_title") if results else None
    retrieval_hit = False
    if expected_guide:
        retrieval_hit = any((r.get("guide_title") == expected_guide) for r in results[:3])

    answer_text = chat.get("answer") or ""
    answer_present = bool(answer_text.strip())
    has_sources_line = _has_sources_line(answer_text) if answer_present else False
    has_safety_word = _has_safety_word(answer_text) if answer_present else False

    return CaseResult(
        case_id=str(case["id"]),
        strategy=strategy,
        guardrail_blocked=guardrail_blocked,
        retrieval_hit=retrieval_hit,
        retrieval_top1_guide=top1_guide,
        runtime_ms=runtime_ms,
        answer_present=answer_present,
        has_sources_line=has_sources_line,
        has_safety_word=has_safety_word,
    )


def _aggregate(results: list[CaseResult], benchmark: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {c["id"]: c for c in benchmark if "id" in c}
    out: dict[str, Any] = {"strategies": {}}
    for strategy in STRATEGIES:
        strat = [r for r in results if r.strategy == strategy]
        if not strat:
            continue
        guarded_expected = 0
        guarded_correct = 0
        retrieval_cases = 0
        retrieval_hits = 0
        answer_ok = 0
        for r in strat:
            case = by_id.get(r.case_id, {})
            expected_guard = bool(case.get("expected_guardrail_blocked", False))
            if expected_guard:
                guarded_expected += 1
                if r.guardrail_blocked:
                    guarded_correct += 1
            expected_guide = case.get("expected_guide_title")
            if expected_guide:
                retrieval_cases += 1
                if r.retrieval_hit:
                    retrieval_hits += 1
            if r.answer_present and r.has_sources_line and r.has_safety_word:
                answer_ok += 1

        out["strategies"][strategy] = {
            "cases": len(strat),
            "guardrail_expected": guarded_expected,
            "guardrail_correct": guarded_correct,
            "retrieval_cases_with_expected_guide": retrieval_cases,
            "retrieval_top3_hit_rate": (retrieval_hits / retrieval_cases) if retrieval_cases else None,
            "basic_answer_ok_rate": answer_ok / len(strat),
            "avg_runtime_ms": sum(r.runtime_ms for r in strat) / len(strat),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare naive vs reranked vs HyDE strategies.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="eval/benchmark_stage3.json",
        help="Path to benchmark JSON file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path for JSON report (default: eval/report_stage3.json).",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not isinstance(benchmark, list):
        raise SystemExit("Benchmark must be a JSON list.")

    client = TestClient(app)
    results: list[CaseResult] = []
    for case in benchmark:
        if not isinstance(case, dict):
            continue
        for strategy in STRATEGIES:
            results.append(run_case(client=client, case=case, strategy=strategy))

    report = _aggregate(results=results, benchmark=benchmark)
    report["benchmark_path"] = str(benchmark_path)
    report["per_case"] = [r.__dict__ for r in results]

    out_path = Path(args.out or "eval/report_stage3.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # small console table
    print("strategy\tcases\tretrieval_top3_hit_rate\tguardrail_correct/expected\tavg_runtime_ms\tbasic_answer_ok_rate")
    for strategy, row in report["strategies"].items():
        print(
            f"{strategy}\t{row['cases']}\t{row['retrieval_top3_hit_rate']}\t"
            f"{row['guardrail_correct']}/{row['guardrail_expected']}\t{row['avg_runtime_ms']:.1f}\t"
            f"{row['basic_answer_ok_rate']:.2f}"
        )
    print(f"\nWrote report to: {out_path}")


if __name__ == "__main__":
    main()

