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
    "hyde_reranked": {
        "rag": "/api/v1/rag/hyde_reranked",
        "chat": "/api/v1/chat/hyde_reranked",
    },
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
    has_sources: bool
    has_safety_signal: bool


def _has_sources(chat: dict[str, Any], answer: str) -> bool:
    return bool(chat.get("citations")) or "sources:" in answer.lower()


def _has_safety_signal(text: str, chat: dict[str, Any]) -> bool:
    if chat.get("safety_warning") or chat.get("guardrail_blocked"):
        return True
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "safety",
            "unplug",
            "turn off",
            "shut off",
            "disconnect power",
            "professional",
            "technician",
        )
    )


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

    results = rag.get("results") or []
    expected_guide = case.get("expected_guide_title")
    retrieval_hit = False
    if expected_guide:
        retrieval_hit = any(result.get("guide_title") == expected_guide for result in results[:3])

    answer_text = chat.get("answer") or chat.get("message") or ""
    return CaseResult(
        case_id=str(case["id"]),
        strategy=strategy,
        guardrail_blocked=bool(chat.get("guardrail_blocked") or rag.get("guardrail_blocked")),
        retrieval_hit=retrieval_hit,
        retrieval_top1_guide=results[0].get("guide_title") if results else None,
        runtime_ms=runtime_ms,
        answer_present=bool(answer_text.strip()),
        has_sources=_has_sources(chat, answer_text),
        has_safety_signal=_has_safety_signal(answer_text, chat),
    )


def aggregate(results: list[CaseResult], benchmark: list[dict[str, Any]], strategies: list[str]) -> dict[str, Any]:
    cases_by_id = {case["id"]: case for case in benchmark if isinstance(case, dict) and "id" in case}
    report: dict[str, Any] = {"strategies": {}}
    for strategy in strategies:
        strategy_results = [result for result in results if result.strategy == strategy]
        if not strategy_results:
            continue

        guarded_expected = 0
        guarded_correct = 0
        retrieval_expected = 0
        retrieval_hits = 0
        answer_quality = 0

        for result in strategy_results:
            case = cases_by_id.get(result.case_id, {})
            if case.get("expected_guardrail_blocked"):
                guarded_expected += 1
                if result.guardrail_blocked:
                    guarded_correct += 1
            if case.get("expected_guide_title"):
                retrieval_expected += 1
                if result.retrieval_hit:
                    retrieval_hits += 1
            if result.answer_present and result.has_safety_signal:
                answer_quality += 1

        report["strategies"][strategy] = {
            "cases": len(strategy_results),
            "guardrail_expected": guarded_expected,
            "guardrail_correct": guarded_correct,
            "retrieval_cases_with_expected_guide": retrieval_expected,
            "retrieval_top3_hit_rate": retrieval_hits / retrieval_expected if retrieval_expected else None,
            "basic_answer_ok_rate": answer_quality / len(strategy_results),
            "avg_runtime_ms": sum(result.runtime_ms for result in strategy_results) / len(strategy_results),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DIY-Assist retrieval and chat strategies.")
    parser.add_argument("--input", default="data/eval/troubleshooting_queries.json")
    parser.add_argument("--out", default="data/eval/report_stage5.json")
    parser.add_argument("--strategies", nargs="+", default=["naive", "reranked", "hyde"])
    args = parser.parse_args()

    unknown = [strategy for strategy in args.strategies if strategy not in STRATEGIES]
    if unknown:
        raise SystemExit(f"Unknown strategies: {', '.join(unknown)}")

    benchmark_path = Path(args.input)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not isinstance(benchmark, list):
        raise SystemExit("Benchmark must be a JSON list.")

    client = TestClient(app)
    results: list[CaseResult] = []
    for case in benchmark:
        if not isinstance(case, dict):
            continue
        for strategy in args.strategies:
            results.append(run_case(client, case, strategy))

    report = aggregate(results, benchmark, args.strategies)
    report["benchmark_path"] = str(benchmark_path)
    report["per_case"] = [result.__dict__ for result in results]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("strategy\tcases\tretrieval_top3_hit_rate\tguardrail_correct/expected\tavg_runtime_ms\tbasic_answer_ok_rate")
    for strategy, row in report["strategies"].items():
        print(
            f"{strategy}\t{row['cases']}\t{row['retrieval_top3_hit_rate']}\t"
            f"{row['guardrail_correct']}/{row['guardrail_expected']}\t"
            f"{row['avg_runtime_ms']:.1f}\t{row['basic_answer_ok_rate']:.2f}"
        )
    print(f"\nWrote report to: {out_path}")


if __name__ == "__main__":
    main()
