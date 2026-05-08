#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typing
import typing_extensions

if not hasattr(typing_extensions, "TypeAlias"):
    typing_extensions.TypeAlias = typing.TypeAlias

import chromadb
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.services.embeddings import build_embedder  # noqa: E402
from app.services.ingestion import chunk_step_text  # noqa: E402


def _ifixit_timeout_seconds() -> int:
    return int(getattr(settings, "ifixit_timeout_seconds", 30))


def _extract_guides(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("guides", "results", "data"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    return []


def _extract_tools(guide: dict[str, Any]) -> list[str]:
    raw_tools = guide.get("tools")
    if isinstance(raw_tools, list):
        out: list[str] = []
        for tool in raw_tools:
            if isinstance(tool, str) and tool.strip():
                out.append(tool.strip())
            elif isinstance(tool, dict):
                name = str(
                    tool.get("name")
                    or tool.get("title")
                    or tool.get("text")
                    or ""
                ).strip()
                if name:
                    out.append(name)
        return out
    return []


def _extract_step_texts(guide: dict[str, Any]) -> list[dict[str, Any]]:
    steps = guide.get("steps", [])
    out: list[dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        if isinstance(step, str):
            text = step.strip()
            step_tools: list[str] = []
        elif isinstance(step, dict):
            base = (
                step.get("text")
                or step.get("summary")
                or step.get("description")
                or ""
            )
            if not base and isinstance(step.get("lines"), list):
                line_parts: list[str] = []
                for line in step["lines"]:
                    if isinstance(line, dict):
                        candidate = (
                            line.get("text_raw")
                            or line.get("text_rendered")
                            or line.get("text")
                            or ""
                        ).strip()
                        if candidate:
                            line_parts.append(candidate)
                base = " ".join(line_parts)
            title = str(step.get("title") or "").strip()
            text = f"{title}. {base}".strip(". ").strip()
            step_tools = []
            if isinstance(step.get("tools"), list):
                for tool in step["tools"]:
                    if isinstance(tool, str) and tool.strip():
                        step_tools.append(tool.strip())
                    elif isinstance(tool, dict):
                        name = str(tool.get("name") or tool.get("title") or "").strip()
                        if name:
                            step_tools.append(name)
        else:
            text = ""
            step_tools = []
        if text:
            out.append(
                {
                    "step_number": idx,
                    "text": f"Step {idx}: {text}",
                    "tools": step_tools,
                }
            )
    return out


def _extract_guide_text(guide: dict[str, Any]) -> str:
    for key in ("summary", "introduction", "description", "text", "guide_text"):
        value = guide.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_brand_and_model(guide: dict[str, Any]) -> tuple[str, str]:
    brand = str(guide.get("brand") or "").strip()
    model = str(guide.get("model") or "").strip()

    subject = guide.get("subject")
    if isinstance(subject, dict):
        if not brand:
            brand = str(subject.get("brand") or subject.get("manufacturer") or "").strip()
        if not model:
            model = str(subject.get("model") or subject.get("model_number") or "").strip()
    return brand or "unknown", model or "unknown"


def _extract_difficulty(guide: dict[str, Any]) -> str:
    difficulty = guide.get("difficulty")
    if isinstance(difficulty, dict):
        return str(difficulty.get("title") or difficulty.get("value") or "unknown")
    if isinstance(difficulty, str) and difficulty.strip():
        return difficulty.strip()
    return "unknown"


def _infer_appliance_type(guide_title: str, guide_text: str) -> str:
    haystack = f"{guide_title} {guide_text}".lower()
    mapping = {
        "washer": "washer",
        "washing machine": "washer",
        "dryer": "dryer",
        "dishwasher": "dishwasher",
        "refrigerator": "refrigerator",
        "fridge": "refrigerator",
        "oven": "oven",
        "microwave": "microwave",
    }
    for keyword, appliance_type in mapping.items():
        if keyword in haystack:
            return appliance_type
    return "unknown"


def _normalized_guide_record(guide: dict[str, Any], index_hint: int) -> dict[str, Any]:
    brand, model = _extract_brand_and_model(guide)
    category = str(guide.get("category") or guide.get("type") or "unknown").strip() or "unknown"
    tools = _extract_tools(guide)
    steps = _extract_step_texts(guide)
    guide_title = str(guide.get("title") or "Untitled Guide")
    guide_text = _extract_guide_text(guide)
    appliance_type = _infer_appliance_type(guide_title=guide_title, guide_text=guide_text)

    return {
        "guide_id": str(guide.get("guideid") or guide.get("id") or f"guide-{index_hint}"),
        "guide_title": guide_title,
        "appliance_category": category,
        "appliance_type": appliance_type,
        "brand": brand,
        "model": model,
        "difficulty": _extract_difficulty(guide),
        "tools": tools,
        "source_url": str(guide.get("url") or guide.get("public_url") or ""),
        "guide_text": guide_text,
        "steps": steps,
    }


def _fetch_guide_detail(client: httpx.Client, guide_id: str) -> dict[str, Any]:
    url = f"{settings.ifixit_api_base_url}/guides/{guide_id}"
    response = client.get(url, timeout=_ifixit_timeout_seconds())
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def _fetch_guides_from_api(limit: int, category: str | None, query: str | None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit}
    if category:
        params["category"] = category
    if query:
        params["q"] = query

    url = f"{settings.ifixit_api_base_url}/guides"
    with httpx.Client(timeout=_ifixit_timeout_seconds()) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        guides = _extract_guides(response.json())

        enriched: list[dict[str, Any]] = []
        for guide in guides:
            if guide.get("steps"):
                enriched.append(guide)
                continue
            guide_id = guide.get("guideid") or guide.get("id")
            if guide_id is None:
                enriched.append(guide)
                continue
            try:
                detail = _fetch_guide_detail(client, str(guide_id))
                merged = {**guide, **detail}
                enriched.append(merged)
            except Exception:
                enriched.append(guide)
        return enriched[:limit]


def _save_raw_payload(guides: list[dict[str, Any]], source: str) -> Path:
    raw_dir = ROOT / settings.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = raw_dir / f"ifixit_{source}_{ts}.json"
    payload = {"source": source, "guides_count": len(guides), "guides": guides}
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def _upsert_guides(guides: list[dict[str, Any]]) -> tuple[int, int]:
    embedder = build_embedder()
    chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = chroma_client.get_or_create_collection(name=settings.chroma_collection)

    total_chunks = 0
    indexed_guides = 0

    for idx, guide in enumerate(guides, start=1):
        normalized = _normalized_guide_record(guide, index_hint=idx)
        steps = normalized["steps"]
        if not steps:
            continue

        guide_id = normalized["guide_id"]
        guide_title = normalized["guide_title"]
        appliance_category = normalized["appliance_category"]
        source_url = normalized["source_url"]
        brand = normalized["brand"]
        model = normalized["model"]
        appliance_type = normalized["appliance_type"]
        difficulty = normalized["difficulty"]
        tools = normalized["tools"]
        guide_text = normalized["guide_text"]

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []

        for step in steps:
            step_idx = int(step.get("step_number", 0))
            step_text = str(step.get("text") or "")
            if not step_text.strip():
                continue
            combined_tools = sorted(set(tools + step.get("tools", [])))
            chunks = chunk_step_text(step_text)
            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                chunk_id = f"{guide_id}-s{step_idx}-c{chunk_idx}"
                ids.append(chunk_id)
                docs.append(chunk_text)
                metas.append(
                    {
                        "guide_id": guide_id,
                        "guide_title": guide_title,
                        "appliance_category": appliance_category,
                        "appliance_type": appliance_type,
                        "brand": brand,
                        "model": model,
                        "difficulty": difficulty,
                        "tools": ", ".join(combined_tools),
                        "guide_text": guide_text,
                        "step_number": step_idx,
                        "chunk_number": chunk_idx,
                        "source_url": source_url,
                    }
                )

        if not docs:
            continue
        embeddings = embedder.encode(docs)
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        total_chunks += len(docs)
        indexed_guides += 1

    return indexed_guides, total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest iFixit guides into ChromaDB.")
    parser.add_argument("--limit", type=int, default=25, help="Number of guides to ingest.")
    parser.add_argument("--category", type=str, default=None, help="Optional category filter.")
    parser.add_argument("--query", type=str, default=None, help="Optional query keyword.")
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Path to pre-fetched iFixit JSON payload.",
    )
    args = parser.parse_args()

    if args.input_json:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        guides = _extract_guides(payload)
        source = "input"
    else:
        try:
            guides = _fetch_guides_from_api(limit=args.limit, category=args.category, query=args.query)
            source = "api"
        except Exception as exc:
            print(f"iFixit fetch failed: {exc}")
            print("Tip: use --input-json data/raw/sample_ifixit_minimal.json for local/offline testing.")
            raise

    raw_path = _save_raw_payload(guides, source=source)
    indexed_guides, total_chunks = _upsert_guides(guides)
    print(
        f"Ingestion complete. guides_indexed={indexed_guides}, "
        f"chunks_upserted={total_chunks}, raw_saved={raw_path}"
    )


if __name__ == "__main__":
    main()
