#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.services.embeddings import build_embedder  # noqa: E402
from app.services.ingestion import chunk_ifixit_steps  # noqa: E402


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


def _extract_step_texts(guide: dict[str, Any]) -> list[str]:
    steps = guide.get("steps", [])
    out: list[str] = []
    for idx, step in enumerate(steps, start=1):
        if isinstance(step, str):
            text = step.strip()
        elif isinstance(step, dict):
            text = (
                step.get("text")
                or step.get("title")
                or step.get("summary")
                or step.get("description")
                or ""
            ).strip()
        else:
            text = ""
        if text:
            out.append(f"Step {idx}: {text}")
    return out


def _fetch_guide_detail(client: httpx.Client, guide_id: str) -> dict[str, Any]:
    url = f"{settings.ifixit_api_base_url}/guides/{guide_id}"
    response = client.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def _fetch_guides_from_api(limit: int, category: str | None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit}
    if category:
        params["category"] = category

    url = f"{settings.ifixit_api_base_url}/guides"
    with httpx.Client() as client:
        response = client.get(url, params=params, timeout=30)
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

    for guide in guides:
        step_texts = _extract_step_texts(guide)
        if not step_texts:
            continue

        guide_id = str(guide.get("guideid") or guide.get("id") or f"guide-{indexed_guides}")
        guide_title = str(guide.get("title") or "Untitled Guide")
        appliance_category = str(guide.get("category") or "unknown")
        source_url = str(guide.get("url") or guide.get("public_url") or "")
        brand = str(guide.get("brand") or "unknown")
        model = str(guide.get("model") or "unknown")

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []

        for step_idx, step_text in enumerate(step_texts, start=1):
            chunks = chunk_ifixit_steps([step_text])
            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                chunk_id = f"{guide_id}-s{step_idx}-c{chunk_idx}"
                ids.append(chunk_id)
                docs.append(chunk_text)
                metas.append(
                    {
                        "guide_id": guide_id,
                        "guide_title": guide_title,
                        "appliance_category": appliance_category,
                        "brand": brand,
                        "model": model,
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
        guides = _fetch_guides_from_api(limit=args.limit, category=args.category)
        source = "api"

    raw_path = _save_raw_payload(guides, source=source)

    indexed_guides, total_chunks = _upsert_guides(guides)
    print(
        f"Ingestion complete. guides_indexed={indexed_guides}, "
        f"chunks_upserted={total_chunks}, raw_saved={raw_path}"
    )


if __name__ == "__main__":
    main()
