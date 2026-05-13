from dataclasses import dataclass
from typing import Any

import chromadb

from app.core.config import settings
from app.services.embeddings import build_embedder


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict[str, Any]


class NaiveRetriever:
    def __init__(self) -> None:
        self.embedding_model = build_embedder()
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection
        )

    def search(
        self,
        query: str,
        appliance_category: str | None = None,
        appliance_type: str | None = None,
        brand: str | None = None,
        model: str | None = None,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if query_embedding is None:
            query_embedding = self.embedding_model.encode([query])[0]
        where = self._build_where_clause(
            appliance_category=appliance_category,
            appliance_type=appliance_type,
            brand=brand,
            model=model,
        )

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: list[RetrievedChunk] = []
        for doc, metadata, distance in zip(documents, metadatas, distances):
            enriched_metadata = dict(metadata or {})
            raw_distance = float(distance)
            enriched_metadata["previous_steps"] = self._get_previous_steps(
                metadata=enriched_metadata,
            )
            enriched_metadata["raw_distance"] = raw_distance
            results.append(
                RetrievedChunk(
                    text=doc,
                    score=self._normalize_score(raw_distance),
                    metadata=enriched_metadata,
                )
            )
        return results

    @staticmethod
    def _build_where_clause(
        appliance_category: str | None = None,
        appliance_type: str | None = None,
        brand: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        filters: list[dict[str, Any]] = []
        if appliance_category:
            filters.append({"appliance_category": {"$eq": appliance_category}})
        if appliance_type:
            filters.append({"appliance_type": {"$eq": appliance_type}})
        if brand:
            filters.append({"brand": {"$eq": brand}})
        if model:
            filters.append({"model": {"$eq": model}})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    @staticmethod
    def _normalize_score(distance: float) -> float:
        return 1.0 / (1.0 + max(distance, 0.0))

    def _get_previous_steps(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        guide_id = metadata.get("guide_id")
        step_number = metadata.get("step_number")
        if guide_id is None or step_number is None:
            return []

        try:
            current_step = int(step_number)
            if current_step <= 1:
                return []

            raw = self.collection.get(
                where={
                    "$and": [
                        {"guide_id": {"$eq": str(guide_id)}},
                        {"step_number": {"$lt": current_step}},
                    ]
                },
                include=["documents", "metadatas"],
            )
            documents = raw.get("documents", [])
            metadatas = raw.get("metadatas", [])
            stitched: list[dict[str, Any]] = []
            for doc, meta in zip(documents, metadatas):
                if not isinstance(meta, dict):
                    continue
                stitched.append(
                    {
                        "step_number": meta.get("step_number"),
                        "text": doc,
                        "guide_title": meta.get("guide_title"),
                    }
                )

            stitched.sort(key=lambda x: int(x.get("step_number", 0)))
            return stitched[-settings.previous_steps_window :]
        except Exception:
            return []


class CrossEncoderReranker:
    """
    Rerank retrieved chunks using a CrossEncoder.

    Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` by default (configurable),
    but is loaded lazily so tests and environments without HF cache can still run.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or getattr(
            settings, "rerank_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []

        try:
            model = self._get_model()
            pairs = [(query, c.text) for c in candidates]
            scores = model.predict(pairs)
        except Exception as exc:
            fallback: list[RetrievedChunk] = []
            for cand in candidates[: max(top_k, 0)]:
                meta = dict(cand.metadata or {})
                meta["rerank_fallback"] = True
                meta["rerank_error"] = str(exc)
                meta["baseline_score"] = float(cand.score)
                fallback.append(RetrievedChunk(text=cand.text, score=cand.score, metadata=meta))
            return fallback

        enriched: list[RetrievedChunk] = []
        for cand, score in zip(candidates, scores):
            meta = dict(cand.metadata or {})
            meta["rerank_score"] = float(score)
            meta["baseline_score"] = float(cand.score)
            enriched.append(RetrievedChunk(text=cand.text, score=float(score), metadata=meta))

        enriched.sort(key=lambda c: float(c.score), reverse=True)
        return enriched[: max(top_k, 0)]


class RerankedRetriever:
    """
    Two-stage retrieval:
    1) Dense retrieval (Chroma) to get a larger candidate set.
    2) CrossEncoder reranking to choose final top-k.
    """

    def __init__(
        self,
        base: NaiveRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        candidate_k: int | None = None,
    ) -> None:
        self.base = base or NaiveRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_k = int(candidate_k or getattr(settings, "rerank_candidate_k", 20))

    def search(
        self,
        query: str,
        appliance_category: str | None = None,
        appliance_type: str | None = None,
        brand: str | None = None,
        model: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        candidates = self.base.search(
            query=query,
            appliance_category=appliance_category,
            appliance_type=appliance_type,
            brand=brand,
            model=model,
            top_k=max(self.candidate_k, top_k),
        )
        return self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
