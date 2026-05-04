from dataclasses import dataclass
from typing import Any
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
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedding_model.encode([query])[0]

        where: dict[str, Any] | None = None
        if appliance_category:
            where = {"appliance_category": appliance_category}

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
import typing
import typing_extensions

if not hasattr(typing_extensions, "TypeAlias"):
    typing_extensions.TypeAlias = typing.TypeAlias

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
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedding_model.encode([query])[0]

        where: dict[str, Any] | None = None
        if appliance_category:
            where = {"appliance_category": appliance_category}

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
    def _normalize_score(distance: float) -> float:
        # Chroma distance can exceed 1.0 depending on metric.
        # Convert to bounded similarity-like confidence in (0, 1].
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

            stitched.sort(
                key=lambda x: int(x.get("step_number", 0)),
            )
            return stitched[-settings.previous_steps_window :]
        except Exception:
            return []
