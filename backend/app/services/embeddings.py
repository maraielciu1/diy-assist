import hashlib
from typing import Sequence

from app.core.config import settings


class FallbackEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = text.lower().split()
            if not tokens:
                vectors.append(vec)
                continue
            for token in tokens:
                idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


def build_embedder():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(settings.embed_model_name)
    except Exception:
        return FallbackEmbedder(dim=384)
