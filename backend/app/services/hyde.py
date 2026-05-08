from __future__ import annotations

from app.services.llm import SLMWrapper
from app.services.retrieval import NaiveRetriever, RetrievedChunk, CrossEncoderReranker


class HyDERetriever:
    """
    HyDE retrieval:
    1) Generate a hypothetical answer.
    2) Embed that hypothetical answer.
    3) Use it as the dense retrieval query.
    """

    def __init__(self, base: NaiveRetriever | None = None, slm: SLMWrapper | None = None) -> None:
        self.base = base or NaiveRetriever()
        self.slm = slm or SLMWrapper()

    def search(
        self,
        query: str,
        appliance_category: str | None = None,
        appliance_type: str | None = None,
        brand: str | None = None,
        model: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        hypothetical = self.slm.generate_hypothetical_answer(query)
        embedding = self.base.embedding_model.encode([hypothetical])[0]
        hits = self.base.search(
            query=query,
            query_embedding=embedding,
            appliance_category=appliance_category,
            appliance_type=appliance_type,
            brand=brand,
            model=model,
            top_k=top_k,
        )
        for h in hits:
            h.metadata["hyde_query"] = hypothetical
        return hits


class HyDERerankedRetriever:
    """
    HyDE + rerank combination:
    - Dense retrieval is driven by the HyDE embedding.
    - CrossEncoder reranking is still done with the original user query.
    """

    def __init__(
        self,
        base: NaiveRetriever | None = None,
        slm: SLMWrapper | None = None,
        reranker: CrossEncoderReranker | None = None,
        candidate_k: int = 20,
    ) -> None:
        self.base = base or NaiveRetriever()
        self.slm = slm or SLMWrapper()
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        appliance_category: str | None = None,
        appliance_type: str | None = None,
        brand: str | None = None,
        model: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        hypothetical = self.slm.generate_hypothetical_answer(query)
        embedding = self.base.embedding_model.encode([hypothetical])[0]
        candidates = self.base.search(
            query=query,
            query_embedding=embedding,
            appliance_category=appliance_category,
            appliance_type=appliance_type,
            brand=brand,
            model=model,
            top_k=max(self.candidate_k, top_k),
        )
        for c in candidates:
            c.metadata["hyde_query"] = hypothetical
        return self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)

