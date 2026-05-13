from typing import Any

import httpx

from app.core.config import settings


def _ifixit_timeout_seconds() -> int:
    return int(getattr(settings, "ifixit_timeout_seconds", 30))


class IFixitLiveClient:
    def suggest_guides(
        self,
        query: str,
        appliance_category: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": max(limit, 1), "q": query}
        if appliance_category:
            params["category"] = appliance_category

        url = f"{settings.ifixit_api_base_url}/guides"
        try:
            with httpx.Client(timeout=_ifixit_timeout_seconds()) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        guides: list[dict[str, Any]] = []
        if isinstance(payload, list):
            guides = [x for x in payload if isinstance(x, dict)]
        elif isinstance(payload, dict):
            for key in ("guides", "results", "data"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    guides = [x for x in candidate if isinstance(x, dict)]
                    break

        stopwords = {
            "a",
            "an",
            "and",
            "can",
            "do",
            "i",
            "is",
            "it",
            "me",
            "my",
            "near",
            "the",
            "to",
            "what",
            "when",
            "with",
            "you",
        }
        query_terms = {
            token.strip(".,?!:;()[]{}").lower()
            for token in query.split()
            if token.strip(".,?!:;()[]{}").lower() not in stopwords
        }

        ranked: list[dict[str, Any]] = []
        for guide in guides:
            title = str(guide.get("title") or "").strip()
            if not title:
                continue
            title_terms = set(title.lower().split())
            overlap = len(query_terms.intersection(title_terms))
            if overlap <= 0:
                continue
            ranked.append(
                {
                    "guide_id": str(guide.get("guideid") or guide.get("id") or ""),
                    "guide_title": title,
                    "source_url": str(guide.get("url") or guide.get("public_url") or ""),
                    "appliance_category": str(guide.get("category") or ""),
                    "_overlap": overlap,
                }
            )
        ranked.sort(key=lambda x: x["_overlap"], reverse=True)
        for item in ranked:
            item.pop("_overlap", None)
        return ranked[:limit]
