from typing import Any

import httpx

from app.core.config import settings


def _provider() -> str:
    return str(getattr(settings, "slm_provider", "ollama")).lower()


def _model_name() -> str:
    return str(getattr(settings, "slm_model_name", "qwen2.5-3b"))


def _ollama_base() -> str:
    return str(getattr(settings, "ollama_base_url", "http://127.0.0.1:11434"))


def _lmstudio_base() -> str:
    return str(getattr(settings, "lmstudio_base_url", "http://127.0.0.1:1234/v1"))


def _timeout_seconds() -> int:
    return int(getattr(settings, "ollama_timeout_seconds", 60))


class SLMWrapper:
    def generate_answer(
        self,
        user_query: str,
        retrieved_chunks: list[dict[str, Any]],
        live_ifixit_guides: list[dict[str, Any]] | None = None,
    ) -> str:
        if not retrieved_chunks:
            return (
                "I could not find enough indexed repair context yet. "
                "Please ingest more guides or refine the appliance/category."
            )

        prompt = self._build_prompt(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
            live_ifixit_guides=live_ifixit_guides or [],
        )
        answer = self._generate(prompt)
        if answer:
            return answer

        top = retrieved_chunks[0]
        top_text = top.get("text", "")
        return (
            f"Based on retrieved iFixit-style repair documentation, start with: {top_text}\n\n"
            "Safety: disconnect power and water/gas supply (if applicable) before inspection."
        )

    def _generate(self, prompt: str) -> str | None:
        if _provider() == "lmstudio":
            return self._generate_with_lmstudio(prompt)
        return self._generate_with_ollama(prompt)

    def _generate_with_ollama(self, prompt: str) -> str | None:
        url = f"{_ollama_base()}/api/generate"
        payload = {
            "model": _model_name(),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            with httpx.Client(timeout=_timeout_seconds()) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                text = str(data.get("response") or "").strip()
                return text or None
            return None
        except Exception:
            return None

    def _generate_with_lmstudio(self, prompt: str) -> str | None:
        url = f"{_lmstudio_base()}/chat/completions"
        payload = {
            "model": _model_name(),
            "messages": [
                {"role": "system", "content": "You are a safety-first DIY appliance assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        try:
            with httpx.Client(timeout=_timeout_seconds()) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
            data = response.json()
            choices = data.get("choices", []) if isinstance(data, dict) else []
            if not choices:
                return None
            message = choices[0].get("message", {})
            text = str(message.get("content") or "").strip()
            return text or None
        except Exception:
            return None

    def _build_prompt(
        self,
        user_query: str,
        retrieved_chunks: list[dict[str, Any]],
        live_ifixit_guides: list[dict[str, Any]],
    ) -> str:
        context_blocks: list[str] = []
        for i, item in enumerate(retrieved_chunks, start=1):
            meta = item.get("metadata", {})
            previous = meta.get("previous_steps", [])
            previous_text = "\n".join(
                [
                    f"- Step {p.get('step_number')}: {p.get('text')}"
                    for p in previous
                    if isinstance(p, dict)
                ]
            )
            context_blocks.append(
                (
                    f"[Context {i}] guide={meta.get('guide_title')} step={meta.get('step_number')} "
                    f"score={item.get('score')}\n"
                    f"text={item.get('text')}\n"
                    f"previous_steps:\n{previous_text or '- none'}\n"
                    f"source={meta.get('source_url')}\n"
                )
            )

        live_guides = "\n".join(
            [
                f"- {g.get('guide_title')} ({g.get('source_url')})"
                for g in live_ifixit_guides
                if isinstance(g, dict)
            ]
        )
        joined_context = "\n".join(context_blocks)
        return (
            "You are DIY-Assist, a safety-first appliance troubleshooting assistant.\n"
            "Use only provided context and iFixit references. If context is insufficient, say so.\n"
            "Always provide concise actionable steps and include a safety warning.\n\n"
            f"User issue: {user_query}\n\n"
            f"Retrieved manual context:\n{joined_context}\n"
            f"Live iFixit guide candidates:\n{live_guides or '- none'}\n\n"
            "Return answer in 4-8 bullet points and include a final 'Sources:' line."
        )
