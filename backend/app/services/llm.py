from typing import Any


class SLMWrapper:
    """
    Minimal SLM wrapper interface.
    Replace generate_answer internals with local Qwen inference or hosted endpoint.
    """

    def generate_answer(
        self,
        user_query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        if not retrieved_chunks:
            return (
                "I could not find enough indexed repair context yet. "
                "Please ingest more guides or refine the appliance/category."
            )

        top = retrieved_chunks[0]
        top_text = top.get("text", "")
        return (
            f"Based on retrieved repair documentation, start with this step: {top_text}\n\n"
            "Safety: disconnect power and water/gas supply (if applicable) before inspection."
        )
