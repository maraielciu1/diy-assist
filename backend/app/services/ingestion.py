from dataclasses import dataclass


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    appliance_category: str
    guide_title: str
    step_number: int


def chunk_ifixit_steps(steps: list[str], max_chars: int = 1800) -> list[str]:
    """
    Lightweight first-pass chunker for step text.
    Weeks 7-8: replace with token-aware chunking and overlap.
    """
    chunks: list[str] = []
    buffer = ""

    for step in steps:
        candidate = f"{buffer}\n{step}".strip() if buffer else step
        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer)
        buffer = step

    if buffer:
        chunks.append(buffer)

    return chunks
