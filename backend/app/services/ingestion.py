from dataclasses import dataclass
import re


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    appliance_category: str
    guide_title: str
    step_number: int


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
    return parts or [cleaned]


def _with_overlap(previous_chunk: str, overlap_sentences: int) -> str:
    if overlap_sentences <= 0:
        return ""
    previous = _split_sentences(previous_chunk)
    if not previous:
        return ""
    return " ".join(previous[-overlap_sentences:]).strip()


def chunk_step_text(
    step_text: str,
    max_chars: int = 420,
    overlap_sentences: int = 1,
    min_chunk_chars: int = 90,
) -> list[str]:
    """
    Split one guide step into retrieval-quality chunks.

    - Keeps each iFixit step as the natural unit.
    - Splits long steps by sentence boundaries.
    - Adds a light sentence overlap for continuity.
    - Merges very short tail fragments into adjacent chunks.
    """
    sentences = _split_sentences(step_text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    previous_chunk = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            previous_chunk = current

        overlap = _with_overlap(previous_chunk, overlap_sentences)
        current = f"{overlap} {sentence}".strip() if overlap else sentence
        if len(current) > max_chars:
            # Fallback for unusually long no-punctuation text.
            hard = current[:max_chars].strip()
            if hard:
                chunks.append(hard)
            current = current[max_chars:].strip()

    if current:
        chunks.append(current)

    if len(chunks) >= 2 and len(chunks[-1]) < min_chunk_chars:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return chunks


def chunk_ifixit_steps(steps: list[str], max_chars: int = 420) -> list[str]:
    """
    Chunk all steps while preserving step boundaries.
    """
    chunks: list[str] = []
    for step in steps:
        chunks.extend(chunk_step_text(step_text=step, max_chars=max_chars))
    return chunks
