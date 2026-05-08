from app.services.ingestion import chunk_ifixit_steps, chunk_step_text


def test_chunk_step_text_splits_long_steps_with_overlap() -> None:
    text = (
        "Step 1: Unplug the washer and pull it away from the wall. "
        "Inspect the drain hose for crushing and severe kinks. "
        "Straighten the hose and ensure the standpipe height matches the manual. "
        "Run a short drain cycle and listen for pump noise."
    )
    chunks = chunk_step_text(text, max_chars=120, overlap_sentences=1, min_chunk_chars=60)
    assert len(chunks) >= 2
    assert "Inspect the drain hose" in chunks[0]
    assert "Inspect the drain hose" in chunks[1] or "Straighten the hose" in chunks[1]


def test_chunk_step_text_merges_short_tail_fragments() -> None:
    text = (
        "Step 3: Remove debris from the filter and reinstall it securely. "
        "Tighten."
    )
    chunks = chunk_step_text(text, max_chars=70, overlap_sentences=0, min_chunk_chars=30)
    assert len(chunks) == 1
    assert "Tighten." in chunks[0]


def test_chunk_ifixit_steps_keeps_step_units() -> None:
    chunks = chunk_ifixit_steps(
        [
            "Step 1: Disconnect power before touching internal parts.",
            "Step 2: Remove the pump cover and inspect for obstructions.",
        ],
        max_chars=120,
    )
    assert len(chunks) >= 2
    assert any("Step 1:" in chunk for chunk in chunks)
    assert any("Step 2:" in chunk for chunk in chunks)
