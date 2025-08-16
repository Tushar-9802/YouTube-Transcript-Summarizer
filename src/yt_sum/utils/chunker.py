# src/yt_sum/utils/chunker.py
from __future__ import annotations
from typing import List
import re

# -------- Sentence splitting --------
def _split_sentences(text: str) -> List[str]:
    """
    Returns a list of sentences. Uses blingfire if available; otherwise a regex
    that also handles common multilingual sentence terminators (incl. '।').
    """
    if not text:
        return []
    # Prefer blingfire (fast & robust)
    try:
        from blingfire import text_to_sentences
        return [s.strip() for s in text_to_sentences(text).split("\n") if s.strip()]
    except Exception:
        pass

    # Fallback: split on sentence enders followed by space + capital/number
    # Includes: . ? ! । (Devanagari danda)
    pat = re.compile(r"(?<=[\.\?\!।])\s+(?=[^\s])", flags=re.UNICODE)
    sents = [s.strip() for s in pat.split(text) if s.strip()]
    # Merge obviously tiny fragments that can happen with quotes or abbreviations.
    merged: List[str] = []
    buf = ""
    for s in sents:
        if not buf:
            buf = s
        else:
            # If previous chunk is short and likely a tail (e.g., quote mark), merge.
            if len(buf) < 12:
                buf = (buf + " " + s).strip()
            else:
                merged.append(buf)
                buf = s
    if buf:
        merged.append(buf)
    return merged

# -------- Chunking by token budget --------
def chunk_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int = 900,
    overlap: int = 120,
) -> List[str]:
    """
    Greedily pack whole sentences into chunks under `max_tokens` (token IDs),
    adding `overlap` tail tokens between chunks for continuity.

    - Uses batch tokenization of sentences (fast).
    - Overlap is applied in **token space**, then decoded back to text.
    - Pathologically long sentences are split safely by tokens.
    - Returns a list[str] ready to feed into the summarizer.
    """
    sents = _split_sentences(text)
    if not sents:
        return []

    # Batch tokenize sentences once (avoids repeated encode calls).
    enc = tokenizer(
        sents,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    sent_ids: List[List[int]] = enc["input_ids"]

    chunks: List[str] = []
    buf_ids: List[int] = []

    def flush_buffer():
        """Decode current buffer to text and push to chunks."""
        nonlocal buf_ids
        if not buf_ids:
            return
        txt = tokenizer.decode(buf_ids, skip_special_tokens=True).strip()
        if txt:
            chunks.append(txt)

    for ids in sent_ids:
        # If the sentence fits in the current chunk, add it.
        if len(buf_ids) + len(ids) <= max_tokens:
            buf_ids.extend(ids)
            continue

        # Otherwise, flush current buffer as a chunk (if any).
        if buf_ids:
            flush_buffer()
            # prepare token overlap tail
            tail = buf_ids[-overlap:] if overlap > 0 else []
            buf_ids = tail.copy()
        else:
            buf_ids = []

        # Try to place the sentence now; if still too big, split it safely.
        if len(buf_ids) + len(ids) <= max_tokens:
            buf_ids.extend(ids)
        else:
            # Sentence alone is too large: split into safe steps.
            step = max(1, max_tokens - 4)
            for i in range(0, len(ids), step):
                seg = ids[i : i + step]
                if not seg:
                    continue
                # If there is something in the buffer and adding seg would overflow,
                # flush buffer first (with overlap into the next segment when possible).
                if buf_ids and len(buf_ids) + len(seg) > max_tokens:
                    flush_buffer()
                    tail = buf_ids[-overlap:] if overlap > 0 else []
                    buf_ids = tail.copy()

                # If even tail+seg is too big (can happen when overlap is large),
                # start clean for this segment.
                if len(buf_ids) + len(seg) > max_tokens:
                    if buf_ids:
                        flush_buffer()
                    buf_ids = []

                buf_ids.extend(seg)

            # After processing long sentence segments, if buffer >= max, flush now to avoid oversized buffer.
            if len(buf_ids) >= max_tokens:
                flush_buffer()
                buf_ids = buf_ids[-overlap:] if overlap > 0 else []

    # Final flush
    flush_buffer()
    return chunks
