# src/yt_sum/utils/chunker.py
from __future__ import annotations
from typing import List, Tuple
import re

# -------- Sentence splitting --------
def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    Tries blingfire (fast & robust). Falls back to a regex that also
    understands the Devanagari danda '।' used in Hindi and some Indic langs.
    """
    if not text:
        return []
    try:
        from blingfire import text_to_sentences
        return [s.strip() for s in text_to_sentences(text).split("\n") if s.strip()]
    except Exception:
        pass

    # Fallback: split on [.?! or ।] followed by whitespace.
    pat = re.compile(r"(?<=[\.\?\!।])\s+(?=\S)", flags=re.UNICODE)
    sents = [s.strip() for s in pat.split(text) if s.strip()]

    # Light merge: glue very short fragments (common with quotes/abbrevs)
    merged: List[str] = []
    buf = ""
    for s in sents:
        if not buf:
            buf = s
        else:
            if len(buf) < 12:               # tiny tail => merge
                buf = (buf + " " + s).strip()
            else:
                merged.append(buf)
                buf = s
    if buf:
        merged.append(buf)
    return merged

# -------- Internal helpers --------
def _decode(tokenizer, ids: List[int]) -> str:
    """Decode token IDs to plain text (no special tokens)."""
    return tokenizer.decode(ids, skip_special_tokens=True).strip()

def _batch_tokenize_sentences(tokenizer, sents: List[str]) -> List[List[int]]:
    """
    Tokenize sentences once as a batch (faster than per-sentence calls).
    Returns a list of token-id lists, one per sentence.
    """
    enc = tokenizer(
        sents,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    # Fast tokenizers return a dict of lists; fall back if needed
    ids = enc.get("input_ids", None)
    if ids is None or not isinstance(ids, list):
        # very rare: non-fast tokenizer path
        ids = [tokenizer(s, add_special_tokens=False, truncation=False)["input_ids"] for s in sents]
    return ids

# -------- Public API --------
def chunk_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int = 900,
    overlap: int = 120,
    *,
    min_chunk_tokens: int = 120,
    coalesce_tail: bool = True,
) -> List[str]:
    """
    Greedily pack whole sentences into chunks limited by `max_tokens` (token IDs),
    adding `overlap` tail tokens between consecutive chunks to preserve context.

    Why overlap? It gives the next chunk a bit of the previous context so the
    model doesn't miss boundary details.

    Args (plain English):
      - tokenizer: any HuggingFace tokenizer (the 'word piece' splitter).
      - max_tokens: upper bound of tokens per chunk (keep < model limit).
      - overlap: number of tokens to carry over from the end of one chunk to the next.
      - min_chunk_tokens: if the final chunk is shorter than this *and* there is a
        previous chunk, we merge it back (avoids a tiny last chunk).
      - coalesce_tail: turn the merge-on-last-chunk behavior on/off.

    Returns:
      List[str]: text chunks safe to feed into the summarizer.
    """
    sents = _split_sentences(text)
    if not sents:
        return []

    sent_ids = _batch_tokenize_sentences(tokenizer, sents)

    chunks: List[str] = []
    buf_ids: List[int] = []

    def flush_buffer():
        """Decode current buffer to text and push to chunks."""
        nonlocal buf_ids
        if not buf_ids:
            return
        txt = _decode(tokenizer, buf_ids)
        if txt:
            chunks.append(txt)

    for ids in sent_ids:
        # Fits into current buffer => append
        if len(buf_ids) + len(ids) <= max_tokens:
            buf_ids.extend(ids)
            continue

        # Doesn't fit: emit current buffer (if any) and start a new one with overlap tail
        if buf_ids:
            flush_buffer()
            tail = buf_ids[-overlap:] if overlap > 0 else []
            buf_ids = tail.copy()
        else:
            buf_ids = []

        # Try to add the sentence now; if still too big, split that sentence by tokens
        if len(buf_ids) + len(ids) <= max_tokens:
            buf_ids.extend(ids)
        else:
            step = max(1, max_tokens - 4)  # safety margin
            for i in range(0, len(ids), step):
                seg = ids[i : i + step]
                if not seg:
                    continue

                # If adding seg would overflow current buffer, emit buffer then start fresh (with overlap)
                if buf_ids and len(buf_ids) + len(seg) > max_tokens:
                    flush_buffer()
                    tail = buf_ids[-overlap:] if overlap > 0 else []
                    buf_ids = tail.copy()

                # If even tail+seg is too big (rare with large overlap), emit buffer and start clean
                if len(buf_ids) + len(seg) > max_tokens:
                    if buf_ids:
                        flush_buffer()
                    buf_ids = []

                buf_ids.extend(seg)

            # If the buffer grew to the cap while splitting a long sentence, emit now
            if len(buf_ids) >= max_tokens:
                flush_buffer()
                buf_ids = buf_ids[-overlap:] if overlap > 0 else []

    # Final flush
    if buf_ids:
        # Optional: coalesce a very small tail back into the previous chunk
        if coalesce_tail and chunks:
            tail_len = len(buf_ids)
            if tail_len < min_chunk_tokens:
                # merge with previous by re-decoding previous + (without duplicate overlap)
                prev = chunks.pop()
                prev_ids = tokenizer(prev, add_special_tokens=False, truncation=False)["input_ids"]
                # try to avoid re-adding overlap twice
                merged_ids = prev_ids + buf_ids[max(0, overlap - 1):]
                merged_txt = _decode(tokenizer, merged_ids[:max_tokens])
                if merged_txt:
                    chunks.append(merged_txt)
                else:
                    # fallback: push both separately if merge failed
                    chunks.append(prev)
                    chunks.append(_decode(tokenizer, buf_ids))
            else:
                chunks.append(_decode(tokenizer, buf_ids))
        else:
            chunks.append(_decode(tokenizer, buf_ids))

    return chunks

def chunk_by_tokens_with_map(
    text: str,
    tokenizer,
    max_tokens: int = 900,
    overlap: int = 120,
    *,
    min_chunk_tokens: int = 120,
    coalesce_tail: bool = True,
) -> Tuple[List[str], List[List[int]]]:
    """
    Variant that also returns which sentence indices went into each chunk.
    Useful later if you want to attach timestamps/evidence per chunk.

    Returns:
      (chunks, sent_index_map)
        - chunks: List[str]
        - sent_index_map: for each chunk i, a list of sentence indices from `_split_sentences(text)`
    """
    sents = _split_sentences(text)
    if not sents:
        return [], []
    sent_ids = _batch_tokenize_sentences(tokenizer, sents)

    chunks: List[str] = []
    idx_map: List[List[int]] = []
    buf_ids: List[int] = []
    buf_sent_idx: List[int] = []

    def flush_buffer():
        nonlocal buf_ids, buf_sent_idx
        if not buf_ids:
            return
        txt = _decode(tokenizer, buf_ids)
        if txt:
            chunks.append(txt)
            idx_map.append(buf_sent_idx.copy())

    for i, ids in enumerate(sent_ids):
        if len(buf_ids) + len(ids) <= max_tokens:
            buf_ids.extend(ids); buf_sent_idx.append(i); continue

        if buf_ids:
            flush_buffer()
            tail = buf_ids[-overlap:] if overlap > 0 else []
            buf_ids = tail.copy()
            # For the map, we keep the same sentence indices; overlap is token-level,
            # so we don't duplicate sentence indices here.
            buf_sent_idx = []

        if len(buf_ids) + len(ids) <= max_tokens:
            buf_ids.extend(ids); buf_sent_idx.append(i)
        else:
            step = max(1, max_tokens - 4)
            for j in range(0, len(ids), step):
                seg = ids[j : j + step]
                if not seg:
                    continue
                if buf_ids and len(buf_ids) + len(seg) > max_tokens:
                    flush_buffer()
                    tail = buf_ids[-overlap:] if overlap > 0 else []
                    buf_ids = tail.copy()
                    buf_sent_idx = []
                if len(buf_ids) + len(seg) > max_tokens:
                    if buf_ids:
                        flush_buffer()
                    buf_ids = []; buf_sent_idx = []
                buf_ids.extend(seg)
            if len(buf_ids) >= max_tokens:
                flush_buffer()
                buf_ids = buf_ids[-overlap:] if overlap > 0 else []
                buf_sent_idx = []

    if buf_ids:
        # Simple tail coalescing strategy for the map version:
        if coalesce_tail and chunks:
            tail_txt = _decode(tokenizer, buf_ids)
            if len(buf_ids) < min_chunk_tokens:
                prev_txt = chunks.pop()
                prev_ids = tokenizer(prev_txt, add_special_tokens=False, truncation=False)["input_ids"]
                merged_ids = prev_ids + buf_ids[max(0, overlap - 1):]
                merged_txt = _decode(tokenizer, merged_ids[:max_tokens])
                if merged_txt:
                    chunks.append(merged_txt)
                    # merge sentence indices too
                    prev_idx = idx_map.pop()
                    idx_map.append(prev_idx + buf_sent_idx)
                else:
                    chunks.append(prev_txt)
                    chunks.append(tail_txt)
                    idx_map.append(buf_sent_idx)
            else:
                chunks.append(tail_txt)
                idx_map.append(buf_sent_idx)
        else:
            chunks.append(_decode(tokenizer, buf_ids))
            idx_map.append(buf_sent_idx)

    return chunks, idx_map
