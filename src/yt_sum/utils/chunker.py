# chunker.py
from __future__ import annotations
from typing import List
import re

def chunk_text(
    text: str,
    tokenizer,
    chunk_tokens: int = 900,
    chunk_overlap: int = 120
) -> List[str]:
    """
    Sentence-aware chunking with token-based cap and overlap.
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = re.split(r'(?<=[\.!\?])\s+', text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0

    for s in sentences:
        if not s:
            continue
        t = len(tokenizer.encode(s, add_special_tokens=False))
        if cur and cur_tok + t > chunk_tokens:
            chunk = " ".join(cur).strip()
            chunks.append(chunk)
            if chunk_overlap > 0:
                tail_tokens = tokenizer.encode(chunk, add_special_tokens=False)[-chunk_overlap:]
                tail_text = tokenizer.decode(tail_tokens, skip_special_tokens=True)
                cur = [tail_text] if tail_text else []
                cur_tok = len(tokenizer.encode(tail_text, add_special_tokens=False)) if tail_text else 0
            else:
                cur, cur_tok = [], 0
        cur.append(s)
        cur_tok += t

    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks
