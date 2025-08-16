# src/yt_sum/utils/chunker.py
from typing import List
from transformers import PreTrainedTokenizerBase

def chunk_by_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 900,
    overlap: int = 120,
) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    buf: List[str] = []

    def tlen(s: str) -> int:
        return len(tokenizer(s, truncation=False, add_special_tokens=False)["input_ids"])

    for w in words:
        candidate = (" ".join(buf) + (" " if buf else "") + w) if buf else w
        if tlen(candidate) <= max_tokens:
            buf.append(w)
        else:
            if buf:
                chunks.append(" ".join(buf))
                if overlap > 0:
                    ids = tokenizer(chunks[-1], truncation=False, add_special_tokens=False)["input_ids"]
                    seed = tokenizer.decode(ids[-overlap:], skip_special_tokens=True).split()
                    buf = seed
                else:
                    buf = []
                buf.append(w)
            else:
                chunks.append(w)  # pathological long token
                buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks
