from typing import List, Iterable
from transformers import PreTrainedTokenizerBase

def chunk_by_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 900,
    overlap: int = 120,
) -> List[str]:
    words = text.split()
    chunks, buf = [], []
    def tlen(s: str) -> int:
        return len(tokenizer(s, truncation=False, add_special_tokens=False)["input_ids"])
    for w in words:
        candidate = (" ".join(buf) + (" " if buf else "") + w) if buf else w
        if tlen(candidate) <= max_tokens:
            buf.append(w)
        else:
            if buf:
                chunks.append(" ".join(buf))
                # seed next with overlap
                if overlap > 0:
                    ids = tokenizer(chunks[-1], truncation=False, add_special_tokens=False)["input_ids"]
                    seed = tokenizer.decode(ids[-overlap:], skip_special_tokens=True)
                    buf = seed.split()
                else:
                    buf = []
                buf.append(w)
            else:
                chunks.append(w)
                buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks
