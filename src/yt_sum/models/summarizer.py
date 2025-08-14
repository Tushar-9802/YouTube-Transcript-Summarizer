# src/yt_sum/models/summarizer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional
import torch
import logging as log
from yt_sum.utils.chunker import chunk_by_tokens


class BaselineSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", device=None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        log.info(f"Loading summarizer model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def summarize(self, text: str, max_new_tokens=160, min_new_tokens=50) -> str:
        inputs = self.tokenizer(text, truncation=True, max_length=1024, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                  no_repeat_ngram_size=3, do_sample=False)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @torch.no_grad()
    def summarize_long(self, text: str) -> str:
        parts: list[str] = []
        for ch in chunk_by_tokens(text, self.tokenizer, max_tokens=900, overlap=120):
            parts.append(self.summarize(ch, max_new_tokens=120, min_new_tokens=40))
        fused = "\n".join(f"- {p}" for p in parts)
        return self.summarize(fused, max_new_tokens=180, min_new_tokens=60)