# src/yt_sum/models/summarizer.py
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from yt_sum.utils.chunker import chunk_by_tokens

class BaselineSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def summarize(
        self,
        text: str,
        max_new_tokens: int = 160,
        min_new_tokens: int = 50,
        num_beams: Optional[int] = None,
        do_sample: bool = False,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> str:
        inputs = self.tokenizer(text, truncation=True, max_length=1024, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            no_repeat_ngram_size=3,
            do_sample=do_sample,
        )
        if num_beams and num_beams > 0:
            gen_kwargs.update(num_beams=num_beams, early_stopping=True)
        if do_sample:
            if top_p is not None: gen_kwargs["top_p"] = top_p
            if temperature is not None: gen_kwargs["temperature"] = temperature

        out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @torch.no_grad()
    def summarize_long(
        self,
        text: str,
        chunk_tokens: int = 900,
        chunk_overlap: int = 120,
        per_chunk_max: int = 120,
        per_chunk_min: int = 40,
        fuse_max: int = 240,
        fuse_min: int = 80,
        num_beams: Optional[int] = None,
    ) -> str:
        parts: List[str] = []
        for ch in chunk_by_tokens(text, self.tokenizer, max_tokens=chunk_tokens, overlap=chunk_overlap):
            parts.append(self.summarize(ch, max_new_tokens=per_chunk_max, min_new_tokens=per_chunk_min, num_beams=num_beams))
        fused = "\n".join(f"- {p}" for p in parts)
        return self.summarize(fused, max_new_tokens=fuse_max, min_new_tokens=fuse_min, num_beams=num_beams)
