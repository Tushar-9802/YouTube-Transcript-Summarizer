# # src/yt_sum/models/summarizer.py
# from typing import Optional, List
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from yt_sum.utils.chunker import chunk_by_tokens

# class BaselineSummarizer:
#     def __init__(self, model_name: str = "facebook/bart-large-cnn", device: Optional[str] = None):
#         self.model_name = model_name
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#         self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.model.eval()

#     @torch.no_grad()
#     def summarize(
#         self,
#         text: str,
#         max_new_tokens: int = 160,
#         min_new_tokens: int = 50,
#         num_beams: Optional[int] = None,
#         do_sample: bool = False,
#         top_p: Optional[float] = None,
#         temperature: Optional[float] = None,
#     ) -> str:
#         inputs = self.tokenizer(text, truncation=True, max_length=1024, return_tensors="pt")
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         gen_kwargs = dict(
#             max_new_tokens=max_new_tokens,
#             min_new_tokens=min_new_tokens,
#             no_repeat_ngram_size=3,
#             do_sample=do_sample,
#         )
#         if num_beams and num_beams > 0:
#             gen_kwargs.update(num_beams=num_beams, early_stopping=True)
#         if do_sample:
#             if top_p is not None: gen_kwargs["top_p"] = top_p
#             if temperature is not None: gen_kwargs["temperature"] = temperature

#         out = self.model.generate(**inputs, **gen_kwargs)
#         return self.tokenizer.decode(out[0], skip_special_tokens=True)

#     @torch.no_grad()
#     def summarize_long(
#         self,
#         text: str,
#         chunk_tokens: int = 900,
#         chunk_overlap: int = 120,
#         per_chunk_max: int = 120,
#         per_chunk_min: int = 40,
#         fuse_max: int = 240,
#         fuse_min: int = 80,
#         num_beams: Optional[int] = None,
#     ) -> str:
#         parts: List[str] = []
#         for ch in chunk_by_tokens(text, self.tokenizer, max_tokens=chunk_tokens, overlap=chunk_overlap):
#             parts.append(self.summarize(ch, max_new_tokens=per_chunk_max, min_new_tokens=per_chunk_min, num_beams=num_beams))
#         fused = "\n".join(f"- {p}" for p in parts)
#         return self.summarize(fused, max_new_tokens=fuse_max, min_new_tokens=fuse_min, num_beams=num_beams)
# src/yt_sum/models/summarizer.py
# src/yt_sum/summarizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from yt_sum.utils.chunker import chunk_by_tokens

# Friendly keys → HF IDs (feel free to add more later)
MODEL_ALIASES: Dict[str, str] = {
    "bart-cnn": "facebook/bart-large-cnn",
    "pegasus-cnn": "google/pegasus-cnn_dailymail",
    # Optional (keep for future guided runs)
    "pegasus-large": "google/pegasus-large",
    "t5-large": "t5-large",
}

def _resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)  # allow raw HF IDs too

@dataclass
class GenParams:
    # Length control (auto if None)
    summary_ratio: float = 0.18          # ~18% of input tokens per chunk
    reduce_target_tokens: int = 320      # final fuse target
    # Decoding
    num_beams: int = 5
    length_penalty: float = 0.9          # <1.0 favors longer outputs
    repetition_penalty: float = 1.08
    no_repeat_ngram_size: int = 3
    do_sample: bool = False
    temperature: float = 1.0
    # Chunking
    chunk_tokens: int = 900              # keep < model limit (1024) with margin
    chunk_overlap: int = 120
    # Optional guidance (used later / works now too)
    guidance: Optional[str] = None       # e.g., "Summarize with Methods, Results, Limitations."

class BaselineSummarizer:
    """
    Research-friendly abstractive summarizer:
      - summarize(): single-shot for <=1024 tokens with auto budgets
      - summarize_long(): chunked map→reduce with token-ratio control
    """
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
        use_8bit: bool = True,            # optional 8-bit load to save VRAM
        device_map: str = "auto"          # requires accelerate for 8-bit path
    ):
        self.model_name = _resolve_model_name(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.quantized = False
        self.model = None

        # Try efficient 8-bit load; gracefully fall back if not available
        if use_8bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_cfg,
                    device_map=device_map,
                )
                self.quantized = True
            except Exception:
                # fallback to normal FP16/FP32
                pass

        if self.model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)

        self.model.eval()

        # Safe input cap (most seq2seq here support 1024)
        self.max_in_tokens = min(
            1024,
            getattr(getattr(self.model, "config", None), "max_position_embeddings", 1024)
        ) - 32  # margin for special tokens

    # ---------- internal helpers ----------
    def _prepend_guidance(self, text: str, guidance: Optional[str]) -> str:
        if not guidance:
            return text
        # For encoder-decoder models like BART/PEGASUS, prefix guidance in the input
        return f"Instruction: {guidance}\n\nTranscript:\n{text}"

    def _budget_from_len(self, in_len: int, ratio: float) -> Tuple[int, int]:
        # Robust budget from input length
        target = max(120, int(in_len * ratio))
        max_new = min(512, max(target, 220))
        min_new = max(100, min(max_new - 80, int(target * 0.7)))
        return min_new, max_new

    def _generate(self, text: str, p: GenParams) -> str:
        inp = self._prepend_guidance(text, p.guidance)

        enc = self.tokenizer(
            inp, truncation=True, max_length=self.max_in_tokens, return_tensors="pt"
        )
        enc = {k: v.to(self.model.device if self.quantized else self.device) for k, v in enc.items()}

        # Auto length budgets from actual input token count
        in_len = int(enc["input_ids"].shape[1])
        min_new, max_new = self._budget_from_len(in_len, p.summary_ratio)

        gen_kwargs = dict(
            min_new_tokens=min_new,
            max_new_tokens=max_new,
            num_beams=p.num_beams,
            early_stopping=True,
            length_penalty=p.length_penalty,
            repetition_penalty=p.repetition_penalty,
            no_repeat_ngram_size=p.no_repeat_ngram_size,
            do_sample=p.do_sample,
        )
        if p.do_sample:
            gen_kwargs["temperature"] = p.temperature

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    # ---------- public API ----------
    @torch.no_grad()
    def summarize(
        self,
        text: str,
        *,
        params: Optional[GenParams] = None,
    ) -> str:
        if not text or not text.strip():
            return ""
        p = params or GenParams()
        return self._generate(text, p)

    @torch.no_grad()
    def summarize_long(
        self,
        text: str,
        *,
        params: Optional[GenParams] = None,
    ) -> str:
        if not text or not text.strip():
            return ""
        p = params or GenParams()

        # 1) map: sentence-aware, token-budget chunking
        chunks: List[str] = chunk_by_tokens(
            text,
            self.tokenizer,
            max_tokens=min(p.chunk_tokens, self.max_in_tokens),
            overlap=p.chunk_overlap,
        )
        if not chunks:
            return ""

        # 2) summarize each chunk with auto budgets (ratio-based)
        partials: List[str] = []
        for ch in chunks:
            partials.append(self._generate(ch, p))

        if len(partials) == 1:
            return partials[0]

        # 3) reduce: fuse partials, target a controlled final size
        fused_in = "\n\n".join(partials)

        # For the fuse step, keep beams high but shrink budgets to target
        fuse_p = GenParams(
            summary_ratio=p.summary_ratio,
            reduce_target_tokens=p.reduce_target_tokens,
            num_beams=max(5, p.num_beams),
            length_penalty=p.length_penalty,
            repetition_penalty=p.repetition_penalty,
            no_repeat_ngram_size=p.no_repeat_ngram_size,
            do_sample=p.do_sample,
            temperature=p.temperature,
            chunk_tokens=p.chunk_tokens,
            chunk_overlap=p.chunk_overlap,
            guidance=p.guidance,
        )

        # Manually override budgets for fuse pass
        def _fuse(text_in: str) -> str:
            inp = self._prepend_guidance(text_in, fuse_p.guidance)
            enc = self.tokenizer(
                inp, truncation=True, max_length=self.max_in_tokens, return_tensors="pt"
            )
            enc = {k: v.to(self.model.device if self.quantized else self.device) for k, v in enc.items()}

            gen_kwargs = dict(
                min_new_tokens=max(120, fuse_p.reduce_target_tokens - 80),
                max_new_tokens=min(512, fuse_p.reduce_target_tokens),
                num_beams=fuse_p.num_beams,
                early_stopping=True,
                length_penalty=fuse_p.length_penalty,
                repetition_penalty=fuse_p.repetition_penalty,
                no_repeat_ngram_size=fuse_p.no_repeat_ngram_size,
                do_sample=fuse_p.do_sample,
            )
            if fuse_p.do_sample:
                gen_kwargs["temperature"] = fuse_p.temperature

            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kwargs)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)

        # If fused input still too long, multi-stage reduce
        enc_len = len(self.tokenizer(fused_in, add_special_tokens=False).input_ids)
        if enc_len > self.max_in_tokens:
            # second-level map-reduce over partials
            mids: List[str] = []
            mid_chunks = chunk_by_tokens(
                fused_in,
                self.tokenizer,
                max_tokens=self.max_in_tokens,
                overlap=p.chunk_overlap,
            )
            for m in mid_chunks:
                mids.append(_fuse(m))
            fused_in = "\n\n".join(mids)

        return _fuse(fused_in)

    # Optional: change model at runtime (for UI dropdowns)
    def switch_model(self, model_name: str, **kwargs) -> None:
        self.__init__(model_name=model_name, device=str(self.device), **kwargs)
