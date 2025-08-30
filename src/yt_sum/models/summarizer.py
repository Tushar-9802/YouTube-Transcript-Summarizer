# src/yt_sum/models/summarizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from yt_sum.utils.chunker import chunk_by_tokens

# (Optional) friendly aliases; raw HF IDs still work.
MODEL_ALIASES: Dict[str, str] = {
    "bart-cnn": "facebook/bart-large-cnn",
    "t5-large": "t5-large",
}

def _resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)

# ----------------------------
# Generation parameter bundle
# ----------------------------
@dataclass
class GenParams:
    # Length control
    summary_ratio: float = 0.18          # ~% of input tokens to keep per chunk
    reduce_target_tokens: int = 320      # final fuse target (tokens)
    # Decoding (research-friendly defaults)
    num_beams: int = 5
    length_penalty: float = 0.9          # <1.0 biases toward longer outputs
    repetition_penalty: float = 1.08
    no_repeat_ngram_size: int = 3
    do_sample: bool = False
    temperature: float = 1.0
    # Chunking
    chunk_tokens: int = 900              # safe headroom under 1024 typical limit
    chunk_overlap: int = 120
    # Optional guidance (light steering)
    guidance: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "GenParams":
        if not d:
            return cls()
        # Only accept known keys; ignore extras gracefully
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fields})

# ----------------------------
# Summarizer (single-model)
# ----------------------------
class BaselineSummarizer:
    """
    One-model, research-friendly summarizer.

    - summarize():     single-shot for <= model context
    - summarize_long(): map→reduce for long transcripts with ratio-based budgets

    Design choices:
      * Deterministic decoding (no sampling) unless you turn it on
      * 8-bit optional loading on CUDA (VRAM-friendly)
      * Backward-compatible with earlier numeric-arg signature
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
        use_8bit: bool = True,
        device_map: str = "auto",
    ):
        self.model_name = _resolve_model_name(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # Pick device
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Try an 8-bit load on CUDA for memory savings. Fall back transparently.
        self.quantized = False
        try:
            if use_8bit and self.device.type == "cuda":
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_cfg,
                    device_map=device_map,
                )
                self.quantized = True
            else:
                raise RuntimeError("8-bit not requested or not on CUDA")
        except Exception:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)

        self.model.eval()

        # Reproducibility: deterministic kernels + seed
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(42)

        # Safe encoder input cap; keep small margin for special tokens
        max_ctx = getattr(getattr(self.model, "config", None), "max_position_embeddings", 1024)
        cfg = getattr(self.model, "config", None)
        max_len = (
        getattr(cfg, "max_position_embeddings", None)
        or getattr(cfg, "max_encoder_positions", None)  # some models (e.g., BART) expose this
        or 1024
        )
        # keep a little headroom for special tokens
        self.max_in_tokens = max(512, int(max_len) - 32)    

    # ---------------- internal helpers ----------------
    def _prepend_guidance(self, text: str, guidance: Optional[str]) -> str:
        if not guidance:
            return text
        # Simple instruction prefix (works well for BART/T5/LED)
        return f"Instruction: {guidance}\n\nTranscript:\n{text}"

    def _budget_from_len(self, in_len: int, ratio: float) -> Tuple[int, int]:
        """
        Compute min/max new tokens given the *input* token length.
        Heuristic: ensure enough room to be informative but not ramble.
        """
        target = max(120, int(in_len * ratio))
        max_new = min(512, max(target, 220))
        min_new = max(100, min(max_new - 80, int(target * 0.7)))
        return min_new, max_new

    def _generate_once(self, text: str, p: GenParams) -> str:
        inp = self._prepend_guidance(text, p.guidance)
        enc = self.tokenizer(
            inp, truncation=True, max_length=self.max_in_tokens, return_tensors="pt"
        )
        dev = self.model.device if self.quantized else self.device
        enc = {k: v.to(dev) for k, v in enc.items()}

        in_len = int(enc["input_ids"].shape[1])
        min_new, max_new = self._budget_from_len(in_len, p.summary_ratio)

        gen_kwargs = dict(
            min_new_tokens=min_new,
            max_new_tokens=max_new,
            num_beams=max(1, int(p.num_beams)),
            early_stopping=True,
            length_penalty=float(p.length_penalty),
            repetition_penalty=float(p.repetition_penalty),
            no_repeat_ngram_size=int(p.no_repeat_ngram_size),
            do_sample=bool(p.do_sample),
        )
        if p.do_sample:
            gen_kwargs["temperature"] = float(p.temperature)

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    # --------------- public API ---------------
    @torch.no_grad()
    def summarize(
        self,
        text: str,
        *,
        params: Optional[GenParams] = None,
        # --- legacy numeric knobs (kept for backward-compat) ---
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Single-shot summarization. Prefer summarize_long() for long transcripts.
        If legacy numeric args are provided, they override p where applicable.
        """
        if not text or not text.strip():
            return ""
        p = params or GenParams()

        # Legacy overrides (optional)
        if num_beams is not None:           p.num_beams = num_beams
        if length_penalty is not None:      p.length_penalty = length_penalty
        if repetition_penalty is not None:  p.repetition_penalty = repetition_penalty
        if no_repeat_ngram_size is not None:p.no_repeat_ngram_size = no_repeat_ngram_size
        if do_sample is not None:           p.do_sample = do_sample
        if temperature is not None:         p.temperature = temperature

        # If explicit budgets are given, temporarily replace ratio-based budget
        if (max_new_tokens is not None) or (min_new_tokens is not None):
            def _fixed_generate(txt: str) -> str:
                inp = self._prepend_guidance(txt, p.guidance)
                enc = self.tokenizer(inp, truncation=True, max_length=self.max_in_tokens, return_tensors="pt")
                dev = self.model.device if self.quantized else self.device
                enc = {k: v.to(dev) for k, v in enc.items()}
                gk = dict(
                    max_new_tokens=max_new_tokens or 160,
                    min_new_tokens=min_new_tokens or max(0, (max_new_tokens or 160) - 80),
                    num_beams=max(1, int(p.num_beams)),
                    early_stopping=True,
                    length_penalty=float(p.length_penalty),
                    repetition_penalty=float(p.repetition_penalty),
                    no_repeat_ngram_size=int(p.no_repeat_ngram_size),
                    do_sample=bool(p.do_sample),
                )
                if p.do_sample:
                    gk["temperature"] = float(p.temperature)
                with torch.no_grad():
                    out = self.model.generate(**enc, **gk)
                return self.tokenizer.decode(out[0], skip_special_tokens=True)
            return _fixed_generate(text)

        return self._generate_once(text, p)

    @torch.no_grad()
    def summarize_long(
        self,
        text: str,
        *,
        params: Optional[GenParams] = None,
        # --- legacy numeric knobs (kept for backward-compat) ---
        chunk_tokens: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        per_chunk_max: Optional[int] = None,
        per_chunk_min: Optional[int] = None,
        fuse_max: Optional[int] = None,
        fuse_min: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> str:
        """
        Map→reduce summarization:
          1) Split into token-budget chunks with overlap.
          2) Summarize each chunk (ratio-based budgets).
          3) Fuse chunk summaries into a long, coherent final summary.
        """
        if not text or not text.strip():
            return ""
        p = params or GenParams()

        # Legacy overrides (optional)
        if chunk_tokens is not None:    p.chunk_tokens = min(int(chunk_tokens), self.max_in_tokens)
        if chunk_overlap is not None:   p.chunk_overlap = int(chunk_overlap)
        if num_beams is not None:       p.num_beams = int(num_beams)

        # 1) chunk
        chunks: List[str] = chunk_by_tokens(
            text,
            self.tokenizer,
            max_tokens=min(p.chunk_tokens, self.max_in_tokens),
            overlap=p.chunk_overlap,
        )
        if not chunks:
            return ""

        # 2) summarize each chunk (ratio budgets)
        partials: List[str] = [self._generate_once(ch, p) for ch in chunks]
        if len(partials) == 1:
            return partials[0]

        # 3) fuse summaries (explicit fuse budgets if provided; else target reduce_target_tokens)
        fused_in = "\n\n".join(partials)

        def _fuse(text_in: str) -> str:
            inp = self._prepend_guidance(text_in, p.guidance)
            enc = self.tokenizer(inp, truncation=True, max_length=self.max_in_tokens, return_tensors="pt")
            dev = self.model.device if self.quantized else self.device
            enc = {k: v.to(dev) for k, v in enc.items()}

            # Either honor legacy fuse_* or use reduce_target_tokens
            min_new = (fuse_min if fuse_min is not None else max(120, p.reduce_target_tokens - 80))
            max_new = (fuse_max if fuse_max is not None else min(512, p.reduce_target_tokens))

            gk = dict(
                min_new_tokens=int(min_new),
                max_new_tokens=int(max_new),
                num_beams=max(5, int(p.num_beams)),
                early_stopping=True,
                length_penalty=float(p.length_penalty),
                repetition_penalty=float(p.repetition_penalty),
                no_repeat_ngram_size=int(p.no_repeat_ngram_size),
                do_sample=bool(p.do_sample),
            )
            if p.do_sample:
                gk["temperature"] = float(p.temperature)

            with torch.no_grad():
                out = self.model.generate(**enc, **gk)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)

        # If fused text overflows encoder, reduce in stages
        enc_len = len(self.tokenizer(fused_in, add_special_tokens=False)["input_ids"])
        if enc_len > self.max_in_tokens:
            mids: List[str] = []
            for m in chunk_by_tokens(fused_in, self.tokenizer, max_tokens=self.max_in_tokens, overlap=p.chunk_overlap):
                mids.append(_fuse(m))
            fused_in = "\n\n".join(mids)

        return _fuse(fused_in)

    # Convenience if you want to hot-swap models from the UI
    def switch_model(self, model_name: str, **kwargs) -> None:
        self.__init__(model_name=model_name, device=str(self.device), **kwargs)
