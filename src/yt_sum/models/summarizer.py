from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

# Domain-specific guidance
DOMAIN_GUIDANCE = {
    "general": "Write a faithful, comprehensive summary.",
    "medical": (
        "Summarize faithfully for medical professionals. "
        "Preserve clinical endpoints, patient characteristics, interventions, comparators, "
        "outcomes, statistics (e.g., CI, p-values), and safety signals. Avoid speculation."
    ),
    "engineering": (
        "Summarize faithfully for engineers. "
        "Preserve assumptions, design choices, algorithms, parameters, metrics, trade-offs, "
        "and limitations. Avoid speculation."
    ),
    "scientific": (
        "Summarize faithfully for scientists. "
        "Preserve hypotheses, datasets, methodology, controls, results, effect sizes, "
        "uncertainty, and limitations. Avoid speculation."
    ),
}


class Summarizer:
    """
    GPU-first summarizer with:
      - domain-aware guidance
      - map→reduce chunking
      - optional refinement pass
      - min_len / max_len support (from UI sliders)
    """

    def __init__(self, model_name: str | None = None, use_8bit: bool | None = None):
        self.model_name = model_name or self._auto_model_name()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if use_8bit is None:
            use_8bit = torch.cuda.is_available()

        self.quantized = False
        if use_8bit and torch.cuda.is_available():
            try:
                q = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name, quantization_config=q, device_map="auto"
                )
                self.quantized = True
            except Exception:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        if torch.cuda.is_available() and not self.quantized:
            try:
                self.model = self.model.half()
            except Exception:
                pass
            self.model.to("cuda")

        torch.manual_seed(42)
        max_len = getattr(self.tokenizer, "model_max_length", 1024)
        self.max_in = 1024 if (max_len is None or max_len > 100_000) else int(max_len)

        # Default chunking
        self.chunk_tokens = 1200
        self.chunk_overlap = 200

    def _auto_model_name(self) -> str:
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 2**30
            if vram >= 14:
                return "allenai/led-large-16384-arxiv"
            if vram >= 8:
                return "allenai/led-base-16384"
        return "facebook/bart-large-cnn"

    # ------------------- public API -------------------

    def summarize_long(
        self,
        text: str,
        *,
        refinement: bool = True,
        domain: str = "general",
        min_len: int | None = None,
        max_len: int | None = None,
    ) -> str:
        """
        Summarize very long text via hierarchical map→reduce.
        - domain: influences guidance
        - refinement: do a second global pass
        - min_len / max_len: enforce length controls
        """
        if not text.strip():
            return ""

        # Stage 1: chunked summaries
        chunks = self._split_by_tokens(text, self.chunk_tokens, self.chunk_overlap)
        partials = [
            self._generate(c, domain=domain, min_len=min_len, max_len=max_len)
            for c in chunks
        ]
        fused = "\n\n".join(partials)

        # Stage 2: fuse into a mega-summary
        mega = self._generate(fused, domain=domain, min_len=min_len, max_len=max_len)

        # Stage 3: refinement
        if refinement:
            refined = self._refine_summary(
                mega, domain=domain, min_len=min_len, max_len=max_len
            )
            return refined
        return mega

    # ------------------- internals -------------------

    def _generate(
        self,
        text: str,
        *,
        domain: str,
        min_len: int | None = None,
        max_len: int | None = None,
        tighten: bool = False,
    ) -> str:
        """Generate summary with domain guidance + length control."""
        guidance = DOMAIN_GUIDANCE.get(domain.lower(), DOMAIN_GUIDANCE["general"])
        prompt = f"{guidance}\n\nSource:\n{text}"

        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_in
        )
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}

        in_len = enc["input_ids"].shape[1]
        # Use UI values if provided, else fallback
        max_len = max_len or min(800, max(300, int(in_len * 0.35)))
        min_len = min_len or (200 if in_len > 800 else 100)

        gen = {
            "num_beams": 6 if tighten else 5,
            "do_sample": False,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.2 if tighten else 0.9,
            "early_stopping": True,
            "max_length": max_len,
            "min_length": min_len,
        }

        # LED requires global attention
        if getattr(self.model.config, "model_type", "") == "led" and "attention_mask" in enc:
            gmask = torch.zeros_like(enc["attention_mask"])
            gmask[:, 0] = 1
            gen["global_attention_mask"] = gmask

        out_ids = self.model.generate(**enc, **gen)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    def _refine_summary(
        self,
        text: str,
        *,
        domain: str,
        min_len: int | None = None,
        max_len: int | None = None,
    ) -> str:
        """Second pass: expand, clarify, and improve cohesion."""
        prompt = (
            "Task: Improve cohesion and completeness of this summary. "
            "Expand explanations where terse, ensure no technical details are lost. "
            "Keep it faithful to the source.\n\nSummary:\n" + text
        )
        return self._generate(
            prompt, domain=domain, min_len=min_len, max_len=max_len, tighten=True
        )

    def _split_by_tokens(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        sents = [s.strip() for s in text.split(". ") if s.strip()]
        out, cur, cur_tok = [], [], 0
        for s in sents:
            s = s if s.endswith(".") else s + "."
            t = len(self.tokenizer.encode(s, add_special_tokens=False))
            if cur and cur_tok + t > max_tokens:
                out.append(" ".join(cur).strip())
                tail = self._tail_tokens(out[-1], overlap)
                cur, cur_tok = [tail, s], len(
                    self.tokenizer.encode(tail + s, add_special_tokens=False)
                )
            else:
                cur.append(s)
                cur_tok += t
        if cur:
            out.append(" ".join(cur).strip())
        return out

    def _tail_tokens(self, text: str, keep: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        tail = ids[-keep:] if len(ids) > keep else ids
        return self.tokenizer.decode(tail, skip_special_tokens=True)
