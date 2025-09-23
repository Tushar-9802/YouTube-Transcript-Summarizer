# summarizer.py
from __future__ import annotations
from typing import Optional, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

from yt_sum.utils.chunker import chunk_text
from yt_sum.utils.keywords import (
    extract_keywords,
    select_evidence_sentences,
    extract_critical_terms,
)

# Optional embeddings (safe fallback if not installed)
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_ST = True
except Exception:
    _HAS_ST = False


_TEMPLATES = {
    "general": ["Context / Topic", "Key Points", "Important Numbers & Terms", "Takeaways"],
    "medical": [
        "Study/Context", "Population & Setting", "Interventions/Comparators",
        "Outcomes & Metrics", "Results & Effect Sizes", "Safety/Adverse Events", "Limitations"
    ],
    "engineering": [
        "Problem & Requirements", "System/Method", "Implementation Details",
        "Parameters & Metrics", "Results & Benchmarks", "Trade-offs", "Limitations"
    ],
    "scientific": [
        "Hypothesis/Objective", "Methodology", "Experimental Setup",
        "Results & Statistics", "Analysis", "Limitations"
    ],
}

_GUIDE = {
    "general": "Write a faithful, precise summary. Avoid speculation.",
    "medical": "For clinicians: preserve study design, cohorts, interventions, outcomes, effect sizes, CI, p-values, safety signals.",
    "engineering": "For engineers: preserve assumptions, design choices, algorithms, parameters, metrics, trade-offs, limitations.",
    "scientific": "For scientists: preserve hypotheses, methodology, experimental setup, results (with uncertainty), and limitations.",
}

_DOMAIN_ONTO = {
    "medical": ["RCT", "placebo", "double-blind", "randomized", "dose", "mg", "CI", "p-value", "hazard ratio", "odds ratio", "RR", "AE", "SAE", "biomarker", "phase I", "phase II", "phase III", "endpoint"],
    "engineering": ["latency", "throughput", "bandwidth", "Hz", "kHz", "MHz", "GHz", "Watts", "FPGA", "ASIC", "GPU", "CPU", "CUDA", "FLOPS", "complexity", "O(n)", "cache", "pipeline", "benchmark"],
    "scientific": ["hypothesis", "p-value", "confidence interval", "variance", "standard deviation", "effect size", "ANOVA", "regression", "correlation", "control group", "treatment group", "replicate", "blinded"],
}

_MODEL_LADDER = [
    (32, "google/long-t5-tglobal-large", 16384),
    (24, "allenai/led-large-16384-arxiv", 16384),
    (16, "google/long-t5-tglobal-base", 16384),
    (12, "allenai/led-base-16384", 16384),
    (8,  "google/pegasus-large", 1024),
    (6,  "facebook/bart-large-cnn", 1024),
]
_DEFAULT_MODEL = "facebook/bart-large-cnn"
_DEFAULT_MAXLEN = 1024


def _detect_vram_gb() -> float:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return 0.0
    return 0.0


def _choose_model_name(explicit: Optional[str]) -> Tuple[str, int]:
    if explicit:
        return explicit, _DEFAULT_MAXLEN
    vram = _detect_vram_gb()
    for min_gb, name, mlen in _MODEL_LADDER:
        if vram >= min_gb:
            return name, mlen
    return _DEFAULT_MODEL, _DEFAULT_MAXLEN


class _Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.available = _HAS_ST
        if not self.available:
            self.model = None
            return
        try:
            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = None
            self.available = False

    def top_k(self, query: str, sentences: List[str], k: int = 8) -> List[str]:
        if not self.available or not self.model or not sentences:
            return []
        try:
            q_emb = self.model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            s_emb = self.model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
            scores = st_util.cos_sim(q_emb, s_emb)[0]
            topk = torch.topk(scores, min(k, len(sentences)))
            idxs = topk.indices.tolist()
            return [sentences[i] for i in idxs]
        except Exception:
            return []


class Summarizer:
    """
    Hybrid extractive+abstractive hierarchical summarizer:
      - VRAM-adaptive model selection (LED/LongT5/Pegasus/BART) + optional 8-bit
      - Evidence-augmented prompts (keyword/MMR + optional embedding rerank)
      - Ontology-guided retention of numbers/units/acronyms/formulae
      - Domain templates/slots; optional IMRaD prefix
      - Optional: compression ratio, audience tone, output language
    """

    def __init__(self, model_name: Optional[str] = None, use_8bit: Optional[bool] = None):
        chosen, maxlen_hint = _choose_model_name(model_name)
        self.model_name = chosen
        self._maxlen_hint = maxlen_hint

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if use_8bit is None:
            use_8bit = torch.cuda.is_available()
        self.quantized = False
        if use_8bit and torch.cuda.is_available():
            try:
                qconf = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name, quantization_config=qconf, device_map="auto"
                )
                self.quantized = True
            except Exception:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        if torch.cuda.is_available() and not self.quantized:
            try:
                self.model = self.model.half().to("cuda")
            except Exception:
                self.model = self.model.to("cuda")

        self.chunk_tokens = 900
        self.chunk_overlap = 120
        mlen = getattr(self.tokenizer, "model_max_length", self._maxlen_hint)
        if mlen is None or mlen > 100000:
            mlen = self._maxlen_hint
        self.max_input_tokens = int(mlen)

        self._embedder = _Embedder()

    def _slots_for(self, domain: str, imrad: bool) -> List[str]:
        slots = list(_TEMPLATES.get(domain.lower(), _TEMPLATES["general"]))
        if imrad:
            imrad_slots = ["Introduction", "Methods", "Results", "Discussion"]
            remain = [s for s in slots if s not in set(imrad_slots)]
            return imrad_slots + remain
        return slots

    def _audience_tone(self, audience: str) -> str:
        a = (audience or "expert").lower()
        if a.startswith("stud"):
            return "Write for students; explain briefly, keep technical terms but add clarity where needed."
        return "Write for experts; keep dense technical phrasing and precise terminology."

    def _compression_hint(self, compression_ratio: Optional[float]) -> str:
        if not compression_ratio:
            return "Aim for a concise but comprehensive summary."
        r = max(0.05, min(0.8, float(compression_ratio)))
        return f"Target a compression ratio of approximately {int(r*100)}% of the source length."

    def _make_prompt(
    self,
    domain: str,
    imrad: bool,
    retention: List[str],
    evidence: List[str],
    audience: str,
    compression_ratio: Optional[float],
    output_language: Optional[str] = None,
) -> str:
        guide = _GUIDE.get(domain.lower(), _GUIDE["general"])
        slots = self._slots_for(domain, imrad)

        must_keep = ", ".join(retention[:40]) if retention else ""
        ev = "\n- ".join(evidence[:10]) if evidence else ""
        tone = self._audience_tone(audience)
        comp = self._compression_hint(compression_ratio)
        lang = (output_language or "").strip().lower()

        lang_hint = ""
        if lang in {"en", "english"}:
            lang_hint = "Write the summary in English."
        elif lang in {"source", "original", "same"}:
            lang_hint = "Write the summary in the same language as the source."

        # Construct a structured prompt
        return (
        f"You are preparing a structured, domain-aware summary.\n"
        f"Domain guidance: {guide}\n"
        f"Tone: {tone}\n"
        f"Length: {comp}\n"
        f"{lang_hint}\n\n"
        f"Ensure all important numbers, acronyms, formulas, and key terms are preserved verbatim. "
        f"{'Also ensure inclusion of: ' + must_keep if must_keep else ''}\n\n"
        f"Ground the summary in these supporting sentences (do not copy verbatim; paraphrase faithfully):\n- {ev}\n\n"
        f"Now write the summary using the following structure. Fill each section with content from the source:\n\n"
        + "\n\n".join([f"{slot}:\n" for slot in slots])
    )


    def _generate(self, text: str, min_len: Optional[int], max_len: Optional[int]) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        inp_len = int(inputs["input_ids"].shape[1])
        tgt_max = max_len or min(950, max(350, int(inp_len * 0.33)))
        tgt_min = min_len or (160 if inp_len > 600 else 90)
        gen = dict(
            num_beams=5, do_sample=False, no_repeat_ngram_size=3,
            length_penalty=1.0, early_stopping=True,
            min_length=tgt_min, max_length=tgt_max
        )
        if getattr(self.model.config, "model_type", "") == "led" and "attention_mask" in inputs:
            import torch as _t
            gmask = _t.zeros_like(inputs["attention_mask"])
            gmask[:, 0] = 1
            gen["global_attention_mask"] = gmask
        out_ids = self.model.generate(**inputs, **gen)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    def _build_global_evidence(self, transcript: str) -> Tuple[List[str], List[str]]:
        terms = extract_critical_terms(transcript)
        kws = extract_keywords(transcript, top_n=20, method="auto", max_ngram=3)
        ev = select_evidence_sentences(transcript, kws, top_k=16, diversity=0.35)
        return terms, ev

    def _rerank_with_embeddings(self, query: str, candidates: List[str], k: int = 8) -> List[str]:
        if not candidates:
            return []
        if not self._embedder.available:
            return candidates[:k]
        ranked = self._embedder.top_k(query, candidates, k=k)
        return ranked if ranked else candidates[:k]

    def summarize_long(
        self,
        transcript: str,
        *,
        domain: str = "general",
        imrad: bool = False,
        refinement: bool = True,
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
        chunk_tokens: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        compression_ratio: Optional[float] = None,
        audience: str = "expert",
        output_language: Optional[str] = None,
    ) -> str:
        txt = (transcript or "").strip()
        if not txt:
            return ""

        if chunk_tokens:
            self.chunk_tokens = int(chunk_tokens)
        if chunk_overlap:
            self.chunk_overlap = int(chunk_overlap)

        global_terms, global_evidence = self._build_global_evidence(txt)
        onto = _DOMAIN_ONTO.get(domain.lower(), [])
        onto_in_text = [t for t in onto if t in txt]
        retention_global = list(dict.fromkeys((global_terms[:30] + onto_in_text)))[:40]

        chunks = chunk_text(txt, self.tokenizer, self.chunk_tokens, self.chunk_overlap)

        partials: List[str] = []
        for ch in chunks:
            local_kws = extract_keywords(ch, top_n=12, method="auto", max_ngram=3)
            ev_cand = select_evidence_sentences(ch, local_kws, top_k=10, diversity=0.35)
            q = f"{domain} {local_kws[:6]}"
            ev = self._rerank_with_embeddings(str(q), ev_cand, k=8)
            retention = list(dict.fromkeys(retention_global + extract_critical_terms(ch)))[:40]

            prompt = self._make_prompt(
                domain=domain,
                imrad=imrad,
                retention=retention,
                evidence=ev,
                audience=audience,
                compression_ratio=compression_ratio,
                output_language=output_language,
            )
            src = f"{prompt}\n\nSource:\n{ch}\n\nSummary:"
            try:
                s = self._generate(src, min_len, max_len)
                s = self._clean_output(s, self._slots_for(domain, imrad))
            except Exception:
                s = ch[:800] + ("..." if len(ch) > 800 else "")
            partials.append(s.strip())

        combined = "\n\n".join(partials)

        reduce_prompt = self._make_prompt(
            domain=domain,
            imrad=imrad,
            retention=retention_global,
            evidence=self._rerank_with_embeddings("global", global_evidence, k=10)
                     or global_evidence[:10],
            audience=audience,
            compression_ratio=compression_ratio,
            output_language=output_language,
        )
        reduce_src = f"{reduce_prompt}\n\nSegment Summaries:\n{combined}\n\nUnified Summary:"
        try:
            final = self._generate(reduce_src, min_len, max_len)
            final = self._clean_output(final, self._slots_for(domain, imrad))
        except Exception:
            final = combined

        if refinement:
            ref_src = (
                f"{reduce_prompt}\n"
                f"Improve coherence, remove redundancy, and retain key numbers/units/acronyms/formulae verbatim.\n"
                f"Draft:\n{final}\n\nRefined Summary:"
            )
            try:
                refined = self._generate(ref_src, min_len, max_len)
                final = self._clean_output(refined, self._slots_for(domain, imrad))
            except Exception:
                pass

        return final.strip()
