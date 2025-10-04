# src/yt_sum/models/summarizer.py
from __future__ import annotations
from typing import Optional, List, Tuple
import os
import torch
import platform
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

from src.yt_sum.utils.chunker import chunk_text as _token_chunk
from src.yt_sum.utils.logging import get_logger

logger = get_logger("summarizer", level="INFO")

# Domain templates
_TEMPLATES = {
    "general": ["Context", "Key Points", "Takeaways"],
    "medical": ["Background", "Methods", "Results", "Clinical Implications", "Limitations"],
    "engineering": ["Problem", "Technical Approach", "Implementation", "Results", "Trade-offs"],
    "scientific": ["Research Question", "Methodology", "Findings", "Discussion", "Limitations"],
}

_GUIDE = {
    "general": "Summarize ONLY what is explicitly stated. Do NOT add external facts or speculation.",
    "medical": "Extract clinical facts verbatim. Include exact numbers, drug names, outcomes.",
    "engineering": "Document technical decisions and results as stated. Preserve specifications.",
    "scientific": "Report methodology and findings exactly. Include statistical values.",
}

_MODEL_CACHE: dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}


def _slots(domain: str, imrad: bool) -> List[str]:
    s = list(_TEMPLATES.get(domain, _TEMPLATES["general"]))
    if imrad:
        im = ["Introduction", "Methods", "Results", "Discussion"]
        s = im + [x for x in s if x not in set(im)]
    return s


def _clean(text: str) -> str:
    """Remove LLM preambles."""
    if not text:
        return ""
    bad = ("Sure", "Here is", "Here's", "Of course", "I will", "I can", "As an AI",
           "Based on", "According to", "The transcript", "The video", "This video")
    t = text.strip()
    for b in bad:
        if t.startswith(b):
            idx = t.find('\n')
            if 0 < idx < 100:
                t = t[idx+1:].strip()
            else:
                t = t[len(b):].lstrip(" :,-")
    return t.strip()


class Summarizer:
    """
    Mistral 7B summarizer with aggressive quantization for 8GB VRAM.
    
    KEY OPTIMIZATIONS:
    - Forces 4-bit NF4 quantization on <=16GB VRAM
    - Optimized chunk sizes for 32K context window
    - Greedy decoding for consistency
    - Proper memory management
    """
    
    def __init__(
        self,
        domain: str = "general",
        summarizer_model: Optional[str] = None,
        use_8bit: bool = True,
    ):
        self.domain = (domain or "general").lower()
        
        # ALWAYS use Mistral 7B
        self.model_name = summarizer_model or "mistralai/Mistral-7B-Instruct-v0.2"
        
        # Detect VRAM
        vram_gb = 0
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_memory / (1024 ** 3), 1)
            except Exception:
                pass
        
        # Cache key includes quantization
        cache_key = f"{self.model_name}_q{use_8bit}_v{vram_gb}"
        
        if cache_key in _MODEL_CACHE:
            logger.info(f"Using cached model: {cache_key}")
            self.tokenizer, self.model = _MODEL_CACHE[cache_key]
            self._set_chunk_defaults(vram_gb)
            return
        
        # Load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Mistral 7B on {self.device} ({vram_gb:.1f}GB VRAM)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            use_fast=True, 
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # AGGRESSIVE quantization setup
        quant_config = None
        allow_bnb = _HAS_BNB and platform.system() != "Windows"
        
        # FORCE quantization on <=16GB (Mistral 7B needs ~14GB FP16)
        must_quantize = (
            self.device == "cuda" and
            vram_gb > 0 and
            vram_gb <= 16
        )
        
        want_quantize = (
            self.device == "cuda" and
            allow_bnb and
            (use_8bit or must_quantize)
        )
        
        if want_quantize:
            try:
                # 4-bit NF4 with double quantization (most aggressive)
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info(f"Loading with 4-bit NF4 quantization (reduces ~14GB → ~4GB)")
            except Exception as e:
                logger.warning(f"4-bit failed: {e}, trying 8-bit")
                try:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info(f"Loading with 8-bit quantization (~7GB)")
                except Exception:
                    logger.warning("Quantization unavailable, loading FP16 (~14GB)")
                    quant_config = None
        
        # Load model with OOM protection
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                quantization_config=quant_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            logger.info(f"Mistral 7B loaded successfully")
        except torch.cuda.OutOfMemoryError:
            logger.error("OOM during model load")
            raise RuntimeError(
                f"Insufficient VRAM for Mistral 7B. Need at least 5GB free with quantization. "
                "Current VRAM: {vram_gb:.1f}GB. Try closing other GPU applications."
            )
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
        
        # Set chunk sizes for Mistral's 32K context
        self._set_chunk_defaults(vram_gb)
        logger.info(f"Chunk settings: {self.chunk_tokens} tokens, {self.chunk_overlap} overlap")
        
        # Cache
        _MODEL_CACHE[cache_key] = (self.tokenizer, self.model)
    
    def _set_chunk_defaults(self, vram_gb: float = 0):
        """VRAM-adaptive chunk sizes for Mistral 7B (32K context)."""
        if vram_gb <= 8:
            # Conservative for 8GB: smaller chunks, more passes
            self.chunk_tokens, self.chunk_overlap = 1200, 150
        elif vram_gb <= 12:
            # Balanced for 12GB
            self.chunk_tokens, self.chunk_overlap = 1800, 200
        else:
            # Aggressive for 16GB+: leverage full context
            self.chunk_tokens, self.chunk_overlap = 2400, 250
    
    def _build_prompt(
        self,
        text: str,
        *,
        imrad: bool,
        audience: str,
        compression_ratio: float,
        output_language: Optional[str],
    ) -> list:
        """Build Mistral-optimized prompt."""
        slots = _slots(self.domain, imrad)
        guide = _GUIDE.get(self.domain, _GUIDE["general"])
        comp = max(0.05, min(0.8, float(compression_ratio or 0.2)))
        lang = output_language or "source"
        
        system = (
            "You are a faithful summarizer. CRITICAL RULES:\n"
            "1. Use ONLY information in the transcript\n"
            "2. Never add external knowledge\n"
            "3. Never speculate\n"
            "4. Preserve exact numbers, names, terms\n"
            "5. If uncertain, omit\n"
            "6. No preambles or meta-commentary\n"
            "7. State facts directly"
        )
        
        user = (
            f"Domain: {self.domain}\n"
            f"Audience: {audience}\n"
            f"Guidelines: {guide}\n"
            f"Target length: ~{int(comp*100)}% of source\n"
            f"Language: {lang}\n\n"
            f"Structure with these sections:\n"
            + "\n".join([f"## {s}" for s in slots])
            + "\n\n---TRANSCRIPT---\n"
            + text
            + "\n\nProvide faithful summary:"
        )
        
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    
    def _generate(
        self,
        messages: list,
        *,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate with greedy decoding."""
        # Mistral uses chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = "\n\n".join([m["content"] for m in messages])
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=30000  # Leave room for 32K context
        ).to(self.model.device)
        
        # Greedy decoding
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": False,  # Greedy
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response after prompt (Mistral format)
        if "[/INST]" in text:
            text = text.split("[/INST]")[-1]
        elif "\n\nProvide faithful summary:" in text:
            text = text.split("\n\nProvide faithful summary:")[-1]
        
        return _clean(text)
    
    def summarize_long(
        self,
        transcript: str,
        *,
        imrad: bool = False,
        refinement: bool = True,
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
        chunk_tokens: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        compression_ratio: Optional[float] = 0.2,
        audience: str = "expert",
        output_language: Optional[str] = None,
        duration_seconds: Optional[int] = None,
    ) -> str:
        """Summarize with streaming processing."""
        text = (transcript or "").strip()
        if not text:
            return ""
        
        ctoks = int(chunk_tokens or self.chunk_tokens)
        cover = int(chunk_overlap or self.chunk_overlap)
        
        chunks = _token_chunk(text, tokenizer=self.tokenizer, chunk_tokens=ctoks, chunk_overlap=cover)
        
        if not chunks or (len(chunks) == 1 and not chunks[0].strip()):
            logger.warning("No valid chunks")
            return "Unable to generate summary: transcript too short"
        
        per_chunk_tokens = min(768, max(384, int(ctoks * (compression_ratio or 0.2))))
        
        logger.info(f"Processing {len(chunks)} chunks ({per_chunk_tokens} tokens each)")
        
        partials: List[str] = []
        for i, ch in enumerate(chunks, 1):
            if not ch.strip():
                continue
            
            logger.info(f"Chunk {i}/{len(chunks)}")
            
            msgs = self._build_prompt(
                ch,
                imrad=imrad,
                audience=audience,
                compression_ratio=(compression_ratio or 0.2),
                output_language=output_language,
            )
            
            try:
                s = self._generate(msgs, max_new_tokens=per_chunk_tokens)
                if s:
                    partials.append(s)
            except Exception as e:
                logger.warning(f"Chunk {i} failed: {e}")
        
        if not partials:
            return "Summary generation failed"
        
        final = partials[0] if len(partials) == 1 else "\n\n".join(partials)
        
        # Refinement for multi-chunk
        if refinement and len(partials) > 1:
            logger.info("Running refinement pass")
            refine_tokens = max(768, min(1536, per_chunk_tokens * len(partials) // 2))
            
            msgs = [
                {
                    "role": "system",
                    "content": (
                        "Refine for coherence. "
                        "Remove redundancy but DO NOT add new information. "
                        "Maintain all facts exactly."
                    )
                },
                {
                    "role": "user",
                    "content": f"Refine:\n\n{final}"
                },
            ]
            
            try:
                final = self._generate(msgs, max_new_tokens=refine_tokens)
            except Exception as e:
                logger.warning(f"Refinement failed: {e}")
        
        return final.strip()
    
    def unload(self):
        """Unload model and clear cache."""
        try:
            global _MODEL_CACHE
            cache_keys_to_remove = [k for k in list(_MODEL_CACHE.keys()) if self.model_name in k]
            for key in cache_keys_to_remove:
                logger.info(f"Removing {key} from cache")
                del _MODEL_CACHE[key]

            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
            gc.collect()

            logger.info("Summarizer unloaded")
        except Exception as e:
            logger.warning(f"Unload error: {e}")