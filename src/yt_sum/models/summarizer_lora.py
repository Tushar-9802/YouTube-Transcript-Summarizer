# src/yt_sum/models/summarizer_lora.py
from __future__ import annotations
from typing import Optional, List
from pathlib import Path
import torch
import gc
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.yt_sum.utils.chunker import chunk_text as _token_chunk
from src.yt_sum.utils.logging import get_logger

logger = get_logger("summarizer_lora")

try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("bitsandbytes not available - using FP16 loading")


class SummarizerWithLoRA:
    """
    Mistral 7B + Optional LoRA adapters.
    Windows-safe with disk offloading for 8GB VRAM.
    """
    
    def __init__(
        self,
        domain: str = "general",
        use_8bit: bool = False,
        adapter_path: Optional[str] = None,
    ):
        self.domain = domain
        self.adapter_path = adapter_path
        self.adapter_loaded = False
        self.use_8bit = use_8bit and BITSANDBYTES_AVAILABLE
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading Mistral 7B ({'4-bit' if self.use_8bit else 'FP16'})...")
        
        bnb_config = None
        if self.use_8bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        
        # Offload directory
        offload_dir = Path("./offload_cache")
        offload_dir.mkdir(exist_ok=True)
        
        # FIXED: Only valid parameters
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            max_memory={0: "6GB", "cpu": "16GB"} if self.device == "cuda" else None,
            offload_folder=str(offload_dir)  # Only this parameter
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if adapter_path:
            self._load_adapter(adapter_path)
        
        self.chunk_tokens = 1024
        self.chunk_overlap = 150
    
    def _load_adapter(self, adapter_path: str):
        """Load LoRA adapter with disk offloading support"""
        path = Path(adapter_path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        
        logger.info(f"Loading adapter: {path.name}")
        
        offload_dir = Path("./offload_cache")
        offload_dir.mkdir(exist_ok=True)
        
        # FIXED: Use only offload_folder parameter
        self.model = PeftModel.from_pretrained(
            self.model,
            str(path),
            is_trainable=False,
            offload_folder=str(offload_dir)
        )
        self.adapter_loaded = True
        logger.info("✓ Adapter loaded")
    
    def summarize_long(
        self,
        transcript: str,
        compression_ratio: float = 0.2,
        max_summary_length: int = 256,
        **kwargs
    ) -> str:
        """Main summarization method"""
        
        if not transcript.strip():
            return ""
        
        chunks = _token_chunk(
            transcript,
            tokenizer=self.tokenizer,
            chunk_tokens=self.chunk_tokens,
            chunk_overlap=self.chunk_overlap
        )
        
        if not chunks:
            return "Unable to generate summary"
        
        logger.info(f"Processing {len(chunks)} chunks ({'LoRA' if self.adapter_loaded else 'base'})")
        
        partials = []
        for i, chunk in enumerate(chunks, 1):
            if not chunk.strip():
                continue
            
            prompt = f"<s>[INST] Summarize this {self.domain} text:\n\n{chunk} [/INST]"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_summary_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "[/INST]" in summary:
                summary = summary.split("[/INST]")[-1].strip()
            
            partials.append(summary)
        
        return "\n\n".join(partials) if partials else "Summary generation failed"
    
    def unload(self):
        """Cleanup"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()