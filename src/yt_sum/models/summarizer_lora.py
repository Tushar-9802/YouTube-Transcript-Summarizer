# src/yt_sum/models/summarizer_lora.py
"""
LoRA-aware summarizer - backward compatible with existing code.
Drop-in replacement with adapter support.
"""

from __future__ import annotations
from typing import Optional, List
from pathlib import Path
import torch
import gc
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.yt_sum.utils.chunker import chunk_text as _token_chunk
from src.yt_sum.utils.logging import get_logger

logger = get_logger("summarizer_lora")


class SummarizerWithLoRA:
    """
    Mistral 7B + Optional LoRA adapters.
    
    Compatible with existing Summarizer API.
    """
    
    def __init__(
        self,
        domain: str = "general",
        use_8bit: bool = True,
        adapter_path: Optional[str] = None,
    ):
        self.domain = domain
        self.adapter_path = adapter_path
        self.adapter_loaded = False
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        logger.info("Loading Mistral 7B...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ) if use_8bit and self.device == "cuda" else None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load adapter if specified
        if adapter_path:
            self._load_adapter(adapter_path)
        
        # Chunk settings
        self.chunk_tokens = 1024
        self.chunk_overlap = 150
    
    def _load_adapter(self, adapter_path: str):
        """Load LoRA adapter"""
        path = Path(adapter_path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        
        logger.info(f"Loading adapter: {path.name}")
        
        self.model = PeftModel.from_pretrained(
            self.model,
            str(path),
            is_trainable=False
        )
        self.adapter_loaded = True
        logger.info("✓ Adapter loaded")
    
    def summarize_long(
        self,
        transcript: str,
        compression_ratio: float = 0.2,
        **kwargs
    ) -> str:
        """Main summarization method - compatible with existing API"""
        
        if not transcript.strip():
            return ""
        
        # Chunk transcript
        chunks = _token_chunk(
            transcript,
            tokenizer=self.tokenizer,
            chunk_tokens=self.chunk_tokens,
            chunk_overlap=self.chunk_overlap
        )
        
        if not chunks:
            return "Unable to generate summary"
        
        logger.info(f"Processing {len(chunks)} chunks ({'LoRA' if self.adapter_loaded else 'base'})")
        
        # Generate per-chunk summaries
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
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract after [/INST]
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