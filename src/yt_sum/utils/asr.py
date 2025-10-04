# src/yt_sum/utils/asr.py
"""PyTorch Whisper - Balanced anti-hallucination approach"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import torch
import gc
import whisper

logger = logging.getLogger("asr")


def transcribe_audio(
    audio_path: str,
    *,
    model_size: str = "small",
    transcript_dir: Optional[Path] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Balanced approach: Strong ASR parameters + minimal post-filtering.
    
    Strategy:
    1. ASR parameters prevent most hallucinations at source
    2. Post-filter ONLY removes extreme cases (5+ exact repeats)
    3. Preserves 95%+ of legitimate content
    """
    
    model = None
    try:
        logger.info(f"Loading Whisper {model_size}...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_size, device=device)
        
        # STRONG anti-hallucination at ASR level
        options: Dict[str, Any] = {
            "fp16": torch.cuda.is_available(),
            "temperature": 0.0,  # Deterministic
            "compression_ratio_threshold": 1.8,  # Strict
            "logprob_threshold": -0.8,  # Conservative
            "no_speech_threshold": 0.4,  # Permissive (allows more speech detection)
            "condition_on_previous_text": False,  # CRITICAL: prevents context poisoning
        }
        
        if language:
            options["language"] = language
        
        logger.info("Transcribing with anti-hallucination parameters")
        result = model.transcribe(audio_path, **options)
        
        # MINIMAL post-filtering - only extreme cases
        segments = []
        prev_text = ""
        exact_repeat_count = 0
        
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if not text:
                continue
            
            # Only filter EXACT repeats 5+ times (clear hallucination)
            if text == prev_text:
                exact_repeat_count += 1
                if exact_repeat_count >= 5:
                    logger.debug(f"Filtered 5+ exact repeats: {text}")
                    continue
            else:
                exact_repeat_count = 0
            
            segments.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": text,
            })
            prev_text = text
        
        full_text = " ".join(s["text"] for s in segments).strip()
        
        output = {"segments": segments, "text": full_text}
        
        if transcript_dir:
            transcript_dir.mkdir(parents=True, exist_ok=True)
            out_path = Path(transcript_dir) / f"{Path(audio_path).stem}_whisper.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            output["file"] = str(out_path.resolve())
        
        logger.info(f"Complete: {len(segments)} segments, {len(full_text)} chars")
        return output
    
    except Exception as e:
        logger.error(f"Whisper failed: {e}")
        raise
    
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()