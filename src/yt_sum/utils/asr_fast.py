# src/yt_sum/utils/asr_fast.py
"""Faster-Whisper - Balanced anti-hallucination"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import torch
import gc

from faster_whisper import WhisperModel

logger = logging.getLogger("asr_fast")


def transcribe_audio_fast(
    audio_path: str,
    *,
    model_size: str = "medium",
    transcript_dir: Optional[Path] = None,
    language: Optional[str] = None,
    compute_type: str = "int8_float16",
    beam_size: int = 5,
    vad_filter: bool = True,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Balanced: Strong ASR parameters + minimal post-filtering.
    """
    
    model = None
    
    try:
        # Auto-adjust for VRAM
        vram_gb = 0
        if device == "cuda" and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024**3)
            except Exception:
                vram_gb = 0
            
            if vram_gb <= 8 and model_size in ("medium", "large", "large-v2", "large-v3"):
                compute_type = "int8_float16"
            
            if vram_gb <= 8:
                beam_size = 1
        
        logger.info(f"Faster-Whisper: {model_size}, {compute_type}, beam={beam_size}")
        
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            num_workers=1,
        )
        
        # STRONG anti-hallucination at ASR level
        segments, info = model.transcribe(
            audio_path,
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=False,
            
            # Key parameters that prevent hallucinations
            condition_on_previous_text=False,  # CRITICAL
            compression_ratio_threshold=1.8,
            log_prob_threshold=-0.8,
            no_speech_threshold=0.4,
            initial_prompt=None,
            temperature=0.0,
        )
        
        # MINIMAL post-filtering - only extreme cases
        processed = []
        prev_text = ""
        exact_repeat_count = 0
        
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            
            # Only filter 5+ exact repeats
            if text == prev_text:
                exact_repeat_count += 1
                if exact_repeat_count >= 5:
                    logger.debug(f"Filtered 5+ exact repeats: {text}")
                    continue
            else:
                exact_repeat_count = 0
            
            processed.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            })
            prev_text = text
        
        full_text = " ".join(s["text"] for s in processed).strip()
        
        logger.info(f"Complete: {len(processed)} segments, {len(full_text)} chars")
        logger.info(f"Language: {info.language} ({info.language_probability:.2f})")
        
        output = {
            "segments": processed,
            "text": full_text,
            "language": info.language,
            "language_probability": float(info.language_probability),
            "duration": float(info.duration),
        }
        
        if transcript_dir:
            transcript_dir.mkdir(parents=True, exist_ok=True)
            out_path = Path(transcript_dir) / f"{Path(audio_path).stem}_fastwhisper.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            output["file"] = str(out_path.resolve())
        
        return output
    
    except Exception as e:
        logger.error(f"Faster-Whisper failed: {e}")
        
        # Fallback chain
        if compute_type == "int8_float16":
            return transcribe_audio_fast(
                audio_path, model_size=model_size, transcript_dir=transcript_dir,
                language=language, compute_type="int8", beam_size=1,
                vad_filter=vad_filter, device=device
            )
        elif compute_type == "int8" and device == "cuda":
            return transcribe_audio_fast(
                audio_path, model_size=model_size, transcript_dir=transcript_dir,
                language=language, compute_type="int8", beam_size=1,
                vad_filter=vad_filter, device="cpu"
            )
        
        raise
    
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()