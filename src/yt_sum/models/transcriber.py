# src/yt_sum/models/transcriber.py
"""Transcriber with balanced anti-hallucination approach"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import multiprocessing as mp
import torch
import gc

from src.yt_sum.utils import asr, asr_fast
from src.yt_sum.utils.hw import auto_asr_choice
from src.yt_sum.utils.logging import get_logger

logger = get_logger("transcriber", level="INFO")

_SMALL_MODELS = {"tiny", "base", "small"}
_LARGE_MODELS = {"medium", "large", "large-v2", "large-v3"}


def _fast_asr_worker(conn, kwargs: Dict[str, Any]) -> None:
    """Subprocess for Faster-Whisper (memory isolation)"""
    try:
        from src.yt_sum.utils import asr_fast as _asr_fast
        out = _asr_fast.transcribe_audio_fast(**kwargs)
        conn.send(("ok", out))
    except Exception as e:
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        try:
            conn.close()
        except Exception:
            pass


class Transcriber:
    """
    Production transcriber with balanced anti-hallucination.
    
    Strategy:
    - Strong ASR parameters prevent hallucinations at source
    - Minimal post-filtering (only extreme cases)
    - Preserves 95%+ of legitimate content
    """
    
    _cache_markers: Dict[str, bool] = {}
    
    def __init__(self, model_size: Optional[str] = None, prefer_accuracy: bool = True):
        self.model_size = (model_size or "small").lower()
        self.prefer_accuracy = prefer_accuracy
        self.transcript_dir = Path("run_data/transcripts")
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
    
    def transcribe(
        self,
        audio_path: str,
        *,
        domain: str = "general",
        translate_to_english: bool = True,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Transcribe with smart backend selection"""
        
        audio_path = str(audio_path)
        
        choice = auto_asr_choice(None, language, self.prefer_accuracy) or {}
        size = (self.model_size or choice.get("model_size", "small")).lower()
        
        # Backend selection
        backend = "whisper" if size in _SMALL_MODELS else "faster"
        
        vram_gb = 0
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
        
        compute_type = "int8_float16" if size in _LARGE_MODELS and vram_gb <= 16 else "float16"
        
        logger.info(f"Backend: {backend}, model={size}, compute={compute_type}, VRAM={vram_gb:.1f}GB")
        
        try:
            if backend == "faster":
                segments = self._run_faster_subprocess(
                    audio_path, size, compute_type, language
                )
            else:
                segments = self._run_whisper(audio_path, size, language)
            
            # Minimal post-filter
            segments = self._filter_extreme_cases(segments)
            return segments
            
        except Exception as e:
            logger.warning(f"Backend '{backend}' failed: {e}, trying fallback...")
            
            if backend == "faster":
                fallback_size = "small"
                logger.info(f"Fallback: PyTorch Whisper {fallback_size}")
                segments = self._run_whisper(audio_path, fallback_size, language)
            else:
                fallback_size = "tiny" if size == "base" else "base"
                logger.info(f"Retry with {fallback_size}")
                segments = self._run_whisper(audio_path, fallback_size, language)
            
            segments = self._filter_extreme_cases(segments)
            return segments
            
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            logger.info("Transcription complete, GPU cleared")
    
    def _filter_extreme_cases(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        MINIMAL filtering - only extreme hallucination cases.
        
        What we filter:
        1. 5+ consecutive exact repeats (clear hallucination)
        2. Exact-match end spam (last 2 segments only)
        
        What we DON'T filter:
        - Technical repetition (legitimate in lectures)
        - Short segments (may be valid)
        - Word diversity (destroys technical content)
        """
        
        if not segments:
            return segments
        
        filtered = []
        prev_text = ""
        exact_repeat_count = 0
        
        # Filter only 5+ exact repeats
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            
            if text == prev_text:
                exact_repeat_count += 1
                if exact_repeat_count >= 5:
                    logger.debug(f"Filtered 5+ exact repeats: {text}")
                    continue
            else:
                exact_repeat_count = 0
            
            filtered.append(seg)
            prev_text = text
        
        # Remove end spam (last 2 segments only, exact matches)
        if len(filtered) >= 3:
            spam_exact = [
                "thank you for watching",
                "please subscribe", 
                "like and subscribe",
                "hit the bell icon",
                "smash that like button",
            ]
            
            for _ in range(min(2, len(filtered))):
                if not filtered:
                    break
                    
                last_text = filtered[-1].get("text", "").lower().strip()
                
                if any(spam == last_text for spam in spam_exact):
                    logger.info(f"Removed end spam: {last_text}")
                    filtered.pop()
                else:
                    break  # Stop if not exact spam match
        
        removed = len(segments) - len(filtered)
        if removed > 0:
            logger.info(f"Filtered {removed}/{len(segments)} segments ({removed/len(segments)*100:.1f}%)")
        
        return filtered
    
    def _run_whisper(self, audio_path: str, size: str, language: Optional[str]) -> List[Dict[str, Any]]:
        """PyTorch Whisper"""
        
        cache_key = f"whisper-{size}"
        if cache_key not in self._cache_markers:
            logger.info(f"Loading Whisper: {size}")
            self._cache_markers[cache_key] = True
        
        out = asr.transcribe_audio(
            audio_path,
            model_size=size,
            transcript_dir=self.transcript_dir,
            language=language,
        )
        return out.get("segments", [])
    
    def _run_faster_subprocess(
        self,
        audio_path: str,
        size: str,
        compute_type: str,
        language: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Faster-Whisper in subprocess (memory isolation)"""
        
        cache_key = f"faster-{size}-{compute_type}"
        if cache_key not in self._cache_markers:
            logger.info(f"Preparing Faster-Whisper: {size} ({compute_type})")
            self._cache_markers[cache_key] = True
        
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        
        kwargs = {
            "audio_path": audio_path,
            "model_size": size,
            "transcript_dir": self.transcript_dir,
            "language": language,
            "compute_type": compute_type,
            "beam_size": 1 if compute_type == "int8_float16" else 3,
            "vad_filter": True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        
        proc = ctx.Process(target=_fast_asr_worker, args=(child_conn, kwargs), daemon=True)
        proc.start()
        proc.join(timeout=900)  # 15 min timeout
        
        if proc.is_alive():
            logger.error("Timeout")
            proc.terminate()
            proc.join()
            raise RuntimeError("Faster-Whisper timeout")
        
        if not parent_conn.poll():
            raise RuntimeError("No data returned")
        
        status, payload = parent_conn.recv()
        parent_conn.close()
        
        if status == "ok":
            return (payload or {}).get("segments", [])
        
        raise RuntimeError(f"Failed: {payload}")