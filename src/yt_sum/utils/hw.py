# src/yt_sum/utils/hw.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

# -------- Device detection --------
@dataclass
class DeviceInfo:
    cuda: bool
    name: str
    total_gb: float
    free_gb: float

def _bytes_to_gb(b: int) -> float:
    return round(b / (1024**3), 2)

def get_device_info() -> DeviceInfo:
    """
    Returns a best-effort snapshot of CUDA availability, device name, and free/total VRAM.
    Falls back gracefully on CPU or when NVML isn't available.
    """
    name = "CPU"
    total_gb = 0.0
    free_gb = 0.0
    cuda = False

    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            i = 0
            name = torch.cuda.get_device_name(i)
            # Try torch mem API first
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                free_gb = _bytes_to_gb(free_b)
                total_gb = _bytes_to_gb(total_b)
            except Exception:
                # Try NVML for more accurate numbers if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    total_gb = _bytes_to_gb(mem.total)
                    free_gb = _bytes_to_gb(mem.free)
                except Exception:
                    # As a last resort, estimate by torch + heuristic
                    total_gb = 0.0
                    free_gb = 0.0
        else:
            name = "CPU"
    except Exception:
        # torch not available or other error → CPU
        pass

    return DeviceInfo(cuda=cuda, name=name, total_gb=total_gb, free_gb=free_gb)

# -------- ASR auto-choice --------
def auto_asr_choice(duration_sec: Optional[int], language: Optional[str], prefer_accuracy: bool) -> Dict[str, Any]:
    """
    Heuristics for Faster-Whisper size & compute type.
    - prefer_accuracy=True: bias toward larger models
    - duration-aware: longer videos get bigger models (if GPU)
    """
    dev = get_device_info()
    dur = duration_sec or 0

    # Default compute type
    if dev.cuda:
        # If free VRAM is tight, prefer mixed int8_float16; else float16 for max speed
        compute_type = "int8_float16" if dev.free_gb and dev.free_gb < 3.5 else "float16"
    else:
        compute_type = "int8"  # CPU path

    # Duration buckets (tune as you like)
    if prefer_accuracy:
        if dur <= 15 * 60:
            size = "small" if dev.cuda else "base"
        elif dur <= 45 * 60:
            size = "small" if not dev.cuda else "medium"
        elif dur <= 120 * 60:
            size = "medium" if dev.cuda else "small"
        else:
            size = "large-v3" if dev.cuda else "medium"
    else:
        if dur <= 15 * 60:
            size = "tiny"
        elif dur <= 45 * 60:
            size = "base"
        elif dur <= 120 * 60:
            size = "small" if dev.cuda else "base"
        else:
            size = "small" if dev.cuda else "base"

    return {
        "backend": "faster",
        "model_size": size,
        "compute_type": compute_type,
        "language": language,  # None → autodetect
    }

# -------- Summarizer auto-params --------
def _estimate_tokens_from_text(txt: str) -> int:
    """
    Rough BPE token estimate from words.
    Empirical ~1.3–1.4x for English transcripts; good enough to size budgets.
    """
    words = len((txt or "").split())
    return int(words * 1.33)

def auto_summarizer_params(transcript_text: str, language: Optional[str]) -> Dict[str, int]:
    """
    Returns legacy-style knobs used by older pipeline paths:
      - chunk_tokens, chunk_overlap
      - num_beams
      - per_chunk_max/min
      - fuse_max/min
    Newer code uses ratio-based GenParams, but we keep this for compatibility & metadata.
    """
    dev = get_device_info()
    toks = _estimate_tokens_from_text(transcript_text)

    # Choose a base chunk size under 1024 encoder limit
    if not dev.cuda:
        chunk_tokens = 640  # CPU safer
    else:
        # If VRAM is tight, give more headroom
        if dev.free_gb and dev.free_gb < 3.5:
            chunk_tokens = 720
        else:
            chunk_tokens = 900

    # Overlap: more for smaller chunks/CPU to keep continuity
    if chunk_tokens <= 720:
        chunk_overlap = 160
    else:
        chunk_overlap = 120

    # Beams: CPU lower; GPU higher
    num_beams = 3 if not dev.cuda else (6 if (dev.free_gb and dev.free_gb >= 8.0) else 5)

    # Per-chunk length ratio heuristic (slightly shorter for very long talks)
    if toks < 1500:
        ratio = 0.22
    elif toks < 6000:
        ratio = 0.18
    else:
        ratio = 0.15

    # Translate ratio → absolute budgets for legacy path
    per_chunk_target = max(120, int(chunk_tokens * ratio))
    per_chunk_max = min(512, max(180, int(per_chunk_target * 1.1)))
    per_chunk_min = max(80, min(per_chunk_max - 80, int(per_chunk_target * 0.7)))

    # Final fuse target scales with total length
    if toks < 1500:
        fuse_max = 280
    elif toks < 6000:
        fuse_max = 340
    elif toks < 12000:
        fuse_max = 380
    else:
        fuse_max = 420
    fuse_min = max(140, fuse_max - 120)

    return {
        "chunk_tokens": int(chunk_tokens),
        "chunk_overlap": int(chunk_overlap),
        "num_beams": int(num_beams),
        "per_chunk_max": int(per_chunk_max),
        "per_chunk_min": int(per_chunk_min),
        "fuse_max": int(fuse_max),
        "fuse_min": int(fuse_min),
    }
