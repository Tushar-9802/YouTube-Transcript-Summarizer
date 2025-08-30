# src/yt_sum/utils/hw.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

# ---------------------------
# Device / memory inspection
# ---------------------------

@dataclass
class DeviceInfo:
    """Snapshot of the best device we can use for inference."""
    cuda: bool                  # any CUDA device available
    mps: bool                   # Apple Metal / MPS available
    index: int                  # chosen CUDA device index (or -1 if none)
    name: str                   # device name (or "CPU"/"MPS")
    total_gb: float             # total VRAM in GiB (0 if unknown)
    free_gb: float              # free VRAM in GiB (0 if unknown)

def _bytes_to_gb(b: int) -> float:
    return round(b / (1024**3), 2)

def _best_cuda_device_by_free_mem() -> Tuple[int, str, float, float]:
    """
    Returns (index, name, total_gb, free_gb) for the CUDA device with the most free memory.
    Falls back to device 0 if queries fail.
    """
    import torch
    n = torch.cuda.device_count()
    if n <= 0:
        return (-1, "CPU", 0.0, 0.0)

    # Try PyTorch's mem_get_info per device first
    best = (-1, "", 0.0, -1.0)
    for i in range(n):
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = f"CUDA:{i}"
        free_gb = 0.0
        total_gb = 0.0
        try:
            torch.cuda.set_device(i)
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb = _bytes_to_gb(free_b)
            total_gb = _bytes_to_gb(total_b)
        except Exception:
            # Try NVML if available
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                total_gb = _bytes_to_gb(mem.total)
                free_gb = _bytes_to_gb(mem.free)
            except Exception:
                pass

        # pick largest free
        if free_gb > best[3]:
            best = (i, name, total_gb, free_gb)

    if best[0] == -1:
        # As a last resort, pick device 0 with unknown mem
        try:
            name0 = torch.cuda.get_device_name(0)
        except Exception:
            name0 = "CUDA:0"
        return (0, name0, 0.0, 0.0)

    return best

def get_device_info() -> DeviceInfo:
    """
    Detects whether CUDA/MPS are available and returns a snapshot with device name and VRAM.
    We choose the CUDA device with **most free VRAM** when multiple GPUs exist.
    Falls back to MPS (Apple Silicon) or CPU.
    """
    cuda = False
    mps = False
    idx = -1
    name = "CPU"
    total_gb = 0.0
    free_gb = 0.0

    try:
        import torch
        # CUDA path
        if torch.cuda.is_available():
            cuda = True
            idx, name, total_gb, free_gb = _best_cuda_device_by_free_mem()
            return DeviceInfo(cuda=cuda, mps=False, index=idx, name=name, total_gb=total_gb, free_gb=free_gb)

        # Apple Metal (MPS) path – good for some models; we still treat memory as unknown
        try:
            mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception:
            mps = False

        if mps:
            return DeviceInfo(cuda=False, mps=True, index=-1, name="MPS", total_gb=0.0, free_gb=0.0)

        # CPU fallback
        return DeviceInfo(cuda=False, mps=False, index=-1, name="CPU", total_gb=0.0, free_gb=0.0)

    except Exception:
        # torch not present or other failure -> CPU
        return DeviceInfo(cuda=False, mps=False, index=-1, name="CPU", total_gb=0.0, free_gb=0.0)

# --------------------------------
# Faster-Whisper (ASR) auto-choice
# --------------------------------

def auto_asr_choice(duration_sec: Optional[int], language: Optional[str], prefer_accuracy: bool) -> Dict[str, Any]:
    """
    Heuristic picker for Faster-Whisper model size and compute type.

    Intuition (plain English):
      - If you have a GPU with enough free VRAM, use a bigger model and fp16.
      - If VRAM is tight, use int8_float16 (memory-friendlier).
      - On CPU (or MPS where Faster-Whisper may not accelerate), use int8.
      - Longer videos + prefer_accuracy=True => bump model size.

    Returns a dict used by the ASR runner.
    """
    dev = get_device_info()
    dur = int(duration_sec or 0)

    # Compute type: favor float16 on roomy GPUs; otherwise a mixed/int8 path
    if dev.cuda:
        compute_type = "float16" if (dev.free_gb and dev.free_gb >= 3.5) else "int8_float16"
    else:
        # CPU / MPS: use int8 (Faster-Whisper runs on CPU here)
        compute_type = "int8"

    # Duration-aware model size (tweakable thresholds)
    if prefer_accuracy:
        if dur <= 15 * 60:
            size = "small" if dev.cuda else "base"
        elif dur <= 45 * 60:
            size = "medium" if dev.cuda else "small"
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
        "language": language,  # None => autodetect in ASR layer
    }

# ---------------------------------
# Summarizer auto-params (legacy)
# ---------------------------------

def _estimate_tokens_from_text(txt: str) -> int:
    """
    Rough token estimate from words (~1.33× rule of thumb for English).
    We only need a ballpark to size chunk budgets.
    """
    words = len((txt or "").split())
    return max(0, int(words * 1.33))

def auto_summarizer_params(transcript_text: str, language: Optional[str]) -> Dict[str, int]:
    """
    Legacy-style knobs for summarization. Newer code uses GenParams (ratio-based),
    but we keep these for compatibility and for logging in the UI.
    """
    dev = get_device_info()
    toks = _estimate_tokens_from_text(transcript_text)

    # Choose chunk size under typical 1024 encoder window.
    # Smaller chunks on CPU to keep latency sensible.
    if not dev.cuda:
        chunk_tokens = 640
    else:
        # If VRAM is tight, give some headroom; otherwise use 900 (safe with BART/T5).
        if dev.free_gb and dev.free_gb < 3.5:
            chunk_tokens = 720
        else:
            chunk_tokens = 900

    # Overlap: a bit more overlap when chunks are smaller (helps continuity).
    chunk_overlap = 160 if chunk_tokens <= 720 else 120

    # Beams: GPUs can afford more beams; CPUs fewer.
    if dev.cuda:
        # If very roomy GPU, 6 beams; otherwise 5 is a solid default
        num_beams = 6 if (dev.free_gb and dev.free_gb >= 8.0) else 5
    else:
        num_beams = 3

    # Per-chunk ratio heuristic (slightly shrink for very long talks)
    if toks < 1500:
        ratio = 0.22
    elif toks < 6000:
        ratio = 0.18
    else:
        ratio = 0.15

    # Convert ratio into absolute budgets for the older path
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
