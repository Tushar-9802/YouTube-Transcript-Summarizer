# src/yt_sum/utils/hw.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
class DeviceInfo:
    cuda: bool
    name: str
    total_gb: float
    free_gb: float
    torch_cuda: Optional[str]

def _mem_gb(bytes_val: int) -> float:
    return float(bytes_val) / (1024 ** 3)

def get_device_info() -> DeviceInfo:
    if torch.cuda.is_available():
        try:
            total_free = torch.cuda.mem_get_info()
            free_gb = _mem_gb(total_free[0])
            total_gb = _mem_gb(total_free[1])
        except Exception:
            props = torch.cuda.get_device_properties(0)
            total_gb = _mem_gb(props.total_memory)
            free_gb = max(0.0, total_gb - _mem_gb(torch.cuda.memory_allocated(0)))
        return DeviceInfo(
            cuda=True,
            name=torch.cuda.get_device_name(0),
            total_gb=round(total_gb, 2),
            free_gb=round(free_gb, 2),
            torch_cuda=torch.version.cuda,
        )
    return DeviceInfo(cuda=False, name="CPU", total_gb=0.0, free_gb=0.0, torch_cuda=None)

def auto_asr_choice(duration_s: Optional[int], language: Optional[str], prefer_accuracy: bool = True) -> Dict[str, Any]:
    """
    Heuristic picker for Faster‑Whisper size + compute_type (or Whisper size if fallback).
    Uses FREE VRAM (not just total) so it plays nice with other apps.
    """
    dev = get_device_info()
    long_audio = (duration_s or 0) >= 20 * 60  # >=20 min

    # Default targets
    choice = {
        "backend": "faster",             # "faster" or "whisper"
        "model_size": "small",           # tiny/base/small/medium/large-v3
        "compute_type": "int8_float16",  # float16 on good GPUs; int8/… for low VRAM
        "language": language,            # e.g., "en", "hi"; None = autodetect
    }

    if not dev.cuda:
        # CPU path
        choice.update({"backend": "whisper", "model_size": "base", "compute_type": "int8"})
        return choice

    # CUDA path: pick by free VRAM + length + accuracy preference
    free = dev.free_gb
    if free >= 12.0:
        # Plenty of VRAM
        choice["model_size"] = "large-v3" if prefer_accuracy and not long_audio else "medium"
        choice["compute_type"] = "float16"
    elif free >= 8.0:
        choice["model_size"] = "medium" if prefer_accuracy or long_audio else "small"
        choice["compute_type"] = "float16"
    elif free >= 6.0:
        choice["model_size"] = "small"
        choice["compute_type"] = "float16"
    else:
        choice["model_size"] = "base"
        choice["compute_type"] = "int8_float16"

    return choice

def estimate_token_count(text: str, language: Optional[str]) -> int:
    # crude but fast: English ~4 chars/token; Indic scripts ~2 chars/token
    if not text:
        return 0
    per_token = 4 if (language in (None, "en")) else 2
    return max(1, len(text) // per_token)

def auto_summarizer_params(text: str, language: Optional[str]) -> Dict[str, Any]:
    """
    Decide chunking/length/beam settings from transcript length and GPU VRAM.
    """
    dev = get_device_info()
    toks = estimate_token_count(text, language)
    long_doc = toks > 3000
    very_long = toks > 8000

    # VRAM‑aware chunk size
    if dev.cuda and dev.free_gb >= 8.0:
        chunk_tokens = 1000 if long_doc else 800
    elif dev.cuda and dev.free_gb >= 6.0:
        chunk_tokens = 800 if long_doc else 700
    else:
        chunk_tokens = 600 if long_doc else 500

    chunk_overlap = 120 if long_doc else 80

    # Length targets (more detail for long docs)
    per_chunk_max = 160 if long_doc else 120
    per_chunk_min = 60 if long_doc else 40
    fuse_max = 320 if very_long else (240 if long_doc else 200)
    fuse_min = 120 if long_doc else 80

    # Beams (trade accuracy for speed)
    if dev.cuda and dev.free_gb >= 8.0:
        beams = 5
    else:
        beams = 3

    return {
        "chunk_tokens": chunk_tokens,
        "chunk_overlap": chunk_overlap,
        "per_chunk_max": per_chunk_max,
        "per_chunk_min": per_chunk_min,
        "fuse_max": fuse_max,
        "fuse_min": fuse_min,
        "num_beams": beams,
    }
