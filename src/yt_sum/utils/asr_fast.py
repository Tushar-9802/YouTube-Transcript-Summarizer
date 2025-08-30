# src/yt_sum/utils/asr_fast.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

def transcribe_audio_fast(
    *,
    audio_path: str,
    model_size: str = "small",
    transcript_dir: Path,
    language: Optional[str] = None,
    compute_type: str = "float16",   # "float16" | "int8_float16" | "int8" | "fp32"
    beam_size: int = 5,
    vad_filter: bool = True,
    initial_prompt: Optional[str] = None,  # helpful if you know domain terms
) -> Dict[str, Any]:
    """
    Faster-Whisper transcription.

    Returns:
      {
        "text": str,
        "segments": [{"start": float, "end": float, "text": str}, ...],
        "language": str,
        "path": str,      # transcript .txt path
        "note": str?,     # optional note about fallbacks
      }

    Notes (plain English):
      - We prefer GPU if available and pick the GPU with the most free VRAM.
      - If the requested compute_type fails (e.g., not enough VRAM), we fall back
        to a more memory-friendly choice automatically.
    """
    import torch
    from faster_whisper import WhisperModel

    # -------- pick device (Route-1: best single device) --------
    try:
        from yt_sum.utils.hw import get_device_info
        dev = get_device_info()
        if dev.cuda:
            device = "cuda"
            device_index = max(0, dev.index) if getattr(dev, "index", -1) is not None else 0
        else:
            device = "cpu"  # MPS currently not used by faster-whisper
            device_index = None
    except Exception:
        device, device_index = ("cuda" if torch.cuda.is_available() else "cpu"), None

    # -------- build model with robust compute fallback --------
    # We attempt the requested compute_type first; if it fails, try safer ones.
    tried: List[Tuple[str, str]] = []
    note: Optional[str] = None

    def _try_build(ct: str) -> Optional[WhisperModel]:
        try:
            return WhisperModel(
                model_size,
                device=device,
                device_index=device_index,
                compute_type=ct,
            )
        except Exception as e:
            tried.append((ct, str(e)))
            return None

    # Attempt pipeline:
    # 1) requested compute_type
    model = _try_build(compute_type)
    # 2) memory-friendlier fallbacks
    if model is None and device == "cuda":
        for ct in ("int8_float16", "int8", "fp16", "fp32"):
            model = _try_build(ct)
            if model is not None:
                break
    if model is None and device != "cuda":
        for ct in ("int8", "fp32"):
            model = _try_build(ct)
            if model is not None:
                break

    if model is None:
        # Last resort: CPU fp32
        model = WhisperModel(model_size, device="cpu", compute_type="fp32")
        tried.append(("forced_cpu_fp32", "fallback"))

    if tried:
        note_lines = ["ASR: compute_type fallback path used:"] + [f" - {ct}: {err}" for ct, err in tried]
        note = "\n".join(note_lines)

    # -------- VAD & decode settings --------
    vad_params = dict(min_silence_duration_ms=250) if vad_filter else None

    # Beam search is usually better for accuracy; temperature not used with beam search.
    # For very long audios you can reduce beam_size via caller knobs if needed.
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,                 # None => auto
        beam_size=max(1, int(beam_size)),
        vad_filter=vad_filter,
        vad_parameters=vad_params,
        word_timestamps=False,
        condition_on_previous_text=True,
        initial_prompt=initial_prompt,
    )

    segments: List[Dict[str, Any]] = []
    parts: List[str] = []
    for seg in segments_iter:
        seg_d = {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
        segments.append(seg_d)
        parts.append(seg_d["text"])

    text = " ".join(parts).strip()
    detected_lang = (getattr(info, "language", None) or language or "unknown")

    # -------- save transcript --------
    transcript_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(transcript_dir) / (Path(audio_path).stem + ".txt")
    out_path.write_text(text, encoding="utf-8")

    out: Dict[str, Any] = {
        "text": text,
        "segments": segments,
        "language": detected_lang,
        "path": str(out_path),
    }
    if note:
        out["note"] = note
    return out
