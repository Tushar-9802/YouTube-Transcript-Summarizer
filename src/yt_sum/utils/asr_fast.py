# src/yt_sum/utils/asr_fast.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List

def transcribe_audio_fast(
    *,
    audio_path: str,
    model_size: str = "small",
    transcript_dir: Path,
    language: Optional[str] = None,
    compute_type: str = "float16",   # "float16" | "int8_float16" | "int8" | "fp32"
    beam_size: int = 5,
    vad_filter: bool = True,
) -> Dict[str, Any]:
    """
    Faster-Whisper transcription.
    Returns: {"text", "segments": [{"start","end","text"}], "language"}
    Saves transcript to transcript_dir / "<audio_stem>.txt"
    """
    from faster_whisper import WhisperModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct model with requested compute type
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    # VAD parameters help with noisy YouTube audio
    vad_params = dict(min_silence_duration_ms=250) if vad_filter else None

    # Beam search for accuracy; you can switch to greedy in speed mode
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,         # None => auto
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=vad_params,
        word_timestamps=False,
        condition_on_previous_text=True,
    )

    segments: List[Dict[str, Any]] = []
    parts: List[str] = []
    for seg in segments_iter:
        seg_d = {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
        segments.append(seg_d)
        parts.append(seg_d["text"])

    text = " ".join(parts).strip()
    detected_lang = (info.language if hasattr(info, "language") else language) or "unknown"

    # Save transcript
    transcript_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(transcript_dir) / (Path(audio_path).stem + ".txt")
    out_path.write_text(text, encoding="utf-8")

    return {
        "text": text,
        "segments": segments,
        "language": detected_lang,
        "path": str(out_path),
    }
