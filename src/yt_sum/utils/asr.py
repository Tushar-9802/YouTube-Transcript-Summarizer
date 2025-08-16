# src/yt_sum/utils/asr.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List

def detect_device() -> str:
    """Return 'cuda' if available else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def transcribe_audio(
    audio_path: str,
    model_size: str,
    transcript_dir: Path,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    OpenAI Whisper (reference) fallback.
    Returns: {"text", "segments": [{"start","end","text"}], "language"}
    Saves transcript to transcript_dir / "<audio_stem>.txt"
    """
    import whisper
    import torch

    device = detect_device()
    model = whisper.load_model(model_size, device=device)

    # fp16 only on CUDA
    fp16 = device == "cuda"

    # If language is None, Whisper will auto-detect
    options = dict(
        language=language,
        task="transcribe",
        word_timestamps=False,
        fp16=fp16,
        condition_on_previous_text=True,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        temperature=0.0,  # deterministic
        beam_size=5,
        best_of=5,
        patience=None,
    )

    result = model.transcribe(audio_path, **options)

    # Normalize output
    text: str = (result.get("text") or "").strip()
    raw_segments = result.get("segments") or []
    segments: List[Dict[str, Any]] = []
    for s in raw_segments:
        segments.append({
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": (s.get("text") or "").strip(),
        })

    # language key can be in result["language"] or detected via detect_language
    detected_lang = result.get("language") or language or "unknown"

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
