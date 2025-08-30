# src/yt_sum/utils/asr.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List

def detect_device() -> str:
    """
    Return 'cuda' if a CUDA GPU is available, else 'mps' on Apple Silicon if available,
    otherwise 'cpu'. This keeps callers flexible without breaking older code paths
    that only check for 'cuda'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Apple Metal Performance Shaders (MPS)
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    except Exception:
        return "cpu"

def transcribe_audio(
    audio_path: str,
    model_size: str,
    transcript_dir: Path,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    OpenAI-Whisper transcription (reference / fallback).
    Returns:
      {
        "text": str,
        "segments": [{"start": float, "end": float, "text": str}, ...],
        "language": str,
        "path": str,   # saved transcript .txt
        "note": str?,  # optional info about device/precision used
      }
    """
    import whisper
    import torch

    device = detect_device()

    # Load model on the chosen device.
    # Whisper will pick dtype automatically; we control fp16 flag in decoding options.
    model = whisper.load_model(model_size, device=("cuda" if device == "cuda" else "cpu"))
    # Note: whisper doesn't run the model on MPS directly; torch.compile/mps support
    # varies. Using CPU execution with MPS tensors is not officially supported for whisper.
    # We therefore set device='cpu' for non-CUDA; it is stable and correct.

    # fp16 only when running generation on CUDA; keep False on CPU/MPS for numerical stability
    fp16 = (device == "cuda")

    # Decoding options:
    # - deterministic path first (temperature=0.0, beam search)
    # - then a small temperature ladder as fallback to recover from pathological cases
    #   (these temps are only used internally by whisper when certain thresholds are not met)
    options = dict(
        language=language,                 # None => auto-detect
        task="transcribe",
        word_timestamps=False,
        fp16=fp16,
        condition_on_previous_text=True,
        beam_size=5,
        best_of=5,                         # used in sampling; harmless here
        patience=None,                     # beam search patience (None => default)
        temperature=0.0,                   # deterministic first attempt
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,   # whisper defaults for stability
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )

    # Run transcription
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

    detected_lang = result.get("language") or language or "unknown"

    # Save transcript to disk
    transcript_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(transcript_dir) / (Path(audio_path).stem + ".txt")
    out_path.write_text(text, encoding="utf-8")

    out: Dict[str, Any] = {
        "text": text,
        "segments": segments,
        "language": detected_lang,
        "path": str(out_path),
    }

    # Add a small note for debugging/reproducibility
    out["note"] = f"whisper model={model_size} device={device} fp16={fp16}"

    return out
