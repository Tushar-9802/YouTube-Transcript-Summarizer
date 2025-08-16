# src/yt_sum/utils/asr.py
from pathlib import Path
from typing import Dict, Any, Optional
import whisper
import torch

def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def transcribe_audio(
    audio_path: Path,
    model_size: str,
    transcript_dir: Path,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    device = detect_device()
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(str(audio_path), language=language)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    out_txt = transcript_dir / (audio_path.stem + ".txt")
    out_txt.write_text(result["text"], encoding="utf-8")
    result.update({"backend": "openai-whisper", "model_size": model_size, "device": device})
    return result
