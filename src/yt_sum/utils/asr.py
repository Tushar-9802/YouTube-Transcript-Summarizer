# src/yt_sum/utils/asr.py
from pathlib import Path
from typing import Dict, Any
import torch
import whisper

def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def transcribe_audio(audio_path: Path, model_size: str, transcript_dir: Path) -> Dict[str, Any]:
    """
    Runs Whisper transcription and saves a <video_id>.txt.
    Returns Whisper result dict (includes 'text', 'segments', 'language').
    """
    device = detect_device()
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(str(audio_path))
    transcript_dir.mkdir(parents=True, exist_ok=True)

    # filename: <stem>.txt where stem is the video_id (since downloader used id as filename)
    out_txt = transcript_dir / (audio_path.stem + ".txt")
    out_txt.write_text(result["text"], encoding="utf-8")
    return result
