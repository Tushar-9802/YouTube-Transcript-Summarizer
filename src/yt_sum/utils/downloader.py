# src/yt_sum/utils/downloader.py
from pathlib import Path
from typing import Tuple, Dict, Any
import yt_dlp

def download_audio(url: str, audio_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Downloads bestaudio and converts to mp3 using ffmpeg.
    Returns (audio_path, metadata_dict).
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(audio_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = audio_dir / f"{info['id']}.mp3"
        return audio_path, info
