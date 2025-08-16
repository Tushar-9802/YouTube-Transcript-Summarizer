# src/yt_sum/utils/downloader.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any
import shutil
import subprocess

def _maybe_ffmpeg_16k_mono(src: Path) -> Path:
    """If ffmpeg exists, convert to 16k mono WAV for maximal ASR compatibility; else return original."""
    if shutil.which("ffmpeg") is None:
        return src
    dst = src.with_suffix(".wav")
    try:
        # 16 kHz, mono, PCM_s16le
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", str(dst)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return dst
    except Exception:
        return src  # fall back silently

def download_audio(url: str, out_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Download bestaudio using yt-dlp.
    Returns: (audio_path, meta_dict)
    - audio_path: local path to audio file (possibly converted to 16k mono WAV if ffmpeg available)
    - meta: {id, title, uploader, duration, view_count, upload_date}
    """
    from yt_dlp import YoutubeDL

    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # Avoid random re-muxes; we’ll optionally convert ourselves
        "postprocessors": [],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # If a playlist accidentally slips through, take the first entry
        if "entries" in info:
            info = info["entries"][0]

        vid = info.get("id")
        ext = info.get("ext", "m4a")
        dl_path = out_dir / f"{vid}.{ext}"

        meta = {
            "id": vid,
            "title": info.get("title"),
            "uploader": info.get("uploader") or info.get("channel"),
            "duration": info.get("duration"),
            "view_count": info.get("view_count"),
            "upload_date": info.get("upload_date"),
        }

    # Optional: convert to 16k mono WAV if ffmpeg is available
    audio_path = _maybe_ffmpeg_16k_mono(dl_path)
    return audio_path, meta
