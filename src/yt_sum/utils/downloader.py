# src/yt_sum/utils/downloader.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import os
import shutil
import subprocess
import glob

# -----------------------------
# Optional WAV conversion
# -----------------------------
def _maybe_ffmpeg_16k_mono(src: Path) -> Path:
    """
    If ffmpeg exists, convert to 16 kHz mono PCM WAV (broadly compatible/robust for ASR).
    Otherwise return the original file unchanged.
    """
    if shutil.which("ffmpeg") is None:
        return src
    dst = src.with_suffix(".wav")
    try:
        # 16 kHz, mono, PCM_s16le
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(src),
                "-ac", "1",        # mono
                "-ar", "16000",    # 16 kHz
                "-vn",             # drop video
                "-f", "wav",
                str(dst)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return dst
    except Exception:
        # If conversion fails, just use the original file.
        return src

# -----------------------------
# File resolution helpers
# -----------------------------
def _find_downloaded_file(out_dir: Path, video_id: str) -> Optional[Path]:
    """
    yt-dlp may pick different audio container extensions depending on the source.
    Find "<id>.<ext>" across common audio extensions.
    """
    exts = ("m4a", "mp4", "webm", "opus", "mp3", "wav", "aac", "ogg", "flac", "mka", "mkv")
    for ext in exts:
        cand = out_dir / f"{video_id}.{ext}"
        if cand.exists():
            return cand
    # Fallback: glob anything that starts with the id (rare remux cases)
    hits = sorted(Path(p) for p in glob.glob(str(out_dir / f"{video_id}.*")))
    return hits[0] if hits else None

# -----------------------------
# Download entrypoint
# -----------------------------
def download_audio(url: str, out_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Download best available audio with yt-dlp.

    Returns:
      audio_path: Path to the downloaded audio (possibly converted to 16k mono WAV)
      meta: {
        id, title, uploader, duration, view_count, upload_date
      }

    Notes:
      - If you need to access age/region-restricted videos, set an environment variable
        YT_COOKIES=/path/to/cookies.txt (Netscape cookie file). If present, we pass it to yt-dlp.
      - We avoid yt-dlp postprocessors here and optionally do a clean WAV via ffmpeg for ASR stability.
    """
    from yt_dlp import YoutubeDL

    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional cookies (for restricted content)
    cookies_path = os.getenv("YT_COOKIES")
    cookiefile = cookies_path if cookies_path and Path(cookies_path).exists() else None

    # Prefer audio-only formats; let yt-dlp pick the best it can.
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,     # safer filenames across OSes
        "socket_timeout": 30,
        "ignoreerrors": False,         # raise on hard failures
        "geo_bypass": True,
        # No postprocessors: we optionally convert ourselves with ffmpeg
        "postprocessors": [],
    }
    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile

    # Retry loop (handles transient network/CDN issues)
    last_exc: Optional[Exception] = None
    info: Dict[str, Any] = {}
    for attempt in range(2):  # try up to 2 times
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            last_exc = None
            break
        except Exception as e:
            last_exc = e

    if last_exc is not None:
        raise RuntimeError(f"yt-dlp failed to download audio: {last_exc}")

    # If a playlist slipped through, take the first entry
    if "entries" in info and isinstance(info["entries"], list) and info["entries"]:
        info = info["entries"][0]

    vid = info.get("id")
    if not vid:
        raise RuntimeError("Could not resolve video id from yt-dlp info.")

    # Locate the actual downloaded file (extension can vary)
    dl_path = _find_downloaded_file(out_dir, vid)
    if not dl_path:
        # As a last resort, rely on the declared extension
        ext = info.get("ext", "m4a")
        dl_path = out_dir / f"{vid}.{ext}"
        if not dl_path.exists():
            raise RuntimeError("Downloaded audio file not found on disk.")
    info_dict = ydl.extract_info(url, download=True)
    meta = {
        "id": vid,
        "title": info.get("title"),
        "uploader": info.get("uploader") or info.get("channel"),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "upload_date": info.get("upload_date"),
        "url": info_dict.get("webpage_url") or url,
    }

    # Optional: convert to 16k mono WAV (best for ASR consistency)
    audio_path = _maybe_ffmpeg_16k_mono(dl_path)
    return audio_path, meta
