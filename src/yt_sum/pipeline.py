# pipeline.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import os, glob, subprocess, shutil, logging

import torch
from docx import Document

from yt_sum.models.transcriber import Transcriber
from yt_sum.models.summarizer import Summarizer
from yt_sum.utils.keywords import extract_keywords, highlight_sentences, extract_critical_terms
from yt_sum.utils.factcheck import EntailmentScorer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------- YouTube download helpers ----------------------
def _find_downloaded_file(out_dir: Path, video_id: str) -> Optional[Path]:
    exts = ("m4a", "mp4", "webm", "opus", "mp3", "wav", "aac", "ogg", "flac", "mka", "mkv")
    for ext in exts:
        p = out_dir / f"{video_id}.{ext}"
        if p.exists():
            return p
    hits = sorted(Path(p) for p in glob.glob(str(out_dir / f"{video_id}.*")))
    return hits[0] if hits else None


def download_audio(url: str, out_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    from yt_dlp import YoutubeDL
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "socket_timeout": 30,
        "ignoreerrors": False,
        "geo_bypass": True,
        "postprocessors": [],
    }
    cookies = os.getenv("YT_COOKIES")
    if cookies and Path(cookies).exists():
        opts["cookiefile"] = cookies

    info: Dict[str, Any] = {}
    last_exc: Optional[Exception] = None
    for _ in range(2):
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            last_exc = None
            break
        except Exception as e:
            last_exc = e
    if last_exc:
        raise RuntimeError(f"yt-dlp failed to download audio: {last_exc}")

    if "entries" in info and isinstance(info["entries"], list) and info["entries"]:
        info = info["entries"][0]

    vid = info.get("id")
    if not vid:
        raise RuntimeError("Could not resolve video id from yt-dlp info.")

    dl_path = _find_downloaded_file(out_dir, vid)
    if not dl_path:
        ext = info.get("ext", "m4a")
        dl_path = out_dir / f"{vid}.{ext}"
        if not dl_path.exists():
            raise RuntimeError("Downloaded audio file not found on disk.")

    # Normalize to 16kHz mono WAV for ASR
    dst = dl_path.with_suffix(".wav")
    try:
        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(dl_path), "-ac", "1", "-ar", "16000", "-vn", str(dst)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            dl_path = dst
    except Exception as e:
        logger.warning(f"ffmpeg convert failed: {e}")

    meta = {
        "id": vid,
        "title": info.get("title"),
        "uploader": info.get("uploader") or info.get("channel"),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "upload_date": info.get("upload_date"),
    }
    return dl_path, meta


# ---------------------- Export helpers ----------------------
def export_docx(base_filename: str, transcript: str, summary: str) -> str:
    out = Path(f"{base_filename}.docx")
    doc = Document()
    doc.add_heading("Video Summary", 0)
    doc.add_heading("Summary", level=1)
    for para in (summary or "").split("\n"):
        if para.strip():
            doc.add_paragraph(para.strip())
    doc.add_heading("Transcript", level=1)
    for para in (transcript or "").split("\n"):
        if para.strip():
            doc.add_paragraph(para.strip())
    doc.save(out)
    return str(out)


# ---------------------- Pipeline ----------------------
def run_pipeline(
    url: str,
    workdir: Path,
    *,
    domain: str = "general",
    whisper_size: Optional[str] = None,
    prefer_accuracy: bool = True,
    summarizer_model: Optional[str] = None,
    use_8bit: Optional[bool] = None,
    refinement: bool = True,
    imrad: bool = False,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
    chunk_tokens: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    translate_non_english: bool = True,
    compression_ratio: Optional[float] = None,
    audience: str = "expert",
    output_language: Optional[str] = None,
    enable_factcheck: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Orchestrates: download -> transcribe -> summarize -> keywords -> factuality -> export.
    """
    device = {"cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device.update(name=props.name, total_gb=round(props.total_memory / (1024 ** 3), 2))
    else:
        device.update(name="CPU", total_gb=None)

    # 1) Download
    audio_dir = workdir / "audio"
    audio_path, meta = download_audio(url, audio_dir)

    # 2) Transcribe
    tr = Transcriber(model_size=whisper_size, prefer_accuracy=prefer_accuracy)
    translate = translate_non_english
    segments = tr.transcribe(str(audio_path), domain=domain, translate_to_english=translate)
    transcript = " ".join(s["text"] for s in segments).strip()
    if not transcript:
        raise RuntimeError("Empty transcript.")

    # Save transcript
    txt_dir = workdir / "transcripts"
    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_path = txt_dir / f"{meta.get('id','transcript')}.txt"
    txt_path.write_text(transcript, encoding="utf-8")

    # 3) Summarize
    summ = Summarizer(model_name=summarizer_model, use_8bit=use_8bit)
    if chunk_tokens:
        summ.chunk_tokens = chunk_tokens
    if chunk_overlap:
        summ.chunk_overlap = chunk_overlap

    summary_text = summ.summarize_long(
        transcript,
        domain=domain,
        imrad=imrad,
        refinement=refinement,
        min_len=min_len,
        max_len=max_len,
        chunk_tokens=summ.chunk_tokens,
        chunk_overlap=summ.chunk_overlap,
        compression_ratio=compression_ratio,
        audience=audience,
        output_language=output_language,
    )

    # 4) Keywords & Highlights
    keywords = extract_keywords(summary_text, top_n=15, method="auto", max_ngram=3)
    critical_terms = extract_critical_terms(transcript)
    highlights = highlight_sentences(transcript, keywords, top_k=8, diversity=0.35)

    # 5) Factuality
    fact: Dict[str, Any] = {}
    if enable_factcheck:
        try:
            scorer = EntailmentScorer()
            fact = scorer.score_summary_against_transcript(summary_text, transcript, k_support=3)
        except Exception as e:
            fact = {"overall_confidence": 0.0, "sentences": [], "error": str(e)}

    # 6) Export DOCX
    out_dir = workdir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    docx_path = export_docx(str(out_dir / f"{meta.get('id','summary')}"), transcript, summary_text)

    meta_info = {
        "video": {
            "id": meta.get("id"),
            "title": meta.get("title"),
            "uploader": meta.get("uploader"),
            "duration": meta.get("duration"),
            "view_count": meta.get("view_count"),
            "upload_date": meta.get("upload_date"),
            "url": url,
        },
        "device": device,
        "asr": {
            "backend": tr.backend,
            "model_size": tr.model_size,
            "language": tr.language or "unknown",
            "translated_to_english": translate,
        },
        "summarizer": {
            "model": summ.model_name,
            "quantized": summ.quantized,
            "imrad": bool(imrad),
            "refinement": bool(refinement),
            "domain": domain,
            "chunk_tokens": summ.chunk_tokens,
            "chunk_overlap": summ.chunk_overlap,
            "audience": audience,
            "compression_ratio": compression_ratio,
            "output_language": output_language,
        },
        "paths": {
            "audio": str(audio_path.resolve()),
            "transcript": str(txt_path.resolve()),
            "docx": str(Path(docx_path).resolve()),
        },
    }

    results = {
        "language": tr.language or "unknown",
        "transcript": transcript,
        "summary": summary_text,
        "keywords": keywords,
        "critical_terms": critical_terms,
        "highlights": highlights,
        "factuality": fact,
    }
    return meta_info, results
