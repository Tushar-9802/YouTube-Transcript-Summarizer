from __future__ import annotations
from pathlib import Path
from docx import Document
import torch

from yt_sum.models.transcriber import Transcriber
from yt_sum.models.summarizer import Summarizer
from yt_sum.utils.downloader import download_audio


def run_pipeline(
    url: str,
    cfg,
    *,
    domain: str = "general",
    whisper_size: str | None = None,
    language: str | None = None,
    prefer_accuracy: bool = True,
    summarizer_model: str | None = None,
    use_8bit: bool | None = None,
    refinement: bool = True,
    min_len: int | None = None,        # <-- add this
    max_len: int | None = None,        # <-- and this
    chunk_tokens: int | None = None,
    chunk_overlap: int | None = None,
    progress=None,
):

    """
    Download → Transcribe (domain-aware) → Summarize (domain-aware).
    Returns (meta_out, results).
    """
    extra = {}

    def report(stage: str, pct: int):
        if progress:
            progress(stage, max(0, min(100, pct)), extra)

    # ---------------- Device info ----------------
    dev = {"cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        dev.update(name=props.name, total_gb=round(props.total_memory / 2**30, 2))
    else:
        dev.update(name="CPU", total_gb=None)
    extra["device"] = dev
    report("init", 0)

    # ---------------- Download ----------------
    report("download", 5)
    audio_path, meta = download_audio(url, cfg.audio_dir)
    report("download", 100)

    # ---------------- Transcribe ----------------
    report("transcribe", 10)
    trans = Transcriber(
        model_size=whisper_size,
        language=language,
        beam_size=5 if prefer_accuracy else 1,
    )
    segments = trans.transcribe(str(audio_path), domain=domain)
    transcript = " ".join(s["text"] for s in segments).strip()
    if not transcript:
        raise RuntimeError("Empty transcript – download or ASR likely failed.")
    extra["asr_choice"] = {
        "backend": trans.backend,
        "model_size": trans.model_size,
        "language": trans.language,
    }
    report("transcribe", 100)

    # ---------------- Summarize ----------------
    report("summarize", 10)
    summ = Summarizer(model_name=summarizer_model, use_8bit=use_8bit)

    # Override defaults from UI
    if chunk_tokens:
        summ.chunk_tokens = chunk_tokens
    if chunk_overlap:
        summ.chunk_overlap = chunk_overlap

    summary_text = summ.summarize_long(
    transcript,
    refinement=refinement,
    domain=domain,
    min_len=min_len,
    max_len=max_len,
)

    report("summarize", 100)

    # ---------------- Save transcript ----------------
    if meta.get("id"):
        tpath = cfg.transcript_dir / f"{meta['id']}.txt"
        cfg.transcript_dir.mkdir(parents=True, exist_ok=True)
        Path(tpath).write_text(transcript, encoding="utf-8")

    meta_out = {
        "video": {
            "id": meta.get("id"),
            "title": meta.get("title"),
            "uploader": meta.get("uploader"),
            "duration": meta.get("duration"),
            "view_count": meta.get("view_count"),
            "upload_date": meta.get("upload_date"),
        },
        "selections": {
            "device": extra["device"],
            "asr_choice": extra["asr_choice"],
            "summarizer": {
                "model": summ.model_name,
                "quantized": getattr(summ, "quantized", False),
                "refinement": bool(refinement),
                "domain": domain,
                "chunk_tokens": summ.chunk_tokens,
                "chunk_overlap": summ.chunk_overlap,
            },
        },
        "paths": {
            "audio": str(audio_path),
            "transcript": str(
                (cfg.transcript_dir / f"{meta.get('id')}.txt").resolve()
            )
            if meta.get("id")
            else None,
        },
    }

    results = {
        "language": trans.language or "unknown",
        "transcript": transcript,
        "summary": summary_text,
    }
    return meta_out, results


def export_docx(base_filename: str, transcript: str, summary: str) -> str:
    """
    Export transcript + summary to DOCX.
    """
    out = Path(f"{base_filename}.docx")
    doc = Document()
    doc.add_heading("Video Summary", 0)

    doc.add_heading("Summary", level=1)
    for para in summary.split("\n"):
        if para.strip():
            doc.add_paragraph(para.strip())

    doc.add_heading("Transcript", level=1)
    for para in transcript.split("\n"):
        if para.strip():
            doc.add_paragraph(para.strip())

    doc.save(out)
    return str(out)
