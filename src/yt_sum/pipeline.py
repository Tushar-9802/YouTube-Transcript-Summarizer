# src/yt_sum/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple, List

from yt_sum.utils.config import Config
from yt_sum.utils.downloader import download_audio
from yt_sum.utils.hw import get_device_info, auto_asr_choice, auto_summarizer_params
from yt_sum.utils.asr import transcribe_audio as whisper_transcribe
from yt_sum.models.summarizer import BaselineSummarizer

ProgressFn = Callable[[str, int, Dict[str, Any]], None]  # stage, percent, extra

def run_pipeline(
    url: str,
    cfg: Config,
    whisper_size: Optional[str] = None,              # if None -> auto
    summarizer_model: str = "facebook/bart-large-cnn",
    chunked: bool = True,
    max_new_tokens: Optional[int] = None,            # if None -> auto
    min_new_tokens: Optional[int] = None,
    num_beams: Optional[int] = None,
    chunk_tokens: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    language: Optional[str] = None,                  # force language; None = autodetect
    prefer_accuracy: bool = True,                    # auto selector hint
    progress: Optional[ProgressFn] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      meta_out: dict with video/meta + selections + paths
      results:  dict with transcript + summary
    """
    extra = {}
    def p(stage: str, pct: int):
        if progress:
            progress(stage, max(0, min(100, pct)), extra)

    # Stage 0: device info
    dev = get_device_info()
    extra["device"] = {"cuda": dev.cuda, "name": dev.name, "free_gb": dev.free_gb, "total_gb": dev.total_gb}
    p("init", 0)

    # Stage 1: download
    p("download", 5)
    audio_path, meta = download_audio(url, cfg.audio_dir)
    p("download", 100)

    # Decide ASR
    asr_choice = auto_asr_choice(meta.get("duration"), language, prefer_accuracy) if whisper_size is None else {
        "backend": "faster",
        "model_size": whisper_size,
        "compute_type": "float16" if dev.cuda else "int8",
        "language": language,
    }
    extra["asr_choice"] = asr_choice

    # Stage 2: transcribe
    p("transcribe", 10)
    try:
        if asr_choice["backend"] == "faster":
            try:
                from yt_sum.utils.asr_fast import transcribe_audio_fast
                result = transcribe_audio_fast(
                    audio_path=audio_path,
                    model_size=asr_choice["model_size"],
                    transcript_dir=cfg.transcript_dir,
                    language=asr_choice["language"],
                    compute_type=asr_choice["compute_type"],
                )
            except Exception as e:
                # fallback to whisper if faster-whisper not installed or failed
                result = whisper_transcribe(audio_path, asr_choice["model_size"], cfg.transcript_dir, language=asr_choice["language"])
                result["note"] = f"faster-whisper failed, fell back to openai-whisper: {e}"
        else:
            result = whisper_transcribe(audio_path, asr_choice["model_size"], cfg.transcript_dir, language=asr_choice["language"])
    except Exception as e:
        raise RuntimeError(f"ASR failed: {e}")
    p("transcribe", 100)

    transcript_text = (result.get("text") or "").strip()
    if not transcript_text:
        raise RuntimeError("Empty transcript.")

    # Summarization params
    auto_sum = auto_summarizer_params(transcript_text, result.get("language"))
    if chunk_tokens is None:      chunk_tokens = auto_sum["chunk_tokens"]
    if chunk_overlap is None:     chunk_overlap = auto_sum["chunk_overlap"]
    if num_beams is None:         num_beams = auto_sum["num_beams"]
    if max_new_tokens is None:    max_new_tokens = auto_sum["per_chunk_max"]
    if min_new_tokens is None:    min_new_tokens = auto_sum["per_chunk_min"]
    fuse_max = auto_sum["fuse_max"]
    fuse_min = auto_sum["fuse_min"]

    extra["summarizer"] = {
        "model": summarizer_model,
        "chunked": chunked,
        "chunk_tokens": chunk_tokens,
        "chunk_overlap": chunk_overlap,
        "num_beams": num_beams,
        "per_chunk_max": max_new_tokens,
        "per_chunk_min": min_new_tokens,
        "fuse_max": fuse_max,
        "fuse_min": fuse_min,
    }

    # Stage 3: summarize
    p("summarize", 10)
    from yt_sum.utils.asr import detect_device
    device = detect_device()
    summarizer = BaselineSummarizer(model_name=summarizer_model, device=device if device == "cuda" else None)
    if num_beams and num_beams > 0:
        summarizer.model.config.num_beams = num_beams
        summarizer.model.config.early_stopping = True

    if chunked:
        # chunked path with progress per-chunk (if callback provided)
        from yt_sum.utils.chunker import chunk_by_tokens
        chunks = chunk_by_tokens(transcript_text, summarizer.tokenizer, max_tokens=chunk_tokens, overlap=chunk_overlap)
        n = max(1, len(chunks))
        parts: List[str] = []
        for i, ch in enumerate(chunks, 1):
            parts.append(summarizer.summarize(ch, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, num_beams=num_beams))
            p("summarize", 10 + int(80 * i / n))
        fused = "\n".join(f"- {p_}" for p_ in parts)
        summary = summarizer.summarize(fused, max_new_tokens=fuse_max, min_new_tokens=fuse_min, num_beams=num_beams)
    else:
        summary = summarizer.summarize(transcript_text, max_new_tokens=fuse_max, min_new_tokens=fuse_min, num_beams=num_beams)
        p("summarize", 95)
    p("summarize", 100)

    meta_out = {
        "video": {
            "id": meta.get("id"),
            "title": meta.get("title"),
            "uploader": meta.get("uploader"),
            "duration": meta.get("duration"),
            "view_count": meta.get("view_count"),
            "upload_date": meta.get("upload_date"),
        },
        "selections": extra,
        "paths": {
            "audio": str(audio_path),
            "transcript": str((cfg.transcript_dir / f"{meta.get('id')}.txt").resolve())
        }
    }
    results = {"language": result.get("language"), "transcript": transcript_text, "summary": summary}
    return meta_out, results
