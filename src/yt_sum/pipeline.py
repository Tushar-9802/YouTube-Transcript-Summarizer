# src/yt_sum/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple, List

from yt_sum.utils.config import Config
from yt_sum.utils.downloader import download_audio
from yt_sum.utils.hw import get_device_info, auto_asr_choice, auto_summarizer_params
from yt_sum.utils.asr import transcribe_audio as whisper_transcribe
from yt_sum.models.summarizer import BaselineSummarizer, GenParams

ProgressFn = Callable[[str, int, Dict[str, Any]], None]  # stage, percent, extra

def run_pipeline(
    url: str,
    cfg: Config,
    *,
    # --- ASR / model selection ---
    whisper_size: Optional[str] = None,              # None => auto pick based on GPU/length
    summarizer_model: str = "facebook/bart-large-cnn",
    chunked: bool = True,                            # True => map→reduce path (recommended)

    # --- Legacy knobs (kept for backward-compat) ---
    max_new_tokens: Optional[int] = None,
    min_new_tokens: Optional[int] = None,
    num_beams: Optional[int] = None,
    chunk_tokens: Optional[int] = None,
    chunk_overlap: Optional[int] = None,

    # --- New summarizer params (preferred, a dict from the UI) ---
    summarizer_params: Optional[Dict[str, Any]] = None,

    # --- ASR options ---
    language: Optional[str] = None,                  # None => autodetect
    prefer_accuracy: bool = True,                    # Bigger/fairer ASR if True
    progress: Optional[ProgressFn] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    High-level pipeline:
      1) Detect device, download audio.
      2) Transcribe (Faster-Whisper if possible; Whisper fallback).
      3) Summarize with a single general model (map→reduce if long).
      Returns:
        meta_out: video/meta + selections + file paths
        results : transcript text + summary text
    """

    # Helper to push progress to Streamlit
    extra: Dict[str, Any] = {}
    def p(stage: str, pct: int):
        if progress:
            progress(stage, max(0, min(100, pct)), extra)

    # -------------------------
    # 0) Device / environment
    # -------------------------
    dev = get_device_info()  # GPU/CPU info + VRAM snapshot
    extra["device"] = {"cuda": dev.cuda, "name": dev.name, "free_gb": dev.free_gb, "total_gb": dev.total_gb}
    p("init", 0)

    # -------------------------
    # 1) Download audio
    # -------------------------
    p("download", 5)
    audio_path, meta = download_audio(url, cfg.audio_dir)  # uses yt-dlp; returns file path + metadata
    p("download", 100)

    # -------------------------
    # 2) Transcribe
    # -------------------------
    # Pick ASR model/compute automatically unless the user forced one.
    asr_choice = auto_asr_choice(meta.get("duration"), language, prefer_accuracy) if whisper_size is None else {
        "backend": "faster",
        "model_size": whisper_size,
        "compute_type": "float16" if dev.cuda else "int8",
        "language": language,
    }
    extra["asr_choice"] = asr_choice

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
                # Safe fallback to OpenAI Whisper
                result = whisper_transcribe(
                    audio_path, asr_choice["model_size"], cfg.transcript_dir, language=asr_choice["language"]
                )
                result["note"] = f"faster-whisper failed, fell back to openai-whisper: {e}"
        else:
            result = whisper_transcribe(
                audio_path, asr_choice["model_size"], cfg.transcript_dir, language=asr_choice["language"]
            )
    except Exception as e:
        raise RuntimeError(f"ASR failed: {e}")
    p("transcribe", 100)

    transcript_text = (result.get("text") or "").strip()
    if not transcript_text:
        raise RuntimeError("Empty transcript.")

    # ---------------------------------------------------
    # 3) Summarization settings (auto + user parameters)
    # ---------------------------------------------------
    # Auto budgets from transcript length (used as defaults and logged for reproducibility)
    auto_sum = auto_summarizer_params(transcript_text, result.get("language"))
    # Legacy fallbacks (still computed for metadata / compatibility)
    if chunk_tokens is None:      chunk_tokens = auto_sum["chunk_tokens"]
    if chunk_overlap is None:     chunk_overlap = auto_sum["chunk_overlap"]
    if num_beams is None:         num_beams = auto_sum["num_beams"]
    if max_new_tokens is None:    max_new_tokens = auto_sum["per_chunk_max"]
    if min_new_tokens is None:    min_new_tokens = auto_sum["per_chunk_min"]
    fuse_max = auto_sum["fuse_max"]
    fuse_min = auto_sum["fuse_min"]

    # New-style params (preferred): robust, ratio-based GenParams
    sp = summarizer_params or {}
    use_8bit = bool(sp.get("use_8bit", dev.cuda))  # default to 8-bit on CUDA

    gen_params = GenParams(
        summary_ratio=float(sp.get("summary_ratio", 0.18)),
        reduce_target_tokens=int(sp.get("reduce_target_tokens", fuse_max)),
        num_beams=int(sp.get("num_beams", num_beams or 5)),
        length_penalty=float(sp.get("length_penalty", 0.9)),
        repetition_penalty=float(sp.get("repetition_penalty", 1.08)),
        no_repeat_ngram_size=int(sp.get("no_repeat_ngram_size", 3)),
        do_sample=bool(sp.get("do_sample", False)),
        temperature=float(sp.get("temperature", 1.0)),
        chunk_tokens=int(sp.get("chunk_tokens", chunk_tokens or 900)),
        chunk_overlap=int(sp.get("chunk_overlap", chunk_overlap or 120)),
        guidance=(sp.get("guidance") or None),
    )

    # Log all the choices so the UI can display them (useful for a paper appendix)
    extra["summarizer"] = {
        "model": summarizer_model,
        "chunked": chunked,
        "legacy": {  # auto baselines for reference
            "chunk_tokens_auto": chunk_tokens,
            "chunk_overlap_auto": chunk_overlap,
            "num_beams_auto": num_beams,
            "per_chunk_max_auto": max_new_tokens,
            "per_chunk_min_auto": min_new_tokens,
            "fuse_max_auto": fuse_max,
            "fuse_min_auto": fuse_min,
        },
        "active": {  # actually used
            "reduce_target_tokens": gen_params.reduce_target_tokens,
            "summary_ratio": gen_params.summary_ratio,
            "num_beams": gen_params.num_beams,
            "chunk_tokens": gen_params.chunk_tokens,
            "chunk_overlap": gen_params.chunk_overlap,
            "no_repeat_ngram_size": gen_params.no_repeat_ngram_size,
            "length_penalty": gen_params.length_penalty,
            "repetition_penalty": gen_params.repetition_penalty,
            "do_sample": gen_params.do_sample,
            "temperature": gen_params.temperature if gen_params.do_sample else None,
            "use_8bit": use_8bit,
            "guidance": (gen_params.guidance[:80] + "…") if (gen_params.guidance and len(gen_params.guidance) > 80) else gen_params.guidance,
        },
        "impl": "single-model, map→reduce, ratio-based length control",
    }

    # -------------------------
    # 4) Summarize
    # -------------------------
    p("summarize", 10)

    # Pick the device for the summarizer (CUDA if available)
    from yt_sum.utils.asr import detect_device
    device = detect_device()

    summarizer = BaselineSummarizer(
        model_name=summarizer_model,              # one base model (Route-1)
        device=device if device == "cuda" else None,
        use_8bit=use_8bit,                        # save VRAM on GPUs
    )

    # Prefer map→reduce for coverage, or single-shot for short transcripts
    use_long = chunked or (len(transcript_text) > 4000)  # char-length heuristic is okay here
    if use_long:
        summary = summarizer.summarize_long(transcript_text, params=gen_params)
    else:
        summary = summarizer.summarize(transcript_text, params=gen_params)

    p("summarize", 100)

    # -------------------------
    # 5) Package outputs
    # -------------------------
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
    results = {
        "language": result.get("language"),
        "transcript": transcript_text,
        "summary": summary,
    }
    return meta_out, results
