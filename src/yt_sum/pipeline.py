# src/yt_sum/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple, List

from yt_sum.utils.config import Config
from yt_sum.utils.downloader import download_audio
from yt_sum.utils.hw import get_device_info, auto_asr_choice, auto_summarizer_params
from yt_sum.utils.asr import transcribe_audio as whisper_transcribe
from yt_sum.models.summarizer import BaselineSummarizer, GenParams  # <-- NEW

ProgressFn = Callable[[str, int, Dict[str, Any]], None]  # stage, percent, extra

def run_pipeline(
    url: str,
    cfg: Config,
    whisper_size: Optional[str] = None,              # if None -> auto
    summarizer_model: str = "facebook/bart-large-cnn",
    chunked: bool = True,

    # ---- Legacy knobs (still supported for backward-compat) ----
    max_new_tokens: Optional[int] = None,
    min_new_tokens: Optional[int] = None,
    num_beams: Optional[int] = None,
    chunk_tokens: Optional[int] = None,
    chunk_overlap: Optional[int] = None,

    # ---- New summarizer params (preferred) ----
    summarizer_params: Optional[Dict[str, Any]] = None,

    language: Optional[str] = None,                  # force language; None = autodetect
    prefer_accuracy: bool = True,                    # auto selector hint
    progress: Optional[ProgressFn] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      meta_out: dict with video/meta + selections + paths
      results:  dict with transcript + summary
    """
    extra: Dict[str, Any] = {}
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

    # ----------------------------
    # Build summarization settings
    # ----------------------------
    # Auto defaults (used if user didn't provide new params)
    auto_sum = auto_summarizer_params(transcript_text, result.get("language"))
    # Legacy -> defaults (still computed for compatibility/metadata)
    if chunk_tokens is None:      chunk_tokens = auto_sum["chunk_tokens"]
    if chunk_overlap is None:     chunk_overlap = auto_sum["chunk_overlap"]
    if num_beams is None:         num_beams = auto_sum["num_beams"]
    if max_new_tokens is None:    max_new_tokens = auto_sum["per_chunk_max"]
    if min_new_tokens is None:    min_new_tokens = auto_sum["per_chunk_min"]
    fuse_max = auto_sum["fuse_max"]
    fuse_min = auto_sum["fuse_min"]

    # New-style params from UI (preferred)
    sp = summarizer_params or {}
    # Efficiency: default to 8-bit on CUDA unless explicitly disabled
    use_8bit = bool(sp.get("use_8bit", dev.cuda))

    # Compose GenParams (ratio-based budgets). We keep sensible fallbacks.
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

    # Record selections (old + new) for the UI
    extra["summarizer"] = {
        "model": summarizer_model,
        "chunked": chunked,
        # legacy/autos (for reference)
        "legacy": {
            "chunk_tokens_auto": chunk_tokens,
            "chunk_overlap_auto": chunk_overlap,
            "num_beams_auto": num_beams,
            "per_chunk_max_auto": max_new_tokens,
            "per_chunk_min_auto": min_new_tokens,
            "fuse_max_auto": fuse_max,
            "fuse_min_auto": fuse_min,
        },
        # new/active
        "active": {
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
        "impl": "map-reduce, ratio-based length control",
    }

    # Stage 3: summarize
    p("summarize", 10)
    from yt_sum.utils.asr import detect_device
    device = detect_device()

    # Init summarizer with GPU + 8-bit preferences
    summarizer = BaselineSummarizer(
        model_name=summarizer_model,
        device=device if device == "cuda" else None,
        use_8bit=use_8bit,
    )

    # Heuristic: if chunked flag is on OR transcript is lengthy, use summarize_long
    use_long = chunked or (len(transcript_text) > 4000)

    if use_long:
        summary = summarizer.summarize_long(transcript_text, params=gen_params)
    else:
        summary = summarizer.summarize(transcript_text, params=gen_params)

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
