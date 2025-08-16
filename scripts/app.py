# scripts/app.py
# --- make ./src importable no matter how Streamlit starts the process ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st
import torch
from docx import Document

from yt_sum.utils.config import Config
from yt_sum.pipeline import run_pipeline

st.set_page_config(page_title="YouTube Transcript Summarizer", layout="wide")
st.title("🎥 YouTube Transcript Summarizer")

cfg = Config("configs/default.yaml")
cfg.ensure_dirs()

# -----------------------
# Sidebar: Inputs/Controls
# -----------------------
with st.sidebar:
    st.header("Settings")

    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste the full video URL. The app will fetch metadata, audio, and transcript (via Whisper if needed)."
    )

    # Models (keep HF IDs for compatibility; your summarizer also accepts these)
    summarizer_model = st.selectbox(
        "Summarizer Model",
        ["facebook/bart-large-cnn",
         "google/pegasus-cnn_dailymail",
         "google/pegasus-large",
         "t5-large"],
        index=0,
        help="Prefer CNN/DailyMail checkpoints for richer, longer summaries. "
             "Avoid XSum (ultra-brief)."
    )

    # Core summarization controls
    st.subheader("Summary Controls")
    final_tokens = st.slider(
        "Final summary size (tokens)",
        min_value=160, max_value=600, value=320, step=20,
        help="Target length for the FINAL fused summary.\n"
             "Tip: 280–360 is a good paper-style abstract; raise for long lectures."
    )
    per_chunk_ratio = st.slider(
        "Per-chunk ratio",
        min_value=0.08, max_value=0.35, value=0.18, step=0.01,
        help="Approximate fraction of each chunk to keep in the first-pass summaries. "
             "Higher = more detail carried into the final fuse."
    )
    beams = st.slider(
        "Detail level (num_beams)",
        min_value=1, max_value=8, value=5, step=1,
        help="Beam search width. Higher = more thorough but slower. 4–6 is a solid default."
    )
    chunk_overlap = st.slider(
        "Chunk overlap (tokens)",
        min_value=64, max_value=256, value=120, step=16,
        help="How much the chunks overlap to avoid cutting sentences. "
             "Increase slightly if you see context being dropped between chunks."
    )
    chunked = st.checkbox(
        "Chunked summarization (recommended)",
        value=True,
        help="Enable map→reduce summarization for long transcripts. Prevents truncation and improves coverage."
    )

    # Decoding style
    st.subheader("Decoding Style")
    use_sampling = st.checkbox(
        "Enable sampling (more creative)",
        value=False,
        help="Turns on sampling for more varied phrasing. For research/reproducibility, leave OFF."
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.2, max_value=1.5, value=0.9, step=0.1,
        help="Higher = more randomness. Ignored if sampling is OFF.",
        disabled=not use_sampling
    )

    # Efficiency / precision
    st.subheader("Performance & ASR")
    use_8bit = st.checkbox(
        "Use 8-bit (GPU) quantization",
        value=True,
        help="Loads the summarizer in 8-bit to save VRAM with minimal quality loss. "
             "Requires a CUDA GPU. Falls back automatically if unavailable."
    )
    prefer_accuracy = st.checkbox(
        "Prefer accuracy over speed (auto ASR)",
        value=True,
        help="Lets the pipeline choose a larger Whisper model / safer compute for better transcription quality."
    )
    force_whisper_size = st.selectbox(
        "ASR size override (optional)",
        [None, "tiny", "base", "small", "medium", "large-v3"],
        index=0,
        help="Force a specific Whisper size. Leave at None for the app to auto-pick based on GPU/length."
    )
    language = st.selectbox(
        "Force language (optional)",
        [None, "en", "hi", "fr", "de", "es"],
        index=0,
        help="If the video language is known, set it here to speed up and improve ASR. Otherwise leave None."
    )

    # Guided / domain-aware hook (already usable)
    st.subheader("Guided Summary (optional)")
    guidance = st.text_area(
        "Guidance / focus",
        placeholder="e.g., Produce a sectioned summary: Background, Methods, Results, Limitations.",
        height=90,
        help="Add light instructions to steer the summary. Works best with BART/PEGASUS/T5. "
             "Keep it concise and factual for research outputs."
    )

# In-UI explanations / docs
with st.expander("📘 What 'Auto' does & control guide"):
    st.markdown(
        """
**Auto choices:**  
- Detects GPU & free VRAM, then chooses a suitable **Faster-Whisper** size and compute type.  
- Tunes **chunk size/overlap**, **beam width**, and **length budgets** relative to transcript length, aiming for stable coverage.

**Controls:**  
- **Final summary size (tokens):** Target length for the final output after fusing chunk summaries.  
- **Per-chunk ratio:** How much of each chunk is kept in its first-pass summary (higher → more detail → longer final).  
- **Detail level (num_beams):** Wider beam search improves thoroughness but costs time.  
- **Chunk overlap:** Overlaps chunk boundaries to avoid context loss.  
- **Sampling & temperature:** For creative phrasing; keep off for reproducible research.  
- **8-bit (GPU):** Saves VRAM with little quality loss on modern GPUs.  
- **Guidance:** Add a short instruction to shape structure/emphasis (e.g., Methods/Results/Limitations).
        """
    )

# ---------------
# Progress display
# ---------------
stage = st.empty()
progress = st.progress(0)

def progress_cb(which: str, pct: int, extra):
    names = {"init": "Init", "download": "Downloading", "transcribe": "Transcribing", "summarize": "Summarizing"}
    stage.info(f"{names.get(which, which)}… {pct}%")
    progress.progress(pct)
    if which == "init":
        dev = extra.get("device", {})
        st.caption(f"Device: {dev.get('name')} (CUDA={dev.get('cuda')}) • Free {dev.get('free_gb')} / {dev.get('total_gb')} GB")

# --------------
# DOCX export fn
# --------------
def export_docx(title: str, transcript: str, summary: str) -> Path:
    out = cfg.outputs_dir / f"{title}.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    doc.add_heading("YouTube Transcript Summary", level=0)
    doc.add_paragraph(f"Title: {title}")
    doc.add_paragraph("")
    doc.add_heading("Summary", level=1); doc.add_paragraph(summary)
    doc.add_heading("Transcript", level=1); doc.add_paragraph(transcript)
    doc.save(out)
    return out

# -----
# RUN
# -----
if st.button("▶️ Run"):
    if not youtube_url.strip():
        st.error("Enter a YouTube URL")
        st.stop()

    st.info(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'} • Model: {summarizer_model}")

    # Package summarizer (map→reduce) parameters for the pipeline / summarizer
    summarizer_params = {
        # length control
        "reduce_target_tokens": int(final_tokens),
        "summary_ratio": float(per_chunk_ratio),
        # decoding
        "num_beams": int(beams),
        "no_repeat_ngram_size": 3,
        "length_penalty": 0.9,
        "repetition_penalty": 1.08,
        "do_sample": bool(use_sampling),
        "temperature": float(temperature) if use_sampling else 1.0,
        # chunking
        "chunk_tokens": 900,          # leave headroom under 1024 encoder limit
        "chunk_overlap": int(chunk_overlap),
        # guided
        "guidance": guidance or None,
        # efficiency
        "use_8bit": bool(use_8bit),
    }

    # NOTE: Update your pipeline to accept **summarizer_params** and forward them
    # to the Summarizer/GenParams (or translate as needed).
    meta, res = run_pipeline(
        url=youtube_url,
        cfg=cfg,
        whisper_size=force_whisper_size,   # None => auto
        summarizer_model=summarizer_model,
        chunked=chunked,
        language=language,
        prefer_accuracy=prefer_accuracy,
        summarizer_params=summarizer_params,   # <— NEW
        progress=progress_cb,
    )

    stage.success("Done."); progress.progress(100)
    title = meta["video"]["title"] or meta["video"]["id"]

    with st.expander("🔎 Auto Choices"):
        sel = meta.get("selections", {})
        st.write({
            "Device": sel.get("device", {}),
            "ASR": sel.get("asr_choice", {}),
            "Summarizer": {
                **sel.get("summarizer", {}),
                "final_target_tokens": final_tokens,
                "per_chunk_ratio": per_chunk_ratio,
                "num_beams": beams,
                "chunk_overlap": chunk_overlap,
                "use_8bit": use_8bit,
                "sampling": use_sampling,
                "temperature": temperature if use_sampling else None,
                "guidance": (guidance[:80] + "…") if guidance and len(guidance) > 80 else guidance
            },
        })

    st.subheader("Summary")
    st.text_area("Summary Output", res["summary"], height=260)

    st.subheader("Transcript")
    st.text_area("Full Transcript", res["transcript"], height=260)

    if st.button("💾 Export as DOCX"):
        safe = (title or "video").replace(":", " ").replace("/", " ")
        out = export_docx(safe, res["transcript"], res["summary"])
        st.success(f"Saved: {out}")
        with open(out, "rb") as f:
            st.download_button(
                "⬇️ Download DOCX",
                f,
                file_name=Path(out).name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
