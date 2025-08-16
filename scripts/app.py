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

# Sidebar
youtube_url = st.sidebar.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
summarizer_model = st.sidebar.selectbox("Summarizer Model", ["facebook/bart-large-cnn", "google/pegasus-xsum"], index=0)
chunked = st.sidebar.checkbox("Chunked summarization (recommended)", value=True)
prefer_accuracy = st.sidebar.checkbox("Prefer accuracy over speed (auto ASR)", value=True)
force_whisper_size = st.sidebar.selectbox("ASR size override (optional)", [None, "tiny", "base", "small", "medium", "large-v3"], index=0)
language = st.sidebar.selectbox("Force language (optional)", [None, "en", "hi", "fr", "de", "es"], index=0)

with st.expander("📘 What 'Auto' does"):
    st.markdown(
        "- Detects GPU + free VRAM, then picks **Faster-Whisper** size + compute type.\n"
        "- Tunes summarization **chunk size/overlap**, **beam width**, and **length** from transcript length & VRAM."
    )

stage = st.empty()
progress = st.progress(0)

def progress_cb(which: str, pct: int, extra):
    names = {"init": "Init", "download": "Downloading", "transcribe": "Transcribing", "summarize": "Summarizing"}
    stage.info(f"{names.get(which, which)}… {pct}%")
    progress.progress(pct)
    if which == "init":
        dev = extra.get("device", {})
        st.caption(f"Device: {dev.get('name')} (CUDA={dev.get('cuda')}) • Free {dev.get('free_gb')} / {dev.get('total_gb')} GB")

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

if st.button("▶️ Run"):
    if not youtube_url.strip():
        st.error("Enter a YouTube URL")
        st.stop()

    st.info(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'} • Model: {summarizer_model}")

    meta, res = run_pipeline(
        url=youtube_url,
        cfg=cfg,
        whisper_size=force_whisper_size,  # None => auto
        summarizer_model=summarizer_model,
        chunked=chunked,
        language=language,
        prefer_accuracy=prefer_accuracy,
        progress=progress_cb,
    )

    stage.success("Done."); progress.progress(100)
    title = meta["video"]["title"] or meta["video"]["id"]

    with st.expander("🔎 Auto Choices"):
        sel = meta["selections"]
        st.write({
            "Device": sel.get("device", {}),
            "ASR": sel.get("asr_choice", {}),
            "Summarizer": sel.get("summarizer", {}),
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
