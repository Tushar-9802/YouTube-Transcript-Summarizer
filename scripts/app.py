import os
from pathlib import Path
import time
from typing import Optional, List

import streamlit as st
import torch
from docx import Document
import yt_dlp

# ---- your project imports (make sure PYTHONPATH=./src) ----
from yt_sum.utils.config import Config
from yt_sum.utils.logging import get_logger
from yt_sum.models.summarizer import BaselineSummarizer
from yt_sum.utils.chunker import chunk_by_tokens  # if you haven't added chunker.py yet, disable chunked mode
from yt_sum.utils.asr import transcribe_audio, detect_device

log = get_logger("yt-sum-ui")
cfg = Config("configs/default.yaml")
cfg.ensure_dirs()

# ---------------- UI boilerplate ----------------
st.set_page_config(page_title="YouTube Transcript Summarizer", layout="wide")
st.title("🎥 YouTube Transcript Summarizer")

# CSS: fixed GPU widget (bottom-left)
GPU_BOX_CSS = """
<style>
.gpu-box {
  position: fixed;
  left: 16px;
  bottom: 16px;
  background: #111;
  color: #ddd;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 12px;
  opacity: 0.9;
  z-index: 9999;
}
.gpu-box .ok { color: #7CFC00; }   /* green */
.gpu-box .warn { color: #ffcc00; } /* yellow */
.gpu-box .bad { color: #ff6666; }  /* red */
</style>
"""
st.markdown(GPU_BOX_CSS, unsafe_allow_html=True)
gpu_box = st.empty()  # we'll update this with markdown during the run


def format_vram_box() -> str:
    """Return HTML snippet for the GPU widget."""
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024**3)
        # note: allocated = tensors; reserved = caching allocator
        alloc = torch.cuda.memory_allocated(0) / (1024**3)
        reserv = torch.cuda.memory_reserved(0) / (1024**3)
        pct = (alloc / total) * 100 if total > 0 else 0
        cls = "ok" if pct < 50 else "warn" if pct < 80 else "bad"
        return f"""
<div class="gpu-box">
  <div><b>GPU:</b> {dev}</div>
  <div><b>VRAM:</b> {alloc:.2f} / {total:.1f} GB (alloc)</div>
  <div><span class="{cls}"><b>Usage:</b> {pct:.0f}%</span></div>
</div>
"""
    else:
        return """
<div class="gpu-box">
  <div><b>GPU:</b> CPU mode</div>
  <div>CUDA not available</div>
</div>
"""


def update_gpu_widget():
    gpu_box.markdown(format_vram_box(), unsafe_allow_html=True)


# ---------------- helpers ----------------
def download_audio_with_progress(url: str, audio_dir: Path, progress_slot, status_slot) -> Path:
    """
    Download bestaudio and convert to mp3, reporting progress into Streamlit.
    Returns audio file path.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(audio_dir / "%(id)s.%(ext)s")

    # yt-dlp progress hook
    prog = {"p": 0}
    def hook(d):
        if d["status"] == "downloading":
            # tdl: d.get('_percent_str') like ' 23.4%'
            try:
                pct = float(d.get("_percent_str", "0%").strip().replace("%", ""))
            except Exception:
                pct = 0.0
            prog["p"] = max(0, min(100, pct))
            progress_slot.progress(int(prog["p"]))
            status_slot.info(f"Downloading… {prog['p']:.1f}%")
            update_gpu_widget()
        elif d["status"] == "finished":
            progress_slot.progress(100)
            status_slot.success("Download finished. Converting to mp3…")
            update_gpu_widget()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "progress_hooks": [hook],
        "quiet": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = audio_dir / f"{info['id']}.mp3"
        status_slot.success(f"Audio saved: {audio_path.name}")
        return audio_path, info


def export_docx(title: str, transcript: str, summary: str) -> Path:
    out = cfg.outputs_dir / f"{title}.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    doc.add_heading("YouTube Transcript Summary", level=0)
    doc.add_paragraph(f"Title: {title}")
    doc.add_paragraph("")
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(summary)
    doc.add_heading("Transcript", level=1)
    doc.add_paragraph(transcript)
    doc.save(out)
    return out


# ---------------- sidebar controls ----------------
st.sidebar.header("Input")
youtube_url = st.sidebar.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
whisper_size = st.sidebar.selectbox("Whisper Size", ["tiny", "base", "small", "medium", "large"], index=2)
summarizer_model = st.sidebar.selectbox(
    "Summarizer Model",
    ["facebook/bart-large-cnn", "google/pegasus-xsum"],
    index=0,
)
chunked = st.sidebar.checkbox("Chunked summarization (recommended)", value=True)

st.sidebar.header("Advanced Summary Controls")
max_new_tokens = st.sidebar.slider("Max New Tokens", 50, 600, 300, step=10)
min_new_tokens = st.sidebar.slider("Min New Tokens", 10, 400, 120, step=10)
num_beams = st.sidebar.slider("Beam Width (num_beams)", 1, 8, 4)
chunk_tokens = st.sidebar.slider("Tokens per Chunk (chunked)", 200, 1500, 1000, step=50)
chunk_overlap = st.sidebar.slider("Overlap Tokens (chunked)", 0, 300, 120, step=10)

with st.expander("📘 Summary Controls Guide"):
    st.markdown("""
- **Max New Tokens**: higher = longer, more detailed summaries (too high may repeat).
- **Min New Tokens**: ensures the model doesn't stop too early.
- **Beam Width**: higher = more faithful/consistent, slower (4–6 is a sweet spot).
- **Tokens per Chunk**: more context per chunk; too large risks VRAM OOM on smaller GPUs.
- **Overlap Tokens**: keeps continuity between chunks; 80–160 is typical.
    """)

# ---------------- main action ----------------
col_left, col_right = st.columns([2, 3])
with col_left:
    run = st.button("▶️ Run")
with col_right:
    export_button_slot = st.empty()

stage_status = st.empty()
progress_bar = st.progress(0)
update_gpu_widget()  # draw initial widget

transcript_area = st.empty()
summary_area = st.empty()

if run:
    if not youtube_url.strip():
        st.error("Please enter a YouTube URL")
        st.stop()

    # Stage 0: device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Using device: **{device.upper()}**  |  Whisper: **{whisper_size}**  |  Summarizer: **{summarizer_model}**")
    update_gpu_widget()

    # Stage 1: download
    stage_status.info("Stage 1/4 • Downloading audio…")
    progress_bar.progress(5)
    update_gpu_widget()
    audio_path, meta = download_audio_with_progress(youtube_url, cfg.audio_dir, progress_bar, stage_status)
    title = meta.get("title", meta.get("id", "video")).replace(":", " ").replace("/", " ")
    progress_bar.progress(25)
    update_gpu_widget()

    # Stage 2: transcribe (Whisper)
    stage_status.info("Stage 2/4 • Transcribing with Whisper…")
    update_gpu_widget()
    # use our existing ASR util; it auto-picks device
    result = transcribe_audio(audio_path, whisper_size, cfg.transcript_dir)
    transcript_text = result.get("text", "").strip()
    if not transcript_text:
        st.error("Transcription produced empty text. Check audio/ffmpeg/whisper setup.")
        st.stop()
    stage_status.success(f"Transcribed • Detected language: {result.get('language')}")
    progress_bar.progress(55)
    update_gpu_widget()

    # Stage 3: summarization
    stage_status.info("Stage 3/4 • Summarizing…")
    device = detect_device()
    summarizer = BaselineSummarizer(model_name=summarizer_model, device=device if device == "cuda" else None)

    # configure decoding knobs
    if num_beams and num_beams > 0:
        summarizer.model.config.num_beams = num_beams
        summarizer.model.config.early_stopping = True

    if chunked:
        # chunk, summarize each with per-chunk progress
        chunks: List[str] = chunk_by_tokens(
            transcript_text,
            summarizer.tokenizer,
            max_tokens=chunk_tokens,
            overlap=chunk_overlap,
        )
        n = len(chunks)
        partials: List[str] = []
        if n == 0:
            st.error("No chunks produced. Try smaller Tokens per Chunk.")
            st.stop()

        for i, ch in enumerate(chunks, 1):
            stage_status.info(f"Stage 3/4 • Summarizing chunk {i}/{n}…")
            update_gpu_widget()
            part = summarizer.summarize(ch, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
            partials.append(part)
            # progress: from 55 up to 95 across chunks
            pct = 55 + int(40 * (i / n))
            progress_bar.progress(min(pct, 95))
        fused_input = "\n".join(f"- {p}" for p in partials)
        summary = summarizer.summarize(fused_input, max_new_tokens=max_new_tokens + 100, min_new_tokens=min_new_tokens + 40)
    else:
        summary = summarizer.summarize(transcript_text, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
        progress_bar.progress(95)
    update_gpu_widget()

    # Stage 4: show + export
    stage_status.success("Stage 4/4 • Done.")
    progress_bar.progress(100)

    st.subheader("Transcript")
    transcript_area.text_area("Full Transcript", transcript_text, height=280)

    st.subheader("Summary")
    summary_area.text_area("Summary Output", summary, height=280)

    with export_button_slot.container():
        if st.button("💾 Export as DOCX"):
            out_path = export_docx(title, transcript_text, summary)
            st.success(f"Saved: {out_path}")
            with open(out_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download DOCX",
                    data=f,
                    file_name=f"{Path(out_path).name}",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
