# scripts/app.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from yt_sum.utils.config import Config
from yt_sum.pipeline import run_pipeline, export_docx

st.set_page_config(page_title="Research Summarizer", layout="wide")

st.title("YouTube Research Summarizer (GPU)")

# ---------------- Input URL ----------------
url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Paste the full YouTube video link you want to summarize."
)

# ---------------- Basic Controls ----------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    domain = st.selectbox(
        "Domain",
        ["general", "medical", "engineering", "scientific"],
        index=0,
        help="Guides ASR and summarizer to preserve technical terms."
    )
with col2:
    refinement = st.toggle(
        "Refinement Pass",
        value=True,
        help="If enabled, a second global summarization pass improves cohesion and readability."
    )
with col3:
    prefer_accuracy = st.toggle(
        "Prefer Accuracy (slower ASR)",
        value=True,
        help="If enabled, uses beam search and larger Whisper models for higher transcription accuracy."
    )
with col4:
    whisper_size = st.selectbox(
        "Whisper Model Size",
        ["", "tiny", "base", "small", "medium", "large-v2"],
        index=0,
        help="Force a specific Whisper ASR model. Leave blank for auto-selection."
    )

# ---------------- Advanced Options ----------------
advanced = st.expander("Advanced Controls")
with advanced:
    summarizer_model = st.text_input(
        "Summarizer Model (HF Hub)",
        value="",
        help="e.g. facebook/bart-large-cnn, allenai/led-large-16384-arxiv. Leave blank for auto."
    )
    use_8bit = st.toggle(
        "Quantized 8-bit Summarizer",
        value=True,
        help="Loads summarizer in 8-bit precision to save VRAM. Slight speed trade-off."
    )
    min_len = st.slider(
        "Minimum summary length (tokens)",
        min_value=0, max_value=500, value=150, step=10,
        help="Forces summary to be at least this long."
    )
    max_len = st.slider(
        "Maximum summary length (tokens)",
        min_value=100, max_value=1000, value=700, step=50,
        help="Caps the maximum length of the summary."
    )
    chunk_tokens = st.slider(
        "Chunk size (tokens)",
        min_value=300, max_value=2000, value=1200, step=50,
        help="How many tokens per chunk before summarization. Larger = more context, more VRAM."
    )
    chunk_overlap = st.slider(
        "Chunk overlap (tokens)",
        min_value=0, max_value=500, value=200, step=10,
        help="Overlap between chunks to reduce boundary information loss."
    )

cfg = Config()  # ensures data/audio and data/transcripts dirs exist

# ---------------- Run Button ----------------
if st.button("Run"):
    if not url.strip():
        st.error("Please paste a YouTube URL.")
        st.stop()

    with st.spinner("Processing… may take several minutes for long videos"):
        def progress(stage, pct, extra):
            st.info(f"{stage}: {pct}%")

        meta, res = run_pipeline(
            url,
            cfg,
            domain=domain,
            refinement=refinement,
            whisper_size=(whisper_size or None),
            summarizer_model=(summarizer_model or None),
            use_8bit=use_8bit,
            prefer_accuracy=prefer_accuracy,
            progress=progress,
            min_len=min_len,
            max_len=max_len,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
        )

        # Export DOCX
        docx_path = export_docx(
            f"outputs/{meta['video'].get('id','summary')}",
            res["transcript"],
            res["summary"]
        )
        with open(docx_path, "rb") as f:
            docx_bytes = f.read()
        res["docx_bytes"] = docx_bytes

    # ---------------- Video Metadata ----------------
    st.subheader("Video Information")
    v = meta["video"]
    st.write(f"**Name:** {v.get('title','N/A')}")
    st.write(f"**Link:** https://www.youtube.com/watch?v={v.get('id','')}")
    st.write(f"**Time:** {v.get('duration','N/A')} seconds")
    st.write(f"**Uploader:** {v.get('uploader','N/A')}")

    # Simple keywords from summary (first 10 words)
    if isinstance(res["summary"], str):
        words = res["summary"].split()
        st.write("**Keywords:** " + ", ".join(words[:10]))
    else:
        st.write("**Keywords:** Not available for structured summary")

    # ---------------- Results ----------------
    st.subheader("Summary")
    st.write(res["summary"])

    st.subheader("Transcript")
    st.download_button(
        "Download Transcript (.txt)",
        data=res["transcript"],
        file_name=f"{meta['video'].get('id','transcript')}.txt",
        mime="text/plain"
    )
    st.download_button(
        "Download Summary + Transcript (.docx)",
        data=res["docx_bytes"],
        file_name=f"{meta['video'].get('id','summary')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
