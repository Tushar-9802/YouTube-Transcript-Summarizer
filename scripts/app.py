from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import psutil
import platform

try:
    import cpuinfo
    cpu_name = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
except ImportError:
    cpu_name = platform.processor() or "Unknown CPU"

# ---------------- Path Setup ----------------
ROOT = Path(__file__).resolve().parents[1]  # project root (YT-S)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yt_sum.pipeline import run_pipeline

# ---------------- Page Config ----------------
st.set_page_config(page_title="Research Summarizer Pro", layout="wide")
st.title("YouTube Research Summarizer")

# ---------------- Session State ----------------
if "use_defaults" not in st.session_state:
    st.session_state.use_defaults = True

def disable_defaults():
    """Turn off default settings if user changes a parameter."""
    st.session_state.use_defaults = False

# ---------------- Inputs ----------------
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

c1, c2, c3, c4 = st.columns(4)
with c1:
    domain = st.selectbox(
        "Domain",
        ["general", "medical", "engineering", "scientific"],
        index=0,
        help="Select the knowledge domain. Guides transcription and summarization to preserve technical terms.",
        on_change=disable_defaults,
    )
with c2:
    imrad = st.toggle(
        "IMRaD Sections",
        value=False,
        help="Organize summary into Introduction, Methods, Results, and Limitations.",
        on_change=disable_defaults,
    )
with c3:
    refinement = st.toggle(
        "Refinement Pass",
        value=True,
        help="Run a second pass to improve cohesion, reduce redundancy, and ensure jargon retention.",
        on_change=disable_defaults,
    )
with c4:
    prefer_accuracy = st.toggle(
        "Prefer Accuracy (ASR)",
        value=True,
        help="Enable beam search and larger Whisper models for higher transcription accuracy (slower).",
        on_change=disable_defaults,
    )

# ---------------- Defaults Toggle ----------------
st.session_state.use_defaults = st.checkbox(
    "Use Recommended Defaults",
    value=st.session_state.use_defaults,
    help="Enable this for balanced configuration (8GB+ GPU). Adjusting any parameter will turn this off."
)

# ---------------- Advanced Controls ----------------
adv = st.expander("Advanced Controls")
with adv:
    whisper_size = st.selectbox(
        "Whisper Model Size",
        ["", "tiny", "base", "small", "medium", "large-v2"],
        index=5 if st.session_state.use_defaults else 0,
        help="Choose the ASR model size. Larger models = more accurate, more VRAM.",
        on_change=disable_defaults,
    )
    summarizer_model = st.text_input(
        "Summarizer Model (HF Hub)",
        value="" if st.session_state.use_defaults else "facebook/bart-large-cnn",
        help="Select summarizer model. Leave blank for auto-selection based on GPU.",
        on_change=disable_defaults,
    )
    use_8bit = st.toggle(
        "Quantized 8-bit Summarizer",
        value=True if st.session_state.use_defaults else False,
        help="Use 8-bit precision for summarizer. Saves VRAM, small performance trade-off.",
        on_change=disable_defaults,
    )
    min_len = st.slider(
        "Minimum summary length (tokens)",
        0, 800, 140 if st.session_state.use_defaults else 200, 10,
        help="Lower = shorter summary, higher = more detailed.",
        on_change=disable_defaults,
    )
    max_len = st.slider(
        "Maximum summary length (tokens)",
        200, 1400, 600 if st.session_state.use_defaults else 800, 25,
        help="Upper bound on summary length. Increase for detail, reduce for conciseness.",
        on_change=disable_defaults,
    )
    chunk_tokens = st.slider(
        "Chunk size (tokens)",
        400, 2400, 900 if st.session_state.use_defaults else 1200, 50,
        help="Tokens per transcript chunk. Larger = more context, needs more VRAM.",
        on_change=disable_defaults,
    )
    chunk_overlap = st.slider(
        "Chunk overlap (tokens)",
        0, 600, 120 if st.session_state.use_defaults else 200, 10,
        help="Overlap tokens between chunks. Prevents context loss at boundaries.",
        on_change=disable_defaults,
    )
    translate_non_english = st.toggle(
        "Translate non-English",
        value=True if st.session_state.use_defaults else False,
        help="Translate non-English transcripts into English before summarization.",
        on_change=disable_defaults,
    )
    compression_ratio = st.slider(
        "Compression ratio (%)",
        5, 80, 20 if st.session_state.use_defaults else 30, 5,
        help="How much to condense transcript into summary. Low = detailed, High = concise.",
        on_change=disable_defaults,
    ) / 100.0
    audience = st.selectbox(
        "Audience",
        ["expert", "student"],
        index=0 if st.session_state.use_defaults else 1,
        help="expert = jargon-rich, student = simplified.",
        on_change=disable_defaults,
    )
    output_language = st.selectbox(
        "Output language",
        ["auto", "en", "source"],
        index=0,
        help="auto = system decides, en = always English, source = keep original language.",
        on_change=disable_defaults,
    )
    enable_factcheck = st.toggle(
        "Enable factuality check",
        value=True if st.session_state.use_defaults else False,
        help="Validate summary sentences against transcript evidence with an NLI model.",
        on_change=disable_defaults,
    )

workdir = ROOT / "run_data"
workdir.mkdir(parents=True, exist_ok=True)

# ---------------- Run Button ----------------
if st.button("Run"):
    if not url.strip():
        st.error("Please paste a YouTube URL.")
        st.stop()

    with st.spinner("Processing…"):
        meta, res = run_pipeline(
            url,
            workdir,
            domain=domain,
            whisper_size=(whisper_size or None),
            prefer_accuracy=prefer_accuracy,
            summarizer_model=(summarizer_model or None),
            use_8bit=use_8bit,
            refinement=refinement,
            imrad=imrad,
            min_len=min_len or None,
            max_len=max_len or None,
            chunk_tokens=chunk_tokens or None,
            chunk_overlap=chunk_overlap or None,
            translate_non_english=translate_non_english,
            compression_ratio=compression_ratio,
            audience=audience,
            output_language=None if output_language == "auto" else output_language,
            enable_factcheck=enable_factcheck,
        )

    st.success("Done!")
   # ---------------- Device & Resource Info ----------------
    st.subheader("System Resource Usage")

    # CPU
    cpu_count = psutil.cpu_count(logical=True)
    st.write(f"**CPU:** {cpu_name} ({cpu_count} cores)")

    # RAM
    ram = psutil.virtual_memory()
    st.write(
        f"**RAM:** {ram.total / (1024**3):.1f} GB total, "
    )

    # GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        total_vram = gpu_props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)

        st.write(
            f"**GPU:** {gpu_name} "
            f"— VRAM {total_vram:.1f} GB total | "
        )
    else:
        st.write("**GPU:** None detected (running on CPU).")



    # ---------------- Display ----------------
    st.subheader("Video Info")
    st.write(f"**Title:** {meta['video']['title']}")
    st.write(f"**Uploader:** {meta['video']['uploader']}")
    st.write(f"**Duration:** {meta['video']['duration']}s")
    st.write(f"**Views:** {meta['video']['view_count']}")
    st.write(f"**Date:** {meta['video']['upload_date']}")
    st.write(f"**URL:** {meta['video']['url']}")

    st.subheader("Summary")
    st.write(res["summary"])

    st.subheader("Keywords")
    st.write(", ".join(res["keywords"]) if res["keywords"] else "—")

    st.subheader("Critical Terms (numbers, acronyms, formulae)")
    st.write(", ".join(res["critical_terms"]) if res["critical_terms"] else "—")

    st.subheader("Extractive Highlights")
    for h in res["highlights"]:
        st.markdown(f"- {h}")

    if res["factuality"]:
        st.subheader("Factuality")
        st.write(f"**Overall confidence:** {res['factuality'].get('overall_confidence', 0.0):.3f}")
        with st.expander("Per-sentence evidence"):
            for item in res["factuality"].get("sentences", []):
                st.markdown(f"- **Summary:** {item['sentence']}")
                st.markdown(f"  - Confidence: `{item['confidence']:.3f}`")
                for ev in item.get("supports", []):
                    st.markdown(f"  - Evidence (e={ev['entailment']:.3f}): {ev['evidence']}")

    st.subheader("Transcript")
    st.download_button(
        "Download Transcript (.txt)",
        data=res["transcript"],
        file_name=f"{meta['video'].get('id','transcript')}.txt",
        mime="text/plain",
    )

    st.subheader("Exports")
    try:
        with open(meta["paths"]["docx"], "rb") as f:
            st.download_button(
                "Download Summary + Transcript (.docx)",
                data=f.read(),
                file_name=f"{meta['video'].get('id','summary')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
    except Exception:
        st.warning("DOCX export missing.")

# ---------------- Detailed Help Section ----------------
with st.expander("Help & Recommended Settings", expanded=False):
    st.markdown("""
    ## ASR (Whisper)
    - **tiny (39M, ~1GB VRAM):** fastest, lowest accuracy  
    - **base (74M, ~1.3GB):** light, decent accuracy  
    - **small (244M, ~2GB):** balanced speed/accuracy  
    - **medium (769M, ~5GB):** strong accuracy, slower  
    - **large-v2 (1550M, ~10GB):** best accuracy, needs high-end GPU  

    Use **Prefer Accuracy (ASR)** for beam search and higher fidelity.  
    For quick drafts, disable it for speed.

    ---

    ## Summarizer Models
    - **facebook/bart-large-cnn:** default, good for short/medium transcripts.  
    - **allenai/led-large-16384-arxiv:** handles long texts (16GB+ VRAM).  
    - **google/long-t5-local-base:** best for very long contexts (20GB+ VRAM).  
    - **Quantized 8-bit:** saves VRAM, slight slowdown, minimal accuracy loss.  

    ---

    ## Chunking & Overlap
    - **Chunk size:** maximum transcript tokens per piece. Larger = more context, more VRAM.  
    - **Overlap:** repeated tokens between chunks so no details are lost. Typical: 100–200.  

    ---

    ## Summary Length & Compression
    - **Min/Max length:** bounds for output size (tokens).  
    - **Compression ratio:** target reduction of transcript → summary.  
      - Low (10–20%): more detail.  
      - High (50–70%): very compressed.  

    ---

    ## Factuality Check
    Uses an NLI model to validate summary sentences against transcript evidence.  
    - Confidence closer to 1.0 = well grounded.  
    - Useful for research/academic videos.  

    ---

    ## Recommended Defaults (GPU tiers)
    - **8 GB:** Whisper `small` or `medium`, BART-large-CNN (8-bit), chunk=900, overlap=120.  
    - **16 GB:** Whisper `medium`, LED-16384, chunk=1200, overlap=200.  
    - **24+ GB:** Whisper `large-v2`, LongT5, chunk=1800+, overlap=300.  

    These are pre-set when you tick **Use Recommended Defaults**.
    """)
# ---------------- End ----------------