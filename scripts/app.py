# Scripts/app.py
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import psutil
import platform
import torch

try:
    import cpuinfo
    CPU_NAME = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
except ImportError:
    CPU_NAME = platform.processor() or "Unknown CPU"

# Path setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yt_sum.pipeline import run_pipeline

# Page config
st.set_page_config(page_title="YouTube Summarizer", layout="wide", initial_sidebar_state="collapsed")

# Title with VRAM indicator
col1, col2 = st.columns([3, 1])
with col1:
    st.title("YouTube Research Summarizer")
with col2:
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        st.metric("VRAM", f"{vram_gb:.1f} GB")

# Session state
if "results" not in st.session_state:
    st.session_state.results = None
if "meta" not in st.session_state:
    st.session_state.meta = None

# Main input
url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Paste any YouTube video URL (works with videos up to 2 hours)"
)

# Quick settings (most common)
col1, col2, col3, col4 = st.columns(4)

with col1:
    domain = st.selectbox(
        "Domain",
        ["general", "medical", "engineering", "scientific"],
        help="Guides terminology preservation"
    )

with col2:
    whisper_size = st.selectbox(
        "Transcription Quality",
        [("Auto", ""), ("Fast", "small"), ("Balanced", "medium"), ("Best", "large-v2")],
        format_func=lambda x: x[0],
        help="Fast = 2x speed, Best = highest accuracy"
    )[1]

with col3:
    summarizer = st.selectbox(
    "Summarizer",
    [("Auto (Mistral 7B)", ""), ("Mistral 7B", "mistralai/Mistral-7B-Instruct-v0.2")],
    format_func=lambda x: x[0],
    help="Auto enables 4-bit quantization on 8-16GB VRAM"
    )[1]

with col4:
    detail_level = st.select_slider(
        "Detail Level",
        options=["Concise", "Balanced", "Detailed"],
        value="Balanced",
        help="How much detail to include"
    )

# Map detail to compression ratio
compression_map = {"Concise": 0.15, "Balanced": 0.20, "Detailed": 0.30}
compression_ratio = compression_map[detail_level]

# Advanced settings in expander (less commonly changed)
with st.expander("⚙️ Advanced Settings"):
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        prefer_accuracy = st.checkbox("High Accuracy Mode", value=True, help="Slower but more accurate transcription")
        refinement = st.checkbox("Refinement Pass", value=True, help="Polish summary for coherence")
        imrad = st.checkbox("IMRaD Structure", value=False, help="Intro/Methods/Results/Discussion format")
        
    with adv_col2:
        audience = st.radio("Target Audience", ["expert", "student"], help="Expert = technical, Student = simplified")
        output_language = st.radio("Output Language", ["auto", "en", "source"], help="Force English or keep original")
        use_8bit = st.checkbox("Force Quantization", value=True, help="Save VRAM (recommended for 8GB GPUs)")

# Work directory
workdir = ROOT / "run_data"
workdir.mkdir(parents=True, exist_ok=True)

# Run button with clear status
if st.button("🚀 Summarize", type="primary", use_container_width=True):
    if not url.strip():
        st.error("Please enter a YouTube URL")
        st.stop()
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Stage indicators
            status_text.info("📥 Downloading audio...")
            progress_bar.progress(10)
            
            status_text.info("🎤 Transcribing (this may take a few minutes)...")
            progress_bar.progress(30)
            
            # Run pipeline
            meta, res = run_pipeline(
                url,
                workdir,
                domain=domain,
                whisper_size=whisper_size,
                prefer_accuracy=prefer_accuracy,
                summarizer_model=summarizer,
                use_8bit=use_8bit,
                refinement=refinement,
                imrad=imrad,
                compression_ratio=compression_ratio,
                audience=audience,
                output_language=None if output_language == "auto" else output_language,
            )
            
            progress_bar.progress(100)
            status_text.success("✅ Complete!")
            
            # Store in session state
            st.session_state.results = res
            st.session_state.meta = meta
            
        except Exception as e:
            progress_bar.progress(0)
            status_text.error(f"❌ Error: {str(e)}")
            st.exception(e)
            st.stop()

# Display results if available
if st.session_state.results and st.session_state.meta:
    res = st.session_state.results
    meta = st.session_state.meta
    
    st.divider()
    
    # Video info in compact format
    with st.expander("📹 Video Information", expanded=False):
        v = meta["video"]
        st.write(f"**{v.get('title', 'Unknown')}** by {v.get('uploader', 'Unknown')}")
        st.caption(f"Duration: {v.get('duration', 0)//60}:{v.get('duration', 0)%60:02d} | Views: {v.get('view_count', 0):,} | Date: {v.get('upload_date', 'Unknown')}")
    
    # Main content: Summary
    st.subheader("📝 Summary")
    st.markdown(res["summary"])
    
    # Side-by-side: Keywords and Terms
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔑 Keywords")
        if res["keywords"]:
            st.write(", ".join(res["keywords"]))
        else:
            st.caption("No keywords extracted")
    
    with col2:
        st.subheader("🔢 Critical Terms")
        if res["critical_terms"]:
            st.write(", ".join(res["critical_terms"]))
        else:
            st.caption("No critical terms found")
    
    # Highlights
    if res["highlights"]:
        st.subheader("✨ Key Highlights")
        for h in res["highlights"]:
            st.markdown(f"• {h}")
    
    # Downloads section
    st.divider()
    st.subheader("💾 Downloads")
    
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    
    with dl_col1:
        st.download_button(
            "📄 Transcript (.txt)",
            data=res["transcript"],
            file_name=f"{meta['video'].get('id', 'transcript')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with dl_col2:
        if meta["paths"].get("docx"):
            try:
                with open(meta["paths"]["docx"], "rb") as f:
                    st.download_button(
                        "📘 Summary + Transcript (.docx)",
                        data=f.read(),
                        file_name=f"{meta['video'].get('id', 'summary')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
            except Exception:
                st.caption("DOCX not available")
    
    with dl_col3:
        # Summary only download
        summary_text = f"# {meta['video'].get('title', 'Summary')}\n\n{res['summary']}\n\n## Keywords\n{', '.join(res['keywords'])}\n\n## Critical Terms\n{', '.join(res['critical_terms'])}"
        st.download_button(
            "📋 Summary Only (.md)",
            data=summary_text,
            file_name=f"{meta['video'].get('id', 'summary')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    # System info in sidebar
    with st.sidebar:
        st.subheader("💻 System Resources")
        
        # CPU
        cpu_count = psutil.cpu_count(logical=True)
        st.metric("CPU", f"{CPU_NAME[:30]}...", f"{cpu_count} cores")
        
        # RAM
        ram = psutil.virtual_memory()
        ram_used_pct = ram.percent
        st.metric("RAM", f"{ram.total / (1024**3):.1f} GB", f"{ram_used_pct:.0f}% used")
        
        # GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_vram = torch.cuda.memory_allocated(0) / (1024**3)
            vram_pct = (allocated_vram / total_vram) * 100
            
            st.metric(
                "GPU",
                f"{gpu_name[:25]}...",
                f"{allocated_vram:.1f}/{total_vram:.1f} GB ({vram_pct:.0f}%)"
            )
        else:
            st.metric("GPU", "None", "CPU mode")
        
        st.divider()
        st.caption("Clear results to free memory:")
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.results = None
            st.session_state.meta = None
            st.rerun()

# Footer
st.divider()
st.caption("💡 Tip: For best results on 8GB GPUs, use 'Fast' or 'Balanced' transcription with 'Auto' summarizer")