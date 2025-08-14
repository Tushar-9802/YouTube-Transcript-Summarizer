# scripts/yt_sum.py
import argparse, json
from pathlib import Path
import torch
from yt_sum.utils.config import Config
from yt_sum.utils.logging import get_logger
from yt_sum.utils.downloader import download_audio
from yt_sum.utils.asr import transcribe_audio, detect_device
from yt_sum.models.summarizer import BaselineSummarizer

def main():
    parser = argparse.ArgumentParser(description="YouTube → Transcript → Summary")
    parser.add_argument(
    "--whisper_size",
    type=str,
    default="small",
    choices=["tiny", "base", "small", "medium", "large"],
    help="Size of Whisper ASR model."
    )

    parser.add_argument(
    "--summarizer_model",
    type=str,
    default="facebook/bart-large-cnn",
    help="Hugging Face model name for summarization."
    )

    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--model", default=None, help="Override summarizer model name")
    parser.add_argument("--max_tokens", type=int, default=160, help="Max new tokens in summary")
    args = parser.parse_args()

    cfg = Config(args.config)
    cfg.ensure_dirs()
    log = get_logger(level=cfg.log_level)

    log.info("Step 1: Downloading audio…")
    audio_path, meta = download_audio(args.url, cfg.audio_dir)
    log.info(f"Downloaded: {audio_path.name} | title='{meta.get('title')}'")

    log.info("Step 2: Transcribing with Whisper…")
    result = transcribe_audio(audio_path, cfg.whisper_model_size, cfg.transcript_dir)
    transcript_text = result.get("text", "").strip()
    if not transcript_text:
        raise RuntimeError("Empty transcript. Check audio/ffmpeg/whisper setup.")

    log.info(f"Detected language: {result.get('language')}, device: {detect_device()}")

    log.info("Step 3: Baseline summarization…")
    device = detect_device()
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log.info(f"CUDA is available ✅ | Device: {device_name} | VRAM: {total_mem:.1f} GB")
    else:
        log.warning("CUDA not available — running on CPU 🚨")
    model_name = args.model or "facebook/bart-large-cnn"
    summarizer = BaselineSummarizer(model_name=model_name, device=device if device == "cuda" else None)

    # For now: summarize the first 1024 tokens-worth of text
    # (We'll add a chunker next)
    summary = summarizer.summarize_long(transcript_text)

    # Save a compact JSON with key info
    out = {
        "video_id": meta.get("id"),
        "title": meta.get("title"),
        "uploader": meta.get("uploader"),
        "duration": meta.get("duration"),
        "view_count": meta.get("view_count"),
        "upload_date": meta.get("upload_date"),
        "language": result.get("language"),
        "summary": summary,
    }
    out_path = cfg.outputs_dir / f"{meta.get('id')}.json"
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(f"✅ Done. Summary saved to: {out_path}")
    print("\n--- SUMMARY ---\n")
    print(summary)

if __name__ == "__main__":
    main()
