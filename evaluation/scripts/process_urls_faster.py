"""
Process URLs with faster-whisper (more reliable)
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import os
from tqdm import tqdm
from openai import OpenAI
import yt_dlp
import logging
from faster_whisper import WhisperModel
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_audio(url: str, output_dir: Path):
    """Download audio from YouTube"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = output_dir / f"{info['id']}.wav"
        return audio_file, info

def transcribe_faster(audio_path: Path):
    """Transcribe with faster-whisper"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel("small", device=device, compute_type="int8_float16" if device == "cuda" else "int8")
    
    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    
    text = " ".join([s.text.strip() for s in segments]).strip()
    return text

def process_domain(domain: str, url_file: Path, output_dir: Path, api_key: str):
    """Process one domain"""
    with open(url_file) as f:
        urls = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Processing {len(urls)} URLs for {domain}")
    
    openai_client = OpenAI(api_key=api_key)
    audio_dir = output_dir / domain / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for url in tqdm(urls, desc=domain):
        try:
            audio_path, meta = download_audio(url, audio_dir)
            transcript = transcribe_faster(audio_path)
            
            if len(transcript.split()) < 500:
                logger.warning(f"Short transcript, skipping: {url}")
                audio_path.unlink(missing_ok=True)
                continue
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a technical writer."},
                    {"role": "user", "content": f"Create 150-250 word summary:\n\n{transcript[:4000]}"}
                ],
                temperature=0.3,
                max_tokens=400,
            )
            
            summary = response.choices[0].message.content.strip()
            
            if len(summary.split()) < 50:
                logger.warning(f"Short summary, skipping: {url}")
                audio_path.unlink(missing_ok=True)
                continue
            
            results.append({
                "video_id": meta['id'],
                "url": url,
                "title": meta.get('title', ''),
                "domain": domain,
                "duration": meta.get('duration', 0),
                "transcript": transcript,
                "transcript_length": len(transcript.split()),
                "reference_summary": summary,
                "channel": meta.get('uploader', ''),
            })
            
            audio_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Failed {url}: {e}")
    
    output_file = output_dir / f"{domain}_test.jsonl"
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    logger.info(f"✓ Saved {len(results)} to {output_file}")

if __name__ == "__main__":
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY")
    
    output_dir = Path("data/youtube_test")
    
    for domain in ['medical', 'engineering', 'scientific']:  # Add 'medical' when ready
        url_file = output_dir / f"{domain}_urls.txt"
        if url_file.exists():
            process_domain(domain, url_file, output_dir, api_key)
