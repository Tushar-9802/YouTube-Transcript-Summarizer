"""
Process URLs with openai-whisper
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
import whisper
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_audio(url: str, output_dir: Path):
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
        return output_dir / f"{info['id']}.wav", info

def transcribe_whisper(audio_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("small", device=device)
    
    result = model.transcribe(
        str(audio_path),
        temperature=0.0,
        condition_on_previous_text=False,
        compression_ratio_threshold=1.8,
        logprob_threshold=-0.8,
    )
    
    return result['text'].strip()

def process_domain(domain: str, url_file: Path, output_dir: Path, api_key: str):
    with open(url_file) as f:
        urls = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Processing {len(urls)} for {domain}")
    
    openai_client = OpenAI(api_key=api_key)
    audio_dir = output_dir / domain / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for url in tqdm(urls, desc=domain):
        try:
            audio_path, meta = download_audio(url, audio_dir)
            transcript = transcribe_whisper(audio_path)
            
            if len(transcript.split()) < 500:
                audio_path.unlink(missing_ok=True)
                continue
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Technical writer."},
                    {"role": "user", "content": f"150-250 word summary:\n{transcript[:4000]}"}
                ],
                temperature=0.3,
                max_tokens=400,
            )
            
            summary = response.choices[0].message.content.strip()
            
            if len(summary.split()) < 50:
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
    
    for domain in ['scientific', 'engineering']:
        url_file = output_dir / f"{domain}_urls.txt"
        if url_file.exists():
            process_domain(domain, url_file, output_dir, api_key)
