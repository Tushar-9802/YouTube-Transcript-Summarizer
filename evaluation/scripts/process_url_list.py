"""
Process manual URL lists for test set creation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
from tqdm import tqdm
from openai import OpenAI
from src.yt_sum.models.transcriber import Transcriber
from src.yt_sum.utils.downloader import download_audio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_url_file(url_file: Path, domain: str, output_dir: Path, api_key: str):
    """Process URLs from file"""
    
    with open(url_file) as f:
        urls = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Processing {len(urls)} URLs for {domain}")
    
    transcriber = Transcriber(model_size="small", prefer_accuracy=True)
    openai_client = OpenAI(api_key=api_key)
    
    audio_dir = output_dir / domain / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for url in tqdm(urls, desc=f"{domain}"):
        try:
            # Download
            audio_path, meta = download_audio(url, audio_dir)
            
            # Transcribe
            segments = transcriber.transcribe(str(audio_path), domain=domain, translate_to_english=True)
            transcript = " ".join(s.get("text", "").strip() for s in segments).strip()
            
            if len(transcript.split()) < 500:
                logger.warning(f"Transcript too short, skipping: {url}")
                Path(audio_path).unlink(missing_ok=True)
                continue
            
            # Generate summary
            prompt = f"""Create a 150-250 word technical summary for evaluation purposes.

Video: {meta.get('title', 'Unknown')}
Domain: {domain}

Transcript (first 4000 chars):
{transcript[:4000]}

Summary:"""
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a technical writer creating reference summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
            )
            
            summary = response.choices[0].message.content.strip()
            
            if len(summary.split()) < 50:
                logger.warning(f"Summary too short, skipping: {url}")
                Path(audio_path).unlink(missing_ok=True)
                continue
            
            # Save result
            results.append({
                "video_id": meta.get('id', ''),
                "url": url,
                "title": meta.get('title', ''),
                "domain": domain,
                "duration": meta.get('duration', 0),
                "transcript": transcript,
                "transcript_length": len(transcript.split()),
                "reference_summary": summary,
                "channel": meta.get('uploader', 'Unknown'),
            })
            
            # Cleanup audio
            Path(audio_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Failed {url}: {e}")
            continue
    
    # Save to JSONL
    output_file = output_dir / f"{domain}_test.jsonl"
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"✓ Saved {len(results)} videos to {output_file}")
    
    # Cleanup
    try:
        audio_dir.rmdir()
    except:
        pass

def main():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error("Set OPENAI_API_KEY environment variable")
        return
    
    output_dir = Path("data/youtube_test")
    
    for domain in ['medical', 'engineering', 'scientific']:
        url_file = output_dir / f"{domain}_urls.txt"
        if url_file.exists():
            process_url_file(url_file, domain, output_dir, api_key)
        else:
            logger.warning(f"Missing {url_file}")

if __name__ == "__main__":
    main()
