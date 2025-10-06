"""
YouTube Test Set Curation - OPTIMIZED
Compatible with existing YT-S transcriber
"""

import sys
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging
from urllib.parse import urlparse, parse_qs

import yt_dlp
from openai import OpenAI
from tqdm import tqdm

# Import existing transcriber
from src.yt_sum.models.transcriber import Transcriber

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VideoCandidate:
    video_id: str
    url: str
    title: str
    duration: int
    domain: str
    channel: str
    description: str = ""
    
@dataclass
class ProcessedVideo:
    video_id: str
    url: str
    title: str
    domain: str
    duration: int
    transcript: str
    transcript_length: int
    reference_summary: str
    channel: str

class YouTubeCurator:
    """Test set curation using existing infrastructure"""
    
    DOMAIN_QUERIES = {
        'medical': [
            'medical lecture pathophysiology',
            'clinical medicine grand rounds',
            'pharmacology lecture series',
        ],
        'engineering': [
            'engineering tutorial advanced',
            'electrical engineering lecture',
            'mechanical engineering principles',
        ],
        'scientific': [
            'physics lecture advanced',
            'quantum mechanics explained',
            'scientific research presentation',
        ]
    }
    
    MIN_DURATION = 900  # 15 minutes
    MAX_DURATION = 3600  # 60 minutes
    MIN_VIEWS = 1000
    
    def __init__(self, output_dir: Path = Path("data/youtube_test"), openai_api_key: str = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transcriber = Transcriber(model_size="small", prefer_accuracy=True)
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        for domain in self.DOMAIN_QUERIES.keys():
            (self.output_dir / domain).mkdir(exist_ok=True)
    
    def search_videos(self, domain: str, target_count: int = 20) -> List[VideoCandidate]:
        """Search for candidate videos"""
        logger.info(f"Searching videos for domain: {domain}")
        
        candidates = []
        seen_ids = set()
        
        ydl_opts = {'quiet': True, 'extract_flat': True, 'skip_download': True}
        
        for query in self.DOMAIN_QUERIES[domain]:
            search_url = f"ytsearch{target_count}:{query}"
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(search_url, download=False)
                    
                    for entry in info.get('entries', []):
                        if entry is None or entry.get('id') in seen_ids:
                            continue
                        
                        duration = entry.get('duration', 0)
                        view_count = entry.get('view_count', 0)
                        
                        if not (self.MIN_DURATION <= duration <= self.MAX_DURATION):
                            continue
                        if view_count < self.MIN_VIEWS:
                            continue
                        
                        video_id = entry.get('id')
                        candidates.append(VideoCandidate(
                            video_id=video_id,
                            url=f"https://www.youtube.com/watch?v={video_id}",
                            title=entry.get('title', ''),
                            duration=duration,
                            domain=domain,
                            channel=entry.get('uploader', 'Unknown'),
                            description=entry.get('description', '')[:500]
                        ))
                        
                        seen_ids.add(video_id)
                        if len(candidates) >= target_count:
                            break
                
                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")
                    continue
            
            if len(candidates) >= target_count:
                break
        
        logger.info(f"Found {len(candidates)} candidates for {domain}")
        return candidates[:target_count]
    
    def download_audio(self, video_url: str, output_path: Path) -> Optional[Path]:
        """Download audio from YouTube"""
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(output_path / '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Find downloaded file
            audio_files = list(output_path.glob("*.wav"))
            return audio_files[0] if audio_files else None
        except Exception as e:
            logger.error(f"Download failed for {video_url}: {e}")
            return None
    
    def transcribe_video(self, audio_path: Path, domain: str) -> Optional[str]:
        """Transcribe using existing Transcriber"""
        try:
            segments = self.transcriber.transcribe(
                str(audio_path),
                domain=domain,
                translate_to_english=True,
            )
            
            transcript = " ".join(s.get("text", "").strip() for s in segments).strip()
            return transcript if len(transcript.split()) >= 500 else None
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
    
    def generate_reference_summary(self, transcript: str, domain: str, title: str) -> Optional[str]:
        """Generate reference summary using GPT-4"""
        if not self.openai_client:
            logger.warning("OpenAI client not configured")
            return None
        
        domain_instructions = {
            'medical': 'Focus on clinical concepts, pathophysiology, diagnostic criteria, and treatment approaches. Preserve medical terminology.',
            'engineering': 'Focus on technical principles, methodologies, design considerations, and mathematical concepts. Preserve engineering terminology.',
            'scientific': 'Focus on theoretical concepts, experimental methods, findings, and implications. Preserve scientific terminology.',
        }
        
        prompt = f"""You are a domain expert creating a high-quality reference summary for evaluation purposes.

Video Title: {title}
Domain: {domain.capitalize()}

Instructions:
- Create a concise summary (150-250 words) that captures key concepts and main points
- {domain_instructions.get(domain, 'Preserve technical terminology')}
- Structure: brief introduction, main concepts/methods, key findings/takeaways
- Write in clear, technical language appropriate for domain experts
- Do NOT include meta-commentary (e.g., "this video discusses...")

Transcript:
{transcript[:4000]}

Summary:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a technical writing expert creating reference summaries for research evaluation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
            )
            
            summary = response.choices[0].message.content.strip()
            return summary if len(summary.split()) >= 50 else None
            
        except Exception as e:
            logger.error(f"GPT-4 summary generation failed: {e}")
            return None
    
    def process_candidate(self, candidate: VideoCandidate, audio_dir: Path) -> Optional[ProcessedVideo]:
        """Complete processing pipeline for one video"""
        logger.info(f"Processing: {candidate.title[:60]}...")
        
        # Download
        audio_path = self.download_audio(candidate.url, audio_dir)
        if not audio_path or not audio_path.exists():
            return None
        
        # Transcribe
        transcript = self.transcribe_video(audio_path, candidate.domain)
        if not transcript:
            audio_path.unlink()
            return None
        
        # Generate summary
        reference_summary = self.generate_reference_summary(transcript, candidate.domain, candidate.title)
        if not reference_summary:
            audio_path.unlink()
            return None
        
        # Cleanup
        audio_path.unlink()
        
        return ProcessedVideo(
            video_id=candidate.video_id,
            url=candidate.url,
            title=candidate.title,
            domain=candidate.domain,
            duration=candidate.duration,
            transcript=transcript,
            transcript_length=len(transcript.split()),
            reference_summary=reference_summary,
            channel=candidate.channel,
        )
    
    def curate_domain(self, domain: str, target_count: int = 20) -> List[ProcessedVideo]:
        """Complete curation for one domain"""
        logger.info(f"=== Starting curation for {domain.upper()} ===")
        
        candidates = self.search_videos(domain, target_count=int(target_count * 1.5))
        if not candidates:
            logger.error(f"No candidates found for {domain}")
            return []
        
        # Save candidates
        candidates_file = self.output_dir / f"{domain}_candidates.json"
        with open(candidates_file, 'w') as f:
            json.dump([asdict(c) for c in candidates], f, indent=2)
        
        # Process
        audio_dir = self.output_dir / domain / 'audio'
        audio_dir.mkdir(exist_ok=True)
        
        processed = []
        for candidate in tqdm(candidates, desc=f"Processing {domain}"):
            result = self.process_candidate(candidate, audio_dir)
            if result:
                processed.append(result)
            if len(processed) >= target_count:
                break
        
        # Save processed
        output_file = self.output_dir / f"{domain}_test.jsonl"
        with open(output_file, 'w') as f:
            for video in processed:
                f.write(json.dumps(asdict(video)) + '\n')
        
        logger.info(f"✓ Processed {len(processed)}/{target_count} videos for {domain}")
        
        # Cleanup
        try:
            audio_dir.rmdir()
        except:
            pass
        
        return processed
    
    def curate_all_domains(self, videos_per_domain: int = 20):
        """Execute complete curation"""
        logger.info("=== STARTING FULL TEST SET CURATION ===")
        
        all_results = {}
        for domain in self.DOMAIN_QUERIES.keys():
            results = self.curate_domain(domain, target_count=videos_per_domain)
            all_results[domain] = results
        
        logger.info("=== CURATION COMPLETE ===")
        return all_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Curate YouTube test set')
    parser.add_argument('--domain', type=str, choices=['medical', 'engineering', 'scientific', 'all'], default='all')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--openai-key', type=str, default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('data/youtube_test'))
    
    args = parser.parse_args()
    
    api_key = args.openai_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error("No OpenAI API key provided. Set OPENAI_API_KEY env var or use --openai-key")
        return
    
    curator = YouTubeCurator(output_dir=args.output_dir, openai_api_key=api_key)
    
    if args.domain == 'all':
        curator.curate_all_domains(videos_per_domain=args.count)
    else:
        curator.curate_domain(args.domain, target_count=args.count)

if __name__ == "__main__":
    main()