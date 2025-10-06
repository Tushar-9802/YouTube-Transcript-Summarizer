"""
Simple Progress Monitor - No external dependencies
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import pandas as pd
from datetime import datetime, timedelta

def monitor_evaluation(results_file: Path = Path('results/evaluation/detailed_results.csv'), 
                       total_samples: int = 60, refresh_sec: float = 5.0):
    """Simple text-based monitoring"""
    
    print("=" * 60)
    print("EVALUATION MONITOR")
    print("=" * 60)
    
    start_time = time.time()
    
    while True:
        if not results_file.exists():
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Waiting for results file...", end='', flush=True)
            time.sleep(refresh_sec)
            continue
        
        try:
            df = pd.read_csv(results_file)
            completed = len(df) // 2  # Each sample has base + lora
            
            if completed >= total_samples:
                print(f"\n\n✓ Evaluation Complete! {completed}/{total_samples} samples processed")
                break
            
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining_sec = (total_samples - completed) / rate if rate > 0 else 0
            eta = datetime.now() + timedelta(seconds=remaining_sec)
            
            # Progress bar
            pct = (completed / total_samples * 100) if total_samples > 0 else 0
            bar_len = 40
            filled = int(bar_len * completed / total_samples)
            bar = '█' * filled + '░' * (bar_len - filled)
            
            # Metrics
            avg_rouge = df['rouge1_f'].mean() if 'rouge1_f' in df else 0
            
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {bar} {pct:.1f}% | "
                  f"{completed}/{total_samples} | ETA: {eta.strftime('%H:%M:%S')} | "
                  f"ROUGE-1: {avg_rouge:.3f}", end='', flush=True)
            
        except Exception as e:
            print(f"\rError reading results: {e}", end='', flush=True)
        
        time.sleep(refresh_sec)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results/evaluation/detailed_results.csv'))
    parser.add_argument('--total', type=int, default=60)
    args = parser.parse_args()
    
    monitor_evaluation(args.results, args.total)