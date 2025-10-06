"""
Performance Profiling - OPTIMIZED
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import torch
import psutil
from typing import Dict
from dataclasses import dataclass, asdict
import json

@dataclass
class PerformanceMetrics:
    stage: str
    duration_sec: float
    vram_allocated_gb: float
    vram_reserved_gb: float
    cpu_percent: float
    ram_used_gb: float

class SimpleProfiler:
    """Lightweight performance profiler"""
    
    def __init__(self):
        self.metrics = []
        self.process = psutil.Process()
    
    def get_vram(self) -> Dict:
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9
            }
        return {'allocated': 0, 'reserved': 0}
    
    def profile_stage(self, stage_name: str):
        """Context manager for profiling"""
        class ProfileContext:
            def __init__(ctx_self, profiler, name):
                ctx_self.profiler = profiler
                ctx_self.name = name
                ctx_self.start_time = None
            
            def __enter__(ctx_self):
                ctx_self.start_time = time.time()
                return ctx_self
            
            def __exit__(ctx_self, *args):
                duration = time.time() - ctx_self.start_time
                vram = ctx_self.profiler.get_vram()
                cpu = ctx_self.profiler.process.cpu_percent(interval=0.1)
                ram = ctx_self.profiler.process.memory_info().rss / 1e9
                
                metric = PerformanceMetrics(
                    stage=ctx_self.name,
                    duration_sec=duration,
                    vram_allocated_gb=vram['allocated'],
                    vram_reserved_gb=vram['reserved'],
                    cpu_percent=cpu,
                    ram_used_gb=ram
                )
                
                ctx_self.profiler.metrics.append(metric)
                print(f"[{ctx_self.name}] {duration:.2f}s | VRAM: {vram['allocated']:.2f}GB")
        
        return ProfileContext(self, stage_name)
    
    def save_report(self, output_file: Path = Path('results/profiling/profile_report.json')):
        """Save profiling results"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)
        
        print(f"\nProfile saved to: {output_file}")
        
        # Print summary
        print("\nPerformance Summary:")
        total_time = sum(m.duration_sec for m in self.metrics)
        max_vram = max(m.vram_allocated_gb for m in self.metrics) if self.metrics else 0
        
        print(f"Total Time: {total_time:.2f}s")
        print(f"Peak VRAM: {max_vram:.2f}GB")
        
        slowest = max(self.metrics, key=lambda m: m.duration_sec) if self.metrics else None
        if slowest:
            print(f"Slowest Stage: {slowest.stage} ({slowest.duration_sec:.2f}s)")

if __name__ == "__main__":
    print("Profiler ready. Import and use in your scripts:")
    print("\nfrom scripts.profile_performance import SimpleProfiler")
    print("\nprofiler = SimpleProfiler()")
    print("with profiler.profile_stage('model_loading'):")
    print("    # your code here")
    print("profiler.save_report()")