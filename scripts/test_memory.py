#!/usr/bin/env python3
"""
Memory Management Test Script
Tests the complete transcription → summarization pipeline in isolation.
"""

import sys
from pathlib import Path
import torch
import gc
import time

# Add project root to path
  # Go up from Scripts to YT-S
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def print_vram(label: str):
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        print(f"{label}")
        print(f"  Free: {free_gb:.2f} GB / {total_gb:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
    else:
        print(f"{label}: CPU mode (no VRAM)")

def aggressive_cleanup():
    """Triple cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()

def test_imports():
    """Test 1: Verify all imports work."""
    print("\n" + "="*60)
    print("TEST 1: Imports")
    print("="*60)
    
    try:
        from src.yt_sum.models.transcriber import Transcriber
        print("✓ Transcriber import OK")
    except Exception as e:
        print(f"✗ Transcriber import FAILED: {e}")
        return False
    
    try:
        from src.yt_sum.models.summarizer import Summarizer
        print("✓ Summarizer import OK")
    except Exception as e:
        print(f"✗ Summarizer import FAILED: {e}")
        return False
    
    try:
        from transformers import BitsAndBytesConfig
        print("✓ BitsAndBytesConfig available")
    except Exception as e:
        print(f"✗ BitsAndBytesConfig NOT available: {e}")
        print("  Install: pip install bitsandbytes")
        return False
    
    return True

def test_whisper_load_unload():
    """Test 2: Whisper loading and cleanup."""
    print("\n" + "="*60)
    print("TEST 2: Whisper Load/Unload")
    print("="*60)
    
    try:
        from src.yt_sum.models.transcriber import Transcriber
        
        print("\n[1] Initial state:")
        print_vram("Before Whisper")
        
        print("\n[2] Loading Whisper small...")
        t = Transcriber(model_size="small", prefer_accuracy=False)
        print("✓ Loaded")
        print_vram("After Whisper load")
        
        print("\n[3] Cleaning up...")
        del t
        aggressive_cleanup()
        time.sleep(1)
        print("✓ Cleaned")
        print_vram("After cleanup")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Whisper test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_summarizer_load_unload():
    """Test 3: Summarizer loading with quantization."""
    print("\n" + "="*60)
    print("TEST 3: Summarizer Load/Unload (Quantized)")
    print("="*60)
    
    try:
        from src.yt_sum.models.summarizer import Summarizer
        
        print("\n[1] Initial state:")
        print_vram("Before Summarizer")
        
        print("\n[2] Loading Phi-3 Mini with 4-bit quantization...")
        s = Summarizer(
            summarizer_model="microsoft/Phi-3-mini-4k-instruct",
            use_8bit=True
        )
        print("✓ Loaded")
        print_vram("After Summarizer load")
        
        print("\n[3] Testing generation...")
        test_text = "This is a test. The quick brown fox jumps over the lazy dog."
        result = s.summarize_long(test_text, refinement=False, compression_ratio=0.5)
        print(f"✓ Generated: {len(result)} chars")
        
        print("\n[4] Cleaning up...")
        s.unload()
        del s
        aggressive_cleanup()
        time.sleep(1)
        print("✓ Cleaned")
        print_vram("After cleanup")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Summarizer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_sequence():
    """Test 4: Full sequence (Whisper → cleanup → Summarizer)."""
    print("\n" + "="*60)
    print("TEST 4: Full Sequence")
    print("="*60)
    
    try:
        from src.yt_sum.models.transcriber import Transcriber
        from src.yt_sum.models.summarizer import Summarizer
        
        print("\n[1] Initial state:")
        print_vram("Start")
        
        print("\n[2] Loading Whisper...")
        t = Transcriber(model_size="small", prefer_accuracy=False)
        print("✓ Loaded")
        print_vram("After Whisper")
        
        print("\n[3] Clearing Whisper...")
        del t
        aggressive_cleanup()
        time.sleep(1)
        print("✓ Cleared")
        print_vram("After Whisper cleanup")
        
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
            if free_gb < 3.0:
                print(f"\n⚠ WARNING: Only {free_gb:.2f} GB free")
                print("  Phi-3 needs ~2.5GB minimum")
        
        print("\n[4] Loading Summarizer...")
        s = Summarizer(
            summarizer_model="microsoft/Phi-3-mini-4k-instruct",
            use_8bit=True
        )
        print("✓ Loaded")
        print_vram("After Summarizer")
        
        print("\n[5] Testing generation...")
        test_text = "AI is transforming technology. ML enables computers to learn."
        result = s.summarize_long(test_text, refinement=False)
        print(f"✓ Generated: {len(result)} chars")
        
        print("\n[6] Final cleanup...")
        s.unload()
        del s
        aggressive_cleanup()
        time.sleep(1)
        print("✓ Cleaned")
        print_vram("Final state")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Full sequence FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("YOUTUBE SUMMARIZER - MEMORY TEST")
    print("="*60)
    
    print("\nSystem Information:")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024**3)
        print(f"VRAM: {total_vram:.2f} GB")
    else:
        print("CUDA: Not available")
    
    results = {
        "Imports": test_imports(),
        "Whisper Load/Unload": False,
        "Summarizer Load/Unload": False,
        "Full Sequence": False,
    }
    
    if results["Imports"]:
        results["Whisper Load/Unload"] = test_whisper_load_unload()
        
        if results["Whisper Load/Unload"]:
            results["Summarizer Load/Unload"] = test_summarizer_load_unload()
            
            if results["Summarizer Load/Unload"]:
                results["Full Sequence"] = test_full_sequence()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("System ready for full pipeline.")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Check errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())