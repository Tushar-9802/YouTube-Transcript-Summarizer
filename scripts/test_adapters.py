"""Adapter validation - confirms LoRA models load and generate summaries"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yt_sum.models.summarizer_lora import SummarizerWithLoRA
import torch

def test_adapter(domain: str):
    """Test single adapter loading and inference"""
    print(f"\n{'='*60}\nTesting {domain.upper()} adapter\n{'='*60}")
    
    adapter_path = f"./models/adapters/{domain}"
    
    # FIXED: Disable quantization (bitsandbytes not available on Windows)
    summarizer = SummarizerWithLoRA(
        domain=domain,
        adapter_path=adapter_path,
        use_8bit=False  # ← KEY FIX
    )
    
    test_texts = {
        "medical": """
        Diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia 
        resulting from defects in insulin secretion, insulin action, or both. Type 1 diabetes 
        is caused by autoimmune destruction of pancreatic beta cells, while Type 2 diabetes 
        involves insulin resistance and relative insulin deficiency. Management includes 
        glycemic control through diet, exercise, oral hypoglycemics, and insulin therapy.
        Complications include retinopathy, nephropathy, neuropathy, and cardiovascular disease.
        """,
        "engineering": """
        The finite element method (FEM) is a numerical technique for solving differential 
        equations in engineering. It subdivides a large system into smaller parts called 
        finite elements. The simple equations modeling these elements are assembled into 
        a larger system that models the entire problem. FEM is used in structural analysis, 
        heat transfer, fluid dynamics, and electromagnetic simulations. Modern implementations 
        use adaptive mesh refinement to optimize computational efficiency while maintaining 
        accuracy in regions of high gradient.
        """,
        "scientific": """
        Quantum entanglement is a physical phenomenon where particles become correlated such 
        that the quantum state of one particle cannot be described independently of others. 
        This correlation persists regardless of distance between particles. Bell's theorem 
        demonstrates that no local hidden variable theory can reproduce all predictions of 
        quantum mechanics. Entanglement is foundational for quantum computing, quantum 
        cryptography, and quantum teleportation protocols.
        """
    }
    
    text = test_texts[domain]
    
    print(f"Input text ({len(text.split())} words):\n{text[:200]}...\n")
    
    summary = summarizer.summarize_long(text)
    
    print(f"Generated Summary ({len(summary.split())} words):\n{summary}\n")
    print(f"✓ Adapter loaded: {summarizer.adapter_loaded}")
    print(f"✓ Device: {summarizer.device}")
    print(f"✓ Quantization: {'4-bit' if summarizer.use_8bit else 'FP16'}")
    
    summarizer.unload()
    torch.cuda.empty_cache()
    
    return summary

if __name__ == "__main__":
    domains = ["medical", "engineering", "scientific"]
    
    print("ADAPTER VALIDATION TEST")
    print("="*60)
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Mode: FP16 (bitsandbytes unavailable)")
    
    results = {}
    for domain in domains:
        try:
            results[domain] = test_adapter(domain)
            print(f"✓ {domain.upper()} PASSED\n")
        except Exception as e:
            print(f"✗ {domain.upper()} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            results[domain] = None
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v is not None)
    print(f"Passed: {passed}/3")
    
    if passed == 3:
        print("\n✓ All adapters functional")
        print("✓ Ready for evaluation pipeline")
    else:
        print(f"\n⚠ {3-passed} adapters failed - check errors above")
    
    print("="*60)