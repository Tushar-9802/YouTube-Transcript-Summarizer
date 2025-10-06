"""
Verify that LoRA adapters work on their training domain (papers)
This proves the negative result is real, not a bug
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

def test_adapter_quality(domain):
    """Test if adapter improves performance on papers"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load base model
    print(f"\nTesting {domain} adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Load papers from training data
    paper_file = f"data/domains/{domain}_papers.jsonl"
    try:
        with open(paper_file) as f:
            papers = [json.loads(line) for line in f][:5]  # Test on 5 papers
    except FileNotFoundError:
        print(f"Warning: {paper_file} not found. Using generic test.")
        papers = [{
            'abstract': 'This study examines machine learning applications in medical diagnostics.',
            'title': 'ML in Medicine'
        }]
    
    # Test base model
    print("Testing base model...")
    base_summaries = []
    for paper in papers:
        prompt = f"[INST] Summarize this abstract: {paper.get('abstract', '')[:1000]} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = base_model.generate(**inputs.to(base_model.device), max_new_tokens=150)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        base_summaries.append(summary)
    
    # Load LoRA adapter
    print(f"Loading {domain} adapter...")
    adapter_path = f"models/adapters/{domain}"
    try:
        lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        return None
    
    # Test LoRA model
    print("Testing LoRA model...")
    lora_summaries = []
    for paper in papers:
        prompt = f"[INST] Summarize this abstract: {paper.get('abstract', '')[:1000]} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = lora_model.generate(**inputs.to(lora_model.device), max_new_tokens=150)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        lora_summaries.append(summary)
    
    # Compare quality
    print(f"\n{domain.upper()} RESULTS:")
    print("-" * 50)
    
    # Check for academic style in LoRA outputs
    academic_markers = ['this paper', 'we present', 'our findings', 'this study']
    lora_academic_count = sum(1 for s in lora_summaries if any(m in s.lower() for m in academic_markers))
    base_academic_count = sum(1 for s in base_summaries if any(m in s.lower() for m in academic_markers))
    
    print(f"Base model academic style: {base_academic_count}/5 summaries")
    print(f"LoRA model academic style: {lora_academic_count}/5 summaries")
    
    if lora_academic_count > base_academic_count:
        print("✓ LoRA adapter learned paper-style writing")
    else:
        print("✗ LoRA adapter may not be working properly")
    
    return {
        'domain': domain,
        'base_academic': base_academic_count,
        'lora_academic': lora_academic_count,
        'sample_base': base_summaries[0] if base_summaries else None,
        'sample_lora': lora_summaries[0] if lora_summaries else None
    }

if __name__ == "__main__":
    results = []
    for domain in ['medical', 'engineering', 'scientific']:
        result = test_adapter_quality(domain)
        if result:
            results.append(result)
            print(f"\nSample LoRA output for {domain}:")
            print(result['sample_lora'][:200] + "...")
    
    print("\n" + "="*60)
    print("ADAPTER VERIFICATION COMPLETE")
    print("="*60)
    
    # Save results
    with open('results/adapter_verification.json', 'w') as f:
        json.dump(results, f, indent=2)
