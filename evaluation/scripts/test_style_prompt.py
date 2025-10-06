#!/usr/bin/env python3
"""
Style-Constrained Prompting: Use LoRA with explicit anti-academic instructions
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from evaluate import load
import pandas as pd

print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=bnb_config,
    device_map="auto",
)

rouge = load('rouge')

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def style_controlled_summary(transcript, domain, adapter_path):
    """Use LoRA with explicit style constraints"""
    
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Strong anti-academic prompt
    prompt = f"""[INST] Summarize this YouTube video transcript naturally and conversationally.

IMPORTANT RULES:
- Write as if explaining to a friend what the video is about
- Use phrases like "the video shows", "the speaker explains", "this tutorial covers"
- Do NOT use academic language like "this paper", "we propose", "the authors", "this study"
- Do NOT include citations or references like [1] or "et al."
- Keep it casual and accessible

Transcript: {transcript[:1200]}

Summary: [/INST]"""
    
    summary = generate(lora_model, prompt)
    
    del lora_model
    torch.cuda.empty_cache()
    
    return summary

# Load error analysis
with open('results/error_analysis/failure_analysis.json') as f:
    error_data = json.load(f)

problematic_videos = [
    e for e in error_data['lora_errors'] 
    if 'academic_framing' in e['categories']
]

print(f"\nTesting style-controlled prompting on {len(problematic_videos)} problematic videos")
print("="*70)

results = []

for video_info in problematic_videos:
    video_id = video_info['video_id']
    domain = video_info['domain']
    
    # Load video
    test_file = Path(f"data/youtube_test/{domain}_test.jsonl")
    with open(test_file) as f:
        videos = [json.loads(line) for line in f]
    
    video = next(v for v in videos if v['video_id'] == video_id)
    
    print(f"\nVideo: {video_id} ({domain})")
    print("-" * 70)
    
    # Generate with style control
    adapter_path = f"models/adapters/{domain}"
    controlled_sum = style_controlled_summary(video['transcript'], domain, adapter_path)
    
    # Check if academic framing is gone
    has_academic = any(phrase in controlled_sum.lower() for phrase in 
                      ['this paper', 'this study', 'we propose', 'the authors', 'this research', 'this work'])
    
    # Score
    original_lora_rouge = video_info['rouge1']
    
    controlled_score = rouge.compute(
        predictions=[controlled_sum],
        references=[video['reference_summary']]
    )
    
    improvement = controlled_score['rouge1'] - original_lora_rouge
    
    print(f"Summary: {controlled_sum[:200]}...")
    print(f"Academic framing removed: {'✓ YES' if not has_academic else '✗ NO'}")
    print(f"ROUGE-1: {original_lora_rouge:.3f} → {controlled_score['rouge1']:.3f} ({improvement:+.3f})")
    
    results.append({
        'video_id': video_id,
        'domain': domain,
        'original_rouge1': original_lora_rouge,
        'style_controlled_rouge1': controlled_score['rouge1'],
        'improvement': improvement,
        'academic_removed': not has_academic
    })

# Summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

avg_original = sum(r['original_rouge1'] for r in results) / len(results)
avg_controlled = sum(r['style_controlled_rouge1'] for r in results) / len(results)
avg_improvement = avg_controlled - avg_original

print(f"\nAverage ROUGE-1:")
print(f"  Original LoRA:       {avg_original:.3f}")
print(f"  Style-Controlled:    {avg_controlled:.3f}")
print(f"  Change:              {avg_improvement:+.3f} ({avg_improvement/avg_original*100:+.1f}%)")

academic_fixed = sum(1 for r in results if r['academic_removed'])
print(f"\nAcademic framing fixed: {academic_fixed}/{len(results)} videos")

# Save
results_df = pd.DataFrame(results)
results_df.to_csv('results/style_controlled_results.csv', index=False)
print(f"\n✓ Saved: results/style_controlled_results.csv")
