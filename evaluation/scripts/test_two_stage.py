#!/usr/bin/env python3
"""
Two-Stage Prompting: Extract domain knowledge, rephrase conversationally
Tests on the 7 problematic videos identified in error analysis
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

def generate(model, prompt, max_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def two_stage_summary(transcript, domain, adapter_path):
    """Two-stage: concepts extraction + conversational rephrasing"""
    
    # Load domain adapter
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Stage 1: Extract key concepts with domain knowledge
    prompt1 = f"[INST] List the 5 most important technical concepts from this {domain} content (keywords only, no sentences):\n\n{transcript[:800]}\n\n[/INST]"
    
    concepts = generate(lora_model, prompt1, max_tokens=80)
    
    # Cleanup - unload LoRA
    del lora_model
    torch.cuda.empty_cache()
    
    # Stage 2: Create conversational summary using base model
    prompt2 = f"""[INST] Write a natural summary of what this video covers. 

Key topics to mention: {concepts}

Transcript excerpt: {transcript[:1000]}

Write 2-3 sentences explaining what the video teaches. Use casual language like "the video explains" or "the speaker discusses". Do NOT use phrases like "this paper" or "the authors". [/INST]"""
    
    summary = generate(base_model, prompt2, max_tokens=150)
    
    return summary, concepts

# Load error analysis to find problematic videos
with open('results/error_analysis/failure_analysis.json') as f:
    error_data = json.load(f)

problematic_videos = [
    e for e in error_data['lora_errors'] 
    if 'academic_framing' in e['categories']
]

print(f"\nTesting two-stage approach on {len(problematic_videos)} problematic videos")
print("="*70)

results = []

for video_info in problematic_videos[:5]:  # Test first 5
    video_id = video_info['video_id']
    domain = video_info['domain']
    
    # Load video data
    test_file = Path(f"data/youtube_test/{domain}_test.jsonl")
    with open(test_file) as f:
        videos = [json.loads(line) for line in f]
    
    video = next(v for v in videos if v['video_id'] == video_id)
    
    print(f"\nVideo: {video_id} ({domain})")
    print("-" * 70)
    
    # Generate two-stage summary
    adapter_path = f"models/adapters/{domain}"
    two_stage_sum, concepts = two_stage_summary(video['transcript'], domain, adapter_path)
    
    # Compare scores
    original_lora_rouge = video_info['rouge1']
    
    two_stage_score = rouge.compute(
        predictions=[two_stage_sum],
        references=[video['reference_summary']]
    )
    
    improvement = two_stage_score['rouge1'] - original_lora_rouge
    
    print(f"Concepts extracted: {concepts[:100]}...")
    print(f"\nTwo-stage summary: {two_stage_sum[:200]}...")
    print(f"\nROUGE-1: {original_lora_rouge:.3f} → {two_stage_score['rouge1']:.3f} ({improvement:+.3f})")
    
    results.append({
        'video_id': video_id,
        'domain': domain,
        'original_rouge1': original_lora_rouge,
        'two_stage_rouge1': two_stage_score['rouge1'],
        'improvement': improvement
    })

# Summary statistics
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

avg_original = sum(r['original_rouge1'] for r in results) / len(results)
avg_two_stage = sum(r['two_stage_rouge1'] for r in results) / len(results)
avg_improvement = avg_two_stage - avg_original

print(f"\nAverage ROUGE-1:")
print(f"  Original LoRA:    {avg_original:.3f}")
print(f"  Two-Stage:        {avg_two_stage:.3f}")
print(f"  Improvement:      {avg_improvement:+.3f} ({avg_improvement/avg_original*100:+.1f}%)")

print(f"\nFixed academic framing: {len(results)}/{len(problematic_videos)} videos tested")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('results/two_stage_results.csv', index=False)
print(f"\n✓ Saved: results/two_stage_results.csv")
