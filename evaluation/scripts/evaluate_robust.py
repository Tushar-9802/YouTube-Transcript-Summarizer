"""
Robust LoRA Evaluation - Aggressive memory management for 8GB VRAM
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats as scipy_stats
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
import logging
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aggressive_cleanup():
    """Maximum memory cleanup"""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()

class SafeEvaluator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def load_model(self, adapter_path=None):
        """Load with max memory constraints"""
        aggressive_cleanup()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Strict memory limits for 8GB
        max_memory = {0: "7GB", "cpu": "16GB"}
        
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        if adapter_path and Path(adapter_path).exists():
            logger.info(f"Loading adapter: {Path(adapter_path).name}")
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        
        return model
    
    def generate(self, model, transcript: str) -> str:
        """Generate with error handling"""
        try:
            prompt = f"[INST] Summarize:\n\n{transcript[:1800]}\n\nSummary: [/INST]"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=1,  # Greedy for memory
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            summary = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return summary.strip()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("OOM during generation")
                aggressive_cleanup()
                return "[OOM ERROR]"
            raise
    
    def evaluate_domain(self, domain: str):
        """Evaluate one domain with aggressive cleanup between phases"""
        test_file = Path(f'data/youtube_test/{domain}_test.jsonl')
        
        if not test_file.exists():
            logger.warning(f"No test file for {domain}")
            return None
        
        # Load samples
        samples = []
        with open(test_file) as f:
            for line in f:
                samples.append(json.loads(line))
        
        logger.info(f"{domain}: {len(samples)} samples")
        results = []
        
        # Phase 1: Base model
        logger.info("Phase 1/2: Base model")
        model = self.load_model()
        base_summaries = []
        
        for i, sample in enumerate(tqdm(samples, desc="Base")):
            summary = self.generate(model, sample['transcript'])
            base_summaries.append(summary)
            
            rouge = self.rouge.score(sample['reference_summary'], summary)
            results.append({
                'video_id': sample['video_id'],
                'domain': domain,
                'model_type': 'base',
                'summary': summary,
                'rouge1_f': rouge['rouge1'].fmeasure,
                'rouge2_f': rouge['rouge2'].fmeasure,
                'rougeL_f': rouge['rougeL'].fmeasure,
            })
            
            # Periodic cleanup
            if (i + 1) % 5 == 0:
                aggressive_cleanup()
        
        del model
        aggressive_cleanup()
        logger.info("Base model unloaded")
        
        # Phase 2: LoRA model
        logger.info("Phase 2/2: LoRA model")
        adapter_path = Path(f'models/adapters/{domain}')
        
        if not adapter_path.exists():
            logger.error(f"Adapter not found: {adapter_path}")
            return pd.DataFrame(results)
        
        model = self.load_model(adapter_path)
        lora_summaries = []
        
        for i, sample in enumerate(tqdm(samples, desc="LoRA")):
            summary = self.generate(model, sample['transcript'])
            lora_summaries.append(summary)
            
            rouge = self.rouge.score(sample['reference_summary'], summary)
            results.append({
                'video_id': sample['video_id'],
                'domain': domain,
                'model_type': 'lora',
                'summary': summary,
                'rouge1_f': rouge['rouge1'].fmeasure,
                'rouge2_f': rouge['rouge2'].fmeasure,
                'rougeL_f': rouge['rougeL'].fmeasure,
            })
            
            if (i + 1) % 5 == 0:
                aggressive_cleanup()
        
        del model
        aggressive_cleanup()
        logger.info("LoRA model unloaded")
        
        # BERTScore (separate, memory-intensive)
        logger.info("Computing BERTScore...")
        references = [s['reference_summary'] for s in samples]
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        P, R, F1 = bert_score(base_summaries, references, lang="en", device=device, verbose=False, batch_size=4)
        base_bert = F1.mean().item()
        
        aggressive_cleanup()
        
        P, R, F1 = bert_score(lora_summaries, references, lang="en", device=device, verbose=False, batch_size=4)
        lora_bert = F1.mean().item()
        
        aggressive_cleanup()
        
        # Add BERTScore to results
        for res in results:
            res['bertscore_f'] = base_bert if res['model_type'] == 'base' else lora_bert
        
        return pd.DataFrame(results)
    
    def run_evaluation(self):
        """Run complete evaluation"""
        output_dir = Path('results/evaluation')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for domain in ['medical', 'engineering', 'scientific']:
            logger.info(f"\n{'='*60}\nEvaluating {domain.upper()}\n{'='*60}")
            
            df = self.evaluate_domain(domain)
            if df is not None:
                all_results.append(df)
            
            aggressive_cleanup()
        
        if not all_results:
            logger.error("No results generated")
            return
        
        # Combine and save
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_dir / 'detailed_results.csv', index=False)
        logger.info(f"Saved {len(final_df)} results")
        
        # Statistics
        stats = {}
        for domain in final_df['domain'].unique():
            for metric in ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bertscore_f']:
                domain_df = final_df[final_df['domain'] == domain]
                base = domain_df[domain_df['model_type'] == 'base'][metric].values
                lora = domain_df[domain_df['model_type'] == 'lora'][metric].values
                
                if len(base) > 0 and len(lora) > 0:
                    t_stat, p_value = scipy_stats.ttest_rel(lora, base)
                    stats[f"{domain}_{metric}"] = {
                        'base_mean': base.mean(),
                        'lora_mean': lora.mean(),
                        'p_value': p_value,
                        'improvement': ((lora.mean() - base.mean()) / base.mean() * 100),
                    }
        
        pd.DataFrame(stats).T.to_csv(output_dir / 'statistical_summary.csv')
        logger.info(f"\nEvaluation complete! Results in {output_dir}")

if __name__ == "__main__":
    evaluator = SafeEvaluator()
    evaluator.run_evaluation()
