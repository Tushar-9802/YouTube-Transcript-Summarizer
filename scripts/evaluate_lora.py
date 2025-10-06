"""
LoRA Adapter Evaluation Pipeline - OPTIMIZED
Compatible with existing YT-S codebase structure
"""

import sys
from pathlib import Path

# Path setup for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from scipy import stats
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    """Evaluation configuration"""
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    adapters_dir: Path = Path("models/adapters")
    test_data_dir: Path = Path("data/youtube_test")
    output_dir: Path = Path("results/evaluation")
    domains: List[str] = None
    use_4bit: bool = True
    max_length: int = 512
    num_beams: int = 4
    
    def __post_init__(self):
        self.domains = self.domains or ["medical", "engineering", "scientific"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

class MetricsCalculator:
    """Compute ROUGE and BERTScore"""
    
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_rouge(self, prediction: str, reference: str) -> Dict:
        scores = self.rouge.score(reference, prediction)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        P, R, F1 = bert_score(predictions, references, lang="en", device=device, verbose=False)
        return {
            'bertscore_f': F1.mean().item(),
            'bertscore_p': P.mean().item(),
            'bertscore_r': R.mean().item(),
        }

class LoRAEvaluator:
    """Main evaluation orchestrator"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.metrics_calc = MetricsCalculator()
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def load_base_model(self):
        """Load base model with 4-bit quantization"""
        logger.info("Loading base model (4-bit)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        return model
    
    def load_lora_model(self, domain: str):
        """Load base + LoRA adapter"""
        logger.info(f"Loading {domain} adapter...")
        base_model = self.load_base_model()
        adapter_path = self.config.adapters_dir / domain
        
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        
        model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
        return model
    
    def generate_summary(self, model, transcript: str) -> str:
        """Generate summary from transcript"""
        prompt = f"""[INST] Summarize the following technical content concisely, preserving key concepts and terminology:

{transcript[:2000]}

Summary: [/INST]"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_length,
                num_beams=self.config.num_beams,
                temperature=0.7,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        summary = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        return summary
    
    def evaluate_domain(self, domain: str) -> pd.DataFrame:
        """Evaluate base vs LoRA for one domain"""
        logger.info(f"Evaluating domain: {domain}")
        
        # Load test data
        test_file = self.config.test_data_dir / f"{domain}_test.jsonl"
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        samples = []
        with open(test_file) as f:
            for line in f:
                samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(samples)} test samples")
        results = []
        
        # Base model evaluation
        logger.info("Evaluating base model...")
        base_model = self.load_base_model()
        base_summaries = []
        
        for sample in tqdm(samples, desc="Base model"):
            summary = self.generate_summary(base_model, sample['transcript'])
            base_summaries.append(summary)
            rouge = self.metrics_calc.compute_rouge(summary, sample['reference_summary'])
            results.append({
                'video_id': sample['video_id'],
                'domain': domain,
                'model_type': 'base',
                'summary': summary,
                **rouge
            })
        
        del base_model
        torch.cuda.empty_cache()
        
        # LoRA evaluation
        logger.info("Evaluating LoRA model...")
        lora_model = self.load_lora_model(domain)
        lora_summaries = []
        
        for sample in tqdm(samples, desc="LoRA model"):
            summary = self.generate_summary(lora_model, sample['transcript'])
            lora_summaries.append(summary)
            rouge = self.metrics_calc.compute_rouge(summary, sample['reference_summary'])
            results.append({
                'video_id': sample['video_id'],
                'domain': domain,
                'model_type': 'lora',
                'summary': summary,
                **rouge
            })
        
        del lora_model
        torch.cuda.empty_cache()
        
        # BERTScore
        references = [s['reference_summary'] for s in samples]
        base_bert = self.metrics_calc.compute_bertscore(base_summaries, references)
        lora_bert = self.metrics_calc.compute_bertscore(lora_summaries, references)
        
        for i, res in enumerate(results):
            bert_scores = base_bert if res['model_type'] == 'base' else lora_bert
            res.update(bert_scores)
        
        return pd.DataFrame(results)
    
    def statistical_comparison(self, df: pd.DataFrame, metric: str) -> Dict:
        """Paired t-test and effect size"""
        base_scores = df[df['model_type'] == 'base'][metric].values
        lora_scores = df[df['model_type'] == 'lora'][metric].values
        
        t_stat, p_value = stats.ttest_rel(lora_scores, base_scores)
        diff = lora_scores - base_scores
        pooled_std = np.sqrt((np.var(base_scores) + np.var(lora_scores)) / 2)
        cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
        
        return {
            'base_mean': base_scores.mean(),
            'base_std': base_scores.std(),
            'lora_mean': lora_scores.mean(),
            'lora_std': lora_scores.std(),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'improvement': ((lora_scores.mean() - base_scores.mean()) / base_scores.mean() * 100),
        }
    
    def generate_latex_table(self, stats: Dict) -> str:
        """Generate LaTeX table"""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Base vs LoRA Comparison}",
            r"\begin{tabular}{lcccccc}",
            r"\hline",
            r"Domain & Metric & Base & LoRA & $\Delta$\% & $p$ & Cohen's $d$ \\",
            r"\hline",
        ]
        
        for domain in self.config.domains:
            for metric in ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bertscore_f']:
                key = f"{domain}_{metric}"
                if key in stats:
                    s = stats[key]
                    sig = "***" if s['p_value'] < 0.001 else "**" if s['p_value'] < 0.01 else "*" if s['p_value'] < 0.05 else ""
                    lines.append(
                        f"{domain.capitalize()} & {metric.replace('_', '-')} & "
                        f"{s['base_mean']:.3f} & {s['lora_mean']:.3f} & "
                        f"{s['improvement']:+.1f}\% & {s['p_value']:.4f}{sig} & {s['cohens_d']:.2f} \\\\"
                    )
        
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)
    
    def plot_comparison(self, stats: Dict):
        """Generate comparison bar charts"""
        metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bertscore_f']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            domains, base_means, lora_means = [], [], []
            
            for domain in self.config.domains:
                key = f"{domain}_{metric}"
                if key in stats:
                    domains.append(domain.capitalize())
                    base_means.append(stats[key]['base_mean'])
                    lora_means.append(stats[key]['lora_mean'])
            
            x = np.arange(len(domains))
            width = 0.35
            ax.bar(x - width/2, base_means, width, label='Base', alpha=0.8)
            ax.bar(x + width/2, lora_means, width, label='LoRA', alpha=0.8)
            ax.set_ylabel('Score')
            ax.set_title(metric.replace('_', '-').upper())
            ax.set_xticks(x)
            ax.set_xticklabels(domains)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'comparison_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_evaluation(self):
        """Execute complete evaluation"""
        all_results = []
        all_stats = {}
        
        for domain in self.config.domains:
            df = self.evaluate_domain(domain)
            all_results.append(df)
            
            for metric in ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bertscore_f']:
                domain_df = df[df['domain'] == domain]
                stats = self.statistical_comparison(domain_df, metric)
                all_stats[f"{domain}_{metric}"] = stats
        
        # Save results
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(self.config.output_dir / 'detailed_results.csv', index=False)
        
        stats_df = pd.DataFrame(all_stats).T
        stats_df.to_csv(self.config.output_dir / 'statistical_summary.csv')
        
        latex_table = self.generate_latex_table(all_stats)
        (self.config.output_dir / 'results_table.tex').write_text(latex_table)
        
        self.plot_comparison(all_stats)
        
        logger.info("Evaluation complete!")
        return final_df, all_stats

if __name__ == "__main__":
    config = EvalConfig()
    evaluator = LoRAEvaluator(config)
    results, stats = evaluator.run_full_evaluation()