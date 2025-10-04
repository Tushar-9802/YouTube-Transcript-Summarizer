# src/yt_sum/train/evaluate_lora.py
"""
Evaluate base vs LoRA adapters with ROUGE + BERTScore
Compatible with Mistral-based pipeline
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import torch
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.yt_sum.models.summarizer_lora import SummarizerWithLoRA


def evaluate_model(
    domain: str,
    test_data: List[Dict],
    adapter_path: Optional[str] = None,
):
    """
    Evaluate model on test data.
    
    test_data format:
    [
        {
            "video_id": "id",
            "transcript": "...",
            "reference_summary": "..."
        },
        ...
    ]
    """
    
    model_type = "LoRA" if adapter_path else "Base"
    print(f"\n{'='*80}")
    print(f"Evaluating: {domain} ({model_type})")
    print(f"{'='*80}\n")
    
    # Load model
    summarizer = SummarizerWithLoRA(
        domain=domain,
        adapter_path=adapter_path
    )
    
    # Initialize scorers
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bert = BERTScorer(
        model_type="bert-base-uncased", 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    results = []
    
    for i, sample in enumerate(tqdm(test_data, desc=f"Evaluating {model_type}"), 1):
        # Generate summary
        generated = summarizer.summarize_long(sample["transcript"])
        
        # Compute metrics
        rouge_scores = rouge.score(sample["reference_summary"], generated)
        P, R, F = bert.score([generated], [sample["reference_summary"]])
        
        results.append({
            "video_id": sample["video_id"],
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "bertscore": F.item(),
            "generated": generated,
        })
    
    # Aggregate
    avg_scores = {
        "rouge1": np.mean([r["rouge1"] for r in results]),
        "rouge2": np.mean([r["rouge2"] for r in results]),
        "rougeL": np.mean([r["rougeL"] for r in results]),
        "bertscore": np.mean([r["bertscore"] for r in results]),
    }
    
    summarizer.unload()
    
    return results, avg_scores


def plot_comparison(comparison: Dict, output_path: Path, domain: str):
    """Generate comparison plots"""
    
    metrics = ["rouge1", "rouge2", "rougeL", "bertscore"]
    base_vals = [comparison[m]["base_mean"] for m in metrics]
    lora_vals = [comparison[m]["lora_mean"] for m in metrics]
    
    # Bar plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_vals, width, label='Base Model', color='#6C757D')
    bars2 = ax.bar(x + width/2, lora_vals, width, label='LoRA Adapted', color='#28A745')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{domain.capitalize()} Domain: Base vs LoRA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path / "comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Delta plot
    fig, ax = plt.subplots(figsize=(10, 6))
    deltas = [comparison[m]["delta"] for m in metrics]
    colors = ['#DC3545' if d < 0 else '#28A745' for d in deltas]
    
    bars = ax.bar(metrics, deltas, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Delta (LoRA - Base)', fontsize=12)
    ax.set_title(f'{domain.capitalize()} Domain: LoRA Improvement', fontsize=14, fontweight='bold')
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.grid(alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path / "delta_plot.png", dpi=150, bbox_inches='tight')
    plt.close()


def compare_models(
    domain: str,
    test_data: List[Dict],
    adapter_path: str,
    output_dir: str = "./evaluation_results"
):
    """Compare base vs LoRA with statistical tests"""
    
    # Evaluate base
    print("\n" + "="*80)
    print("EVALUATING BASE MODEL")
    print("="*80)
    base_results, base_avg = evaluate_model(domain, test_data)
    
    # Evaluate LoRA
    print("\n" + "="*80)
    print("EVALUATING LORA MODEL")
    print("="*80)
    lora_results, lora_avg = evaluate_model(domain, test_data, adapter_path)
    
    # Statistical comparison
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)
    
    metrics = ["rouge1", "rouge2", "rougeL", "bertscore"]
    
    comparison = {}
    for metric in metrics:
        base_scores = [r[metric] for r in base_results]
        lora_scores = [r[metric] for r in lora_results]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(lora_scores, base_scores)
        
        delta = lora_avg[metric] - base_avg[metric]
        
        comparison[metric] = {
            "base_mean": base_avg[metric],
            "lora_mean": lora_avg[metric],
            "delta": delta,
            "delta_pct": (delta / base_avg[metric]) * 100 if base_avg[metric] > 0 else 0,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
        
        sig_marker = "✓" if p_value < 0.05 else "✗"
        print(f"\n{metric.upper()}:")
        print(f"  Base:  {base_avg[metric]:.4f}")
        print(f"  LoRA:  {lora_avg[metric]:.4f}")
        print(f"  Delta: {delta:+.4f} ({comparison[metric]['delta_pct']:+.2f}%)")
        print(f"  p-value: {p_value:.4f} {sig_marker} {'Significant' if p_value < 0.05 else 'Not significant'}")
    
    # Save results
    output_path = Path(output_dir) / domain
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "comparison.json", "w") as f:
        json.dump({
            "domain": domain,
            "base_results": base_results,
            "lora_results": lora_results,
            "base_avg": base_avg,
            "lora_avg": lora_avg,
            "comparison": comparison,
        }, f, indent=2)
    
    # Generate plots
    plot_comparison(comparison, output_path, domain)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(domain, comparison)
    with open(output_path / "table.tex", "w") as f:
        f.write(latex_table)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"  - comparison.json (detailed results)")
    print(f"  - comparison_plot.png (bar chart)")
    print(f"  - delta_plot.png (improvement chart)")
    print(f"  - table.tex (LaTeX table)")
    
    return comparison


def generate_latex_table(domain: str, comparison: Dict) -> str:
    """Generate LaTeX table for paper"""
    
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Evaluation Results: {domain.capitalize()} Domain}}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\hline\n"
    latex += "Metric & Base & LoRA & $\\Delta$ & p-value \\\\\n"
    latex += "\\hline\n"
    
    for metric in ["rouge1", "rouge2", "rougeL", "bertscore"]:
        comp = comparison[metric]
        sig = "*" if comp["significant"] else ""
        
        latex += f"{metric.upper()} & "
        latex += f"{comp['base_mean']:.3f} & "
        latex += f"{comp['lora_mean']:.3f} & "
        latex += f"{comp['delta']:+.3f}{sig} & "
        latex += f"{comp['p_value']:.3f} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\label{tab:" + domain + "}\n"
    latex += "\\end{table}\n"
    
    return latex


def load_test_data_from_jsonl(jsonl_path: str, max_samples: int = 200) -> List[Dict]:
    """Load test data from JSONL file"""
    
    test_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            item = json.loads(line)
            
            # Auto-detect columns
            text = item.get("text") or item.get("article") or item.get("document") or ""
            summary = item.get("summary") or item.get("abstract") or item.get("highlights") or ""
            
            if text and summary:
                test_data.append({
                    "video_id": item.get("id", f"sample_{i}"),
                    "transcript": text,
                    "reference_summary": summary
                })
    
    return test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapters against base model")
    parser.add_argument("domain", choices=["medical", "engineering", "scientific"],
                       help="Domain to evaluate")
    parser.add_argument("--test-data", required=True,
                       help="Path to test JSONL file")
    parser.add_argument("--adapter-path", required=True,
                       help="Path to trained LoRA adapter")
    parser.add_argument("--output-dir", default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=200,
                       help="Maximum test samples")
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_data = load_test_data_from_jsonl(args.test_data, args.max_samples)
    print(f"Loaded {len(test_data)} test samples")
    
    # Compare models
    compare_models(
        domain=args.domain,
        test_data=test_data,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir
    )