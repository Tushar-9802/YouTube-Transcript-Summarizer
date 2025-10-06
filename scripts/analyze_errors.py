"""
Error Analysis and Failure Mode Detection - OPTIMIZED
Compatible with evaluation pipeline
"""

import sys
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict
import re
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ErrorCase:
    video_id: str
    domain: str
    model_type: str
    generated_summary: str
    reference_summary: str
    rouge1_f: float
    bertscore_f: float
    error_categories: List[str]
    severity: str

class ErrorAnalyzer:
    """Automated error detection and categorization"""
    
    HALLUCINATION_PATTERNS = [
        r'(?i)(the video discusses|this video presents|the speaker mentions)',
        r'(?i)(in this lecture|in this presentation)',
        r'(?i)(as shown in|as demonstrated)',
    ]
    
    def __init__(self, results_csv: Path, test_data_dir: Path, output_dir: Path):
        self.results_csv = results_csv
        self.test_data_dir = test_data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = pd.read_csv(results_csv)
        self.test_data = self._load_test_data()
    
    def _load_test_data(self) -> Dict:
        """Load original test data"""
        data = {}
        for domain in ['medical', 'engineering', 'scientific']:
            test_file = self.test_data_dir / f"{domain}_test.jsonl"
            if test_file.exists():
                with open(test_file) as f:
                    for line in f:
                        sample = json.loads(line)
                        data[sample['video_id']] = sample
        return data
    
    def detect_hallucinations(self, summary: str) -> bool:
        """Detect meta-commentary hallucinations"""
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, summary):
                return True
        return False
    
    def detect_repetition(self, summary: str) -> bool:
        """Detect repetitive patterns"""
        sentences = summary.split('.')
        if len(sentences) < 2:
            return False
        
        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[i+1:]:
                if sent1.strip() and sent1.strip() == sent2.strip():
                    return True
        
        return False
    
    def categorize_error(self, row: pd.Series, test_sample: Dict) -> List[str]:
        """Categorize error types"""
        categories = []
        
        summary = row['summary']
        
        if self.detect_hallucinations(summary):
            categories.append('hallucination')
        
        if self.detect_repetition(summary):
            categories.append('repetition')
        
        ref_len = len(test_sample['reference_summary'].split())
        gen_len = len(summary.split())
        length_ratio = gen_len / ref_len if ref_len > 0 else 0
        
        if length_ratio < 0.5:
            categories.append('too_short')
        elif length_ratio > 2.0:
            categories.append('too_long')
        
        if row['bertscore_f'] < 0.6:
            categories.append('semantic_drift')
        
        if row['rouge2_f'] < 0.1:
            categories.append('too_generic')
        
        return categories
    
    def determine_severity(self, rouge1: float, bertscore: float) -> str:
        """Determine error severity"""
        if rouge1 < 0.2 or bertscore < 0.5:
            return 'high'
        elif rouge1 < 0.35 or bertscore < 0.7:
            return 'medium'
        else:
            return 'low'
    
    def analyze_all_errors(self, score_threshold: float = 0.4) -> List[ErrorCase]:
        """Analyze all errors"""
        error_cases = []
        
        for _, row in self.results.iterrows():
            if row['rouge1_f'] < score_threshold:
                video_id = row['video_id']
                test_sample = self.test_data.get(video_id, {})
                
                if not test_sample:
                    continue
                
                categories = self.categorize_error(row, test_sample)
                severity = self.determine_severity(row['rouge1_f'], row['bertscore_f'])
                
                error_cases.append(ErrorCase(
                    video_id=video_id,
                    domain=row['domain'],
                    model_type=row['model_type'],
                    generated_summary=row['summary'],
                    reference_summary=test_sample.get('reference_summary', ''),
                    rouge1_f=row['rouge1_f'],
                    bertscore_f=row['bertscore_f'],
                    error_categories=categories,
                    severity=severity,
                ))
        
        return error_cases
    
    def generate_error_distribution_plot(self, error_cases: List[ErrorCase]):
        """Plot error distribution"""
        category_counts = defaultdict(lambda: {'base': 0, 'lora': 0})
        
        for case in error_cases:
            for cat in case.error_categories:
                category_counts[cat][case.model_type] += 1
        
        categories = list(category_counts.keys())
        base_counts = [category_counts[cat]['base'] for cat in categories]
        lora_counts = [category_counts[cat]['lora'] for cat in categories]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, base_counts, width, label='Base Model', alpha=0.8)
        ax.bar(x + width/2, lora_counts, width, label='LoRA Model', alpha=0.8)
        
        ax.set_xlabel('Error Category')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution: Base vs LoRA')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_error_report(self, error_cases: List[ErrorCase]):
        """Generate comprehensive error report"""
        report = ["# Error Analysis Report\n"]
        
        total_errors = len(error_cases)
        base_errors = [c for c in error_cases if c.model_type == 'base']
        lora_errors = [c for c in error_cases if c.model_type == 'lora']
        
        report.append(f"## Overall Statistics\n")
        report.append(f"- Total Error Cases: {total_errors}")
        report.append(f"- Base Model Errors: {len(base_errors)}")
        report.append(f"- LoRA Model Errors: {len(lora_errors)}\n")
        
        # Severity distribution
        severity_dist = defaultdict(int)
        for case in error_cases:
            severity_dist[case.severity] += 1
        
        report.append(f"## Severity Distribution\n")
        for severity in ['high', 'medium', 'low']:
            count = severity_dist[severity]
            pct = (count / total_errors * 100) if total_errors > 0 else 0
            report.append(f"- {severity.capitalize()}: {count} ({pct:.1f}%)")
        report.append("\n")
        
        # Top error categories
        all_categories = defaultdict(int)
        for case in error_cases:
            for cat in case.error_categories:
                all_categories[cat] += 1
        
        report.append(f"## Top Error Categories\n")
        sorted_cats = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)
        for cat, count in sorted_cats[:5]:
            report.append(f"- {cat}: {count} occurrences")
        report.append("\n")
        
        with open(self.output_dir / 'error_analysis_report.md', 'w') as f:
            f.write('\n'.join(report))
    
    def export_error_cases_csv(self, error_cases: List[ErrorCase]):
        """Export error cases to CSV"""
        data = []
        for case in error_cases:
            data.append({
                'video_id': case.video_id,
                'domain': case.domain,
                'model_type': case.model_type,
                'severity': case.severity,
                'error_categories': '|'.join(case.error_categories),
                'rouge1_f': case.rouge1_f,
                'bertscore_f': case.bertscore_f,
                'generated_summary': case.generated_summary[:200],
                'reference_summary': case.reference_summary[:200],
            })
        
        df = pd.DataFrame(data)
        df.to_csv(self.output_dir / 'error_cases.csv', index=False)
    
    def run_full_analysis(self):
        """Execute complete error analysis"""
        logger.info("Analyzing errors...")
        
        error_cases = self.analyze_all_errors(score_threshold=0.4)
        logger.info(f"Found {len(error_cases)} error cases")
        
        logger.info("Generating visualizations...")
        self.generate_error_distribution_plot(error_cases)
        
        logger.info("Generating error report...")
        self.generate_error_report(error_cases)
        
        logger.info("Exporting error cases...")
        self.export_error_cases_csv(error_cases)
        
        logger.info(f"Error analysis complete! Output saved to: {self.output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Error Analysis')
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--test-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/error_analysis'))
    
    args = parser.parse_args()
    
    analyzer = ErrorAnalyzer(
        results_csv=args.results,
        test_data_dir=args.test_data,
        output_dir=args.output
    )
    
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()