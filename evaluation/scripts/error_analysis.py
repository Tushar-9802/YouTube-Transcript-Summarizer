#!/usr/bin/env python3
"""
Comprehensive Error Analysis for LoRA Video Summarization
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

class ErrorAnalyzer:
    """Automated failure detection"""
    
    ACADEMIC_PHRASES = [
        r'(?i)\b(this|the) (paper|study|research|work|article)\b',
        r'(?i)\bwe (present|propose|demonstrate|show)\b',
        r'(?i)\b(authors?|researchers?)\b',
        r'(?i)\bet al\.?',
    ]
    
    def __init__(self, results_file, test_data_dir, output_dir):
        self.results = pd.read_csv(results_file)
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_academic_framing(self, text):
        """Detect academic writing patterns"""
        if pd.isna(text):
            return False, []
        
        found = []
        for pattern in self.ACADEMIC_PHRASES:
            matches = re.findall(pattern, str(text))
            if matches:
                found.extend(matches)
        return len(found) > 0, found
    
    def categorize_failure(self, row):
        """Analyze single summary"""
        summary = row['summary']
        
        errors = {
            'video_id': row['video_id'],
            'domain': row['domain'],
            'categories': [],
            'rouge1': row['rouge1_f'],
            'bertscore': row['bertscore_f']
        }
        
        # Check academic framing
        has_academic, phrases = self.detect_academic_framing(summary)
        if has_academic:
            errors['categories'].append('academic_framing')
            errors['examples'] = phrases[:3]
        
        # Check length
        word_count = len(str(summary).split())
        if word_count < 50:
            errors['categories'].append('too_short')
        elif word_count > 300:
            errors['categories'].append('too_long')
        
        # Check for citation artifacts
        if re.search(r'\[\d+\]', summary) or 'et al' in summary.lower():
            errors['categories'].append('citation_artifacts')
        
        return errors
    
    def run_analysis(self):
        """Execute analysis"""
        print("="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        
        lora_df = self.results[self.results['model_type'] == 'lora']
        base_df = self.results[self.results['model_type'] == 'base']
        
        print(f"\nAnalyzing {len(lora_df)} LoRA summaries")
        print(f"Comparing with {len(base_df)} base summaries")
        
        # Analyze all LoRA outputs
        lora_errors = []
        for _, row in lora_df.iterrows():
            error_info = self.categorize_failure(row)
            lora_errors.append(error_info)
        
        # Analyze base for comparison
        base_errors = []
        for _, row in base_df.iterrows():
            error_info = self.categorize_failure(row)
            base_errors.append(error_info)
        
        # Count frequencies
        lora_category_counts = defaultdict(int)
        base_category_counts = defaultdict(int)
        
        for err in lora_errors:
            for cat in err['categories']:
                lora_category_counts[cat] += 1
        
        for err in base_errors:
            for cat in err['categories']:
                base_category_counts[cat] += 1
        
        # Print results
        print(f"\nERROR FREQUENCIES:")
        print(f"\n{'Category':<25} {'LoRA':>10} {'Base':>10} {'Difference':>12}")
        print("-" * 60)
        
        all_categories = set(lora_category_counts.keys()) | set(base_category_counts.keys())
        total_lora = len(lora_errors)
        total_base = len(base_errors)
        
        for cat in sorted(all_categories):
            lora_count = lora_category_counts[cat]
            base_count = base_category_counts[cat]
            lora_pct = (lora_count/total_lora)*100
            base_pct = (base_count/total_base)*100
            diff = lora_pct - base_pct
            
            print(f"{cat:<25} {lora_count:3d} ({lora_pct:4.1f}%) {base_count:3d} ({base_pct:4.1f}%) {diff:+6.1f}%")
        
        # Save results
        output_file = self.output_dir / "failure_analysis.json"
        with open(output_file, 'w') as f:
            json.dump({
                'total_lora': len(lora_errors),
                'total_base': len(base_errors),
                'lora_errors': lora_errors,
                'base_errors': base_errors,
                'lora_statistics': dict(lora_category_counts),
                'base_statistics': dict(base_category_counts)
            }, f, indent=2)
        
        print(f"\n✓ Saved: {output_file}")
        
        # Create case studies - most severe LoRA failures
        print("\n" + "="*60)
        print("EXAMPLE FAILURES (LoRA-specific issues)")
        print("="*60)
        
        # Find cases with academic framing in LoRA but not base
        severe_cases = []
        for lora_err in lora_errors:
            if 'academic_framing' in lora_err['categories']:
                # Check if base had same issue
                base_err = next((e for e in base_errors if e['video_id'] == lora_err['video_id']), None)
                if base_err and 'academic_framing' not in base_err['categories']:
                    severe_cases.append((lora_err, base_err))
        
        for i, (lora_err, base_err) in enumerate(severe_cases[:3], 1):
            print(f"\nCase {i} - {lora_err['video_id']} ({lora_err['domain']})")
            print(f"  LoRA issues: {', '.join(lora_err['categories'])}")
            print(f"  LoRA ROUGE-1: {lora_err['rouge1']:.3f} | Base ROUGE-1: {base_err['rouge1']:.3f}")
            if 'examples' in lora_err:
                print(f"  Academic phrases: {lora_err['examples']}")
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"LoRA summaries with issues: {sum(1 for e in lora_errors if e['categories'])} / {total_lora}")
        print(f"Base summaries with issues: {sum(1 for e in base_errors if e['categories'])} / {total_base}")
        
        if lora_category_counts:
            most_common = max(lora_category_counts.items(), key=lambda x: x[1])
            print(f"\nMost common LoRA issue: {most_common[0]} ({most_common[1]} occurrences, {most_common[1]/total_lora*100:.1f}%)")

if __name__ == "__main__":
    analyzer = ErrorAnalyzer(
        results_file="results/evaluation/detailed_results.csv",
        test_data_dir="data/youtube_test",
        output_dir="results/error_analysis"
    )
    analyzer.run_analysis()
