"""
Quality Assurance and Validation Suite - OPTIMIZED
Pre-flight checks before evaluation
"""

import sys
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import json
from typing import Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    passed: bool
    message: str
    details: Dict = None

class QAValidator:
    """Comprehensive validation suite"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or ROOT
        self.results = []
    
    def log_result(self, passed: bool, test_name: str, message: str, details: Dict = None):
        result = ValidationResult(passed, message, details)
        self.results.append((test_name, result))
        
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name} - {message}")
        
        if details:
            for key, value in details.items():
                logger.info(f"  {key}: {value}")
    
    def validate_cuda(self) -> bool:
        """Validate CUDA availability"""
        logger.info("=== Validating CUDA ===")
        
        if not torch.cuda.is_available():
            self.log_result(False, "CUDA", "CUDA not available")
            return False
        
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        details = {
            "Device": device_name,
            "Total Memory": f"{total_mem:.2f} GB",
            "CUDA Version": torch.version.cuda,
        }
        
        if total_mem < 6:
            self.log_result(False, "CUDA Memory", f"Insufficient VRAM: {total_mem:.1f}GB (need 6GB minimum)", details)
            return False
        
        self.log_result(True, "CUDA", f"CUDA functional with {total_mem:.1f}GB VRAM", details)
        return True
    
    def validate_bitsandbytes(self) -> bool:
        """Validate bitsandbytes 4-bit quantization"""
        logger.info("=== Validating Bitsandbytes ===")
        
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            
            version = bnb.__version__
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            details = {"Version": version, "4-bit Support": "Yes", "NF4 Quantization": "Available"}
            self.log_result(True, "Bitsandbytes", "4-bit quantization available", details)
            return True
            
        except ImportError as e:
            self.log_result(False, "Bitsandbytes", f"Import failed: {e}")
            return False
        except Exception as e:
            self.log_result(False, "Bitsandbytes", f"Configuration failed: {e}")
            return False
    
    def validate_adapters(self) -> bool:
        """Validate LoRA adapter files"""
        logger.info("=== Validating LoRA Adapters ===")
        
        adapters_dir = self.project_root / "models" / "adapters"
        required_domains = ["medical", "engineering", "scientific"]
        
        all_valid = True
        
        for domain in required_domains:
            adapter_path = adapters_dir / domain
            
            if not adapter_path.exists():
                self.log_result(False, f"Adapter: {domain}", "Directory not found")
                all_valid = False
                continue
            
            config_file = adapter_path / "adapter_config.json"
            model_file = adapter_path / "adapter_model.safetensors"
            
            if not config_file.exists():
                self.log_result(False, f"Adapter: {domain}", "Missing adapter_config.json")
                all_valid = False
                continue
            
            if not model_file.exists():
                self.log_result(False, f"Adapter: {domain}", "Missing adapter_model.safetensors")
                all_valid = False
                continue
            
            try:
                with open(config_file) as f:
                    config = json.load(f)
                
                details = {
                    "Rank": config.get('r', 'N/A'),
                    "Alpha": config.get('lora_alpha', 'N/A'),
                    "Target Modules": ', '.join(config.get('target_modules', [])),
                    "Model Size": f"{model_file.stat().st_size / 1e6:.1f} MB",
                }
                
                self.log_result(True, f"Adapter: {domain}", "Valid adapter", details)
                
            except Exception as e:
                self.log_result(False, f"Adapter: {domain}", f"Config validation failed: {e}")
                all_valid = False
        
        return all_valid
    
    def validate_test_data(self) -> bool:
        """Validate test dataset"""
        logger.info("=== Validating Test Data ===")
        
        test_dir = self.project_root / "data" / "youtube_test"
        required_domains = ["medical", "engineering", "scientific"]
        
        all_valid = True
        
        for domain in required_domains:
            test_file = test_dir / f"{domain}_test.jsonl"
            
            if not test_file.exists():
                self.log_result(False, f"Test Data: {domain}", "Test file not found (run curation first)")
                all_valid = False
                continue
            
            try:
                samples = []
                with open(test_file) as f:
                    for line in f:
                        samples.append(json.loads(line))
                
                if len(samples) == 0:
                    self.log_result(False, f"Test Data: {domain}", "Empty test file")
                    all_valid = False
                    continue
                
                required_fields = ['video_id', 'url', 'transcript', 'reference_summary', 'domain']
                sample = samples[0]
                missing_fields = [f for f in required_fields if f not in sample]
                
                if missing_fields:
                    self.log_result(False, f"Test Data: {domain}", f"Missing fields: {missing_fields}")
                    all_valid = False
                    continue
                
                avg_transcript_len = sum(len(s['transcript'].split()) for s in samples) / len(samples)
                
                details = {
                    "Sample Count": len(samples),
                    "Avg Transcript Length": f"{int(avg_transcript_len)} words",
                }
                
                self.log_result(True, f"Test Data: {domain}", "Valid test set", details)
                
            except json.JSONDecodeError as e:
                self.log_result(False, f"Test Data: {domain}", f"Invalid JSON: {e}")
                all_valid = False
            except Exception as e:
                self.log_result(False, f"Test Data: {domain}", f"Validation failed: {e}")
                all_valid = False
        
        return all_valid
    
    def validate_dependencies(self) -> bool:
        """Validate Python dependencies"""
        logger.info("=== Validating Dependencies ===")
        
        required_packages = {
            'torch': 'PyTorch',
            'transformers': 'Transformers',
            'peft': 'PEFT',
            'bitsandbytes': 'Bitsandbytes',
            'rouge_score': 'ROUGE Score',
            'bert_score': 'BERTScore',
            'yt_dlp': 'yt-dlp',
        }
        
        all_valid = True
        versions = {}
        
        for package, name in required_packages.items():
            try:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                versions[name] = version
            except ImportError:
                self.log_result(False, f"Dependency: {name}", "Not installed")
                all_valid = False
        
        if all_valid:
            self.log_result(True, "Dependencies", "All dependencies installed", versions)
        
        return all_valid
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        logger.info("=" * 60)
        logger.info("STARTING QUALITY ASSURANCE VALIDATION")
        logger.info("=" * 60)
        
        checks = [
            ("Dependencies", self.validate_dependencies),
            ("CUDA", self.validate_cuda),
            ("Bitsandbytes", self.validate_bitsandbytes),
            ("Adapters", self.validate_adapters),
            ("Test Data", self.validate_test_data),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            passed = check_func()
            if not passed:
                all_passed = False
            logger.info("")
        
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        passed_count = sum(1 for _, r in self.results if r.passed)
        total_count = len(self.results)
        
        logger.info(f"Total Checks: {total_count}")
        logger.info(f"Passed: {passed_count}")
        logger.info(f"Failed: {total_count - passed_count}")
        
        if all_passed:
            logger.info("")
            logger.info("✓ ALL VALIDATIONS PASSED - READY FOR EVALUATION")
        else:
            logger.info("")
            logger.info("✗ SOME VALIDATIONS FAILED - FIX ISSUES BEFORE PROCEEDING")
            logger.info("")
            logger.info("Failed checks:")
            for test_name, result in self.results:
                if not result.passed:
                    logger.info(f"  - {test_name}: {result.message}")
        
        logger.info("=" * 60)
        
        return all_passed

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='QA Validation Suite')
    parser.add_argument('--project-root', type=Path, default=None)
    
    args = parser.parse_args()
    
    validator = QAValidator(project_root=args.project_root)
    success = validator.run_full_validation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()