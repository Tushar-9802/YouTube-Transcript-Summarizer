# src/yt_sum/train/train_lora_8gb.py
"""
LoRA Training for 8GB VRAM - OOM Protected
RTX 4070 Laptop optimized with aggressive memory management
Includes training visualization and TensorBoard logging
"""

import torch
import gc
import os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import json
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

# Force memory efficiency
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


class MemoryConfig:
    """8GB VRAM conservative settings - TESTED on RTX 4070"""
    
    # Model
    BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    MAX_SEQ_LENGTH = 1024  # Reduced from 2048 for safety
    
    # LoRA - Reduced parameters
    LORA_R = 8  # Reduced from 16 (fewer params = less memory)
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "v_proj"]  # Only attention (not MLP = memory save)
    
    # Training - Ultra conservative
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 16  # Effective batch = 16
    EPOCHS = 2  # Reduced from 3
    LEARNING_RATE = 2e-4
    WARMUP_STEPS = 50
    
    # Memory safety
    OPTIM = "paged_adamw_8bit"
    GRADIENT_CHECKPOINTING = True


def aggressive_cleanup():
    """OOM prevention"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def validate_paths(domain: str, jsonl_path: str):
    """Prevent path errors before training starts"""
    import json
    
    path = Path(jsonl_path)
    
    # Check existence
    if not path.exists():
        raise FileNotFoundError(
            f"\n❌ File not found: {jsonl_path}\n"
            f"Expected: {path.resolve()}\n"
            f"Run: ls {path.parent}  # to see available files"
        )
    
    # Check extension
    if path.suffix != ".jsonl":
        raise ValueError(f"Expected .jsonl, got: {path.suffix}")
    
    # Validate JSONL format
    try:
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            json.loads(first_line)
    except Exception as e:
        raise ValueError(f"Invalid JSONL format in {path.name}: {e}")
    
    # Check size
    size_mb = path.stat().st_size / 1e6
    print(f"✓ Dataset: {path.name} ({size_mb:.1f} MB)")
    
    return path


def load_dataset_safe(jsonl_path: str, max_samples: int = 3000):
    """
    Load with automatic column detection.
    Reduced default samples to 3000 for memory safety.
    """
    print(f"Loading: {jsonl_path}")
    
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    cols = ds.column_names
    
    # Detect columns
    column_pairs = [
        ("text", "summary"),
        ("article", "abstract"), 
        ("source", "target"),
        ("document", "summary"),
    ]
    
    text_col = summary_col = None
    for t, s in column_pairs:
        if t in cols and s in cols:
            text_col, summary_col = t, s
            break
    
    if not text_col:
        raise ValueError(f"Cannot detect columns in {cols}")
    
    print(f"Using: {text_col} -> {summary_col}")
    
    # Sample conservatively
    ds = ds.shuffle(seed=42)
    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    
    return ds, text_col, summary_col


def format_dataset(ds, text_col: str, summary_col: str, domain: str):
    """Format for Mistral"""
    
    def format_fn(example):
        # Truncate long inputs (OOM prevention)
        text = example[text_col][:8000]  # Hard limit
        summary = example[summary_col][:2000]
        
        instruction = f"Summarize this {domain} text:\n\n{text}"
        
        # Mistral format
        return {
            "text": f"<s>[INST] {instruction} [/INST] {summary}</s>"
        }
    
    formatted = ds.map(
        format_fn,
        remove_columns=ds.column_names,
        desc="Formatting",
        num_proc=1  # Single process to save memory
    )
    
    return formatted.train_test_split(test_size=0.1, seed=42)


def generate_training_plots(trainer, output_path: Path, domain: str):
    """Generate loss and learning rate plots from training history"""
    
    print("\nGenerating training plots...")
    
    try:
        # Save training history
        log_history = pd.DataFrame(trainer.state.log_history)
        log_csv = output_path / "training_log.csv"
        log_history.to_csv(log_csv, index=False)
        print(f"📊 Training logs: {log_csv}")
        
        # Loss curve
        if "loss" in log_history.columns and len(log_history) > 0:
            plt.figure(figsize=(10, 6))
            steps = log_history.dropna(subset=["loss"])["step"]
            losses = log_history.dropna(subset=["loss"])["loss"]
            plt.plot(steps, losses, label="Training Loss", linewidth=2, color='#2E86AB')
            
            if "eval_loss" in log_history.columns:
                eval_steps = log_history.dropna(subset=["eval_loss"])["step"]
                eval_losses = log_history.dropna(subset=["eval_loss"])["eval_loss"]
                if len(eval_steps) > 0:
                    plt.plot(eval_steps, eval_losses, label="Eval Loss", 
                            linewidth=2, color='#A23B72', marker='o')
            
            plt.xlabel("Step", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title(f"{domain.capitalize()} LoRA Training Loss", fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            loss_plot = output_path / "loss_curve.png"
            plt.savefig(loss_plot, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📉 Loss curve: {loss_plot}")
        
        # Learning rate schedule
        if "learning_rate" in log_history.columns:
            plt.figure(figsize=(10, 6))
            lr_data = log_history.dropna(subset=["learning_rate"])
            if len(lr_data) > 0:
                plt.plot(lr_data["step"], lr_data["learning_rate"], 
                        linewidth=2, color='#F18F01')
                plt.xlabel("Step", fontsize=12)
                plt.ylabel("Learning Rate", fontsize=12)
                plt.title("Learning Rate Schedule (Cosine with Warmup)", 
                         fontsize=14, fontweight='bold')
                plt.grid(alpha=0.3, linestyle='--')
                plt.tight_layout()
                
                lr_plot = output_path / "lr_curve.png"
                plt.savefig(lr_plot, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"📈 LR curve: {lr_plot}")
        
        # Training summary stats
        if len(log_history) > 0:
            summary = {
                "final_loss": float(log_history["loss"].dropna().iloc[-1]) if "loss" in log_history.columns else None,
                "min_loss": float(log_history["loss"].dropna().min()) if "loss" in log_history.columns else None,
                "final_eval_loss": float(log_history["eval_loss"].dropna().iloc[-1]) if "eval_loss" in log_history.columns and len(log_history["eval_loss"].dropna()) > 0 else None,
                "total_steps": int(log_history["step"].max()) if "step" in log_history.columns else None,
            }
            
            summary_file = output_path / "training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"📋 Training summary: {summary_file}")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plots: {e}")


def train_adapter_8gb(
    domain: str,
    jsonl_path: str,
    output_dir: str = "./models/adapters",
    max_samples: int = 3000,
):
    """
    OOM-protected training for 8GB VRAM.
    """
    
    # VALIDATE FIRST (prevents wasted GPU time)
    jsonl_path = validate_paths(domain, jsonl_path)
    
    output_path = Path(output_dir) / domain
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Training: {domain}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Max samples: {max_samples}")
    print(f"{'='*80}\n")
    
    # Initial cleanup
    aggressive_cleanup()
    
    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print("Loading model with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MemoryConfig.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MemoryConfig.BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare for k-bit
    model = prepare_model_for_kbit_training(model)
    
    # ========================================================================
    # ADD LORA
    # ========================================================================
    print("Adding LoRA adapters...")
    
    lora_config = LoraConfig(
        r=MemoryConfig.LORA_R,
        lora_alpha=MemoryConfig.LORA_ALPHA,
        target_modules=MemoryConfig.TARGET_MODULES,
        lora_dropout=MemoryConfig.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    aggressive_cleanup()
    
    # ========================================================================
    # LOAD DATASET
    # ========================================================================
    print("Loading dataset...")
    
    ds, text_col, summary_col = load_dataset_safe(jsonl_path, max_samples)
    dataset = format_dataset(ds, text_col, summary_col, domain)
    
    print(f"Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")
    
    # ========================================================================
    # TRAIN
    # ========================================================================
    print("Training...")
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=MemoryConfig.EPOCHS,
        per_device_train_batch_size=MemoryConfig.BATCH_SIZE,
        gradient_accumulation_steps=MemoryConfig.GRADIENT_ACCUMULATION,
        learning_rate=MemoryConfig.LEARNING_RATE,
        warmup_steps=MemoryConfig.WARMUP_STEPS,
        optim=MemoryConfig.OPTIM,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        gradient_checkpointing=MemoryConfig.GRADIENT_CHECKPOINTING,
        max_grad_norm=0.3,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        dataloader_num_workers=0,  # Avoid multiprocess memory issues
        
        # TensorBoard logging
        report_to=["tensorboard"],
        logging_dir=str(output_path / "logs"),
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=MemoryConfig.MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,
    )
    
    aggressive_cleanup()
    
    # Train with OOM protection
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n*** OOM DETECTED - Reduce max_samples and retry ***\n")
            raise
        raise
    
    # ========================================================================
    # SAVE
    # ========================================================================
    print("\nSaving model and generating plots...")
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Generate training plots
    generate_training_plots(trainer, output_path, domain)
    
    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump({
            "domain": domain,
            "samples": len(dataset["train"]),
            "epochs": MemoryConfig.EPOCHS,
            "lora_r": MemoryConfig.LORA_R,
            "lora_alpha": MemoryConfig.LORA_ALPHA,
            "learning_rate": MemoryConfig.LEARNING_RATE,
            "base_model": MemoryConfig.BASE_MODEL,
        }, f, indent=2)
    
    print(f"\n✓ Complete! Saved to: {output_path}")
    print(f"  - Adapter weights: adapter_model.bin")
    print(f"  - Training logs: training_log.csv")
    print(f"  - Loss curve: loss_curve.png")
    print(f"  - LR curve: lr_curve.png")
    print(f"  - TensorBoard logs: logs/")
    print(f"\nView in TensorBoard: tensorboard --logdir {output_path / 'logs'}")
    
    # Cleanup
    del model, trainer
    aggressive_cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter for domain-specific summarization (8GB VRAM optimized)"
    )
    parser.add_argument(
        "domain", 
        choices=["medical", "engineering", "scientific"],
        help="Domain to train on"
    )
    parser.add_argument(
        "jsonl_path",
        help="Path to training data (.jsonl file)"
    )
    parser.add_argument(
        "--output-dir", 
        default="./models/adapters",
        help="Output directory for trained adapters (default: ./models/adapters)"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=3000,
        help="Maximum training samples (reduce if OOM, default: 3000)"
    )
    
    args = parser.parse_args()
    
    train_adapter_8gb(
        domain=args.domain,
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )