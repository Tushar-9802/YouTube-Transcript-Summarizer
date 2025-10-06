"""
LoRA Fine-tuning for Domain-Specific Video Summarization
Optimized for Google Colab T4 GPU (15GB VRAM)

Training method: QLoRA (4-bit quantization + LoRA adapters)
Base model: Mistral-7B-Instruct-v0.2
Hardware: NVIDIA T4 GPU, ~11GB VRAM usage, ~1 hour per domain

Usage:
1. Upload dataset (medical.jsonl/scientific.jsonl/engineering.jsonl) to Colab
2. Run setup cell: !pip install -q bitsandbytes peft sentencepiece
3. Execute this script with appropriate dataset filename
"""

from huggingface_hub import notebook_login
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import pandas as pd
import matplotlib.pyplot as plt

# Authenticate with HuggingFace (required for Mistral-7B-Instruct-v0.2)
notebook_login()

# Configuration
DOMAIN = "medical"  # Change to: "scientific" or "engineering"
DATASET_FILE = f"{DOMAIN}.jsonl"
OUTPUT_DIR = f"./{DOMAIN}_adapter"
MAX_SAMPLES = 2000
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]  # Attention projection layers only
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

# 4-bit quantization configuration (reduces 14GB model to ~4GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4-bit (optimal for LLMs)
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Nested quantization for better precision
)

# Load model with 4-bit quantization
print(f"Loading {BASE_MODEL} with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# Enable gradient checkpointing (saves memory)
model.gradient_checkpointing_enable()

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

# Add LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load and prepare dataset
print(f"Loading {DATASET_FILE}...")
ds = load_dataset("json", data_files=DATASET_FILE, split="train")
ds = ds.shuffle(seed=42).select(range(min(len(ds), MAX_SAMPLES)))

def format_instruction(examples):
    """Format data with Mistral instruction template"""
    prompts = [
        f"<s>[INST] Summarize this {DOMAIN} text:\n\n{text[:3000]} [/INST] {summary[:800]}</s>"
        for text, summary in zip(examples["text"], examples["summary"])
    ]
    return tokenizer(
        prompts, 
        truncation=True, 
        max_length=MAX_SEQ_LENGTH, 
        padding=False
    )

# Tokenize dataset
tokenized = ds.map(
    format_instruction, 
    batched=True, 
    remove_columns=ds.column_names
)

# Train/eval split
dataset_split = tokenized.train_test_split(test_size=0.1, seed=42)
print(f"Train samples: {len(dataset_split['train'])}, Eval samples: {len(dataset_split['test'])}")

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",  # 8-bit optimizer (saves memory)
    warmup_steps=10,
    report_to="none",  # Disable wandb
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train
print("\nStarting training...")
trainer.train()

# Save adapter
print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Generate training visualizations
print("Generating training plots...")
log_history = pd.DataFrame(trainer.state.log_history)
log_history.to_csv(f"{OUTPUT_DIR}/training_log.csv", index=False)

if "loss" in log_history.columns:
    plt.figure(figsize=(10, 6))
    train_data = log_history.dropna(subset=["loss"])
    
    plt.plot(
        train_data["step"], 
        train_data["loss"], 
        linewidth=2, 
        color='#2E86AB',
        label="Training Loss"
    )
    
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"{DOMAIN.capitalize()} Domain LoRA Training", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3, linestyle='--')
    plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to {OUTPUT_DIR}/loss_curve.png")

print("\nTraining complete!")
print(f"Adapter saved to: {OUTPUT_DIR}")
print(f"Trainable parameters: {model.get_nb_trainable_parameters()}")

# Download instructions (for Colab)
print("\nTo download adapter, run:")
print(f"!zip -r {DOMAIN}_adapter.zip {OUTPUT_DIR}")
print(f"from google.colab import files; files.download('{DOMAIN}_adapter.zip')")