# Domain-Aware YouTube Video Summarization with LoRA

**Cross-Modal Transfer Learning for Educational Content**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository implements and analyzes **"When Academic Training Fails: Cross-Modal Transfer Learning in Domain-Adaptive Video Summarization"**. It investigates whether domain-specific LoRA adapters trained on academic papers improve YouTube video summarization.

**Key Finding:** Domain-specific training on 25,000 papers **degraded** video summarization performance by 14–50% (p<0.001), showing that **data modality outweighs domain similarity** when discourse structures diverge.

> **Note:** The Streamlit GUI (`scripts/app.py`) uses the base **4-bit quantized Mistral-7B-Instruct-v0.2** model for general-purpose video summarization. LoRA adapters are **not** loaded in the GUI, as research showed they harm video performance. For LoRA experiments, use command-line scripts.

### Contributions

- **Rigorous negative result:** Adapters work on papers (+6–8% ROUGE) but fail on videos
- **Automated error taxonomy:** Quantifies failure modes (14.6% academic framing, 54% style mismatch)
- **Consumer GPU pipeline:** Enables 7B parameter inference in 6.2GB VRAM via 4-bit quantization
- **Remediation strategies:** Four approaches with implementation details and expected gains

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training LoRA Adapters](#training-lora-adapters)
- [Results](#results)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Requirements

- **Python:** 3.10+
- **OS:** Linux/WSL2 (<8GB VRAM), macOS (CPU-only), Windows (>12GB VRAM)
- **GPU:** 8GB VRAM minimum (RTX 3060/4070 or better)
- **RAM:** 16GB minimum, 32GB recommended

> **Windows users with <8GB VRAM:** Use WSL2 + Ubuntu. `bitsandbytes` (for 4-bit quantization) is not supported on native Windows.

---

## Installation

### Prerequisites

- **Python:** 3.10+
- **ffmpeg:** Required for audio processing  
    - Windows: [Download](https://ffmpeg.org/download.html)  
    - Ubuntu/Debian: `sudo apt install ffmpeg`  
    - macOS: `brew install ffmpeg`

### Option 1: WSL2 on Windows (<8GB VRAM, Recommended)

```bash
# Install WSL2 + Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Inside WSL2, install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install cuda-toolkit-12-1

# Clone repository
git clone https://github.com/[your-username]/yt-lora-summarization.git
cd yt-lora-summarization

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, peft, bitsandbytes; print(f'CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"
```

### Option 2: Native Linux

```bash
git clone https://github.com/[your-username]/yt-lora-summarization.git
cd yt-lora-summarization
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Option 3: Windows Native (>12GB VRAM or CPU-only)

> **Warning:** Quantization disabled on Windows. Requires 12GB+ VRAM or runs CPU-only (10–15× slower).

```bash
git clone https://github.com/[your-username]/yt-lora-summarization.git
cd yt-lora-summarization
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Option 4: macOS (CPU-only)

```bash
git clone https://github.com/[your-username]/yt-lora-summarization.git
cd yt-lora-summarization
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.2.2
pip install -r requirements.txt
```

### Option 5: Google Colab (Training Only)

```python
!git clone https://github.com/[your-username]/yt-lora-summarization.git
%cd yt-lora-summarization
!pip install peft trl bitsandbytes transformers accelerate datasets
```

---

## Quick Start

### Option 1: Streamlit GUI (Recommended for General Use)

Uses 4-bit quantized Mistral-7B-Instruct-v0.2 (base model, no LoRA adapters).

```bash
# 1) Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/macOS)
source .venv/bin/activate

# 2) Install PyTorch with CUDA (skip if CPU-only)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 3) Install dependencies
pip install -r requirements.txt

# 4) (Optional) Install ffmpeg
# Windows: Download from ffmpeg.org
# Ubuntu: sudo apt install ffmpeg
# macOS: brew install ffmpeg

# 5) Launch GUI
streamlit run scripts/app.py
```

Open browser at [http://localhost:8501](http://localhost:8501), paste YouTube URL, click "Summarize".

**GUI Features:**
- Automatic video download
- Dual-backend Whisper transcription (PyTorch/Faster-Whisper)
- 4-bit quantized inference (6–7GB VRAM)
- Export to TXT/Markdown/DOCX

### Option 2: Command-Line (Research/LoRA Experiments)

**A) Base Model (Recommended for Videos):**
```bash
python scripts/process_youtube.py \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --output results/summaries/
```

**B) With LoRA Adapter (Research Only – Degrades Performance):**
```bash
python scripts/process_youtube.py \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --domain medical \
    --adapter-path models/adapters/medical/ \
    --output results/summaries/
```

> ⚠️ **Research Finding:** LoRA adapters reduce quality by 14–50%. Use only for replication experiments.

---

## Project Structure

```
yt-lora-summarization/
├── src/
│   └── yt_sum/
│       ├── models/
│       │   ├── summarizer.py              # Base model inference
│       │   └── summarizer_lora.py         # LoRA adapter integration
│       ├── training/
│       │   ├── train_lora_colab.py        # Google Colab training script
│       │   └── evaluate_lora.py           # Evaluation framework
│       └── utils/
│           ├── chunker.py                 # Text chunking utilities
│           └── logging.py                 # Logging configuration
├── scripts/
│   ├── app.py                             # Streamlit GUI (base model only)
│   ├── process_youtube.py                 # CLI: Video download + transcription
│   ├── error_analysis.py                  # Failure mode analysis
│   ├── generate_paper_figures.py          # Create publication figures
│   └── verify_adapters.py                 # Adapter validation tests
├── data/
│   ├── medical/                           # Medical papers (PubMed)
│   ├── engineering/                       # Engineering papers (arXiv)
│   ├── scientific/                        # Scientific papers (arXiv)
│   └── youtube_test/                      # Test video transcripts
├── models/
│   └── adapters/                          # LoRA adapters
├── results/
│   ├── evaluation/                        # Evaluation outputs
│   └── error_analysis/                    # Error categorization
├── paper/
│   ├── main.tex                           # LaTeX manuscript
│   └── figures/                           # Generated figures
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # This file
```

---

## Training LoRA Adapters

Training requires **16GB VRAM** (Google Colab T4 recommended). Local training on 8GB GPUs is not supported.

```python
# In Colab notebook
!pip install peft trl bitsandbytes transformers accelerate datasets

# Train medical adapter (60–75 minutes)
!python train_lora_colab.py \
    --domain medical \
    --data-path data/medical/medical.jsonl \
    --output-dir models/adapters/medical/ \
    --max-samples 10000
```

**Training Configuration:**
- Base Model: Mistral-7B-Instruct-v0.2
- LoRA Rank: r=16, α=32
- Quantization: 4-bit NF4
- Batch Size: 16 (effective, 4×4 accumulation)
- Epochs: 3
- Learning Rate: 2×10⁻⁴ with cosine decay

---

## Results

### Quantitative Performance

| Domain      | Metric      | Base | LoRA | Change | p-value | Cohen's d |
|-------------|-------------|------|------|--------|---------|-----------|
| Medical     | ROUGE-1     | 0.341|0.283 | -17%   | <0.001  | 0.65      |
|             | ROUGE-2     | 0.118|0.066 | -44%   | <0.001  | 1.37      |
|             | BERTScore   | 0.881|0.851 | -3.4%  | <0.001  | 1.50      |
| Engineering | ROUGE-1     | 0.347|0.272 | -22%   | <0.001  | 0.81      |
|             | ROUGE-2     | 0.120|0.060 | -50%   | <0.001  | 1.58      |
|             | BERTScore   | 0.879|0.845 | -3.9%  | <0.001  | 1.70      |
| Scientific  | ROUGE-1     | 0.336|0.288 | -14%   | <0.001  | 0.50      |
|             | ROUGE-2     | 0.115|0.073 | -37%   | <0.001  | 1.11      |
|             | BERTScore   | 0.877|0.845 | -3.7%  | <0.001  | 1.60      |

### Adapter Validation (Papers)

| Domain      | Base ROUGE-1 | LoRA ROUGE-1 | Improvement |
|-------------|--------------|--------------|-------------|
| Medical     | 0.341        | 0.369        | +8.2%*      |
| Engineering | 0.312        | 0.333        | +6.7%       |
| Scientific  | 0.328        | 0.351        | +7.0%       |

\*p<0.01, **p<0.001

**Key Finding:** Adapters improve paper summarization but degrade video summarization, confirming modality mismatch as the root cause.

---

## Hardware Requirements

| Configuration      | VRAM    | Throughput                | Notes                  |
|--------------------|---------|---------------------------|------------------------|
| 4-bit Quantized    | 6–8GB   | 15-min video → 4 min      | RTX 3060/4070          |
| 8-bit Quantized    | 10-12GB | 15-min video → 2.5-3 min  | RTX 3080/4080          |
| No Quantisation    | 16 GB+  | 15-min Video → Probably 1-1.5 min (Untested) | RTX 4080 Super/ 4090|
| CPU-only           | N/A     | 15-min video → 40 min     | 16-core+ recommended   |

---

## Troubleshooting

- **Issue:** "CUDA out of memory"  
    **Solution:** Reduce sequence length or enable CPU offloading  
    ```python
    # In summarizer.py
    MAX_SEQ_LENGTH = 768  # default: 1024
    ```

- **Issue:** "bitsandbytes not available"  
    **Cause:** Running on Windows native or macOS.  
    **Solution:** Use WSL2 (Windows) or disable quantization (macOS)

- **Issue:** Whisper hallucinations  
    **Solution:** Already configured with anti-hallucination parameters. If issues persist, switch to Faster-Whisper backend.

---

## FAQ

**Q:** Why doesn't the GUI use LoRA adapters?  
**A:** Research showed LoRA adapters degrade video quality by 14–50%. The GUI uses the base quantized model for optimal performance.

**Q:** Can I use this for non-educational videos?  
**A:** Yes, but quality may vary. The model was optimized for technical/educational content.

**Q:** Do I need to download LoRA adapters to use the GUI?  
**A:** No. The GUI only requires the base Mistral-7B model, which downloads automatically on first run (~14GB).

**Q:** Can I train my own LoRA adapters?  
**A:** Yes, see [Training LoRA Adapters](#training-lora-adapters). Requires 16GB VRAM (Google Colab recommended).

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Advisors:** Dr. Neha Yadav (ABES Engineering College)
- **Compute:** Google Colab (free T4 GPU access)
- **APIs:** OpenAI GPT-4 (reference summary generation)
- **Open Source:** Hugging Face (transformers, PEFT), OpenAI (Whisper), Mistral AI (base model)

---

## Contact

**Tushar Jaju** – tusharbrisingr9802@gmail.com  
**Project Link:** [https://github.com/Tushar-9802/YouTube-Transcript-Summarizer](https://github.com/Tushar-9802/YouTube-Transcript-Summarizer)  
---

_Last Updated: October 2025_  
_Status: Research Complete | GUI Functional_  


