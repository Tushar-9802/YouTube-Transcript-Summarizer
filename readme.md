# Domain-Aware YouTube Video Summarization with LoRA

**Cross-Modal Transfer Learning for Educational Content**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository implements and analyzes **"Cross-Modal Transfer Learning in Domain-Adaptive Video Summarization"**. It investigates whether domain-specific LoRA adapters trained on academic papers improve YouTube video summarization.

**Key Finding:** LoRA adapters trained on 1,800 academic documents per domain scored **below the base model** on 48 YouTube videos (ROUGE-1 −14 to −22%, ROUGE-2 −37 to −50%, p<0.01). Domain similarity alone was not sufficient for transfer. The cause of the degradation is an open question.

> **Note:** The Streamlit GUI (`scripts/app.py`) uses the base **4-bit quantized Mistral-7B-Instruct-v0.2** model for general-purpose video summarization. LoRA adapters are **not** loaded in the GUI, because they lowered video summarization scores in this study. For LoRA experiments, use command-line scripts.

### Contributions

- **Negative result:** Adapters scored below the base model on video summarization
- **Failure analysis:** Repetitive output (9/48 vs 1/48), academic framing (7/48 vs 2/48), and a null result on passive voice
- **Consumer GPU pipeline:** 7B parameter inference in 6.2GB VRAM via 4-bit quantization
- **Candidate remedies:** Four strategies, none tested here

## Status: Presented at IMPACT 2025 (December 2025). Springer proceedings in production, DOI will be added when assigned.

## Corrections

The results in this README were corrected in November 2026 after an audit against the recorded experiments. Earlier versions reported values and analyses that did not match the saved outputs.

- Table values are now recomputed from `evaluation/results/evaluation/detailed_results.csv` by `evaluation/scripts/recompute_chapter_tables.py`, with results in `evaluation/results/recomputed.json`.
- The paper-abstract validation table was removed. No recorded run supports it.
- The passive-voice, nominalization, and perplexity results were re-run on the saved summaries. Passive voice did not change between models.
- Training used 1,800 documents per domain, not 25,000 papers. References came from `gpt-4o-mini`. Test transcripts came from Whisper small.
- BERTScore in `detailed_results.csv` is a group mean repeated on each row, kept as recorded. Per-video values are in the recompute script, and `evaluate_robust.py` now stores them.
- Four test transcripts are failed ASR outputs and are kept as recorded: `j3mhkYbznBk`, `LBUFrBHAjcs`, `EjaGktVQdNg`, `iUjVnV5FsYo`.

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
git clone https://github.com/Tushar-9802/YouTube-Transcript-Summarizer.git
cd YouTube-Transcript-Summarizer

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
git clone https://github.com/Tushar-9802/YouTube-Transcript-Summarizer.git
cd YouTube-Transcript-Summarizer
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Option 3: Windows Native (>12GB VRAM or CPU-only)

> **Warning:** Quantization disabled on Windows. Requires 12GB+ VRAM or runs CPU-only (10–15× slower).

```bash
git clone https://github.com/Tushar-9802/YouTube-Transcript-Summarizer.git
cd YouTube-Transcript-Summarizer
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Option 4: macOS (CPU-only)

```bash
git clone https://github.com/Tushar-9802/YouTube-Transcript-Summarizer.git
cd YouTube-Transcript-Summarizer
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.2.2
pip install -r requirements.txt
```

### Option 5: Google Colab (Training Only)

```python
!git clone https://github.com/Tushar-9802/YouTube-Transcript-Summarizer.git
%cd YouTube-Transcript-Summarizer
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

### Option 2: Research Scripts

Evaluation scripts run from `evaluation/` with the repository root on the Python path:

```bash
cd evaluation
PYTHONPATH=.. python scripts/evaluate_robust.py
```

Recompute the reported tables from saved outputs, from the repository root:

```bash
python evaluation/scripts/recompute_chapter_tables.py . evaluation/results/recomputed.json
python scripts/generate_figures.py
```

The recompute script needs `spacy` with `en_core_web_sm`, which is not in `requirements.txt`.

> **Finding:** LoRA adapters lowered video summarization scores in this study. Use them only for replication experiments.

---

## Project Structure

```
YouTube-Transcript-Summarizer/
├── src/yt_sum/
│   ├── models/                            # Summarizer, LoRA summarizer, transcriber
│   ├── training/                          # Colab training script and dataset prep
│   ├── utils/                             # ASR, chunking, keywords, config, logging
│   └── pipeline.py                        # End-to-end pipeline
├── scripts/
│   ├── app.py                             # Streamlit GUI (base model only)
│   └── generate_figures.py                # Figures from recomputed results
├── evaluation/
│   ├── data/youtube_test/                 # 48 test videos with transcripts and references
│   ├── models/adapters/                   # LoRA adapters and training logs
│   ├── results/                           # Recorded outputs and recomputed tables
│   └── scripts/                           # Evaluation, analysis, and recompute scripts
├── configs/default.yaml
├── requirements.txt
├── LICENSE
└── readme.md
```

---

## Training LoRA Adapters

Training requires **16GB VRAM** (Google Colab T4). Local training on 8GB GPUs is not supported.

Run `src/yt_sum/training/training_lora_colab.py` in Colab. Set `DOMAIN` at the top of the script and upload `{DOMAIN}.jsonl`. Each adapter took about 57–59 minutes.

**Training Configuration (from saved adapters and logs):**
- Base Model: Mistral-7B-Instruct-v0.2
- LoRA: r=16, α=32, dropout 0.05, target modules q_proj and v_proj (6,815,744 trainable parameters)
- Quantization: 4-bit NF4 with double quantization, float16 compute
- Data: 2,000 sampled documents per domain, 1,800 for training and 200 held out
- Batch Size: 8 (effective, 2×4 accumulation), sequence length 512
- Epochs: 1 (225 steps)
- Learning Rate: 2×10⁻⁴ with linear decay, paged 8-bit AdamW

The training JSONL files are not in this repository, so exact retraining is not possible from the repository alone. Medical data came from MS^2, and engineering and scientific data came from arXiv.

---

## Results

### Quantitative Performance

Paired t-tests on 48 videos (16 medical, 15 engineering, 17 scientific). References are `gpt-4o-mini` summaries.

| Domain | Metric | Base | LoRA | Change | p-value | Cohen's d |
|---|---|---|---|---|---|---|
| Medical     | ROUGE-1   | 0.469 | 0.390 | -17% | <0.001 | 1.06 |
|             | ROUGE-2   | 0.167 | 0.094 | -44% | <0.001 | 1.10 |
|             | ROUGE-L   | 0.270 | 0.200 | -26% | <0.001 | 1.14 |
|             | BERTScore | 0.885 | 0.854 | -3.4% | <0.001 | 2.02 |
| Engineering | ROUGE-1   | 0.440 | 0.345 | -22% | <0.001 | 0.78 |
|             | ROUGE-2   | 0.181 | 0.090 | -50% | <0.001 | 1.29 |
|             | ROUGE-L   | 0.264 | 0.204 | -23% | 0.006 | 0.75 |
|             | BERTScore | 0.874 | 0.840 | -3.9% | <0.001 | 0.72 |
| Scientific  | ROUGE-1   | 0.449 | 0.385 | -14% | 0.005 | 0.47 |
|             | ROUGE-2   | 0.183 | 0.116 | -37% | <0.001 | 0.91 |
|             | ROUGE-L   | 0.268 | 0.206 | -23% | <0.001 | 0.79 |
|             | BERTScore | 0.881 | 0.848 | -3.7% | <0.001 | 0.77 |

ROUGE-1 decreased on 42 of 48 videos. Excluding failed transcripts and repetitive outputs (n=37) gave declines of 13.8% (ROUGE-1), 38.0% (ROUGE-2), and 22.8% (ROUGE-L), all p<0.001.

### Failure Analysis

| Metric | Base | LoRA | p-value |
|---|---|---|---|
| Passive voice (% of sentences) | 25.3 | 24.4 | 0.84 |
| Nominalization (% of nouns) | 18.9 | 14.6 | 0.006 |
| Perplexity (GPT-2) | 22.0 | 14.2 | <0.001 |
| Unigram overlap (%) | 62.8 | 51.4 | <0.001 |
| Bigram overlap (%) | 23.5 | 12.6 | <0.001 |
| Mean length (words, reference 179) | 112 | 194 | |
| Repetitive outputs | 1/48 | 9/48 | |
| Academic framing | 2/48 | 7/48 | |
| Citation artifacts | 0/48 | 1/48 | |

Repetition lowers perplexity, so the lower LoRA value does not indicate better fluency. No quantitative source-domain validation was completed, so training-related causes cannot be fully excluded.

---

## Hardware Requirements

| Configuration      | VRAM    | Throughput                | Notes                  |
|--------------------|---------|---------------------------|------------------------|
| 4-bit Quantized    | 6–8GB   | 15-min video → 4 min      | RTX 3060/4070          |
| 8-bit Quantized    | 10-12GB | 15-min video → 2.5-3 min  | RTX 3080/4080          |
| CPU-only           | N/A     | 15-min video → 40 min     | 16-core+ recommended   |

---

## Troubleshooting

- **Issue:** "CUDA out of memory"  
    **Solution:** Close other GPU applications, use a smaller Whisper model, or use CPU offloading.

- **Issue:** "bitsandbytes not available"  
    **Cause:** Running on Windows native or macOS.  
    **Solution:** Use WSL2 (Windows) or disable quantization (macOS)

- **Issue:** Whisper hallucinations  
    **Solution:** Already configured with anti-hallucination parameters. If issues persist, switch to Faster-Whisper backend.

---

## FAQ

**Q:** Why doesn't the GUI use LoRA adapters?  
**A:** LoRA adapters lowered video summarization scores in this study, so the GUI uses the base quantized model.

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
- **APIs:** OpenAI `gpt-4o-mini` (reference summary generation)
- **Open Source:** Hugging Face (transformers, PEFT), OpenAI (Whisper), Mistral AI (base model)

---

## Contact

**Tushar Jaju** – tusharbrisingr9802@gmail.com  
**Project Link:** [https://github.com/Tushar-9802/YouTube-Transcript-Summarizer](https://github.com/Tushar-9802/YouTube-Transcript-Summarizer)  
---

_Last Updated: November 2026_  
_Status: Research Complete | GUI Functional_  


