
# YouTube-Transcript-Summarizer

A research-grade, domain-adaptive pipeline for summarizing YouTube videos.

## Currently Under Development

=======
### Quickstart

```bash
# 1) Create venv
python -m venv .venv && . .venv/Scripts/activate  # Windows
# source .venv/bin/activate                        # Linux/Mac

# 2) Install PyTorch (GPU build if you have CUDA; else skip this line)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.8.0

# 3) Install deps
pip install -r requirements.txt

# 4) (optional) Install ffmpeg from your OS package manager

# 5) Run the app
streamlit run scripts/app.py
