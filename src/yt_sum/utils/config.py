# src/yt_sum/utils/config.py
from pathlib import Path
import yaml

class Config:
    def __init__(self, path: str = "configs/default.yaml"):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Config not found: {self._path.resolve()}")
        with self._path.open("r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

        # normalize to Path objects
        p = self._cfg.get("paths", {})
        self.audio_dir = Path(p.get("audio_dir", "data/audio"))
        self.transcript_dir = Path(p.get("transcript_dir", "data/transcripts"))
        self.outputs_dir = Path(p.get("outputs_dir", "outputs"))
        self.adapters_dir = Path(p.get("adapters_dir", "models/adapters"))

        self.whisper_model_size = self._cfg.get("whisper", {}).get("model_size", "base")
        self.log_level = self._cfg.get("logging", {}).get("level", "INFO")

    def ensure_dirs(self):
        for d in [self.audio_dir, self.transcript_dir, self.outputs_dir, self.adapters_dir]:
            d.mkdir(parents=True, exist_ok=True)
