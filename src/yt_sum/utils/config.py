# src/yt_sum/utils/config.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import os
import yaml

_DEFAULT_YAML = {
    "paths": {
        "audio_dir": "data/audio",
        "transcript_dir": "data/transcripts",
        "outputs_dir": "outputs",
        "adapters_dir": "models/adapters",
        # Optional, not required by the app but useful to have:
        "logs_dir": "logs",
    },
    "whisper": {
        "model_size": "base"
    },
    "logging": {
        "level": "INFO"
    }
}

def _expand(p: str) -> Path:
    """Expand ~ and environment variables, then return a Path."""
    return Path(os.path.expandvars(p)).expanduser()

class Config:
    """
    Small YAML-backed config with path normalization.

    Attributes (paths):
      audio_dir, transcript_dir, outputs_dir, adapters_dir, logs_dir (optional)

    Other attributes:
      whisper_model_size, log_level

    Methods:
      ensure_dirs()  -> create the directories if they don't exist
      to_dict()      -> dict view of the config
      save()         -> write current config back to YAML
      update_from_dict(d) -> update fields and re-normalize paths
    """

    def __init__(self, path: str = "configs/default.yaml", create_if_missing: bool = True):
        self._path = _expand(path)
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)

        if not self._path.exists():
            if create_if_missing:
                # write a default file so first-run works out of the box
                with self._path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(_DEFAULT_YAML, f, sort_keys=False)
                self._cfg = _DEFAULT_YAML.copy()
            else:
                raise FileNotFoundError(f"Config not found: {self._path.resolve()}")
        else:
            with self._path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            # merge defaults (so missing keys don't crash)
            self._cfg = _merge_dicts(_DEFAULT_YAML, loaded)

        # normalize to Path objects
        p = self._cfg.get("paths", {})
        self.audio_dir = _expand(p.get("audio_dir", _DEFAULT_YAML["paths"]["audio_dir"]))
        self.transcript_dir = _expand(p.get("transcript_dir", _DEFAULT_YAML["paths"]["transcript_dir"]))
        self.outputs_dir = _expand(p.get("outputs_dir", _DEFAULT_YAML["paths"]["outputs_dir"]))
        self.adapters_dir = _expand(p.get("adapters_dir", _DEFAULT_YAML["paths"]["adapters_dir"]))
        # optional, not required elsewhere
        self.logs_dir = _expand(p.get("logs_dir", _DEFAULT_YAML["paths"]["logs_dir"]))

        self.whisper_model_size = self._cfg.get("whisper", {}).get("model_size", _DEFAULT_YAML["whisper"]["model_size"])
        self.log_level = self._cfg.get("logging", {}).get("level", _DEFAULT_YAML["logging"]["level"])

    # ---------- public API ----------
    def ensure_dirs(self) -> None:
        for d in [self.audio_dir, self.transcript_dir, self.outputs_dir, self.adapters_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": {
                "audio_dir": str(self.audio_dir),
                "transcript_dir": str(self.transcript_dir),
                "outputs_dir": str(self.outputs_dir),
                "adapters_dir": str(self.adapters_dir),
                "logs_dir": str(self.logs_dir),
            },
            "whisper": {
                "model_size": self.whisper_model_size
            },
            "logging": {
                "level": self.log_level
            }
        }

    def save(self) -> None:
        """Write current config back to YAML."""
        with self._path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def update_from_dict(self, d: Dict[str, Any]) -> None:
        """Update fields from a dict, re-normalize paths, and persist in-memory."""
        if not d:
            return
        # paths
        p = d.get("paths", {})
        if "audio_dir" in p:        self.audio_dir = _expand(p["audio_dir"])
        if "transcript_dir" in p:   self.transcript_dir = _expand(p["transcript_dir"])
        if "outputs_dir" in p:      self.outputs_dir = _expand(p["outputs_dir"])
        if "adapters_dir" in p:     self.adapters_dir = _expand(p["adapters_dir"])
        if "logs_dir" in p:         self.logs_dir = _expand(p["logs_dir"])
        # whisper/logging
        w = d.get("whisper", {})
        if "model_size" in w:       self.whisper_model_size = str(w["model_size"])
        lg = d.get("logging", {})
        if "level" in lg:           self.log_level = str(lg["level"])

        # also merge to backing dict so save() reflects the change
        self._cfg = _merge_dicts(self._cfg, self.to_dict())

# ---------- small util ----------
def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dicts without modifying inputs (override wins)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out
