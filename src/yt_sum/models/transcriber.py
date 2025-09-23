# transcriber.py
from __future__ import annotations
from typing import Optional, Dict, Any
import torch

try:
    from faster_whisper import WhisperModel
    _HAS_FASTER = True
except Exception:
    _HAS_FASTER = False

try:
    import whisper as openai_whisper
    _HAS_OPENAI_WHISPER = True
except Exception:
    _HAS_OPENAI_WHISPER = False


_DOMAIN_PROMPTS = {
    "general": "",
    "medical": "Medical lecture transcription. Use correct clinical and pharmacological terminology.",
    "engineering": "Engineering lecture transcription. Use correct technical terms and units.",
    "scientific": "Scientific talk transcription. Use correct research methods terminology and metrics.",
}


class Transcriber:
    """
    Whisper/Faster-Whisper with domain-aware initial prompts and optional translation to English.
    """

    def __init__(self, model_size: Optional[str] = None, prefer_accuracy: bool = True):
        if not model_size:
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 2**30
                if vram >= 10:
                    model_size = "large-v2"
                elif vram >= 5:
                    model_size = "medium"
                else:
                    model_size = "small" if prefer_accuracy else "base"
            else:
                model_size = "base" if prefer_accuracy else "tiny"

        self.model_size = model_size
        self.backend = "faster-whisper" if (_HAS_FASTER and torch.cuda.is_available()) else "openai-whisper"
        self.language: Optional[str] = None

        if self.backend == "faster-whisper":
            self.model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = openai_whisper.load_model(model_size, device=device)

    def transcribe(self, audio_path: str, *, domain: str = "general", translate_to_english: bool = False) -> list[Dict[str, Any]]:
        init_prompt = _DOMAIN_PROMPTS.get(domain.lower(), "")
        segments = []
        detected = None

        if self.backend == "faster-whisper":
            kwargs = {
                "beam_size": 5,
                "temperature": [0.0, 0.2, 0.4],
                "task": "translate" if translate_to_english else "transcribe",
                "initial_prompt": init_prompt or None
            }
            s_iter, info = self.model.transcribe(audio_path, **kwargs)
            for s in s_iter:
                text = (s.text or "").strip()
                if text:
                    segments.append({"start": float(s.start), "end": float(s.end), "text": text})
            detected = getattr(info, "language", None)
        else:
            kwargs = {
                "beam_size": 5,
                "temperature": [0.0, 0.2, 0.4],
                "task": "translate" if translate_to_english else "transcribe",
                "initial_prompt": init_prompt or None,
            }
            res = self.model.transcribe(audio_path, **kwargs)
            for s in res.get("segments", []):
                text = (s.get("text") or "").strip()
                if text:
                    segments.append({
                        "start": float(s.get("start", 0.0)),
                        "end": float(s.get("end", 0.0)),
                        "text": text
                    })
            detected = res.get("language")

        self.language = detected or self.language
        return segments
