from __future__ import annotations
import torch

try:
    from faster_whisper import WhisperModel
    _USE_FASTER = True
except ImportError:
    import whisper  # type: ignore
    _USE_FASTER = False


DOMAIN_HINTS = {
    "general": "",
    "medical": (
        "Medical lecture. Use terms like randomized controlled trial, cohort, "
        "meta-analysis, confidence interval, p-value, adverse events, dosage, "
        "biomarker, sensitivity, specificity, comorbidity."
    ),
    "engineering": (
        "Engineering talk. Use terms like tolerance, load, stress-strain, "
        "signal-to-noise ratio, bandwidth, throughput, latency, PID, PWM, "
        "finite element analysis, CAD, BOM."
    ),
    "scientific": (
        "Scientific seminar. Use terms like hypothesis, methodology, "
        "experimental design, replication, baseline, ablation, statistical significance, "
        "confidence interval, effect size, dataset, benchmark."
    ),
}


class Transcriber:
    """
    GPU-first Whisper/Faster-Whisper transcriber with domain-aware initial prompts,
    beam search, and temperature fallback for high accuracy on technical content.
    """

    def __init__(
        self,
        model_size: str | None = None,
        language: str | None = None,
        beam_size: int = 5,
        prefer_accuracy: bool = True,
    ):
        self.language = language
        self.beam_size = beam_size

        if model_size is None:
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 2**30
                if vram >= 10:
                    model_size = "large-v2"
                elif vram >= 5:
                    model_size = "medium"
                elif vram >= 2:
                    model_size = "small"
                else:
                    model_size = "base"
            else:
                model_size = "base" if prefer_accuracy else "tiny"
        self.model_size = model_size

        if _USE_FASTER:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self.backend = "faster-whisper"
        else:
            self.model = whisper.load_model(model_size, device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
            self.backend = "openai-whisper"

    def transcribe(self, audio_path: str, domain: str = "general") -> list[dict]:
        """
        Transcribe audio → list of {start, end, text}. Applies a domain-aware initial prompt.
        """
        init = DOMAIN_HINTS.get(domain.lower(), DOMAIN_HINTS["general"]).strip() or None

        if _USE_FASTER:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=self.beam_size,
                temperature=[0.0, 0.2, 0.4],
                language=self.language,
                initial_prompt=init,
            )
            out = []
            for s in segments:
                t = (s.text or "").strip()
                if t:
                    out.append({"start": float(s.start), "end": float(s.end), "text": t})
            if getattr(info, "language", None):
                self.language = info.language  # type: ignore
            return out

        # OpenAI whisper path
        result = self.model.transcribe(  # type: ignore
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            temperature=[0.0, 0.2, 0.4],
            initial_prompt=init,
            condition_on_previous_text=True,
        )
        segs = []
        for s in result.get("segments", []):
            t = (s.get("text") or "").strip()
            if t:
                segs.append({"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": t})
        self.language = result.get("language", self.language)
        return segs
