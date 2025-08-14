import whisper
import torch
class WhisperTranscriber:
    def __init__(self, model_size="small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.model = whisper.load_model(model_size, device=self.device)
