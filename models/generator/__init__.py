import numpy as np
import librosa


class AudioGenerator:
    def __init__(self, sr: int = 22050) -> None:
        self.sr = sr

    def __call__(self, spec: np.ndarray) -> tuple[np.ndarray, int]:
        """Convert a spectrogram to audio using Griffin-Lim."""
        spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
        audio_signal = librosa.griffinlim(spec)
        return audio_signal, self.sr
