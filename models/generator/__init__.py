import numpy as np
import librosa

class AudioGenerator:
    def __init__(self, sr: int = 22050) -> None:
        self.sr = sr
        
    def __call__(self, spec_complex: np.ndarray) -> tuple[np.ndarray, int]:
        real_part = np.nan_to_num(spec_complex[0], nan=0.0, posinf=200.0, neginf=0.0)
        imag_part = np.nan_to_num(spec_complex[1], nan=0.0, posinf=200.0, neginf=0.0)
        complex_spec = real_part + 1j * imag_part
        audio_signal = librosa.istft(complex_spec)
        
        return audio_signal, self.sr