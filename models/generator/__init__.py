import numpy as np
import torch
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN

from utils.common import create_path
from utils.config import load_config

config = load_config("Generate Audio")
checkpoint_path = create_path(config.generator.checkpoints)

class AudioGenerator:
<<<<<<< HEAD
    def __init__(self, sr: int = 22050) -> None:
        self.sr = sr
        
    def __call__(self, spec_complex: np.ndarray) -> tuple[np.ndarray, int]:
        real_part = np.nan_to_num(spec_complex[0], nan=0.0, posinf=200.0, neginf=0.0)
        imag_part = np.nan_to_num(spec_complex[1], nan=0.0, posinf=200.0, neginf=0.0)
        complex_spec = real_part + 1j * imag_part
        audio_signal = librosa.istft(complex_spec)
        
        return audio_signal, self.sr
=======
    def __init__(self) -> None:
        self.sr: int = 22050
        self.model = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            savedir=checkpoint_path / "tts-hifigan-ljspeech",
        )

    def __call__(self, spec: torch.Tensor) -> tuple[np.ndarray, int]:
        # spec_db = torchaudio.functional.amplitude_to_DB(
        #     spec, multiplier=20.0, amin=1e-5, db_multiplier=0
        # )
        audio_signal = self.model.decode_batch(spec.squeeze(1))
        audio_signal = audio_signal.cpu().numpy()[0][0]
        return audio_signal, self.sr
>>>>>>> d2f1c6add2976bfdb060ea10cec7bfb2cff4f56e
