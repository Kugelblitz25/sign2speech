import numpy as np
import torch
from speechbrain.inference.vocoders import HIFIGAN

from utils.common import create_path
from utils.config import load_config

config = load_config("Generate Audio")
checkpoint_path = create_path(config.generator.checkpoints)


class AudioGenerator:
    def __init__(self) -> None:
        self.sr: int = 22050
        self.model = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            savedir=checkpoint_path / "tts-hifigan-ljspeech",
        )

    def __call__(self, spec: torch.tensor) -> tuple[np.ndarray, int]:
        audio_signal = self.model.decode_batch(spec.squeeze(1))
        audio_signal = audio_signal.to("cpu").cpu().numpy()[0][0]
        return audio_signal, self.sr
