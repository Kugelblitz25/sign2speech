import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from speechbrain.inference.TTS import Tacotron2
from models.generator import AudioGenerator
from models.generator.preprocessing.spec_gen import generate_spec
from utils.common import create_path
from utils.config import load_config
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def generate_audio(word: str, output_path: Path) -> None:
    config = load_config("Generate Audio")
    checkpoint_path = create_path(config.generator.checkpoints)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tacotron2 = Tacotron2.from_hparams(
        source="speechbrain/tts-tacotron2-ljspeech",
        savedir=checkpoint_path / "tts-tacotron2",
        run_opts={"device": device},
    )
    spec = generate_spec(word, tacotron2)
    audio_generator = AudioGenerator()
    audio_signal, sr = audio_generator(torch.tensor(spec).unsqueeze(0))
    sf.write(output_path, audio_signal, sr)
    print(f"Audio saved at {output_path}")

def generate_audio_dataset(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_pairs = []

    for idx, row in df.iterrows():
        incorrect_path = output_dir / f"incorrect_{idx}.wav"
        correct_path = output_dir / f"correct_{idx}.wav"
        _ = generate_audio(row['incorrect'], incorrect_path)
        _ = generate_audio(row['correct'], correct_path)  
        audio_pairs.append((incorrect_path, correct_path))   
    return audio_pairs


if __name__ == "__main__":
    word = "dog"
    output_audio_path = create_path("experiments/audio/output_audio.wav")

    spec = generate_audio(word, output_audio_path)
    print(f"Spectrogram shape: {spec.shape}")
