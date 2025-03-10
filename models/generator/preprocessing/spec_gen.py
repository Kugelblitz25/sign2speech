from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from speechbrain.inference.TTS import Tacotron2

from utils.common import create_path, get_logger
from utils.config import load_config

logger = get_logger("logs/spectrogram_generation.log")


def generate_spec(word: str, model: Tacotron2, max_length: int = -1) -> np.ndarray:
    mel_output, *_ = model.encode_text(word)

    if max_length <= 0:
        return mel_output.cpu().detach().numpy()

    if mel_output.shape[2] > max_length:
        return -15 * np.ones((80, max_length))

    padded_mel_output = torch.nn.functional.pad(
        mel_output,
        [
            0,
            max(0, max_length - mel_output.shape[2]),
            0,
            max(0, 80 - mel_output.shape[1]),
        ],
        mode="constant",
        value=-15,
    )
    return padded_mel_output.cpu().detach().numpy()


def process_words(
    n_words: int, train_data: pd.DataFrame, model: Tacotron2, max_length: int
) -> pd.DataFrame:
    rows = []

    freq = dict(Counter(train_data.Gloss.to_list()))
    words = sorted(freq, key=lambda x: -freq[x])

    i = 0
    count = 0
    while count < n_words:
        word = words[count][:-1].title()
        spectrogram = generate_spec(word, model, max_length)
        rows.append([words[count]] + spectrogram.flatten().tolist())
        if np.allclose(spectrogram, -15):
            i += 1
        count += 1
    logger.info(f"Bad Audio: {i}/{count}")
    data = pd.DataFrame(
        rows, columns=["word"] + [f"feature_{i}" for i in range(1, 80 * 88 + 1)]
    )

    return data


def main(
    train_data_path: str,
    specs_path: Path,
    classlist_path: Path,
    n_words: int,
    model: Tacotron2 | None,
    max_length: int,
) -> None:
    if model is None:
        raise ValueError("Unable to load model.")

    train_data = pd.read_csv(train_data_path)
    data = process_words(n_words, train_data, model, max_length)
    data.to_csv(specs_path, index=False)

    classes = data["word"].tolist()
    with open(classlist_path, "w") as f:
        f.write("\n".join(classes))


if __name__ == "__main__":
    config = load_config("Generate spectrograms for words")

    checkpoint_path = create_path(config.generator.checkpoints)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    tacotron2 = Tacotron2.from_hparams(
        source="speechbrain/tts-tacotron2-ljspeech",
        savedir=checkpoint_path / "tts-tacotron2",
        run_opts={"device": device},
    )

    specs_path = create_path(config.data.processed.specs)
    classlist_path = create_path(config.data.processed.classlist)

    main(
        config.data.raw.csvs.train,
        specs_path,
        classlist_path,
        config.n_words,
        tacotron2,
        config.generator.max_length,
    )
