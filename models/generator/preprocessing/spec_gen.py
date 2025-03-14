import io
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from gtts import gTTS

from utils.common import create_path, get_logger
from utils.config import load_config

logger = get_logger("logs/spectrogram_generation.log")


def text_to_spectrogram(word: str, max_length: int = -1) -> np.ndarray:
    tts = gTTS(text=word, lang="en")
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    y, _ = sf.read(audio_buffer)

    # Generate spectrogram
    D = np.abs(librosa.stft(y))

    if max_length < 0:
        return D

    # Pad or trim to fixed time length
    if D.shape[1] < max_length:
        pad_width = max_length - D.shape[1]
        D = np.pad(D, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
    else:
        logger.info("Truncating")
        D = D[:, :max_length]

    return D


def process_words(
    n_words: int, train_data: pd.DataFrame, max_length: int
) -> pd.DataFrame:
    rows = []
    freq = dict(Counter(train_data.Gloss.to_list()))
    words = sorted(freq, key=lambda x: -freq[x])

    for count in range(min(n_words, len(words))):
        word = words[count].title()
        spectrogram = text_to_spectrogram(word, max_length)
        rows.append([words[count]] + spectrogram.flatten().tolist())

    data = pd.DataFrame(
        rows,
        columns=["word"] + [f"feature_{i}" for i in range(1, 1025 * max_length + 1)],
    )
    return data


def main(
    train_data_path: str,
    specs_path: Path,
    classlist_path: Path,
    n_words: int,
    max_length: int,
) -> None:
    train_data = pd.read_csv(train_data_path)
    data = process_words(n_words, train_data, max_length)
    data.to_csv(specs_path, index=False)

    classes = data["word"].tolist()
    with open(classlist_path, "w") as f:
        f.write("\n".join(classes))


if __name__ == "__main__":
    config = load_config("Generate spectrograms for words")
    specs_path = create_path(config.data.processed.specs)
    classlist_path = create_path(config.data.processed.classlist)
    main(
        config.data.raw.csvs.train,
        specs_path,
        classlist_path,
        config.n_words,
        config.generator.max_length,
    )
