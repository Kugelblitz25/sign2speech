from collections import Counter

import numpy as np
import pandas as pd
import torch
from speechbrain.inference.TTS import Tacotron2

from utils.config import Config
from utils.model import create_path


def generate_spec(word: str, model):
    mel_output, *_ = model.encode_text(word)
    if mel_output.shape[2] > 88:
        return torch.zeros(80, 88)
    padded_mel_output = torch.nn.functional.pad(
        mel_output,
        [
            0,
            max(0, 88 - mel_output.shape[2]),
            0,
            max(0, 80 - mel_output.shape[1]),
        ],
        mode="constant",
        value=-15,
    )
    return padded_mel_output.cpu().detach().numpy()


def process_words(n_words: int, train_data: pd.DataFrame, model):
    rows = []

    freq = dict(Counter(train_data.Gloss.to_list()))
    words = sorted(freq, key=lambda x: -freq[x])

    i = 0
    count = 0
    while count < n_words:
        word = words[count][:-1].title()
        spectrogram = generate_spec(word, model)
        rows.append([words[count]] + spectrogram.flatten().tolist())
        if np.allclose(spectrogram, 0):
            print(word)
            i += 1
        count += 1
    print(f"Bad Audio: {i}/{count}")
    data = pd.DataFrame(rows)
    data.rename(columns={data.columns[0]: "word"}, inplace=True)

    return data


def main(
    train_data_path: str, specs_path: str, classlist_path: str, n_words: int, model
):
    specs_path = create_path(specs_path)
    classlist_path = create_path(classlist_path)

    train_data = pd.read_csv(train_data_path)
    data = process_words(n_words, train_data, model)
    data.to_csv(specs_path, index=False)

    classes = data["word"].tolist()
    with open(classlist_path, "w") as f:
        f.write("\n".join(classes))


if __name__ == "__main__":
    config = Config("Generate spectrograms for words")

    checkpoint_path = create_path(config.generator.checkpoints)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tacotron2 = Tacotron2.from_hparams(
        source="speechbrain/tts-tacotron2-ljspeech",
        savedir=checkpoint_path / "tts-tacotron2",
        run_opts={"device": device},
    )

    main(
        config.data.raw.csvs.train,
        config.data.processed.specs,
        config.data.processed.classlist,
        config.n_words,
        tacotron2,
    )
