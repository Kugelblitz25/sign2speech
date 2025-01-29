from collections import Counter

import pandas as pd
import torch
from speechbrain.inference.TTS import Tacotron2

from utils import Config, create_path


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
    return padded_mel_output


def process_words(n_words: int, train_data: pd.DataFrame, model):
    rows = []

    freq = dict(Counter(train_data.Gloss.to_list()))
    words = sorted(freq, key=lambda x: -freq[x])

    # i = 0
    # count = 0
    # while count < n_words:
    # Change back after proper gloss

    for i in range(n_words):
        spectrogram = generate_spec(words[i][:-1], model)
        rows.append([words[i]] + spectrogram.cpu().detach().numpy().flatten().tolist())

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
