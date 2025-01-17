import pandas as pd
import torch
import json

from utils.configs import create_path, load_config
from speechbrain.inference.TTS import Tacotron2


def generate_spec(word: str, model):
    mel_output, *_ = model.encode_text(word)
    return mel_output


def process_words(n_words: int, json_data: dict, model):
    spectrograms = {}
    rows = []

    sorted_data = sorted(json_data, key=lambda x: -len(x["instances"]))
    words = [sorted_data[i]["gloss"] for i in range(len(sorted_data))]

    i = 0
    count = 0
    while count < n_words:
        spectrogram = generate_spec(words[i], model)
        if spectrogram.shape[2] > 88:
            i += 1
            continue
        spectrograms[words[i]] = spectrogram
        count += 1
        i += 1
        if (count + 1) % 10 == 0:
            print(f"{count+1} spectorams generated.")

    for word, spectrogram in spectrograms.items():
        padded_spectrogram = torch.nn.functional.pad(
            spectrogram,
            [
                0,
                max(0, 88 - spectrogram.shape[2]),
                0,
                max(0, 80 - spectrogram.shape[1]),
            ],
            mode="constant",
            value=-15,
        )

        rows.append(
            [word] + padded_spectrogram.cpu().detach().numpy().flatten().tolist()
        )

    data = pd.DataFrame(rows)
    data.rename(columns={data.columns[0]: "word"}, inplace=True)

    return data


def main(json_file: str, specs_path: str, classlist_path: str, n_words: int, model):
    specs_path = create_path(specs_path)
    classlist_path = create_path(classlist_path)

    with open(json_file, "r") as f:
        json_data = json.load(f)

    data = process_words(n_words, json_data, model)
    data.to_csv(specs_path, index=False)

    classes = data["word"].tolist()
    with open(classlist_path, "w") as f:
        f.write("\n".join(classes))


if __name__ == "__main__":
    config = load_config(description="Generate spectrograms for words")

    json_file = config["data"]["raw"]["json"]
    specs_path = config["data"]["processed"]["specs"]
    classlist_path = config["data"]["processed"]["classlist"]
    n_words = config["n_words"]
    checkpoint_path = config["generator"]["checkpoints"]

    checkpoint_path = create_path(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tacotron2 = Tacotron2.from_hparams(
        source="speechbrain/tts-tacotron2-ljspeech",
        savedir=checkpoint_path / "tts-tacotron2",
        run_opts={"device": device},
    )

    main(json_file, specs_path, classlist_path, n_words, tacotron2)
