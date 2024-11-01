from pathlib import Path
import pandas as pd
import torch
import json
from nemo.collections.tts.models import Tacotron2Model

tacotron2 = Tacotron2Model.from_pretrained("tts_en_tacotron2")
tacotron2.eval()

def generate_spec(word):
    tokens = tacotron2.parse(word)
    spectrogram = tacotron2.generate_spectrogram(tokens=tokens)
    return spectrogram

def process_words(n_words, json_data):
    spectrograms = {}
    rows = []

    sorted_data = sorted(json_data, key=lambda x: -len(x['instances']))
    words = [sorted_data[i]['gloss'] for i in range(len(sorted_data))]

    i = 0
    count = 0
    while count < n_words:
        spectrogram = generate_spec(words[i])
        if spectrogram.shape[2] > 88:
            i+=1
            continue
        spectrograms[words[i]] = spectrogram
        count += 1
        i += 1


    for word, spectrogram in spectrograms.items():
        padded_spectrogram = torch.nn.functional.pad(
            spectrogram,
            [
                0, max(0, 88 - spectrogram.shape[2]),
                0, max(0, 80 - spectrogram.shape[1])
            ],
            mode='constant', value= -15
        )

        rows.append([word] + padded_spectrogram.cpu().detach().numpy().flatten().tolist())

    data = pd.DataFrame(rows)
    data.rename(columns={data.columns[0]: 'word'}, inplace=True)

    return data

def main(json_file: str, output_path: str, n_words: int):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    with open(json_file, 'r') as f:
        json_data = json.load(f)

    data = process_words(n_words, json_data)
    data.to_csv(output_path/'specs.csv', index=False)

    classes = data['word'].tolist()
    with open(output_path/'classes.txt', 'w') as f:
        f.write('\n'.join(classes))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='data/raw/WLASL_v0.3.json', help='Path WLASL to the json file')
    parser.add_argument('--output_path', type=str, default='data/processed/generator', help='Path to save the processed data')
    parser.add_argument('--n_words', type=int, default=100, help='Number of words to process')

    args = parser.parse_args()
    main(args.json_file, args.output_path, args.n_words)
