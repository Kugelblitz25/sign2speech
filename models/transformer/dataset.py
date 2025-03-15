import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, features_csv: str, spectrograms_csv: str, spec_len: int = 80) -> None:
        self.features_df = pd.read_csv(features_csv)
        feature_cols = [col for col in self.features_df.columns if "feature_" in col]

        self.specs_df = pd.read_csv(spectrograms_csv)
        specs = self.specs_df.drop("word", axis=1).values
        self.spectrograms = {
            word: spec.reshape(-1, 88, spec_len)
            for word, spec in zip(self.specs_df["word"], specs)
            if not np.allclose(spec, -15)
        }
        self.features_df = self.features_df[
            self.features_df.Gloss.isin(self.spectrograms.keys())
        ].reset_index(drop=True)
        self.features = self.features_df[feature_cols].values
        self.words = self.features_df["Gloss"]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        feature = torch.Tensor(self.features[idx])
        word = self.words[idx]
        spectrogram = torch.Tensor(self.spectrograms[word])
        return feature, spectrogram
