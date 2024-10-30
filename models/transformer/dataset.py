import pandas as pd
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, features_csv, spectrograms_csv):
        self.features_df = pd.read_csv(features_csv)
        feature_cols = [col for col in self.features_df.columns if 'feature_' in col]
        self.words = self.features_df['gloss']
        self.features = self.features_df[feature_cols].values
        
        self.specs_df = pd.read_csv(spectrograms_csv)
        specs = self.specs_df.drop('word', axis=1).values
        self.spectrograms = {word: spec.reshape(-1, 128, 88) for word, spec in zip(self.specs_df['word'], specs)}
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.Tensor(self.features[idx])
        word = self.words[idx]
        spectrogram = torch.Tensor(self.spectrograms[word])
        return feature, spectrogram
