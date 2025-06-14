import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset

from models.extractor.dataset import crop_size, num_frames, transform


class S2S_Dataset(Dataset):
    def __init__(
        self,
        videos_csv: str | Path,
        spec_csv: str | Path,
        video_dir: str | Path,
        spec_len: int = 80,
    ) -> None:
        self.video_data = pd.read_csv(videos_csv)
        spec_data = pd.read_csv(spec_csv)
        self.transform = transform

        classes = sorted(self.video_data.Gloss.unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.spectrograms = self.load_spectrograms(spec_data, spec_len)

        self.video_data = self.video_data[
            self.video_data.Gloss.isin(self.spectrograms.keys())
        ]
        self.video_data = self.video_data.reset_index(drop=True)

        video_dir = Path(video_dir)
        self.video_data["Video file"] = (
            self.video_data["Video file"].apply(lambda x: video_dir / x).tolist()
        )

        logging.info(f"Loaded {len(self.video_data)} video entries with spectrograms.")

    def load_spectrograms(self, spec_data: pd.DataFrame, spec_len: int) -> dict:
        specs = spec_data.drop("Gloss", axis=1).values
        spectrograms = {}
        for word, spec in zip(spec_data.Gloss, specs):
            if not np.allclose(spec, 0):
                reshaped_features = spec.reshape(-1, 2)
                real_part = reshaped_features[:, 0].reshape(spec_len, 1025).T
                imag_part = reshaped_features[:, 1].reshape(spec_len, 1025).T
                spectrograms[word] = np.stack([real_part, imag_part], axis=0)
        return spectrograms

    def __len__(self) -> int:
        return len(self.video_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        item = self.video_data.iloc[idx]
        video_path = item["Video file"]
        gloss = item["Gloss"]
        label = self.class_to_idx[gloss]

        try:
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=video.duration)
            video_data = self.transform(video_data)
            video_tensor = video_data["video"]
            spectrogram = torch.tensor(self.spectrograms[gloss], dtype=torch.float32)

            return video_tensor, label, spectrogram
        except Exception as e:
            logging.warning(f"Failed to load video {video_path}: {str(e)}")
            return (
                torch.zeros((3, num_frames, crop_size, crop_size)),
                label,
                torch.tensor(self.spectrograms[gloss]),
            )
