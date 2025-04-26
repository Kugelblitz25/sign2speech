import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo

side_size = 256
crop_size = 256
num_frames = 64
sampling_rate = 8
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 25
clip_duration = (num_frames * sampling_rate) / frames_per_second

transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: (x / 255.0) * 2 - 1),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size=(crop_size, crop_size)),
        ]
    ),
)


class CombinedDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        video_dir: str,
        specs_df: pd.DataFrame,
        spec_len: int = 80,
    ) -> None:
        self.data = data
        self.video_dir = Path(video_dir)
        self.transform = transform
        self.classes = sorted(data.Gloss.unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.spec_len = spec_len

        # Process spectrograms
        specs = specs_df.drop("word", axis=1).values
        self.spectrograms = {}
        for word, spec in zip(specs_df["word"], specs):
            if not np.allclose(spec, 0):
                reshaped_features = spec.reshape(-1, 2)
                real_part = reshaped_features[:, 0]
                imag_part = reshaped_features[:, 1]

                D_real = real_part.reshape(spec_len, 1025).T
                D_imag = imag_part.reshape(spec_len, 1025).T
                self.spectrograms[word] = np.stack([D_real, D_imag], axis=0)

        # Filter data to only include items with spectrograms
        self.data = self.data[
            self.data.Gloss.isin(self.spectrograms.keys())
        ].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.tensor, str, torch.tensor]:
        item = self.data.iloc[idx]
        video_path = self.video_dir / item["Video file"]
        gloss = item["Gloss"]
        label = self.class_to_idx[gloss]

        try:
            # Load and transform video
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=clip_duration)
            video_data = transform(video_data)
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
