import logging
import random
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from pytorchvideo.data.encoded_video import EncodedVideo
from tqdm import tqdm

from utils.config import Config
from utils.model import create_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
csvPaths = namedtuple("Paths", ["train", "test", "val"])


def apply_augmentation(frames: list[np.ndarray]) -> list[np.ndarray]:
    augmented_frames = []

    # Random augmentation parameters (consistent across frames)
    brightness_factor = random.uniform(0.6, 1.6)
    contrast_factor = random.uniform(0.6, 1.6)
    hue_factor = random.uniform(-0.2, 0.2)
    saturation_factor = random.uniform(0.6, 1.6)
    rotation_angle = random.uniform(-15, 15)

    # Random temporal sampling (speed variation)
    temporal_mask = torch.ones(len(frames))
    num_frames_to_drop = random.randint(0, len(frames) // 8)
    drop_indices = random.sample(range(len(frames)), num_frames_to_drop)
    temporal_mask[drop_indices] = 0

    for i, frame in enumerate(frames):
        if temporal_mask[i] == 0:
            continue
        frame_pil = TF.to_pil_image(frame)
        frame_aug = TF.rotate(frame_pil, rotation_angle)
        frame_aug = TF.adjust_brightness(frame_aug, brightness_factor)
        frame_aug = TF.adjust_contrast(frame_aug, contrast_factor)
        frame_aug = TF.adjust_hue(frame_aug, hue_factor)
        frame_aug = TF.adjust_saturation(frame_aug, saturation_factor)
        augmented_frames.append(frame_aug)

    return augmented_frames


def save_video(frames, output_path, fps=25):
    height, width = frames[0].shape[1], frames[0].shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_tensor in frames:
        frame_np = (
            (255 - frame_tensor * 255).byte().permute(1, 2, 0).numpy().astype("uint8")
        )
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


def augment_dataset(
    data: pd.DataFrame, video_root: Path, output_dir: Path, num_augmentations: int = 3
) -> pd.DataFrame:
    augmented_data = []

    for row in tqdm(range(len(data)), desc="Augmenting videos"):
        item = data.iloc[row]
        video_path = video_root / item["Video file"]

        try:
            # Load original video
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=video.duration)
            video_frames = video_data["video"]

            # Process original frames
            processed_frames = []
            # x1, y1, x2, y2 = item["bbox"]
            # w, h = video_frames.shape[2], video_frames.shape[3]
            # x1, x2 = max(0, x1 - 50), min(h, x2 + 50)
            # y1, y2 = max(0, y1 - 50), min(w, y2 + 50)
            for i in range(video_frames.shape[1]):
                frame = video_frames[:, i, :, :]  # y1:y2, x1:x2]
                processed_frames.append(frame)

            # Generate augmented versions
            for i in range(num_augmentations):
                augmented_frames = apply_augmentation(processed_frames)
                video_file = item["Video file"]
                new_video_path = output_dir / f"{i}_{video_file}"
                save_video(augmented_frames, new_video_path)
                new_item = item.copy()
                new_item["Video file"] = f"{i}_{video_file}"
                augmented_data.append(new_item)

        except Exception as e:
            logging.error(f"Failed to augment video {video_path}: {str(e)}")
            continue

    return pd.concat(augmented_data, axis=1).T


def main(
    csvs_path: csvPaths,
    video_root: str,
    output_video_dir: Path,
    num_augmentations: int,
):
    for split in ["train", "test", "val"]:
        csv_path = getattr(csvs_path, split)
        data = pd.read_csv(csv_path)
        augmented_data = augment_dataset(
            data, Path(video_root), output_video_dir, num_augmentations
        )
        augmented_data.to_csv(csv_path, index=False)

        logging.info(f"Created {len(augmented_data)} augmented videos for {csv_path}")


# Example usage
if __name__ == "__main__":
    config = Config("Video Data Augmentation for Sign Language Dataset")

    output_video_dir = create_path(config.data.processed.videos)

    main(
        config.data.processed.csvs,
        config.data.raw.videos,
        output_video_dir,
        config.extractor.num_augmentations,
    )
