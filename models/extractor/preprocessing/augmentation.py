import random
from pathlib import Path

import cv2
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import Permute, ShortSideScale
from torchvision.transforms import ColorJitter, Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
)
from tqdm import tqdm

from utils.common import create_path, get_logger
from utils.config import Splits, load_config

logger = get_logger("logs/video_augmentation.log")


def apply_augmentation(frames: torch.Tensor) -> torch.Tensor:
    side_size = 256
    crop_size = 256
    transform = Compose(
        [
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size=(crop_size, crop_size)),
            Permute((1, 0, 2, 3)),
            Lambda(lambda x: (x / 255.0)),
            ColorJitter(brightness=0.6, contrast=0.6, hue=0.2, saturation=0.6),
            Lambda(lambda x: (x * 255.0)),
        ]
    )
    rotation_angle = random.uniform(-15, 15)
    tot_frames = int(frames.shape[1])
    temporal_mask = torch.ones(tot_frames, dtype=torch.bool)
    num_frames_to_drop = random.randint(0, tot_frames // 8)
    drop_indices = random.sample(range(tot_frames), num_frames_to_drop)
    temporal_mask[drop_indices] = False

    frames_aug = transform(frames)
    frames_aug = frames_aug[temporal_mask, :, :, :]
    frames_aug = TF.rotate(frames_aug, rotation_angle)
    return frames_aug


def save_video(frames, output_path, fps=25):
    height, width = frames.shape[2], frames.shape[3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_tensor in frames:
        frame_np = (frame_tensor).permute(1, 2, 0).cpu().numpy().astype("uint8")
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


def augment_dataset(
    data: pd.DataFrame,
    video_root: Path,
    output_dir: Path,
    num_augmentations: int = 3,
) -> pd.DataFrame:
    augmented_data = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for row in tqdm(range(len(data)), desc="Augmenting videos"):
        item = data.iloc[row]
        video_path = video_root / item["Video file"]

        try:
            # Load original video
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=video.duration)
            video_frames = video_data["video"]

            # Process original frames
            # processed_frames = []
            # x1, y1, x2, y2 = item["bbox"]
            # w, h = video_frames.shape[2], video_frames.shape[3]
            # x1, x2 = max(0, x1 - 50), min(h, x2 + 50)
            # y1, y2 = max(0, y1 - 50), min(w, y2 + 50)
            # for i in range(video_frames.shape[1]):
            #     frame = video_frames[:, i, :, :]  # y1:y2, x1:x2]
            #     processed_frames.append(frame)

            # Generate augmented versions
            for i in range(num_augmentations):
                augmented_frames = apply_augmentation(video_frames.to(device))
                video_file = item["Video file"]
                new_video_path = output_dir / f"{i}_{video_file}"
                save_video(augmented_frames, new_video_path)
                new_item = item.copy()
                new_item["Video file"] = f"{i}_{video_file}"
                augmented_data.append(new_item)

        except Exception as e:
            logger.error(f"Failed to augment video {video_path}: {str(e)}")
            continue

    return pd.concat(augmented_data, axis=1).T


def main(
    csvs_path: Splits,
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

        logger.info(f"Created {len(augmented_data)} augmented videos for {csv_path}")


# Example usage
if __name__ == "__main__":
    config = load_config("Video Data Augmentation for Sign Language Dataset")

    output_video_dir = create_path(config.data.processed.videos)

    main(
        config.data.processed.csvs,
        config.data.raw.videos,
        output_video_dir,
        config.extractor.num_augmentations,
    )
