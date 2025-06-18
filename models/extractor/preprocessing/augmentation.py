import random
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
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


class BackgroundRemover:
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )

    def remove_background(self, frame: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(frame_rgb)

        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_frame = np.zeros(frame.shape, dtype=np.uint8)
        bg_frame[:] = (0, 0, 0)

        output_frame = np.where(condition, frame, bg_frame)
        return output_frame


class MotionDetector:
    def __init__(self):
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def detect_motion_regions(self, frames: List[np.ndarray]) -> List[float]:
        if len(frames) < 2:
            return [0.0] * len(frames)

        motion_scores = [0.0]

        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            p0 = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
            )

            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, p0, None, **self.lk_params
                )

                good_new = p1[st == 1]
                good_old = p0[st == 1]

                if len(good_new) > 0 and len(good_old) > 0:
                    motion_magnitude = np.mean(
                        np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
                    )
                    motion_scores.append(motion_magnitude)
                else:
                    motion_scores.append(0.0)
            else:
                motion_scores.append(0.0)

        return motion_scores

    def get_motion_crop_indices(self, motion_scores: List[float]) -> Tuple[int, int]:
        if not motion_scores:
            return 0, len(motion_scores) - 1

        motion_threshold = np.mean(motion_scores) + np.std(motion_scores) * 0.5
        high_motion_indices = [
            i for i, score in enumerate(motion_scores) if score > motion_threshold
        ]

        if not high_motion_indices:
            return 0, len(motion_scores) - 1

        # start_idx = max(0, min(high_motion_indices) - 5)
        # end_idx = min(len(motion_scores) - 1, max(high_motion_indices) + 5)

        return 0, len(motion_scores) - 1


bg_remover = BackgroundRemover()
motion_detector = MotionDetector()


def preprocess_video_frames(video_frames: torch.Tensor) -> torch.Tensor:
    frames_list = []
    for i in range(video_frames.shape[1]):
        frame_tensor = video_frames[:, i, :, :]
        frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy().astype("uint8")
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        frames_list.append(frame_bgr)

    motion_scores = motion_detector.detect_motion_regions(frames_list)
    start_idx, end_idx = motion_detector.get_motion_crop_indices(motion_scores)

    processed_frames = []
    for i in range(start_idx, end_idx + 1):
        if i < len(frames_list):
            frame_no_bg = bg_remover.remove_background(frames_list[i])
            frame_rgb = cv2.cvtColor(frame_no_bg, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
            processed_frames.append(frame_tensor)

    if not processed_frames:
        processed_frames = [video_frames[:, 0, :, :]]

    processed_video = torch.stack(processed_frames, dim=1)
    return processed_video


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


def save_video(frames: torch.Tensor, output_path: str | Path, fps: int = 25) -> None:
    height, width = frames.shape[2], frames.shape[3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_tensor in frames:
        frame_np = (frame_tensor).permute(1, 2, 0).cpu().numpy().astype("uint8")
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


def augment_dataset(
    data: pd.DataFrame,
    video_root: Path,
    output_dir: Path,
    num_augmentations: int,
    device: torch.device,
) -> pd.DataFrame:
    augmented_data = []

    for row in tqdm(range(len(data)), desc="Augmenting videos"):
        item = data.iloc[row]
        video_path = video_root / item["Video file"]

        try:
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=video.duration)
            video_frames = video_data["video"]
            preprocessed_frames = preprocess_video_frames(video_frames)

            for i in range(num_augmentations):
                augmented_frames = apply_augmentation(preprocessed_frames.to(device))
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
    device: torch.device,
) -> None:
    for split in ["train", "test", "val"]:
        csv_path = getattr(csvs_path, split)
        data = pd.read_csv(csv_path)
        augmented_data = augment_dataset(
            data, Path(video_root), output_video_dir, num_augmentations, device
        )
        augmented_data.to_csv(csv_path, index=False)

        logger.info(f"Created {len(augmented_data)} augmented videos for {csv_path}")


if __name__ == "__main__":
    config = load_config("Video Data Augmentation for Sign Language Dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_video_dir = create_path(config.data.processed.videos)

    main(
        config.data.processed.csvs,
        config.data.raw.videos,
        output_video_dir,
        config.extractor.num_augmentations,
        device,
    )
