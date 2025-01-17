import json
import logging
import random
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from tqdm import tqdm

from utils.configs import create_path, load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class VideoAugmenter:
    def apply_augmentation(self, frames):
        """Apply random augmentations to video frames"""
        augmented_frames = []

        # Random augmentation parameters (consistent across frames)
        brightness_factor = random.uniform(0.6, 1.6)
        contrast_factor = random.uniform(0.6, 1.6)
        hue_factor = random.uniform(-0.2, 0.2)
        saturation_factor = random.uniform(0.6, 1.6)
        rotation_angle = random.uniform(-15, 15)

        # Random temporal sampling (speed variation)
        temporal_mask = torch.ones(len(frames))
        num_frames_to_drop = random.randint(0, len(frames) // 4)
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
            frame_aug = TF.to_tensor(frame_aug)
            augmented_frames.append(frame_aug)

        return torch.stack(augmented_frames)

    def save_video(self, frames, output_path, fps=30):
        height, width = frames[0].shape[1], frames[0].shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_tensor in frames:
            frame_np = (
                (255 - frame_tensor * 255)
                .byte()
                .permute(1, 2, 0)
                .numpy()
                .astype("uint8")
            )
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()


def augment_dataset(
    data: list[dict], video_root: str, output_dir: str, num_augmentations: int = 3
):
    # Create augmenter
    augmenter = VideoAugmenter()

    # Transform for loading original videos
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    augmented_data = []

    for item in tqdm(data, desc="Augmenting videos"):
        video_path = video_root / f"{item['video_id']}.mp4"

        try:
            # Load original video
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=video.duration)
            video_frames = video_data["video"]

            # Process original frames
            processed_frames = []
            x1, y1, x2, y2 = item["bbox"]
            w, h = video_frames.shape[2], video_frames.shape[3]
            x1, x2 = max(0, x1 - 50), min(h, x2 + 50)
            y1, y2 = max(0, y1 - 50), min(w, y2 + 50)
            for i in range(video_frames.shape[1]):
                frame = video_frames[:, i, y1:y2, x1:x2]
                processed_frames.append(transform(frame))
            processed_frames = torch.stack(processed_frames, dim=0)

            # Generate augmented versions
            for i in range(num_augmentations):
                augmented_frames = augmenter.apply_augmentation(processed_frames)
                video_id = item["video_id"]
                new_video_path = output_dir / f"{video_id}_{i}.mp4"
                augmenter.save_video(augmented_frames, new_video_path)
                augmented_data.append(
                    {
                        "gloss": item["gloss"],
                        "video_id": f"{video_id}_{i}",
                    }
                )

        except Exception as e:
            logging.error(f"Failed to augment video {video_path}: {str(e)}")
            continue

    return augmented_data


def main(datafile: str, video_root: str, output_video_dir: str, num_augmentations: int):
    with open(datafile) as f:
        data = json.load(f)

    augmented_data = augment_dataset(
        data, Path(video_root), output_video_dir, num_augmentations
    )

    with open(datafile, "w") as f:
        json.dump(augmented_data, f, indent=4)

    logging.info(f"Created {len(augmented_data)} augmented videos for {datafile}")


# Example usage
if __name__ == "__main__":
    config = load_config("Video Data Augmentation for Sign Language Dataset")

    train_datafile = config["data"]["processed"]["train_data"]
    test_datafile = config["data"]["processed"]["test_data"]
    video_root = config["data"]["raw"]["videos"]
    output_videos_dir = config["data"]["processed"]["videos"]
    num_augmentations = config["extractor"]["num_augmentations"]

    output_video_dir = create_path(output_videos_dir)

    main(train_datafile, video_root, output_videos_dir, num_augmentations)
    main(test_datafile, video_root, output_videos_dir, num_augmentations)
