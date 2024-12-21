import json
import random
from pathlib import Path
import cv2
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_crop_video(video_path, bbox, resize_dim=(224, 224)):
    try:
        # Load original video
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=0, end_sec=video.duration)
        video_frames = video_data["video"]

        # Crop and resize frames based on bounding box
        x1, y1, x2, y2 = bbox
        cropped_frames = []
        for i in range(video_frames.shape[1]):
            frame = video_frames[:, i, y1:y2, x1:x2]
            frame_resized = torch.nn.functional.interpolate(
                frame.unsqueeze(0), size=resize_dim
            ).squeeze(0)
            cropped_frames.append(frame_resized)

        return torch.stack(cropped_frames)
    except Exception as e:
        logging.error(f"Error loading or cropping video {video_path}: {str(e)}")
        return None


def save_concatenated_video(frames_list, output_path, fps=30):
    height, width = frames_list[0][0].shape[1], frames_list[0][0].shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frames in frames_list:
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


def select_random_instance(data: list, video_root: str):
    selected_items = []

    for item in data:
        if item.get("instances"):
            while True:
                selected_instance = random.choice(item["instances"])
                if (video_root / f"{selected_instance['video_id']}.mp4").exists():
                    break
            selected_items.append(
                {
                    "gloss": item["gloss"],
                    "bbox": selected_instance["bbox"],
                    "video_id": selected_instance["video_id"],
                }
            )

    return selected_items


def create_combined_videos(
    data: list[dict], video_root: str, output_dir: str, num_videos: int = 5
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(num_videos), desc="Creating combined videos"):
        selected_items = select_random_instance(random.sample(data, 3), video_root)
        frames_list = []

        for item in selected_items:
            video_path = video_root / f"{item['video_id']}.mp4"
            frames = load_and_crop_video(video_path, item["bbox"])
            if frames is not None:
                frames_list.append(frames)

        if len(frames_list) == 3:
            combined_video_path = (
                output_dir
                / f"{'_'.join([item['gloss'] for item in selected_items])}.mp4"
            )
            save_concatenated_video(frames_list, combined_video_path)
            logging.info(f"Saved combined video: {combined_video_path}")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create Combined Sign Language Videos")
    parser.add_argument(
        "--datafile",
        type=str,
        default="data/raw/WLASL_v0.3.json",
        help="Path to JSON file containing video information",
    )
    parser.add_argument(
        "--classlist_path",
        type=str,
        default="data/processed/generator/classes.txt",
        help="Path to the classlist file",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="data/raw/videos",
        help="Directory containing videos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_videos",
        help="Directory to save combined videos",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of combined videos to create (default: 5)",
    )

    args = parser.parse_args()
    with open(args.datafile) as f:
        data = json.load(f)

    with open(args.classlist_path) as f:
        classes = [cls.strip() for cls in f.readlines()]

    data = [i for i in data if i["gloss"] in classes]

    create_combined_videos(
        data, Path(args.video_root), args.output_dir, args.num_videos
    )
