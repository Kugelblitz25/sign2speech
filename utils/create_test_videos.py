import cv2
import numpy as np

import pandas as pd
from pathlib import Path

from utils.config import load_config
from utils.common import create_subset


def combine_videos(video_list: list[str], video_root: Path) -> np.ndarray:
    video_paths = [str(video_root / video) for video in video_list]
    frames = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if frames:
            last_frame = frames[-1]
            for _ in range(10):
                frames.append(last_frame)
    return np.array(frames)


def main(
    num_videos: int,
    wpv: int,
    data: pd.DataFrame,
    video_root: Path,
    output_dir: Path,
):
    for _ in range(num_videos):
        random_signer = np.random.choice(data["Participant ID"].unique())
        sub_data = data[data["Participant ID"] == random_signer]
        random_videos = sub_data.sample(min(wpv, len(sub_data)))
        video_files = random_videos["Video file"].tolist()
        word_list = random_videos["Gloss"].tolist()
        video_array = combine_videos(video_files, video_root)
        file_name = f"test_video_{random_signer}_{'_'.join(word_list)}.mp4"
        output_path = output_dir / file_name
        # Save the combined video
        out = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25,
            (video_array.shape[2], video_array.shape[1]),
        )
        for frame in video_array:
            out.write(frame)
        out.release()


if __name__ == "__main__":
    config = load_config(
        "Create Test Videos",
        num_videos={
            "type": int,
            "default": 10,
            "help": "Number of test videos to create",
        },
        wpv={
            "type": int,
            "default": 10,
            "help": "Number of words per video",
        },
        output_dir={
            "type": str,
            "default": "test_videos",
            "help": "Directory to save test videos",
        },
    )

    video_root = Path(config.data.raw.videos)
    data_file = create_subset(config.data.raw.csvs.test, config.n_words)
    data = pd.read_csv(data_file)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main(config.num_videos, config.wpv, data, video_root, output_dir)
