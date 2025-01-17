import argparse
import json
import yaml
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset, video_transform
from utils import create_path, load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
video_data = list[dict[str, str]]


def process_json(
    json_path: str, video_root: str, classlist_path: str
) -> tuple[video_data, video_data, list[str]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    with open(classlist_path, "r") as f:
        classlist = set([word.strip() for word in f.readlines()])

    data = [item for item in data if item["gloss"] in classlist]
    video_root = Path(video_root)
    train_data = []
    test_data = []
    miss_count = 0
    tot_count = 0
    for item in tqdm(data, desc="Processing JSON"):
        gloss = item["gloss"]
        for instance in item["instances"]:
            tot_count += 1
            if not (video_root / f"{instance['video_id']}.mp4").exists():
                miss_count += 1
                continue

            split = instance["split"]
            data = {
                "gloss": gloss,
                "video_id": instance["video_id"],
                "bbox": instance["bbox"],
            }
            if split == "train":
                train_data.append(data)
            else:
                test_data.append(data)
    logging.info(f"{tot_count - miss_count}/{tot_count} videos found.")
    return train_data, test_data


def verify_videos(data: video_data, video_root: str):
    # Create dataset and dataloader
    dataset = WLASLDataset(data, video_root, transform=video_transform())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Verify videos
    good_videos = []
    for idx, (video_data, _) in enumerate(tqdm(dataloader, desc="Verifying videos")):
        if not torch.all(video_data == 0):
            good_videos.append(data[idx])

    # Report results
    total_videos = len(dataset)
    successful_videos = len(good_videos)
    logging.info(
        f"Verification complete. {successful_videos}/{total_videos} videos loaded successfully."
    )

    return good_videos


def main(
    json_path: str,
    classlist_path: str,
    video_root: str,
    train_data_path: str,
    test_data_path: str,
):
    train_data_path = create_path(train_data_path)
    test_data_path = create_path(test_data_path)

    train_data, test_data = process_json(json_path, video_root, classlist_path)
    train_data = verify_videos(train_data, video_root)
    test_data = verify_videos(test_data, video_root)

    with open(train_data_path, "w") as f:
        json.dump(train_data, f, indent=4)

    with open(test_data_path, "w") as f:
        json.dump(test_data, f, indent=4)


if __name__ == "__main__":
    config = load_config("Process video dataset for classification")

    json_path = config["data"]["raw"]["json"]
    video_root = config["data"]["raw"]["videos"]
    classes_path = config["data"]["processed"]["classlist"]
    train_data_path = config["data"]["processed"]["train_data"]
    test_data_path = config["data"]["processed"]["test_data"]

    main(json_path, classes_path, video_root, train_data_path, test_data_path)
