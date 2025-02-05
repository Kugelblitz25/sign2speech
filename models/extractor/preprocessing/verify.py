from collections import namedtuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset
from utils.common import Config, get_logger
from utils.model import create_path

logger = get_logger("logs/video_verify.log")
csvPaths = namedtuple("Paths", ["train", "test", "val"])


def verify_videos(
    datafile_path: str, video_root: str, classlist: set[str]
) -> pd.DataFrame:
    # Create dataset and dataloader
    data = pd.read_csv(datafile_path)
    dataset = WLASLDataset(data, video_root)
    label_to_class = {val: key for key, val in dataset.class_to_idx.items()}
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Verify videos
    good_videos = []
    for idx, (video_data, label) in enumerate(
        tqdm(dataloader, desc="Verifying videos")
    ):
        if (not torch.all(video_data == 0)) and (
            label_to_class[label.numpy()[0]] in classlist
        ):
            good_videos.append(data.iloc[idx])

    # Report results
    total_videos = len(dataset)
    successful_videos = len(good_videos)
    logger.info(
        f"Verification complete. {successful_videos}/{total_videos} videos loaded successfully."
    )

    return pd.concat(good_videos, axis=1).T


def main(
    csvs_path: csvPaths,
    classlist_path: str,
    video_root: str,
    verified_csvs_path: csvPaths,
):
    with open(classlist_path) as f:
        classes = set([word.strip() for word in f.readlines()])

    for split in ["train", "test", "val"]:
        csv_path = getattr(csvs_path, split)
        verified_csv_path = create_path(getattr(verified_csvs_path, split))
        verified_data = verify_videos(csv_path, video_root, classes)
        verified_data.to_csv(verified_csv_path, index=False)


if __name__ == "__main__":
    config = Config("Process video dataset for classification")

    main(
        config.data.raw.csvs,
        config.data.processed.classlist,
        config.data.raw.videos,
        config.data.processed.csvs,
    )
