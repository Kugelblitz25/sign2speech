import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset
from utils.common import create_path, get_logger, create_subset
from utils.config import Splits, load_config

logger = get_logger("logs/video_verify.log")


def verify_videos(datafile_path: str, video_root: str) -> pd.DataFrame:
    # Create dataset and dataloader
    dataset = WLASLDataset(datafile_path, video_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    data = pd.read_csv(datafile_path)

    # Verify videos
    good_videos = []
    for idx, (video_data, _) in enumerate(tqdm(dataloader, desc="Verifying videos")):
        if not torch.all(video_data == 0):
            good_videos.append(data.iloc[idx])

    # Report results
    total_videos = len(dataset)
    successful_videos = len(good_videos)
    logger.info(
        f"Verification complete. {successful_videos}/{total_videos} videos loaded successfully."
    )

    return pd.concat(good_videos, axis=1).T


def main(
    n_words: int,
    csvs_path: Splits,
    video_root: str,
    verified_csvs_path: Splits,
) -> None:
    for split in ["train", "test", "val"]:
        csv_path = getattr(csvs_path, split)
        subset_csv_path = create_subset(csv_path, n_words)
        verified_csv_path = create_path(getattr(verified_csvs_path, split))
        verified_data = verify_videos(subset_csv_path, video_root)
        verified_data.to_csv(verified_csv_path, index=False)


if __name__ == "__main__":
    config = load_config("Process video dataset for classification")

    main(
        config.n_words,
        config.data.raw.csvs,
        config.data.raw.videos,
        config.data.processed.csvs,
    )
