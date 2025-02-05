from collections import namedtuple
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset
from models.extractor.model import ModifiedI3D
from utils.common import Config, get_logger
from utils.model import create_path, load_model_weights

csvPaths = namedtuple("Paths", ["train", "test", "val"])
logger = get_logger("logs/feature_generation.log")


def extract_features(model, dataloader: DataLoader, save_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_features = []
    all_video_ids = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(
            tqdm(dataloader, desc="Feature Extraction")
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            features, _ = model(inputs)
            features = (
                features.flatten().cpu().numpy().reshape(dataloader.batch_size, -1)
            )

            # Get video IDs for this batch
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min(start_idx + dataloader.batch_size, len(dataloader.dataset))
            batch_video_ids = dataloader.dataset.data.loc[
                start_idx : end_idx - 1, "Video file"
            ].to_list()

            all_features.extend(features)
            all_video_ids.extend(batch_video_ids)

    feature_cols = [f"feature_{i}" for i in range(len(all_features[0]))]
    df = pd.DataFrame(all_features, columns=feature_cols)
    df["Video file"] = all_video_ids
    data = dataloader.dataset.data
    video_to_gloss: dict[str, str] = {
        data.iloc[i]["Video file"]: data.iloc[i]["Gloss"] for i in range(len(data))
    }
    df["Gloss"] = df["Video file"].map(video_to_gloss)
    df.to_csv(save_path, index=False)
    logger.info(f"Features saved to {save_path}")


def main(
    data_path: csvPaths,
    num_words: int,
    video_root: str,
    weights: str,
    save_path: csvPaths,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using Device: {device}")
    model = ModifiedI3D(num_words).to(device)
    model = load_model_weights(model, weights, device)

    for split in ["train", "test", "val"]:
        csv_path = getattr(data_path, split)
        data = pd.read_csv(csv_path)
        dataset = WLASLDataset(data, video_root)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
        output_path = create_path(getattr(save_path, split))
        extract_features(model, dataloader, output_path)


if __name__ == "__main__":
    config = Config("Feature Generation for Spectrogram Generation")

    main(
        config.data.processed.csvs,
        config.n_words,
        config.data.processed.videos,
        config.transformer.extractor_weights,
        config.data.processed.vid_features,
    )
