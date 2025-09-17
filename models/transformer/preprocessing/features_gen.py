from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset
from models.extractor.model import Extractor
from utils.common import create_path, create_subset, get_logger
from utils.config import Splits, load_config
from utils.model import load_model_weights

logger = get_logger("logs/feature_generation.log")


def extract_features(model: Extractor, dataloader: DataLoader, save_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_features = []
    all_video_ids = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(
            tqdm(dataloader, desc="Feature Extraction")
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            features, logits = model(inputs)
            probabilities = F.softmax(logits, dim=1)
            _, predictions = torch.max(probabilities, 1)
            correct_indices = (
                (predictions == predictions).cpu().numpy()
            )  # currently always True
            # to keep only correct predictions, you might want to compare with labels:
            # correct_indices = (predictions == labels).cpu().numpy()

            if any(correct_indices):
                features_np = features.cpu().numpy()
                correct_features = features_np[correct_indices]

                # Get video IDs for this batch
                start_idx = batch_idx * dataloader.batch_size
                end_idx = min(
                    start_idx + dataloader.batch_size, len(dataloader.dataset)
                )
                batch_video_ids = dataloader.dataset.data.loc[
                    start_idx : end_idx - 1, "Video file"
                ].to_list()

                # Only keep video IDs with correct predictions
                correct_video_ids = [
                    vid
                    for i, vid in enumerate(batch_video_ids)
                    if i < len(correct_indices) and correct_indices[i]
                ]

                all_features.extend(correct_features)
                all_video_ids.extend(correct_video_ids)

    if all_features:
        feature_cols = [f"feature_{i}" for i in range(len(all_features[0]))]
        df = pd.DataFrame(all_features, columns=feature_cols)
        df["Video file"] = all_video_ids

        data = dataloader.dataset.data
        video_to_gloss = {
            data.iloc[i]["Video file"]: data.iloc[i]["Gloss"] for i in range(len(data))
        }
        df["Gloss"] = df["Video file"].map(video_to_gloss)
        df.to_csv(save_path, index=False)
        logger.info(
            f"Features saved to {save_path} with {len(df)}/{len(dataloader.dataset)} matching samples"
        )
    else:
        logger.warning(f"No matching predictions found for {save_path}")
        # Create empty dataframe with expected columns
        feature_cols = [f"feature_{i}" for i in range(model.output_dim)]
        df = pd.DataFrame(columns=feature_cols + ["Video file", "Gloss"])
        df.to_csv(save_path, index=False)


def main(
    data_path: Splits,
    num_words: int,
    video_root: str,
    weights: str,
    save_path: Splits,
    device: torch.device,
) -> None:
    logger.debug(f"Using Device: {device}")
    model = Extractor(num_words).to(device)
    load_model_weights(model, weights, device)

    for split in ["train", "test", "val"]:
        csv_path = getattr(data_path, split)
        dataset = WLASLDataset(csv_path, video_root)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
        output_path = create_path(getattr(save_path, split))
        extract_features(model, dataloader, output_path)


if __name__ == "__main__":
    config = load_config(
        "Feature Generation for Spectrogram Generation",
        model_weights={
            "type": str,
            "default": "models/checkpoints/extractor_best.pt",
            "help": "Path to the model weights file",
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subsets = Splits(
        train=create_subset(config.data.processed.csvs.train, config.n_words),
        val=create_subset(config.data.processed.csvs.val, config.n_words),
        test=create_subset(config.data.processed.csvs.test, config.n_words),
    )

    main(
        subsets,
        config.n_words,
        config.data.processed.videos,
        config.model_weights,
        config.data.processed.vid_features,
        device,
    )
