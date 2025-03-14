from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset
from models.extractor.model import Extractor
from utils.common import get_logger, create_path
from utils.config import load_config
from utils.model import load_model_weights

logger = get_logger("logs/extractor_testing.log")


class Tester:
    def __init__(
        self,
        model: Extractor,
        test_data_path: str,
        video_root: str,
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> None:
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using Device: {self.device}")

        self.model = model.to(self.device)
        self.test_data = pd.read_csv(test_data_path)
        self.video_root = video_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.test_loader = self._get_dataloader()

    def _get_dataloader(self) -> DataLoader:
        dataset = WLASLDataset(self.test_data, self.video_root)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return dataloader

    def evaluate(self) -> dict:
        self.model.eval()

        all_labels = []
        all_predictions = []
        all_top5_predictions = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Testing"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, outputs = self.model(inputs)

                # Get top-1 predictions
                _, top1_preds = torch.max(outputs, 1)

                # Get top-5 predictions
                _, top5_preds = torch.topk(outputs, 5, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(top1_preds.cpu().numpy())
                all_top5_predictions.extend(top5_preds.cpu().numpy())

        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_top5_predictions = np.array(all_top5_predictions)

        # Calculate metrics
        results = {}

        # Top-1 accuracy
        top1_acc = (all_predictions == all_labels).mean() * 100
        results["top1_accuracy"] = top1_acc

        # Top-5 accuracy
        top5_correct = 0
        for i, label in enumerate(all_labels):
            if label in all_top5_predictions[i]:
                top5_correct += 1
        top5_acc = (top5_correct / len(all_labels)) * 100
        results["top5_accuracy"] = top5_acc

        # F1 scores
        f1_per_class = f1_score(all_labels, all_predictions, average=None)
        f1_weighted = f1_score(all_labels, all_predictions, average="weighted")

        results["f1_weighted"] = f1_weighted
        results["f1_per_class"] = f1_per_class

        # Detailed classification report
        class_report = classification_report(
            all_labels, all_predictions, output_dict=True
        )
        results["classification_report"] = class_report

        return results

    def print_results(self, results: dict, output_path: Path) -> None:
        logger.info("Test Results:")
        logger.info(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
        logger.info(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
        logger.info(f"Weighted F1-Score: {results['f1_weighted']:.4f}")

        # Print per-class metrics if needed
        # Note: This could be large depending on the number of classes
        logger.info("Per-class metrics available in results['classification_report']")

        # Save results to CSV
        results_df = pd.DataFrame(
            {
                "Class": list(results["classification_report"].keys())[
                    :-3
                ],  # Exclude 'accuracy', 'macro avg', 'weighted avg'
                "Precision": [
                    results["classification_report"][c]["precision"]
                    for c in list(results["classification_report"].keys())[:-3]
                ],
                "Recall": [
                    results["classification_report"][c]["recall"]
                    for c in list(results["classification_report"].keys())[:-3]
                ],
                "F1-Score": [
                    results["classification_report"][c]["f1-score"]
                    for c in list(results["classification_report"].keys())[:-3]
                ],
                "Support": [
                    results["classification_report"][c]["support"]
                    for c in list(results["classification_report"].keys())[:-3]
                ],
            }
        )

        results_df.to_csv(output_path / "test_results_per_class.csv", index=False)
        logger.info("Per-class results saved to 'test_results_per_class.csv'")

        # Save summary metrics
        summary_df = pd.DataFrame(
            {
                "Metric": ["Top-1 Accuracy", "Top-5 Accuracy", "Weighted F1-Score"],
                "Value": [
                    results["top1_accuracy"],
                    results["top5_accuracy"],
                    results["f1_weighted"],
                ],
            }
        )

        summary_df.to_csv(output_path / "test_results_summary.csv", index=False)
        logger.info("Summary results saved to 'test_results_summary.csv'")


if __name__ == "__main__":
    config = load_config("Test video classification model")

    output_path = create_path("experiments/test_extractor/")

    model = Extractor(
        num_classes=config.n_words,
        base_model=config.extractor.model,
        n_freeze=config.extractor.training.freeze,
    )

    load_model_weights(
        model,
        Path(config.extractor.checkpoints) / f"full_best_{config.extractor.model}.pt",
        torch.device("cuda:1"),
    )

    tester = Tester(
        model=model,
        test_data_path=config.data.processed.csvs.test,
        video_root=config.data.processed.videos,
        batch_size=config.extractor.training.batch_size,
        num_workers=config.extractor.training.num_workers,
    )

    results = tester.evaluate()
    tester.print_results(results)
