from collections import deque
from typing import Generator

import cv2
import matplotlib.pyplot as plt
import torch

from models.extractor import FeatureExtractor
from utils.config import NMSConfig


class NMS:
    def __init__(self, extractor: FeatureExtractor, config: NMSConfig) -> None:
        self.extractor = extractor
        self.win_size = config.win_size
        self.overlap = config.overlap
        self.threshold = config.threshold
        self.hop_length = config.hop_length

    def __call__(
        self, cap: cv2.VideoCapture
    ) -> Generator[tuple[int, torch.Tensor], None, None]:
        window: deque = deque(maxlen=self.win_size)

        if not cap.isOpened():
            raise ValueError("Video Stream not open.")

        for _ in range(self.win_size):
            ret, frame = cap.read()
            if not ret:
                break
            window.append(frame)

        best_window_idx: int = 0
        best_feature, best_confidence = self.extractor(list(window))
        window_index: int = 0

        # For visualization
        all_indices: list[int] = []
        all_confidences: list[float] = []
        selected_indices: list[int] = []

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            window_index += 1

            if not window_index % self.hop_length == 0:
                continue

            window.append(frame)
            feature: torch.Tensor
            confidence: float
            feature, confidence = self.extractor(list(window))

            all_indices.append(window_index)
            all_confidences.append(confidence)

            if confidence < self.threshold:
                continue

            if abs(window_index - best_window_idx) > self.win_size - self.overlap:
                selected_indices.append(window_index)
                yield (best_window_idx, best_feature)
                best_window_idx = window_index
                best_feature = feature
                best_confidence = confidence

            elif confidence > best_confidence:
                best_window_idx = window_index
                best_feature = feature
                best_confidence = confidence

        if best_confidence > self.threshold:
            selected_indices.append(best_window_idx)
            yield (best_window_idx, best_feature)
        
        yield (window_index, None)

        # Create visualization
        if all_indices:
            self.plot_probabilities(all_indices, all_confidences, selected_indices)

    def plot_probabilities(
        self,
        indices: list[int],
        confidences: list[float],
        selected_indices: list[int],
        plot_path: str = "realtime_probs.png",
    ) -> None:
        plt.figure(figsize=(12, 6))

        # Plot all probabilities
        plt.plot(indices, confidences, "b-", alpha=0.5, label="All Windows")
        selected_confidences: list[float] = []
        for idx in selected_indices:
            if idx in indices:
                selected_confidences.append(confidences[indices.index(idx)])

        plt.scatter(
            selected_indices,
            selected_confidences,
            color="red",
            s=50,
            label="Selected Windows",
        )

        # Add threshold line
        plt.axhline(
            y=self.threshold,
            color="green",
            linestyle="--",
            label=f"Threshold ({self.threshold})",
        )

        # Add window regions
        for idx in selected_indices:
            plt.axvspan(idx, idx + self.win_size, alpha=0.2, color="lightgreen")

        plt.xlabel("Frame Index")
        plt.ylabel("Confidence Score")
        plt.title("Realtime NMS Window Selection")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
