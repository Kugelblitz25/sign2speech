import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.extractor import FeatureExtractor
from utils.config import NMSConfig

class NMS:
    def __init__(self, extractor: FeatureExtractor, config: NMSConfig) -> None:
        self.extractor = extractor
        self.hop_length = config.hop_length
        self.win_size = config.win_size
        self.overlap = config.overlap
        self.threshold = config.threshold
    
    def predict(
        self, frames: list[np.ndarray]
    ) -> dict[int, tuple[torch.tensor, float]]:
        features = {}
        for i in tqdm(range(0, len(frames) - self.win_size // 2 + 1, self.hop_length), "Getting Features"):
            ft, conf = self.extractor(frames[i : i + self.win_size])
            features[i] = (ft, conf)
        return features
    
    def __call__(self, frames: list[np.ndarray], plot_path="probs.png") -> dict[int, torch.Tensor]:
        features = self.predict(frames)
        lambda_ = 0.0
        
        # All frame indices and their confidences for plotting
        all_indices = sorted(features.keys())
        all_confidences = [features[idx][1] for idx in all_indices]
        frame_idxs = []
        for i in range(1, len(all_confidences)):
            all_confidences[i] = min(1, lambda_*all_confidences[i-1] + all_confidences[i])
            if all_confidences[i] >= self.threshold:
                frame_idxs.append((all_confidences[i], -all_indices[i]))
        
        # Get high confidence frames
        frame_idxs = sorted(frame_idxs)
        
        good_preds = []
        with tqdm(total=len(frame_idxs), desc="Applying NMS") as pbar:
            while len(frame_idxs) > 0:
                before = len(frame_idxs)
                _, frame_idx = frame_idxs.pop()
                frame_idx *= -1
                good_preds.append(frame_idx)
                frame_idxs = [
                    (prob, idx)
                    for prob, idx in frame_idxs
                    if abs(-1*idx - frame_idx) > self.win_size - self.overlap
                ]
                pbar.update(before - len(frame_idxs))
        
        # Create the plot
        self.plot_probabilities(all_indices, all_confidences, good_preds, plot_path)
        
        return {idx: features[idx][0] for idx in good_preds}
    
    def plot_probabilities(self, indices, confidences, selected_indices, plot_path):
        plt.figure(figsize=(12, 6))
        
        # Plot all probabilities
        plt.plot(indices, confidences, 'b-', alpha=0.5, label='All Windows')
        
        # Highlight selected windows
        selected_confidences = [conf for idx, conf in zip(indices, confidences) if idx in selected_indices]
        plt.scatter(selected_indices, [confidences[indices.index(idx)] for idx in selected_indices], 
                   color='red', s=50, label='Selected Windows')
        
        # Add threshold line
        plt.axhline(y=self.threshold, color='green', linestyle='--', label=f'Threshold ({self.threshold})')
        
        # Add window regions
        for idx in selected_indices:
            plt.axvspan(idx, idx + self.win_size, alpha=0.2, color='lightgreen')
        
        plt.xlabel('Frame Index')
        plt.ylabel('Confidence Score')
        plt.title('NMS Window Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()