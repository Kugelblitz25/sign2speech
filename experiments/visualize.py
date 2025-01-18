import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.extractor.dataset import WLASLDataset, video_transform
from models.extractor.model import ModifiedI3D
from pathlib import Path
from utils import load_model_weights

def plot_tsne(features, labels, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    features_embedded = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    scatter = plt.scatter(features_embedded[:, 0], features_embedded[:, 1], 
                         c=[np.where(unique_labels == l)[0][0] for l in labels])
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Features")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.savefig(save_path / "tsne_visualization.png")
    plt.close()
    print(f"t-SNE plot saved to {save_path / 'tsne_visualization.png'}")

def plot_confusion_matrix(true_labels, predicted_labels, save_path):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved to {save_path / 'confusion_matrix.png'}")

def extract_features(model, test_loader, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_features = []
    all_video_ids = []
    all_labels = []
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(
            tqdm(test_loader, desc=f"Feature Extraction")
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            features, outputs = model(inputs)
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            features = features.cpu().numpy()
            # Get video IDs for this batch
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + inputs.size(0)
            batch_video_ids = [
                test_loader.dataset.data[i]["video_id"]
                for i in range(start_idx, min(end_idx, len(test_loader.dataset)))
            ]
            
            # Store features and metadata
            all_features.extend(features)
            all_video_ids.extend(batch_video_ids)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    # Convert to numpy arrays
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    accuracy = np.mean(all_labels == all_predictions)

    classes = sorted(list(set(item["gloss"] for item in test_loader.dataset.data)))
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}
    all_labels = [idx_to_class[label] for label in all_labels]
    all_predictions = [idx_to_class[pred] for pred in all_predictions]
    
    # Generate visualizations
    plot_tsne(all_features, all_labels, save_path)
    plot_confusion_matrix(all_labels, all_predictions, save_path)
    
    # Calculate and save accuracy
    print(f"Overall Accuracy: {accuracy:.4f}")
    

def main(test_data: str, video_root: str, weights: str, save_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    with open(test_data) as f:
        test_data = json.load(f)
        
    transform = video_transform()
    test_dataset = WLASLDataset(test_data, video_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    num_classes = 10
    model = ModifiedI3D(num_classes).to(device)
    model = load_model_weights(model, weights)
    print(f"Num Classes: {num_classes}")

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    
    extract_features(model, test_loader, save_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Feature Generation for Spectrogram Generation"
    )
    parser.add_argument(
        "--datafile",
        type=str,
        default="data/processed/extractor/train_100.json",
        help="Path to the testing data JSON file",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="data/processed/extractor/videos",
        help="Directory containing videos",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="models/extractor/checkpoints/checkpoint_final.pt",
        help="Path to load the model weights",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments",
        help="Path to save the extracted features CSV file",
    )
    args = parser.parse_args()
    main(args.datafile, args.video_root, args.weights_path, args.save_path)
