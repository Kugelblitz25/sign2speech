from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset
from models.extractor.model import ModifiedI3D
from utils.model import load_model_weights, create_path
from utils.config import Config


def plot_tsne(
    train_features: list[np.array],
    test_features: list[np.array],
    train_labels: list[str],
    test_labels: list[str],
    save_path: Path,
):
    # Convert to numpy arrays
    train_features = np.array(train_features, dtype=np.float64)
    test_features = np.array(test_features, dtype=np.float64)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(
        np.concatenate([train_features, test_features])
    ).astype(np.float64)
    train_embedding = embedded[: len(train_features)]
    test_embedding = embedded[len(train_features) :]

    unique_labels = np.unique(train_labels + test_labels)
    test_colors = [np.where(unique_labels == label)[0][0] for label in test_labels]

    # Perform clustering on training embeddings
    from sklearn.svm import SVC

    clf = SVC(kernel="rbf", random_state=42)
    clf.fit(train_embedding, train_labels)

    # Create a figure
    plt.figure(figsize=(20, 20))

    # Create a meshgrid to visualize cluster regions
    x_min, x_max = train_embedding[:, 0].min() - 1, train_embedding[:, 0].max() + 1
    y_min, y_max = train_embedding[:, 1].min() - 1, train_embedding[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, (x_max - x_min) / 300),
        np.arange(y_min, y_max, (y_max - y_min) / 300),
    )

    # Predict cluster labels for all points in meshgrid
    mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
    Z = clf.predict(mesh_points)
    Z = np.array([np.where(unique_labels == z)[0][0] for z in Z])
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    plt.contourf(xx, yy, Z, levels=10, alpha=0.5, cmap="tab10")
    plt.contour(xx, yy, Z, colors="black", alpha=0.4, linewidths=1)

    # Plot test points
    plt.scatter(
        test_embedding[:, 0],
        test_embedding[:, 1],
        c=test_colors,
        marker="o",
        cmap="tab10",
        alpha=1,
        edgecolors="black",
        linewidth=1,
        s=200,
    )

    plt.title("t-SNE Visualization with Clustering Regions")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.axis("equal")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"t-SNE plot with clustering saved to {save_path}")


def plot_confusion_matrix(
    true_labels: list[str], predicted_labels: list[str], save_path: Path
):
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_percentage = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100).round(
        1
    )  # Plot confusion matrix as percentages
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt=".1f",
        cmap="plasma",
        yticklabels=np.unique(true_labels),
        xticklabels=np.unique(true_labels),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def extract_features(model, dataloader: DataLoader):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    features = []
    labels = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for inputs, lbls in tqdm(dataloader, desc="Feature Extraction"):
            inputs, lbls = inputs.to(device), lbls.to(device)
            fts, outputs = model(inputs)

            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            fts = fts.flatten().cpu().numpy().reshape(dataloader.batch_size, -1)

            # Store features and metadata
            features.extend(fts)
            labels.extend(lbls.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    # Convert to numpy arrays
    idx_to_class = {idx: cls for cls, idx in dataloader.dataset.class_to_idx.items()}
    labels = [idx_to_class[int(label)] for label in labels]
    predictions = [idx_to_class[int(pred)] for pred in predictions]

    return features, labels, predictions


def main(
    train_data_path: str,
    test_data_path: str,
    video_root: str,
    weights: str,
    output_path: str,
):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    train_dataset = WLASLDataset(train_data, video_root)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_dataset = WLASLDataset(test_data, video_root)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    num_classes = 10
    model = ModifiedI3D(num_classes).to(device)
    model = load_model_weights(model, weights, device)
    print(f"Num Classes: {num_classes}")

    save_path = create_path(output_path)

    train_features, train_labels, train_predictions = extract_features(
        model, train_loader
    )
    test_features, test_labels, test_predictions = extract_features(model, test_loader)

    plot_tsne(
        train_features, test_features, train_labels, test_labels, save_path / "tsne.png"
    )
    plot_confusion_matrix(train_labels, train_predictions, save_path / "train_cfm.png")
    plot_confusion_matrix(test_labels, test_predictions, save_path / "test_cfm.png")


if __name__ == "__main__":
    config = Config("Visualization")

    main(
        config.data.processed.csvs.train,
        config.data.processed.csvs.test,
        config.data.processed.videos,
        config.pipeline.extractor_weights,
        "experiments",
    )
