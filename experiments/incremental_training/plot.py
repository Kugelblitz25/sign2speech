import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from utils.common import create_path


def parse_log_file(file_path):
    """Parse the log file and extract metrics for different experiments."""
    with open(file_path, "r") as f:
        log_content = f.read()

    experiment_headers = re.findall(
        r"\[INFO\] \d+/\d+/\d+ \d+:\d+ - Processing (\d+) words", log_content
    )  # [15:-3]

    experiment_blocks = re.split(
        r"\[INFO\] \d+/\d+/\d+ \d+:\d+ - Processing \d+ words", log_content
    )[1:]  # [16:-3]

    experiments = []

    for exp_name, block in zip(experiment_headers, experiment_blocks):
        metrics = re.findall(
            r"\[INFO\] \d+/\d+/\d+ \d+:\d+ - Train Loss: ([\d\.]+), Val Loss: ([\d\.]+), Train Accuracy: ([\d\.]+)%, Val Accuracy: ([\d\.]+)%",
            block,
        )

        final_accs = re.findall(
            r"\[INFO\] \d+/\d+/\d+ \d+:\d+ - Train Acc: ([\d\.]+), Val Acc: ([\d\.]+)",
            block,
        )[0]

        time_taken = re.findall(
            r"\[INFO\] \d+/\d+/\d+ \d+:\d+ - Time taken: ([\d\.]+)s", block
        )

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        for train_loss, val_loss, train_acc, val_acc in metrics:
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            train_accs.append(float(train_acc))
            val_accs.append(float(val_acc))

        experiments.append(
            {
                "n_words": int(exp_name),
                "epochs": len(train_accs),
                "final_accs": final_accs,
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_acc": train_accs,
                "val_acc": val_accs,
                "time_taken": float(time_taken[0]) if time_taken else None,
            }
        )

    return experiments


def plot_metrics(experiments, output_dir):
    """Plot training and validation metrics for all experiments."""
    output_dir = create_path(output_dir)
    max_epochs = max([exp["epochs"] for exp in experiments])

    colors = ["b", "r", "g", "c", "m", "y", "k"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    axes = axes.flatten()

    axes[0].set_title("Training Accuracy")
    axes[1].set_title("Validation Accuracy")

    for i, exp_data in enumerate(experiments):
        color = colors[i % len(colors)]
        n_words = exp_data["n_words"]
        epochs = range(1, exp_data["epochs"] + 1)
        axes[0].plot(
            epochs,
            exp_data["train_acc"],
            f"{color}-",
            label=f"{n_words} words",
            marker=".",
        )
        axes[1].plot(
            epochs,
            exp_data["val_acc"],
            f"{color}--",
            label=f"{n_words} words",
            marker=".",
        )

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim((0, 110))
        ax.set_xticks(range(1, max_epochs + 2, 1))
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="lower right")

    fig.suptitle("Model Accuracy across Training Epochs", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    plt.savefig(output_dir / "accuracy_plot.png", dpi=300, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    axes = axes.flatten()

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")

    for i, exp_data in enumerate(experiments):
        color = colors[i % len(colors)]
        n_words = exp_data["n_words"]
        epochs = range(1, exp_data["epochs"] + 1)
        axes[0].plot(
            epochs,
            exp_data["train_loss"],
            f"{color}-",
            label=f"{n_words} words",
            marker=".",
        )
        axes[1].plot(
            epochs,
            exp_data["val_loss"],
            f"{color}--",
            label=f"{n_words} words",
            marker=".",
        )

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(range(1, max_epochs + 2, 1))
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper right")

    fig.suptitle("Model Loss across Training Epochs", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    plt.savefig(output_dir / "loss_plot.png", dpi=300, bbox_inches="tight")

    words = [exp["n_words"] for exp in experiments]
    time = [exp["time_taken"] / 3600 for exp in experiments]

    X = np.array(words).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, time)

    x = np.linspace(0, 500, 50)
    y = model.predict(x.reshape(-1, 1))

    plt.figure(figsize=(10, 8))
    plt.scatter(words, time, marker="o", color="r")
    plt.plot(x, y, "g--", label="best fit")
    plt.xlabel("Number of Words")
    plt.ylabel("Time Taken(hr)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title("Time Taken for Training", fontsize=16)
    plt.suptitle(f"Total time taken: {sum(time):.2f} hr")

    plt.savefig(output_dir / "time_plot.png", dpi=300, bbox_inches="tight")

    final_train_acc = [float(exp["final_accs"][0]) for exp in experiments]
    final_val_acc = [float(exp["final_accs"][1]) for exp in experiments]

    plt.figure(figsize=(10, 6))
    plt.plot(words, final_train_acc, marker="o", color="g", label="Train")
    plt.plot(words, final_val_acc, marker="s", color="b", label="Val")
    plt.xlabel("Number of Words")
    plt.ylabel("Accuracy(%)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(0, 110)
    plt.title("Accuracy vs Number of Words", fontsize=16)

    plt.savefig(output_dir / "acc_v_n_plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    experiments = parse_log_file("logs/standalone_train.log")
    plot_metrics(experiments, "experiments/incremental_training/std")
