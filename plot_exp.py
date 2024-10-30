import re
import matplotlib.pyplot as plt


def plot(ax: plt.Axes, x: list[float], y_train: list[float],  y_val: list[float], label: str) -> None:
    ax.plot(x, y_train, label=f'Train {label}')
    ax.plot(x, y_val, label=f'Validation {label}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.set_title(f'Training and Validation {label}')
    ax.legend()


def parse_log(log_content: str) -> tuple[list]:
    # Initialize lists to store extracted data
    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # Regular expression pattern for extracting data
    pattern = r"Epoch (\d+)/\d+, Train Loss: ([\d.]+), Val Loss: ([\d.]+), Train Accuracy: ([\d.]+)%, Val Accuracy: ([\d.]+)%"

    # Extract data
    for match in re.finditer(pattern, log_content):
        epochs.append(int(match.group(1)))
        train_loss.append(float(match.group(2)))
        val_loss.append(float(match.group(3)))
        train_acc.append(float(match.group(4)))
        val_acc.append(float(match.group(5)))
    
    return epochs, train_loss, val_loss, train_acc, val_acc

def plot_experiment(log_path: str) -> None:
    with open(log_path) as f:
        log_content = f.read()

    epochs, train_loss, val_loss, train_acc, val_acc = parse_log(log_content)

    fig, ax = plt.subplots(1, 2)
    ax=ax.flatten()
    plot(ax[0], epochs, train_loss, val_loss, 'Loss')
    plot(ax[1], epochs, train_acc, val_acc, 'Accuracy')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_experiment('experiments/output_aug_5.log')