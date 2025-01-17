import torch

def save_model(model, config, loss, path):
    save_data = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "loss": loss,
    }
    torch.save(save_data, path)


def load_model_weights(model, path):
    print(f"Weights Loaded from {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.load(path, weights_only=True, map_location=device)[
        "model_state_dict"
    ]
    model.load_state_dict(weights)
    return model