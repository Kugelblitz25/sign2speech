import argparse
import yaml
from pathlib import Path

def create_path(path_str: str):
    path = Path(path_str)
    if path.is_dir():
        path.mkdir(exists_ok=True, parents=True)
    else:
        path.parent.mkdir(exists_ok=True, parents=True)
    return path


def load_config(desc: str):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    return config