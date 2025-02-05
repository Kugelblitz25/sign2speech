import logging

import argparse
from collections import namedtuple

import yaml


def get_logger(loc: str):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(loc)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(message)s", datefmt="%d/%m/%Y %H:%M"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class Config:
    def __init__(self, desc: str):
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

        self.config = self.parse_dict("RootCFG", config)

    def parse_dict(self, name: str, dictionary: dict):
        nt = namedtuple(name, dictionary.keys())
        processed_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                processed_dict[key] = self.parse_dict(key, value)
            elif isinstance(value, list):
                processed_dict[key] = [
                    self.parse_dict(f"item{i}", item)
                    if isinstance(item, dict)
                    else item
                    for i, item in enumerate(value)
                ]
            else:
                processed_dict[key] = value

        return nt(**processed_dict)

    def __getattr__(self, name: str):
        return getattr(self.config, name)
