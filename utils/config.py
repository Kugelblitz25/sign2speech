import argparse
from dataclasses import dataclass

import yaml


@dataclass
class Splits:
    train: str
    val: str
    test: str


@dataclass
class RawData:
    videos: str
    csvs: Splits


@dataclass
class ProcessedData:
    videos: str
    csvs: Splits
    classlist: str
    specs: str
    vid_features: Splits


@dataclass
class Data:
    raw: RawData
    processed: ProcessedData


@dataclass
class Training:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    scheduler_patience: int
    scheduler_factor: float
    enable_earlystop: bool
    num_workers: int


@dataclass
class ExtractorTraining(Training):
    freeze: int


@dataclass
class Extractor:
    num_augmentations: int
    checkpoints: str
    model: str
    training: ExtractorTraining


@dataclass
class TransformerTraining(Training):
    pass


@dataclass
class Transformer:
    extractor_weights: str
    checkpoints: str
    training: TransformerTraining


@dataclass
class Generator:
    checkpoints: str


@dataclass
class NMS:
    win_size: int
    hop_length: int
    overlap: int
    threshold: float


@dataclass
class Pipeline:
    extractor_weights: str
    transformer_weights: str


@dataclass
class Config:
    n_words: int
    data: Data
    extractor: Extractor
    transformer: Transformer
    generator: Generator
    nms: NMS
    pipeline: Pipeline


# Function to load config from YAML file
def load_config(desc: str) -> Config:
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    # Create nested objects
    raw_csvs = config_dict["data"]["raw"]["csvs"]
    raw_data = RawData(
        videos=config_dict["data"]["raw"]["videos"],
        csvs=Splits(
            train=raw_csvs["train"], val=raw_csvs["val"], test=raw_csvs["test"]
        ),
    )

    processed_csvs = config_dict["data"]["processed"]["csvs"]
    fearures_csvs = config_dict["data"]["processed"]["vid_features"]
    processed_data = ProcessedData(
        videos=config_dict["data"]["processed"]["videos"],
        csvs=Splits(
            train=processed_csvs["train"],
            val=processed_csvs["val"],
            test=processed_csvs["test"],
        ),
        classlist=config_dict["data"]["processed"]["classlist"],
        specs=config_dict["data"]["processed"]["specs"],
        vid_features=Splits(
            train=fearures_csvs["train"],
            val=fearures_csvs["val"],
            test=fearures_csvs["test"],
        ),
    )

    data = Data(raw=raw_data, processed=processed_data)

    extractor_training = ExtractorTraining(
        freeze=config_dict["extractor"]["training"]["freeze"],
        epochs=config_dict["extractor"]["training"]["epochs"],
        batch_size=config_dict["extractor"]["training"]["batch_size"],
        lr=config_dict["extractor"]["training"]["lr"],
        weight_decay=config_dict["extractor"]["training"]["weight_decay"],
        patience=config_dict["extractor"]["training"]["patience"],
        scheduler_patience=config_dict["extractor"]["training"]["scheduler_patience"],
        scheduler_factor=config_dict["extractor"]["training"]["scheduler_factor"],
        enable_earlystop=config_dict["extractor"]["training"]["enable_earlystop"],
        num_workers=config_dict["extractor"]["training"]["num_workers"],
    )

    extractor = Extractor(
        num_augmentations=config_dict["extractor"]["num_augmentations"],
        checkpoints=config_dict["extractor"]["checkpoints"],
        model=config_dict["extractor"]["model"],
        training=extractor_training,
    )

    transformer_training = TransformerTraining(
        epochs=config_dict["transformer"]["training"]["epochs"],
        batch_size=config_dict["transformer"]["training"]["batch_size"],
        lr=config_dict["transformer"]["training"]["lr"],
        weight_decay=config_dict["transformer"]["training"]["weight_decay"],
        patience=config_dict["transformer"]["training"]["patience"],
        scheduler_patience=config_dict["transformer"]["training"]["scheduler_patience"],
        scheduler_factor=config_dict["transformer"]["training"]["scheduler_factor"],
        enable_earlystop=config_dict["transformer"]["training"]["enable_earlystop"],
        num_workers=config_dict["transformer"]["training"]["num_workers"],
    )

    transformer = Transformer(
        extractor_weights=config_dict["transformer"]["extractor_weights"],
        checkpoints=config_dict["transformer"]["checkpoints"],
        training=transformer_training,
    )

    generator = Generator(checkpoints=config_dict["generator"]["checkpoints"])

    nms = NMS(
        win_size=config_dict["nms"]["win_size"],
        hop_length=config_dict["nms"]["hop_length"],
        overlap=config_dict["nms"]["overlap"],
        threshold=config_dict["nms"]["threshold"],
    )

    pipeline = Pipeline(
        extractor_weights=config_dict["pipeline"]["extractor_weights"],
        transformer_weights=config_dict["pipeline"]["transformer_weights"],
    )

    return Config(
        n_words=config_dict["n_words"],
        data=data,
        extractor=extractor,
        transformer=transformer,
        generator=generator,
        nms=nms,
        pipeline=pipeline,
    )
