# Sign Language to Speech Synthesis

This project translates sign language videos into audible speech. It uses a deep learning pipeline consisting of a video feature extractor, a transformer model to convert visual features into spectrograms, and an audio generator to synthesize speech from these spectrograms.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Running the Gradio Interface](#running-the-gradio-interface)
  - [Running Tests](#running-tests)
- [Configuration](#configuration)
- [Training](#training)
  - [Dataset](#dataset)
  - [Training Pipeline](#training-pipeline)
  - [Individual Model Training](#individual-model-training)
- [Utilities](#utilities)
- [License](#license)

## Features

-   **Sign Language to Speech:** Converts input sign language videos into spoken audio.
-   **Gradio Interface:** Easy-to-use web interface for demonstrations.
-   **Modular Architecture:** Composed of distinct models for feature extraction, transformation, and audio generation.
-   **Configurable:** Training and model parameters can be adjusted via a central configuration file.
-   **Extensible:** Provides scripts for training, testing, and utility functions for further development.

## Project Structure

Here's a high-level overview of the key directories and files:

-   `config.yaml`: Central configuration file for data paths, model parameters, and training settings.
-   `models/`: Contains the core deep learning models.
    -   `__init__.py`: Defines the main `Sign2Speech` pipeline class.
    -   `extractor/`: Video feature extraction model (e.g., I3D).
    -   `transformer/`: Model to transform video features into spectrograms.
    -   `generator/`: Model/utility to generate audio from spectrograms.
    -   `nms.py`: Non-Maximum Suppression for selecting relevant video frames/windows.
-   `utils/`: Utility scripts for configuration loading, logging, model saving/loading, plotting, etc.
-   `ui.py`: Runs the Gradio web interface for live demonstration.
-   `test.py`: Script for evaluating the end-to-end Sign2Speech pipeline.
-   `trainer.sh`: Shell script to run the complete training pipeline.
-   `transformer.sh`: Shell script for specific training/testing tasks related to the transformer model.
-   `requirements.txt`: List of Python dependencies (users should prefer `pyproject.toml`).
-   `pyproject.toml`: Project definition file, includes dependencies (preferred for setup).
-   `LICENSE`: Project license (Apache License 2.0).

## Installation

### Prerequisites

-   Python (>=3.10 recommended, as per `pyproject.toml`).
-   It's recommended to use a virtual environment.
-   [uv](https://github.com/astral-sh/uv): A fast Python package installer and resolver (optional but recommended for consistency with project scripts like `transformer.sh`). If you don't use `uv`, you can use `pip`.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and activate a virtual environment:**
    Using `venv` (standard Python):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    Or using Conda:
    ```bash
    conda create -n sign2speech_env python=3.10
    conda activate sign2speech_env
    ```

3.  **Install dependencies:**
    The preferred method is using `uv` with `pyproject.toml` for development or general use:
    ```bash
    # Install uv if you haven't already:
    # pip install uv
    # or follow instructions on uv's GitHub page.

    # Install project dependencies using uv:
    uv pip install -e .[test]
    ```
    The `.[test]` will install main dependencies plus those needed for testing, as defined in `pyproject.toml`.
    If you prefer not to use `uv`, you can use `pip` with `pyproject.toml` (requires a recent version of pip that supports PEP 517):
    ```bash
    pip install -e .[test]
    ```
    Alternatively, you can use the `requirements.txt` (though `pyproject.toml` is more up-to-date):
    ```bash
    uv pip install -r requirements.txt
    # or
    # pip install -r requirements.txt
    ```

    *Note on `PYTHONPATH`*: The original README and `trainer.sh` mention setting `export PYTHONPATH=$(pwd)`. If you install the package in editable mode (`pip install -e .` or `uv pip install -e .`), this should generally not be necessary as the project's modules will be discoverable.

## Usage

### Running the Gradio Interface

To start the Gradio web interface for a live demonstration:

```bash
python ui.py
```

This will launch a local web server, and you can access the interface through your browser (usually at `http://127.0.0.1:7860`). You can upload a sign language video to get the synthesized audio.

### Running Tests

The project includes a script to evaluate the performance of the Sign2Speech pipeline:

```bash
python test.py --config_file config.yaml
```

-   The `test.py` script uses settings from `config.yaml` to load data and models.
-   It typically processes test videos, generates audio, transcribes it using an ASR model (like Whisper), and calculates metrics such as WER, CER, and BLEU against reference glosses.
-   Ensure that paths to test data and model weights in your `config.yaml` are correctly set up.

There is also a test script specific to the transformer model located at `models/transformer/test.py`. This can be run via `python models/transformer/test.py` or using the `transformer.sh` script.

## Configuration

The main configuration for the project is done through the `config.yaml` file. This file is structured using YAML and includes sections for:

-   `n_words`: Number of words/classes the models are trained on.
-   `data`: Paths for raw and processed datasets, including videos, CSV files for splits (train, validation, test), class lists, spectrograms, and video features.
    -   *Note on Dataset:* The `config.yaml` currently refers to `data/asl-citizen/`. The original README mentioned WLASL-2000. You will need to ensure your dataset is structured as expected by the configuration and processing scripts.
-   `extractor`: Settings for the feature extractor model (e.g., I3D), including augmentation, checkpoint paths, and training parameters.
-   `transformer`: Settings for the feature transformer model, including paths to extractor weights, checkpoint paths, and training parameters.
-   `generator`: Settings for the audio generator, including checkpoint paths and maximum audio length.
-   `pipeline`: Configuration for the end-to-end inference pipeline, including NMS (Non-Maximum Suppression) parameters and paths to the pre-trained weights for the extractor and transformer models.
    -   *Note on Model Weights:* The `pipeline` section in `config.yaml` specifies paths for `extractor_weights` and `transformer_weights`. Ensure these paths point to valid, trained model checkpoint files. The default configuration might point to paths like `experiments/combined/checkpoints/...`. If the `experiments` folder is not available or these weights are not present, you will need to train your own models or obtain pre-trained weights and update these paths accordingly.

The `utils/config.py` script provides dataclasses that define the structure of this configuration and helper functions to load it.

## Training

The project provides scripts and a framework for training the models from scratch or fine-tuning them.

### Dataset

-   The training process requires a dataset of sign language videos with corresponding glosses (text transcriptions of the signs).
-   The `config.yaml` file specifies paths to training, validation, and test data. You will need to prepare your dataset according to the expected format and update these paths. Common datasets for sign language include WLASL, ASLLVD, etc. The current `config.yaml` seems set up for a dataset named "asl-citizen".
-   Preprocessing scripts are available in `models/extractor/preprocessing/` and `models/generator/preprocessing/`. These handle tasks like:
    -   `models/generator/preprocessing/spec_gen.py`: Generating spectrograms.
    -   `models/extractor/preprocessing/verify.py`: Verifying video files and creating data splits.
    -   `models/extractor/preprocessing/augmentation.py`: Augmenting video data to increase dataset size.
    -   `models/transformer/preprocessing/features_gen.py`: Generating features from videos using a trained feature extractor.

### Training Pipeline

The `trainer.sh` script provides an automated way to run the entire training process:

```bash
chmod +x trainer.sh
./trainer.sh
```

This script typically performs the following steps (as seen in its content):
1.  Generates spectrograms.
2.  Verifies and splits video data.
3.  Augments video data.
4.  Trains the feature extractor (`models/extractor/train.py`).
5.  Generates features for the transformer model (`models/transformer/preprocessing/features_gen.py`).
6.  Trains the feature transformer (`models/transformer/train.py`).
7.  Runs a test (`test.py`).

Ensure your `config.yaml` is correctly set up before running `trainer.sh`.

### Individual Model Training

You can also train individual components of the pipeline:

-   **Feature Extractor:**
    ```bash
    python models/extractor/train.py --config_file config.yaml
    ```
-   **Feature Transformer:**
    ```bash
    python models/transformer/train.py --config_file config.yaml
    # Or for cosine annealing variant:
    # python models/transformer/train_cosine.py --config_file config.yaml
    ```
    The `transformer.sh` script also provides examples of running these training scripts, potentially with `uv`.

Training parameters such as epochs, batch size, learning rate, etc., can be adjusted in the `config.yaml` file under the respective model's `training` section.

## Utilities

The `utils/` directory contains several helpful scripts:

-   `common.py`: Logger setup (`get_logger`) and path creation utilities (`create_path`).
-   `config.py`: Loads and parses the `config.yaml` file into typed dataclasses.
-   `create_test_videos.py`: Script to generate concatenated test videos from a dataset, useful for creating specific evaluation samples.
-   `model.py`: Includes an `EarlyStopping` class for training, and functions to `save_model` and `load_model_weights`.
-   `plot_exp.py`: Parses training log files (loss, accuracy) and plots them using `matplotlib`.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
