<div align="center">

<h1> IsoSignVid2Aud </h1>

### Sign Language Video to Audio Conversion without Text Intermediaries

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2510.07837) [![Project Page](https://img.shields.io/badge/Project_Page-Visit-green.svg)](https://kugelblitz25.github.io/sign2speech/)
<br/>

**Harsh Kavediya**<sup>&dagger;,1</sup>, 
**Vighnesh Nayak**<sup>&dagger;,1</sup>, 
**Bheeshm Sharma**<sup>2</sup>
**Balamurugan Palaniappan**<sup>2</sup>

<sup>1</sup>Department of Mechanical Engineering &nbsp;&nbsp; <sup>2</sup>Department of IEOR

Indian Institute of Technology Bombay

**Accepted at 5th International Conference on AI-ML Systems [2025]**


</div>

## 📝 Abstract

> Sign language to spoken language audio translation is important to connect the hearing- and speech-challenged humans with others. We consider sign language videos with isolated sign sequences rather than continuous grammatical signing. Such videos are useful in educational applications and sign prompt interfaces. Towards this, we propose IsoSignVid2Aud, a novel end-to-end framework that translates sign language videos with a sequence of possibly non-grammatic continuous signs to speech without requiring intermediate text representation, providing immediate communication benefits while avoiding the latency and cascading errors inherent in multi-stage translation systems. Our approach combines an I3D-based feature extraction module with a specialized feature transformation network and an audio generation pipeline, utilizing a novel Non-Maximal Suppression (NMS) algorithm for the temporal detection of signs in non-grammatic continuous sequences. Experimental results demonstrate competitive performance on ASL-Citizen-1500 and WLASL-100 datasets with Top-1 accuracies of 72.01% and 78.67%, respectively, and audio quality metrics (PESQ: 2.67, STOI: 0.73) indicating intelligible speech output.

## 📋Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#🚀-usage)
  - [Running the Gradio Interface](#running-the-gradio-interface)
  - [Running Tests](#running-tests)
- [Configuration](#configuration)
- [Training](#training)
  - [Dataset](#dataset)
  - [Combined Training](#combined-training)
  - [Individual Model Training](#individual-model-training)
  - [Training Pipeline](#training-pipeline)
- [Utilities](#utilities)
- [Citation](#citation)
- [License](#license)

## ✨ Features

-   **Sign Language to Speech:** Converts input sign language videos into spoken audio.
-   **Gradio Interface:** Easy-to-use web interface for demonstrations.
-   **Modular Architecture:** Composed of distinct models for feature extraction, transformation, and audio generation.
-   **Configurable:** Training and model parameters can be adjusted via a central configuration file.
-   **Extensible:** Provides scripts for training, testing, and utility functions for further development.

## 📂 Project Structure

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

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kugelblitz25/sign2speech
    cd sign2speech
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
    # Install project dependencies using uv:
    uv pip install -r pyproject.toml --extra test
    ```
    The `.[test]` will install main dependencies plus those needed for testing, as defined in `pyproject.toml`.
    If you prefer not to use `uv`, you can use `pip` with `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

>Note: If you get `ModuleNotFoundError: No module named 'models'`, run
>```bash
>export PYTHONPATH=$(pwd)
>```

## 🚀 Usage

### Running the Gradio Interface

To start the Gradio web interface for a live demonstration:

```bash
python ui.py
```

This will launch a local web server, and you can access the interface through your browser (usually at http://127.0.0.1:7860). You can upload a sign language video to get the synthesized audio.

### Running Tests

The project includes a script to evaluate the performance of the Sign2Speech pipeline:

- Create set of test videos with specified number of words per videos
```bash
python utils/create_test_videos.py --num_videos 10 --wpv 10 --output_dir test_videos
```
- Run tests
```bash
python test.py --videos_loc test_videos
```

-   The `test.py` script uses settings from `config.yaml` to load data and models.
-   It processes test videos, generates audio, transcribes it using an ASR model (like Whisper), and calculates metrics such as WER, CER, and BLEU against reference glosses.
-   Ensure that paths to test data and model weights in your `config.yaml` are correctly set up.

There are also a test scripts specific to the feature extractor and  feature transformer models located at `models/extractor/test.py` and `models/transformer/test.py` respectfully.

## ⚙️ Configuration

The main configuration for the project is done through the `config.yaml` file. This file is structured using YAML and includes sections for:

-   `n_words`: Number of words/classes the models are trained on.
-   `data`: Paths for raw and processed datasets, including videos, CSV files for splits (train, validation, test).
-   `extractor`: Settings for the feature extractor model (e.g., I3D), including augmentation, checkpoint paths, and training parameters.
-   `transformer`: Settings for the feature transformer model, including paths to extractor weights, checkpoint paths, and training parameters.
-   `generator`: Settings for the audio generator, including checkpoint paths and maximum audio length.
-   `pipeline`: Configuration for the end-to-end inference pipeline, including NMS (Non-Maximum Suppression) parameters and paths to the pre-trained weights for the extractor and transformer models.

The `utils/config.py` script provides dataclasses that define the structure of this configuration and helper functions to load it.

## 🏋️ Training

The project provides scripts and a framework for training the models from scratch or fine-tuning them.

### Dataset

- The training process requires a dataset of sign language videos with corresponding glosses (text transcriptions of the signs).
- The `config.yaml` file specifies paths to training, validation, and test data csvs. Prepare the dataset according to the expected format and update these paths. Common datasets for sign language include WLASL, ASL-Citizen, etc.

### Preprocessing

- Generate audio spectrograms for glosses
  ```bash
  python models/generator/preprocessing/spec_gen.py
  ```
- Remove corrupted videos
  ```bash
  python models/extractor/preprocessing/verify.py
  ```
- Augment the videos to generate more data
  ```bash
  python models/extractor/preprocessing/augmentation.py
  ```

### Combined Training

Run the combined training using following command 
  ```bash
  python models/train.py --config_file config.yaml
  ```

Modify the optimizer and scheduler parameters for the feature extractor and feature transformer in their corresponding sections in `config.yaml` file. Epochs and batch sizes are defined in separate combined section.

### Individual Model Training

You can also train individual components of the pipeline:

- **Feature Extractor:**
    ```bash
    python models/extractor/train.py --config_file config.yaml
    ```
- Generate features to train feature transformer
    ```bash
    python models/transformer/preprocessing/features_gen.py
    ```
- **Feature Transformer:**
    ```bash
    python models/transformer/train.py --config_file config.yaml
    # Or for cosine annealing variant:
    # python models/transformer/train.py --use_cosine true --config_file config.yaml
    ```

Training parameters such as epochs, batch size, learning rate, etc., can be adjusted in the `config.yaml` file under the respective model's `training` section.

### Training Pipeline

The `trainer.sh` script provides an automated way to run the individual training process:

```bash
chmod +x trainer.sh
./trainer.sh
```

This script performs the following steps:
1.  Generates spectrograms.
2.  Verifies and splits video data.
3.  Augments video data.
4.  Trains the feature extractor (`models/extractor/train.py`).
5.  Generates features for the feature transformer model (`models/transformer/preprocessing/features_gen.py`).
6.  Trains the feature transformer (`models/transformer/train.py`).

Ensure your `config.yaml` is correctly set up before running `trainer.sh`.

## 🛠️ Utilities

The `utils/` directory contains several helpful scripts:

-   `common.py`: Logger setup (`get_logger`) and subset creation utilities (`create_subset`).
-   `config.py`: Loads and parses the `config.yaml` file into typed dataclasses.
-   `create_test_videos.py`: Script to generate concatenated test videos from a dataset, useful for creating specific evaluation samples.
-   `model.py`: Includes an `EarlyStopping` class for training, and functions to `save_model` and `load_model_weights`.

## 🔗 Citation

If you find this work useful in your research, please cite:
```bib
@misc{kavediya2025isosignvid2audsignlanguagevideo,
      title={IsoSignVid2Aud: Sign Language Video to Audio Conversion without Text Intermediaries},
      author={Harsh Kavediya and Vighnesh Nayak and Bheeshm Sharma and Balamurugan Palaniappan},
      year={2025},
      eprint={2510.07837},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.07837}, 
}
```

## 📄 License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
