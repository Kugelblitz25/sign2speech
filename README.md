# Sign Language Video to Audio without text

## Training

### Install dependancies

---

- Create a virtual environment and install the project dependancies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download data

---

- Download the raw WLASL-2000 Dataset from kaggle

```shell
curl -L -o archive.zip 'https://www.kaggle.com/api/v1/datasets/download/risangbaskoro/wlasl-processed'
```

- Unzip the `archive.zip` into `data/raw` folder.

- Configurations are stored in `config.yaml` file.

### Feature Extractor (Modified I3D)

- Generate the spectrograms and select Top K classes with highest number of instances in the dataset

```shell
python3 models/generator/preprocessing/spec_gen.py
```

- Verify the videos and split into train and validation sets

```shell
python3 models/extractor/preprocessing/verify.py
```

- Augment the videos to generate more data

```shell
python3 models/extractor/preprocessing/augmentation.py
```

- Train the model

```shell
python3 models/extractor/train.py
```

> **Note**: If you get `ModuleNotFoundError: No module named 'models'`, run
>
> ```shell
> export PYTHONPATH=$(pwd)
> ```
>

### Feature Transformer

---

- Save the features from the WLASL dataset using the above trained model.

```shell
python3 models/transformer/preprocessing/features_gen.py
```

- Train the model

```shell
python3 models/transformer/train.py
```

Adjust the train parameters in `models/transformer/config.json`

### Training Script

---

Alternatively, after installing the dependacies and downloading the dataset into `data/raw` folder you can run `trainer.sh` to doc the complete training.

```bash
chmod +x trainer.sh
./trainer.sh
```

## Pre-trained weights

Download our pretrained weights from [Google Drive](https://drive.google.com/drive/folders/150wd1GsVxnIXq3btG0EEhhXS9gBYnJ2f?usp=sharing). Save the extractor weights to `models/extractor/checkpoints` and transformer weights to `models/transformer/checkpoints`.

## Testing

Change the input video location and output audio location in `test.py`

```shell
python3 test.py
```

## Gradio Interface

For gradio interface run

```shell
python3 ui.py
```
