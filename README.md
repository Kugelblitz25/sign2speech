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

### Feature Extractor (Modified I3D)

---

- Download the raw WLASL-2000 Dataset from kaggle

```shell
curl -L -o archive.zip 'https://www.kaggle.com/api/v1/datasets/download/risangbaskoro/wlasl-processed'
```

- Unzip the `archive.zip` into `data/raw` folder.

- Generate the spectrograms and select Top 100 classes with highest number of instances in the dataset

```shell
python3 models/generator/preprocessing/spec_gen.py
```

- Verify the videos and split into train and validation sets

```shell
python3 models/extractor/preprocessing/verify.py
```

- Augment the videos to generate more data

```shell
python3 models/extractor/preprocessing/augmentation.py --datafile data/raw/train_100.json
python3 models/extractor/preprocessing/augmentation.py --datafile data/raw/test_100.json
```

- Train the model

```shell
python3 models/extractor/train.py
```

Adjust the train parameters in `models/extractor/config.json`

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

## Testing

Chenge the input video location and output audio location in `test.py`

```shell
python3 test.py
```
