# Sign Language Video to Audio without text

## Training

### Feature Extractor (Modified I3D)

---

- Download the raw WLASL-2000 Dataset from kaggle

```shell
curl -L -o archive.zip https://www.kaggle.com/api/v1/datasets/download/risangbaskoro/wlasl-processed`
```

- Unzip the `archive.zip` into `data/raw` folder.

- Verify the videos and split into train and validation sets with 100 classes

```shell
python3 models/extractor/preprocessing/verify.py
```

- Augment the videos to generate more data

```shell
mkdir -p data/processed/extractor
python3 models/extractor/preprocessing/augmentation.py --datafile data/raw/train_100.json
python3 models/extractor/preprocessing/augmentation.py --datafile data/raw/test_100.json
```

- Train the model

```shell
python3 models/extractor/train.py
```

Adjust the train parameters in `models/extractor/config.json`

> **Note**: If you get `ModuleNotFoundErrorModuleNotFoundError: No module named 'models'`, run
>
> ```shell
> export PYTHONPATH=$(pwd)
> ```
>

### Feature Transformer

---

- Save the features from the WLASL dataset using the above trained model.

```shell
mkdir -p data/processed/transformer
python3 models/transformer/preprocessing/features_gen.py
```

- Train the model

```shell
python3 models/transformer/train.py
```

Adjust the train parameters in `models/transformer/config.json`

## Testing

Chenge the input video location and output audio location in `test.py`

```shell
python3 test.py
```
