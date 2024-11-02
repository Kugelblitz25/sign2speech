#!/bin/bash

source .venv/bin/activate
export PYTHONPATH=$(pwd)

echo "Generate spectrograms for Top 100 words"
python3 models/generator/preprocessing/spec_gen.py

echo "Loading and Verifying videos for 100 signs"
python3 models/extractor/preprocessing/verify.py

echo "Augment the videos to generate more data"
python3 models/extractor/preprocessing/augmentation.py --datafile data/processed/extractor/train_100.json
python3 models/extractor/preprocessing/augmentation.py --datafile data/processed/extractor/test_100.json

echo "Training Feature Extractor"
python3 models/extractor/train.py

echo "Creating Features"
python3 models/transformer/preprocessing/features_gen.py

echo "Training Feature Transformer"
python3 models/transformer/train.py