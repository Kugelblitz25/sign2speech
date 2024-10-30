import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset, video_transform
from models.extractor.model import ModifiedI3D
from utils import load_model_weights


def extract_features(model, test_loader, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    all_features = []
    all_video_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc=f"Feature Extraction")):
            inputs, labels = inputs.to(device), labels.to(device)
            features, _ = model(inputs)
            features = features.cpu().numpy()
            
            # Get video IDs for this batch
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + inputs.size(0)
            batch_video_ids = [test_loader.dataset.data[i]['video_id'] for i in range(start_idx, min(end_idx, len(test_loader.dataset)))]
            
            # Store features and metadata
            all_features.extend(features)
            all_video_ids.extend(batch_video_ids)
    
    feature_cols = [f'feature_{i}' for i in range(features.shape[1])]
    df = pd.DataFrame(all_features, columns=feature_cols)
    df['video_id'] = all_video_ids
    video_to_gloss = {item['video_id']: item['gloss'] for item in test_loader.dataset.data}
    df['gloss'] = df['video_id'].map(video_to_gloss)

    xtrain, xval = train_test_split(df, test_size=0.2)
    xtrain.to_csv(save_path / 'features_train.csv', index=False)
    xval.to_csv(save_path / 'features_val.csv', index=False)
    print(f"Features saved to {save_path}")

def main(test_data: str, video_root: str, weights: str, save_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    with open(test_data) as f:
        test_data = json.load(f)
    
    transform = video_transform()
    test_dataset = WLASLDataset(test_data, video_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    num_classes = 100
    model = ModifiedI3D(num_classes).to(device)
    model = load_model_weights(model, weights)
    print(f"Num Classes: {num_classes}")
    
    extract_features(model, test_loader, save_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Generation for Spectrogram Generation')
    
    parser.add_argument('--datafile', type=str, default='data/raw/train_100.json', help='Path to the testing data JSON file')
    parser.add_argument('--video_root', type=str, default='data/raw/videos', help='Directory containing videos')
    parser.add_argument('--weights_path', type=str, default='models/extractor/checkpoints/checkpoint_final.pt', help='Path to load the model weights')
    parser.add_argument('--save_path', type=str, default='data/processed/transformer', help='Path to save the extracted features CSV file')
    
    args = parser.parse_args()
    main(args.datafile, args.video_root, args.weights_path, args.save_path)
