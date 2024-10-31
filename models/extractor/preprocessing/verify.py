import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset, video_transform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
video_data = list[dict[str, str]]

def process_json(json_path: str, video_root: str, classlist_path: str) -> tuple[video_data, video_data, list[str]]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    with open(classlist_path, 'r') as f:
        classlist = f.readlines()

    data = [item for item in data if item['gloss'] in classlist]

    videos = os.listdir(video_root)
    train_data = []
    test_data = []
    miss_count = 0
    tot_count = 0
    for item in tqdm(data, desc="Processing JSON"):
        gloss = item['gloss']
        for instance in item['instances']:
            tot_count += 1
            if f'{instance["video_id"]}.mp4' not in videos:
                miss_count += 1
                continue
            
            split = instance['split']
            data = {
                    'gloss': gloss,
                    'video_id': instance["video_id"],
                    'bbox': instance['bbox']
                   }
            if split == 'train':
                train_data.append(data)
            else:
                test_data.append(data) 
    logging.info(f"{tot_count-miss_count}/{tot_count} videos found.")
    return train_data, test_data, len(classlist)


def verify_videos(data: video_data, video_root: str):    
    # Create dataset and dataloader
    dataset = WLASLDataset(data, video_root, transform=video_transform())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Verify videos
    good_videos = []
    for idx, (video_data, label) in enumerate(tqdm(dataloader, desc="Verifying videos")):
        if not torch.all(video_data == 0):
            good_videos.append(data[idx])

        video_data, label = video_data.to(device), label.to(device)
    
    # Report results
    total_videos = len(dataset)
    successful_videos = len(good_videos)
    logging.info(f"Verification complete. {successful_videos}/{total_videos} videos loaded successfully.")
    
    return good_videos

def main(json_path: str, classlist_path: str, video_root: str, output_folder: str):
    train_data, test_data, n_classes = process_json(json_path, video_root, classlist_path)
    train_data = verify_videos(train_data, video_root)
    test_data = verify_videos(test_data, video_root)

    train_data_path = os.path.join(output_folder, f"train_{n_classes}.json")
    test_data_path = os.path.join(output_folder, f"test_{n_classes}.json")

    with open(train_data_path, 'w') as f:
        json.dump(train_data, f, indent=4)
    
    with open(test_data_path, 'w') as f:
        json.dump(test_data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video dataset for classification')

    parser.add_argument('--json_path', type=str, default='data/raw/WLASL_v0.3.json', help='Path to the input JSON file')
    parser.add_argument('--classlist_path', type=str, default='data/processed/generator/classes.txt', help='Path to the classlist file')
    parser.add_argument('--video_root', type=str, default='data/raw/videos', help='Directory containing videos')
    parser.add_argument('--output_folder', type=str, default='data/raw', help='Directory to save outputs')
    args = parser.parse_args()

    main(args.json_path, args.classlist_path, args.video_root, args.output_folder, args.n_classes)
