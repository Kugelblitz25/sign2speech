from pathlib import Path
import torch
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
import torchvision.transforms as transforms
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WLASLDataset(Dataset):
    def __init__(self, data, video_dir, transform=None) -> None:
        self.data = data
        self.video_dir = Path(video_dir)
        self.transform = transform
        self.classes = sorted(list(set(item['gloss'] for item in data)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.tensor, int]:
        item = self.data[idx]
        video_path = self.video_dir / f"{item['video_id']}.mp4"
        label = self.class_to_idx[item['gloss']]
        try:
            video = EncodedVideo.from_path(video_path)
            video_data = video.get_clip(start_sec=0, end_sec=video.duration)
            video_data = video_data['video']
            video_data = preprocess_video(video_data, self.transform)
        except Exception as e:
            logging.warning(f"Failed to load video {video_path}: {str(e)}")
            return torch.zeros((3, 32, 224, 224)), label
        return video_data, label

def video_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])   
        
def preprocess_video(video_data, transform = video_transform()):
    if transform:
        transformed_frames = []
        for i in range(video_data.shape[1]):
            frame = video_data[:, i, :, :]
            transformed_frames.append(transform(frame))
        video_data = torch.stack(transformed_frames, dim=1)
    
    # Ensure we have exactly 32 frames
    if video_data.shape[1] > 32:
        step = video_data.shape[1] // 32
        video_data = video_data[:, ::step, :, :][:, :32, :, :]
    elif video_data.shape[1] < 32:
        padding = torch.zeros((3, 32 - video_data.shape[1], 224, 224))
        video_data = torch.cat([video_data, padding], dim=1)
    
    return video_data

