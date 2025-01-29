import time
from pathlib import Path

from models import Sign2Speech
from utils import Config

import soundfile as sf
import cv2

config = Config("Generate Audio")

model = Sign2Speech(
    num_words=config.n_words,
    hop_length=config.nms.hop_length,
    win_size=config.nms.win_size,
    overlap=config.nms.overlap,
    threshold=config.nms.threshold,
    extractor_checkpoint=config.pipline.extractor_checkpoint,
    transformer_checkpoint=config.pipline.transformer_checkpoint,
)


def predict(file):
    filename = Path(file).stem
    frames = []
    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    audio = model(frames)

    sf.write(f"outputs/{filename}.wav", audio, 22050)


print("Starting Predictions:")
t1 = time.time()
test_path = Path("test_videos")
for video in test_path.iterdir():
    predict(video)
t2 = time.time()
print(f"Completed Predictions in {t2 - t1:.2f}s")
