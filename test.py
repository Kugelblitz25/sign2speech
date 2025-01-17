import time
from pathlib import Path

from models import Sign2Speech
from utils import load_config

import soundfile as sf
import cv2

config = load_config("Generate Audio")

num_words = config["n_words"]
win_size = config["nms"]["win_size"]
hop_length = config["nms"]["hop_length"]
threshold = config["nms"]["threshold"]
overlap = config["nms"]["overlap"]
extractor_checkpoint = config["pipeline"]["extractor_weights"]
transformer_checkpoint = config["pipeline"]["transformer_weights"]

model = Sign2Speech(
    num_words=num_words,
    hop_length=hop_length,
    win_size=win_size,
    overlap=overlap,
    threshold=threshold,
    extractor_checkpoint=extractor_checkpoint,
    transformer_checkpoint=transformer_checkpoint,
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
