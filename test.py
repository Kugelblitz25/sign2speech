import time
from pathlib import Path

from models import Sign2Speech

import soundfile as sf
import cv2

model = Sign2Speech(hop_length=3, win_size=50, threshold=0.7)

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

    sf.write(f'outputs/{filename}.wav', audio, 22050)

print("Starting Predictions:")
t1 = time.time()
test_path = Path('test_videos')
for video in test_path.iterdir():
    predict(video)
t2 = time.time()
print(f"Completed Predictions in {t2-t1:.2f}s")