from models import Sign2Speech

import soundfile as sf
import cv2

model = Sign2Speech()

frames = []
cap = cv2.VideoCapture('data/raw/videos/69241.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

audio, sr = model(frames)

sf.write('output_audio.wav', audio, sr)