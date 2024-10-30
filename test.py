from models.extractor import FeatureExtractor
from models.transformer import FeatureTransformer
from models.generator import AudioGenerator

import cv2

model1 = FeatureExtractor('models/extractor/checkpoints/checkpoint_final.pt')
model2 = FeatureTransformer('models/transformer/checkpoints/checkpoint_final.pt')
model3 = AudioGenerator()

frames = []
cap = cv2.VideoCapture('data/raw/videos/69241.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

ft, conf = model1(frames)
spec = model2(ft)
audio = model3(spec)

import soundfile as sf
sf.write('output_audio.wav', audio, model3.sr)
print(spec.shape)