import gradio as gr
from models import Sign2Speech
import soundfile as sf
import tempfile
import cv2

model = Sign2Speech(hop_length=3, win_size=50, threshold=0.7)


def predict(file: str):
    frames = []
    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    audio = model(frames)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        sf.write(temp_audio_file.name, audio, 22050)
        audio_path = temp_audio_file.name

    return audio_path


interface = gr.Interface(
    fn=predict,
    inputs=gr.Video(label="Input Video"),
    outputs=gr.Audio(label="Generated Audio"),
    title="Sign2Speech",
)

interface.launch(share=True)
