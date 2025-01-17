import gradio as gr
from models import Sign2Speech
import soundfile as sf
from utils.configs import load_config
import tempfile
import cv2

config = load_config("Generate Audio")
win_size = config["nms"]["win_size"]
hop_length = config["nms"]["hop_length"]
threshold = config["nms"]["threshold"]
overlap = config["nms"]["overlap"]
extractor_checkpoint = config["pipeline"]["extractor_weights"]
transformer_checkpoint = config["pipeline"]["transformer_weights"]

model = Sign2Speech(
    hop_length=hop_length,
    win_size=win_size,
    overlap=overlap,
    threshold=threshold,
    extractor_checkpoint=extractor_checkpoint,
    transformer_checkpoint=transformer_checkpoint,
)


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
