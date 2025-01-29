import gradio as gr
from models import Sign2Speech
import soundfile as sf
from utils import Config
import tempfile
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
