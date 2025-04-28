import tempfile

import cv2
import gradio as gr
import numpy as np
import soundfile as sf

from models import Sign2Speech
from utils.config import load_config

config = load_config("Generate Audio")


def predict(file: str):
    video = cv2.VideoCapture(file)
    fps = video.get(cv2.CAP_PROP_FPS)
    model = Sign2Speech(
        num_words=config.n_words,
        spec_len=config.generator.max_length,
        fps=fps,
        config=config.pipeline,
    )

    audio_complete = np.zeros((0,))

    while True:
        ret, frame = video.read()
        if not ret:
            break
        model.process_frame(frame)
        ret, audio = model.buffer.get()

        if not ret:
            print("No audio generated yet.")
            continue

        audio_complete = np.concatenate((audio_complete, audio))

    model.close_stream()
    video.release()

    while True:
        ret, audio = model.buffer.get()
        if not ret:
            break
        audio_complete = np.concatenate((audio_complete, audio))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file_path = temp_file.name
        sf.write(temp_file_path, audio_complete, model.generator.sr)
    return temp_file_path


interface = gr.Interface(
    fn=predict,
    inputs=gr.Video(label="Input Video"),
    outputs=gr.Audio(label="Generated Audio"),
    title="Sign2Speech",
)

interface.launch(share=True)
