import tempfile

import gradio as gr
import numpy as np
import soundfile as sf

from models import Sign2Speech
from utils.config import load_config

config = load_config("Generate Audio")

model = Sign2Speech(
    num_words=config.n_words,
    hop_length=config.nms.hop_length,
    win_size=config.nms.win_size,
    overlap=config.nms.overlap,
    threshold=config.nms.threshold,
    extractor_checkpoint=config.pipeline.extractor_weights,
    transformer_checkpoint=config.pipeline.transformer_weights,
)


def predict(file: str) -> np.ndarray:
    audio = model(file)
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
