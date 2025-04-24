from pathlib import Path

import cv2
import librosa
import numpy as np
from tqdm import tqdm

from models.extractor import FeatureExtractor
from models.generator import AudioGenerator
from models.nms import NMS
from models.transformer import FeatureTransformer
from utils.config import PipelineConfig


class Sign2Speech:
    def __init__(
        self,
        num_words: int,
        spec_len: int,
        config: PipelineConfig,
    ) -> None:
        self.extractor = FeatureExtractor(config.extractor_weights, num_words)
        self.transformer = FeatureTransformer(config.transformer_weights, spec_len)
        self.generator = AudioGenerator()
        self.nms = NMS(self.extractor, config.nms)

    def combine_audio(self, audios: list[tuple[int, np.ndarray]]) -> np.ndarray:
        audio_concat = np.zeros(int((audios[0][0] / self.fps) * self.generator.sr))
        for idx, (frame_idx, audio) in enumerate(tqdm(audios[:-1], "Combining Audio")):
            video_dur = (audios[idx + 1][0] - frame_idx) / self.fps
            audio_dur = len(audio) / self.generator.sr
            if video_dur >= audio_dur:
                silence_pad = np.zeros(int((video_dur - audio_dur) * self.generator.sr))
                audio = np.concatenate((audio, silence_pad))
            else:
                stretch_factor = audio_dur / video_dur
                audio = librosa.effects.time_stretch(
                        y=audio.astype(float), rate=stretch_factor
                    )
            audio_concat = np.concatenate([audio_concat, audio])
        return audio_concat

    def get_frames(self, path: str | Path) -> list[np.ndarray]:
        frame_list = []
        cap = cv2.VideoCapture(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_list.append(frame)
        cap.release()
        return frame_list

    def __call__(self, frames: list[np.ndarray] | str | Path) -> np.ndarray:
        if not isinstance(frames, list):
            frames = self.get_frames(frames)

        predictions = self.nms(frames)

        for frame_idx in tqdm(predictions, desc="Generating Audio"):
            spec = self.transformer(predictions[frame_idx]).cpu().numpy().squeeze(0)
            audio, _ = self.generator(spec)
            predictions[frame_idx] = audio
        audios = [(key, val) for key, val in predictions.items()]
        audios = sorted(audios)
        audios.append((len(frames), -1))
        return self.combine_audio(audios)
