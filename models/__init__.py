from pathlib import Path

import cv2
import librosa
import numpy as np

from models.extractor import FeatureExtractor
from models.generator import AudioGenerator
from models.nms import NMS
from models.transformer import FeatureTransformer
from utils.config import PipelineConfig


class Sign2Speech:
    def __init__(
        self,
        num_words: int,
        config: PipelineConfig,
    ) -> None:
        self.extractor = FeatureExtractor(config.extractor_weights, num_words)
        self.transformer = FeatureTransformer(config.transformer_weights)
        self.generator = AudioGenerator()
        self.fps = 30
        self.nms = NMS(self.extractor, config.nms)

    def combine_audio(self, audios: list[tuple[int, np.ndarray]]) -> np.ndarray:
        audio_concat = np.zeros(int((audios[0][0] / self.fps + 1) * self.generator.sr))
        for idx, (frame_idx, audio) in enumerate(audios[:-1]):
            video_dur = (audios[idx + 1][0] - frame_idx) / self.fps
            audio_dur = len(audio) / self.generator.sr
            if video_dur >= audio_dur:
                silence_pad = np.zeros(int((video_dur - audio_dur) * self.generator.sr))
                audio = np.concatenate((audio, silence_pad))
            else:
                stretch_factor = video_dur / audio_dur
                if stretch_factor > 0.3:
                    audio = librosa.effects.time_stretch(
                        y=audio.astype(float), rate=stretch_factor
                    )
            audio_concat = np.concatenate([audio_concat, audio])
        return audio_concat

    def get_frames(self, path: str | Path) -> list[np.ndarray]:
        frame_list = []
        cap = cv2.VideoCapture(path)
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

        for frame_idx in predictions:
            spec = self.transformer(predictions[frame_idx])
            audio, _ = self.generator(spec)
            predictions[frame_idx] = audio
        audios = [(key, val) for key, val in predictions.items()]
        audios = sorted(audios)
        audios.append((len(frames), -1))
        return self.combine_audio(audios)
