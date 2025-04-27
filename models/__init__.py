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
                if stretch_factor < 0:
                    print(video_dur, audio_dur, stretch_factor)
                audio = librosa.effects.time_stretch(
                    y=audio.astype(float), rate=stretch_factor
                )
            audio_concat = np.concatenate([audio_concat, audio])
        return audio_concat

    def __call__(self, frames: str | Path) -> np.ndarray:
        video = cv2.VideoCapture(frames)
        self.fps = video.get(cv2.CAP_PROP_FPS)

        predictions = []
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=frame_count, desc="Processing Frames") as pbar:
            while True:
                ret, frame = video.read()
                if not ret:
                    break

                index, feature = self.nms(frame)
                if index != -1:
                    spec = self.transformer(feature).cpu().numpy().squeeze(0)
                    audio, _ = self.generator(spec)
                    predictions.append((index, audio))
                pbar.update(1)

        if self.nms.best_window_idx is not None:
            spec = self.transformer(self.nms.best_feature).cpu().numpy().squeeze(0)
            audio, _ = self.generator(spec)
            predictions.append((self.nms.best_window_idx, audio))

        predictions.append((frame_count, None))

        video.release()
        return self.combine_audio(predictions)
