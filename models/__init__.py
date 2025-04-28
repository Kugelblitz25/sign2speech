from collections import deque
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


class AudioBuffer:
    def __init__(self, segsize: int) -> None:
        self.buffer = deque()
        self.segsize = segsize
        self.cooldown = 0

    def add(self, audio: np.ndarray) -> None:
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        self.buffer += deque(audio.tolist())

    def get(self) -> np.ndarray:
        if len(self.buffer) == 0:
            return False, np.zeros(self.segsize)
        segment = np.zeros(min(self.segsize, len(self.buffer)))
        count = 0
        while len(self.buffer) > 0 and count < self.segsize:
            segment[count] = self.buffer.popleft()
            count += 1
        return True, segment


class Sign2Speech:
    def __init__(
        self,
        num_words: int,
        spec_len: int,
        fps: int,
        config: PipelineConfig,
    ) -> None:
        self.extractor = FeatureExtractor(config.extractor_weights, num_words)
        self.transformer = FeatureTransformer(config.transformer_weights, spec_len)
        self.generator = AudioGenerator()
        self.nms = NMS(self.extractor, config.nms)
        self.win_size = config.nms.win_size
        self.fps = fps
        self.segsize = int(self.generator.sr / self.fps)
        self.buffer = AudioBuffer(self.segsize)

    def pad_audio(self, audio: np.ndarray) -> np.ndarray:
        audio_dur = len(audio) / self.generator.sr
        video_dur = self.win_size / self.fps
        if video_dur <= audio_dur:
            stretch_factor = audio_dur / video_dur
            audio = librosa.effects.time_stretch(
                y=audio.astype(float), rate=stretch_factor
            )
        else:
            audio = np.pad(
                audio,
                (0, int((video_dur - audio_dur) * self.generator.sr)),
                mode="constant",
            )
        return audio

    def process_frame(self, frame: np.ndarray) -> None:
        index, feature = self.nms(frame)
        if index != -1:
            spec = self.transformer(feature).cpu().numpy().squeeze(0)
            audio, _ = self.generator(spec)
            self.buffer.add(self.pad_audio(audio))
            self.buffer.cooldown = self.win_size
        else:
            self.buffer.add(np.zeros(self.segsize))

    def close_stream(self):
        if self.nms.best_window_idx is not None:
            spec = self.transformer(self.nms.best_feature).cpu().numpy().squeeze(0)
            audio, _ = self.generator(spec)
            audio = self.pad_audio(audio)
            self.buffer.add(audio)

    def __call__(self, frames: str | Path) -> np.ndarray:
        video = cv2.VideoCapture(frames)
        self.fps = video.get(cv2.CAP_PROP_FPS)

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
                    self.buffer.add(self.pad_audio(audio))
                    self.buffer.cooldown = self.win_size - 1
                else:
                    self.buffer.add(np.zeros(self.segsize))
                pbar.update(1)

        if self.nms.best_window_idx is not None:
            spec = self.transformer(self.nms.best_feature).cpu().numpy().squeeze(0)
            audio, _ = self.generator(spec)
            self.buffer.add(self.pad_audio(audio))

        video.release()
        return np.array(self.buffer.buffer)
