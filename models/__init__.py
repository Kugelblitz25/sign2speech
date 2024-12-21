from models.extractor import FeatureExtractor
from models.transformer import FeatureTransformer
from models.generator import AudioGenerator
import numpy as np
import librosa


class NMS:
    def __init__(
        self,
        extractor: FeatureExtractor,
        hop_length: int = 1,
        win_size: int = 50,
        overlap: int = 0,
        threshold: float = 0.9,
    ) -> None:
        self.extractor = extractor
        self.hop_length = hop_length
        self.win_size = win_size
        self.overlap = overlap
        self.threshold = threshold

    def predict(self, frames: list):
        features = {}
        for i in range(0, len(frames), self.hop_length):
            ft, conf, _ = self.extractor(frames[i : i + self.win_size])
            features[i] = [ft, conf.cpu().numpy()[0]]
        return features

    def __call__(self, frames: list):
        features = self.predict(frames)
        frame_idxs = [
            idx for idx, (_, prob) in features.items() if prob > self.threshold
        ]
        frame_idxs = sorted(frame_idxs, key=lambda x: features[x][1])
        good_preds = []
        while len(frame_idxs) > 0:
            frame_idx = frame_idxs.pop()
            good_preds.append(frame_idx)
            frame_idxs = [
                i
                for i in frame_idxs
                if abs(i - frame_idx) > self.win_size - self.overlap
            ]
        return {idx: features[idx][0] for idx in good_preds}


class Sign2Speech:
    def __init__(
        self,
        hop_length: int = 5,
        win_size: int = 64,
        overlap: int = 0,
        threshold: float = 0.6,
        extractor_checkpoint="models/extractor/checkpoints/checkpoint_final.pt",
        transformer_checkpoint="models/transformer/checkpoints/checkpoint_final.pt",
    ):
        self.extractor = FeatureExtractor(extractor_checkpoint)
        self.transformer = FeatureTransformer(transformer_checkpoint)
        self.generator = AudioGenerator()
        self.fps = 30
        self.nms = NMS(self.extractor, hop_length, win_size, overlap, threshold)

    def combine_audio(self, audios):
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

    def __call__(self, frames: list):
        predictions = self.nms(frames)
        for frame_idx in predictions:
            spec = self.transformer(predictions[frame_idx])
            audio, _ = self.generator(spec)
            predictions[frame_idx] = audio
        audios = [[key, val] for key, val in predictions.items()]
        audios = sorted(audios, key=lambda x: x[0])
        audios.append([len(frames), -1])
        return self.combine_audio(audios)
