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
        spec_len: int,
        config: PipelineConfig,
    ) -> None:
        self.extractor = FeatureExtractor(config.extractor_weights, num_words)
        self.transformer = FeatureTransformer(config.transformer_weights, spec_len)
        self.generator = AudioGenerator()
        self.nms = NMS(self.extractor, config.nms)

    def process_frame(self, frame: np.ndarray) -> None:
        index, feature = self.nms(frame)
        if index == -1:
            return False, None

        spec = self.transformer(feature).cpu().numpy().squeeze(0)
        audio, _ = self.generator(spec)
        return True, audio

    def close_stream(self):
        if self.nms.best_window_idx is None:
            return False, None
        spec = self.transformer(self.nms.best_feature).cpu().numpy().squeeze(0)
        audio, _ = self.generator(spec)
        return True, audio
