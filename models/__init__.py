from models.extractor import FeatureExtractor
from models.transformer import FeatureTransformer
from models.generator import AudioGenerator

class NMS:
    def __init__(self, 
                 extractor: FeatureExtractor,
                 hop_length: int = 1, 
                 win_size: int = 64, 
                 overlap: int = 0,
                 threshold: float = 0.8) -> None:
        self.extractor = extractor
        self.hop_length = hop_length
        self.win_size = win_size
        self.overlap = overlap
        self.threshold = threshold
        self.features = {}

    def predict(self, frames: list):
        for i in range(0, len(frames), self.hop_length):
            ft, conf = self.extractor(frames[i:i + self.win_size])
            self.features[i] = [ft, conf.cpu().numpy()[0]]

    def __call__(self, frames: list):
        self.predict(frames)
        frame_idxs = [idx for idx, (_, prob) in self.probs.items() if prob > self.threshold]
        frame_idxs = sorted(frame_idxs, key=lambda x: self.features[x][1])
        good_preds = []
        while len(frame_idxs) > 0:
            frame_idx = frame_idxs.pop()
            good_preds.append(frame_idx)
            frame_idxs = [i for i in frame_idxs if abs(i - frame_idx) > self.win_size - self.overlap]
        return {idx: self.features[idx][0] for idx in good_preds}

class Sign2Speech:
    def __init__(self,
                 hop_length: int = 1,
                 win_size: int = 64,
                 overlap: int = 0,
                 threshold: float = 0.8,
                 extractor_checkpoint='models/extractor/checkpoints/checkpoint_final.pt',
                 transformer_checkpoint='models/transformer/checkpoints/checkpoint_final.pt'):
        extractor = FeatureExtractor(extractor_checkpoint)
        self.transformer = FeatureTransformer(transformer_checkpoint)
        self.generator = AudioGenerator()
        self.nms = NMS(extractor, hop_length, win_size, overlap, threshold)

    def combine_audio(self, predictions: dict):
        pass        
    
    def __call__(self, frames: list):
        predictions = self.nms(frames)
        for frame_idx in predictions:
            spec = self.transformer(predictions[frame_idx])
            audio = self.generator(spec)
            predictions[frame_idx] = audio
        return self.combine_audio(predictions)