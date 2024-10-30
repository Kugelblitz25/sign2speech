from models.extractor import FeatureExtractor
from models.transformer import FeatureTransformer
from models.generator import AudioGenerator

class Sign2Speech:
    def __init__(self,
                 extractor_checkpoint='models/extractor/checkpoints/checkpoint_final.pt',
                 transformer_checkpoint='models/transformer/checkpoints/checkpoint_final.pt'):
        self.extractor = FeatureExtractor(extractor_checkpoint)
        self.transformer = FeatureTransformer(transformer_checkpoint)
        self.generator = AudioGenerator()
    
    def __call__(self, frames: list):
        ft, conf = self.extractor(frames)
        spec = self.transformer(ft)
        audio = self.generator(spec)

        return audio, self.model3.sr