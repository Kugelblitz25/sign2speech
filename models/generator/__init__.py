from nemo.collections.tts.models import HifiGanModel
import soundfile as sf


class AudioGenerator:
    def __init__(self):
        self.sr = 22050
        self.model = HifiGanModel.from_pretrained("nvidia/tts_hifigan")

    def save_audio(self, audio_signal, output_path):
        sf.write(output_path, audio_signal, self.sr)
    
    def __call__(self, spec):
        audio_signal = self.model.convert_spectrogram_to_audio(spec=spec.squeeze(1))
        audio_signal = audio_signal.to('cpu').detach().squeeze().numpy()
        return audio_signal, self.sr