from speechbrain.inference.vocoders import HIFIGAN
import soundfile as sf


class AudioGenerator:
    def __init__(self):
        self.sr = 22050
        self.model = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            savedir="models/generator/checkpoints/tts-hifigan-ljspeech",
        )

    def save_audio(self, audio_signal, output_path):
        sf.write(output_path, audio_signal, self.sr)

    def __call__(self, spec):
        audio_signal = self.model.decode_batch(spec.squeeze(1))
        audio_signal = audio_signal.to("cpu").cpu().numpy()[0][0]
        return audio_signal, self.sr
