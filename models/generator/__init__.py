import librosa


class AudioGenerator:
    def __init__(self):
        self.sr = 22050

    def denormalize(self, spectrogram):
        spectrogram = spectrogram.cpu().numpy()[0][0]
        spectrogram = librosa.db_to_power(spectrogram)
        return spectrogram
    
    def __call__(self, spec):
        power_spec = self.denormalize(spec)
        linear_spectrogram = librosa.feature.inverse.mel_to_stft(power_spec, sr=self.sr, n_fft=self.n_fft)
        audio_signal = 2 * librosa.griffinlim(linear_spectrogram, hop_length=self.hop_length, n_iter=self.n_iter)
        return audio_signal, self.sr