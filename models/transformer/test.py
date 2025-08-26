import random

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.generator import AudioGenerator
from models.transformer.dataset import SpectrogramDataset
from models.transformer.model import SpectrogramGenerator
from utils.common import create_path, get_logger
from utils.config import load_config
from utils.model import load_model_weights

logger = get_logger("logs/transformer_testing.log")


def calculate_pesq(reference, generated, sr=22050):
    try:
        reference_16k = librosa.resample(reference, orig_sr=sr, target_sr=16000)
        generated_16k = librosa.resample(generated, orig_sr=sr, target_sr=16000)

        score = pesq(16000, reference_16k, generated_16k, "wb")
        return score
    except Exception as e:
        logger.error(f"Error calculating PESQ: {e}")
        return -0.5


def calculate_stoi(reference, generated, sr=22050):
    try:
        score = stoi(reference, generated, sr, extended=False)
        return score
    except Exception as e:
        logger.error(f"Error calculating STOI: {e}")
        return 0.0


def calculate_snr(reference, generated):
    try:
        noise = reference - generated
        signal_power = np.sum(reference**2)
        noise_power = np.sum(noise**2)

        if noise_power == 0:
            return float("inf")

        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    except Exception as e:
        logger.error(f"Error calculating SNR: {e}")
        return -float("inf")


def get_mel_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    real_part = np.nan_to_num(spectrogram[0], nan=0.0, posinf=200.0, neginf=0.0)
    imag_part = np.nan_to_num(spectrogram[1], nan=0.0, posinf=200.0, neginf=0.0)
    complex_spec = np.abs(real_part + 1j * imag_part)
    mel = librosa.feature.melspectrogram(S=complex_spec, sr=22050)
    return librosa.power_to_db(mel)


def calculate_mcd(reference_aud, generated_aud, n_mfcc=13, sr=22050):
    try:
        mfccs1 = librosa.feature.mfcc(y=reference_aud, sr=sr, n_mfcc=n_mfcc)
        mfccs2 = librosa.feature.mfcc(y=generated_aud, sr=sr, n_mfcc=n_mfcc)

        # Ensure same shape by truncating to shorter length
        min_length = min(mfccs1.shape[1], mfccs2.shape[1])
        mfccs1 = mfccs1[:, :min_length]
        mfccs2 = mfccs2[:, :min_length]

        # Calculate MCD
        diff = mfccs1 - mfccs2
        mcd = np.sqrt(np.mean(diff**2, axis=0))

        return np.mean(mcd)
    except Exception as e:
        logger.error(f"Error calculating MCD: {e}")
        return float("inf")


def spectral_convergence(reference_spec, generated_spec):
    try:
        ref_mag = np.sqrt(reference_spec[0] ** 2 + reference_spec[1] ** 2)
        gen_mag = np.sqrt(generated_spec[0] ** 2 + generated_spec[1] ** 2)

        nom = np.linalg.norm(ref_mag - gen_mag, ord="fro")
        denom = np.linalg.norm(ref_mag, ord="fro")

        if denom == 0:
            return float("inf")

        return nom / denom
    except Exception as e:
        logger.error(f"Error calculating spectral convergence: {e}")
        return float("inf")


class Tester:
    def __init__(
        self,
        test_data_path: str,
        specs_csv: str,
        model: SpectrogramGenerator,
        audio_generator: AudioGenerator,
        output_dir: str,
        device: torch.device,
        batch_size: int = 64,
        num_workers: int = 4,
        num_samples_to_save: int = 10,
    ) -> None:
        self.device = device
        self.output_dir = create_path(output_dir)
        self.samples_dir = create_path(self.output_dir / "audio_samples")

        self.sr = 22050
        self.num_samples_to_save = num_samples_to_save

        logger.debug(f"Using Device: {self.device}")
        logger.debug(f"Using Sample Rate: {self.sr}Hz")
        logger.debug(f"Will save {num_samples_to_save} random audio samples")

        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.audio_generator = audio_generator
        self.test_loader = self.get_dataloader(
            test_data_path, specs_csv, batch_size, num_workers
        )

    def get_dataloader(
        self, features_csv: str, spec_csv: str, batch_size: int, num_workers: int
    ) -> DataLoader:
        dataset = SpectrogramDataset(
            features_csv, spec_csv, config.generator.max_length
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return dataloader

    def save_audio_sample(
        self,
        generated_audio,
        target_audio,
        sample_idx,
        metrics,
    ):
        gen_path = self.samples_dir / f"sample_{sample_idx}_generated.wav"
        target_path = self.samples_dir / f"sample_{sample_idx}_target.wav"
        metrics_path = self.samples_dir / f"sample_{sample_idx}_metrics.txt"

        sf.write(
            gen_path,
            generated_audio,
            self.sr,
            subtype="FLOAT",
        )
        sf.write(
            target_path,
            target_audio,
            self.sr,
            subtype="FLOAT",
        )

        with open(metrics_path, "w") as f:
            f.write(f"PESQ: {metrics['pesq']:.4f}\n")
            f.write(f"STOI: {metrics['stoi']:.4f}\n")
            f.write(f"SNR: {metrics['snr']:.4f} dB\n")
            f.write(f"MCD: {metrics['mcd']:.4f}\n")
            f.write(f"SC: {metrics['sc']:.4f}\n")

        logger.info(f"Saved audio sample {sample_idx} to {self.samples_dir}")

    def test(self) -> dict:
        logger.info("Starting model evaluation...")

        metrics = {"pesq": [], "stoi": [], "snr": [], "mcd": [], "sc": [], "mse": []}

        sample_indices = set()
        total_samples = len(self.test_loader.dataset)

        if total_samples < self.num_samples_to_save:
            logger.warning(
                f"Requested {self.num_samples_to_save} samples but only {total_samples} are available"
            )
            self.num_samples_to_save = total_samples

        sample_indices = set(
            random.sample(range(total_samples), self.num_samples_to_save)
        )
        logger.info(f"Will save audio samples at indices: {sorted(sample_indices)}")

        sample_count = 0
        global_sample_idx = 0
        low_quality_count = 0

        with torch.no_grad():
            for features, target_specs in tqdm(self.test_loader, "Evaluating"):
                features = features.to(self.device)
                target_specs = target_specs.to(self.device)

                generated_specs = self.model(features)

                generated_specs_np = generated_specs.cpu().numpy()
                target_specs_np = target_specs.cpu().numpy()

                batch_size = generated_specs.shape[0]
                for i in range(batch_size):
                    gen_spec = generated_specs_np[i]
                    tgt_spec = target_specs_np[i]

                    metrics["mse"] = np.mean(np.sqrt((gen_spec - tgt_spec) ** 2))

                    # Convert complex spectrograms to audio
                    generated_audio, _ = self.audio_generator(gen_spec)
                    target_audio, _ = self.audio_generator(tgt_spec)

                    # Calculate metrics
                    pesq_score = calculate_pesq(
                        target_audio, generated_audio, sr=self.sr
                    )
                    if pesq_score < 1.5:
                        low_quality_count += 1

                    stoi_score = calculate_stoi(
                        target_audio, generated_audio, sr=self.sr
                    )
                    snr_score = calculate_snr(target_audio, generated_audio)
                    mcd_score = calculate_mcd(target_audio, generated_audio)
                    sc_score = spectral_convergence(tgt_spec, gen_spec)

                    metrics["pesq"].append(pesq_score)
                    metrics["stoi"].append(stoi_score)
                    metrics["snr"].append(snr_score)
                    metrics["mcd"].append(mcd_score)
                    metrics["sc"].append(sc_score)

                    # Save audio sample if it's in our random selection
                    if global_sample_idx in sample_indices:
                        sample_metrics = {
                            "pesq": pesq_score,
                            "stoi": stoi_score,
                            "snr": snr_score,
                            "mcd": mcd_score,
                            "sc": sc_score,
                        }
                        self.save_audio_sample(
                            generated_audio,
                            target_audio,
                            sample_count,
                            sample_metrics,
                        )

                        sample_count += 1

                    global_sample_idx += 1

                # Print low quality percentage after each batch
                if batch_size > 0:
                    logger.info(
                        f"Low quality samples: {low_quality_count * 100 / global_sample_idx:.2f}% ({low_quality_count}/{global_sample_idx})"
                    )

        # Calculate average metrics
        avg_metrics = {
            "mse": np.mean(metrics["mse"]),
            "pesq": np.mean(metrics["pesq"]),
            "stoi": np.mean(metrics["stoi"]),
            "snr": np.mean(metrics["snr"]),
            "mcd": np.mean(metrics["mcd"]),
            "sc": np.mean(metrics["sc"]),
        }

        # Log results
        logger.info("Evaluation Results:")
        logger.info(f"MSE: {avg_metrics['mse']:.4f}")
        logger.info(f"PESQ [-0.5 to 4.5]: {avg_metrics['pesq']:.4f}")
        logger.info(f"STOI [0 to 1]: {avg_metrics['stoi']:.4f}")
        logger.info(f"SNR: {avg_metrics['snr']:.4f} dB")
        logger.info(f"MCD (lower is better): {avg_metrics['mcd']:.4f}")
        logger.info(f"Spectral Convergence (lower is better): {avg_metrics['sc']:.4f}")
        logger.info(f"Saved {sample_count}/{self.num_samples_to_save} audio samples")
        logger.info(
            f"Low quality samples: {low_quality_count * 100 / global_sample_idx:.2f}% ({low_quality_count}/{global_sample_idx})"
        )

        # Save detailed results to file
        results_file = self.output_dir / "test_results.txt"
        with open(results_file, "w") as f:
            f.write("# Spectrogram Generator Test Results\n\n")
            f.write(f"Sample rate: {self.sr}Hz\n")
            f.write("Vocoder: HiFiGAN (speechbrain/tts-hifigan-ljspeech)\n")
            f.write(f"Audio samples: {sample_count} (saved to {self.samples_dir})\n")
            f.write(
                f"Low quality samples: {low_quality_count * 100 / global_sample_idx:.2f}% ({low_quality_count}/{global_sample_idx})\n"
            )
            f.write("\n## Average Metrics\n\n")
            f.write(f"- MSE: {avg_metrics['mse']:.4f}\n")
            f.write(f"- PESQ [-0.5 to 4.5]: {avg_metrics['pesq']:.4f}\n")
            f.write(f"- STOI [0 to 1]: {avg_metrics['stoi']:.4f}\n")
            f.write(f"- SNR: {avg_metrics['snr']:.4f} dB\n")
            f.write(f"- MCD (lower is better): {avg_metrics['mcd']:.4f}\n")
            f.write(
                f"- Spectral Convergence (lower is better): {avg_metrics['sc']:.4f}\n"
            )

        # Save metrics as CSV for easier analysis
        df = pd.DataFrame(metrics)
        df.to_csv(self.output_dir / "metrics.csv", index=False)

        return avg_metrics


if __name__ == "__main__":
    config = load_config(
        "Testing spectrogram generator",
        output_path={
            "type": str,
            "default": "models/transformer/test_results",
            "help": "Path to save test results",
        },
        model_weights={
            "type": str,
            "default": "models/checkpoints/transformer_best.pt",
            "help": "Path to the model weights for testing",
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpectrogramGenerator(spec_len=config.generator.max_length).to(device)

    load_model_weights(
        model,
        config.model_weights,
        device,
    )

    audio_generator = AudioGenerator()

    tester = Tester(
        config.data.processed.vid_features.test,
        config.data.processed.specs,
        model,
        audio_generator,
        config.output_path,
        device,
        batch_size=32,
        num_samples_to_save=10,
    )

    metrics = tester.test()
