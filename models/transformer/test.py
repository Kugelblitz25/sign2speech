import random
from pathlib import Path

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


def calculate_mcd(reference_spec, generated_spec, n_mfcc=13):
    try:
        reference_mfcc = librosa.feature.mfcc(S=reference_spec, n_mfcc=n_mfcc)
        generated_mfcc = librosa.feature.mfcc(S=generated_spec, n_mfcc=n_mfcc)

        min_len = min(reference_mfcc.shape[1], generated_mfcc.shape[1])
        reference_mfcc = reference_mfcc[:, :min_len]
        generated_mfcc = generated_mfcc[:, :min_len]

        diff = reference_mfcc - generated_mfcc
        mcd = np.sqrt(2 * np.sum(diff**2, axis=0))
        return np.mean(mcd)
    except Exception as e:
        logger.error(f"Error calculating MCD: {e}")
        return float("inf")


class Tester:
    def __init__(
        self,
        test_data_path: str,
        specs_csv: str,
        model: SpectrogramGenerator,
        audio_generator: AudioGenerator,
        output_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        num_samples_to_save: int = 10,
    ) -> None:
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.output_dir = create_path(output_dir)
        self.samples_dir = create_path(self.output_dir / "audio_samples")

        self.sr = 22050
        self.num_samples_to_save = num_samples_to_save

        logger.debug(f"Using Device: {self.device}")
        logger.debug(f"Using Sample Rate: {self.sr}Hz")
        logger.debug(f"Will save {num_samples_to_save} random audio samples")

        self.model = model.to(self.device)
        self.audio_generator = audio_generator
        self.test_loader = self.get_dataloader(
            test_data_path, specs_csv, batch_size, num_workers
        )

    def get_dataloader(
        self, features_csv: str, spec_csv: str, batch_size: int, num_workers: int
    ) -> DataLoader:
        dataset = SpectrogramDataset(features_csv, spec_csv)
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
        generated_spec,
        target_spec,
        sample_idx,
        metrics,
    ):
        gen_path = self.samples_dir / f"sample_{sample_idx}_generated.wav"
        target_path = self.samples_dir / f"sample_{sample_idx}_target.wav"
        metrics_path = self.samples_dir / f"sample_{sample_idx}_metrics.txt"
        spec_csv_path = self.output_dir / "spectrograms.csv"

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

        # Convert spectrograms into a DataFrame row format
        gen_row = [f"sample_{sample_idx}_generated"] + generated_spec.flatten().tolist()
        tgt_row = [f"sample_{sample_idx}_target"] + target_spec.flatten().tolist()

        # Load existing CSV or create a new DataFrame
        if spec_csv_path.exists():
            df = pd.read_csv(spec_csv_path)
        else:
            df = pd.DataFrame(
                columns=["id"] + [f"bin_{i}" for i in range(generated_spec.size)]
            )

        # Append new data
        df.loc[len(df)] = gen_row
        df.loc[len(df)] = tgt_row

        # Save DataFrame back to CSV
        df.to_csv(spec_csv_path, index=False)

        logger.info(
            f"Saved audio and spectrogram sample {sample_idx} to {self.samples_dir}"
        )

    def test(self) -> dict:
        logger.info("Starting model evaluation...")

        metrics = {"mse": [], "pesq": [], "stoi": [], "snr": [], "mcd": []}

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

        with torch.no_grad():
            for idx, (features, target_specs) in enumerate(
                tqdm(self.test_loader, "Evaluating")
            ):
                features = features.to(self.device)
                target_specs = target_specs.to(self.device)

                generated_specs = self.model(features)

                mse = torch.nn.functional.mse_loss(generated_specs, target_specs).item()
                metrics["mse"].append(mse)

                generated_specs_np = generated_specs.cpu().numpy()
                target_specs_np = target_specs.cpu().numpy()

                for i in range(generated_specs.shape[0]):
                    generated_audio, _ = self.audio_generator(
                        generated_specs_np[i]
                    )
                    target_audio, _ = self.audio_generator(
                       target_specs_np[i]
                    )

                    generated_audio = generated_audio[0]
                    target_audio = target_audio[0]

                    # Calculate metrics
                    pesq_score = calculate_pesq(
                        target_audio, generated_audio, sr=self.sr
                    )
                    stoi_score = calculate_stoi(
                        target_audio, generated_audio, sr=self.sr
                    )
                    snr_score = calculate_snr(target_audio, generated_audio)
                    mcd_score = calculate_mcd(target_specs_np[i], generated_specs_np[i])

                    metrics["pesq"].append(pesq_score)
                    metrics["stoi"].append(stoi_score)
                    metrics["snr"].append(snr_score)
                    metrics["mcd"].append(mcd_score)

                    # Save audio sample if it's in our random selection
                    if global_sample_idx in sample_indices:
                        sample_metrics = {
                            "pesq": pesq_score,
                            "stoi": stoi_score,
                            "snr": snr_score,
                            "mcd": mcd_score,
                        }
                        self.save_audio_sample(
                            generated_audio,
                            target_audio,
                            generated_specs_np[i],
                            target_specs_np[i],
                            sample_count,
                            sample_metrics,
                        )

                        sample_count += 1

                    global_sample_idx += 1

        # Calculate average metrics
        avg_metrics = {
            "mse": np.mean(metrics["mse"]),
            "pesq": np.mean(metrics["pesq"]),
            "stoi": np.mean(metrics["stoi"]),
            "snr": np.mean(metrics["snr"]),
            "mcd": np.mean(metrics["mcd"]),
        }

        # Log results
        logger.info("Evaluation Results:")
        logger.info(f"MSE: {avg_metrics['mse']:.4f}")
        logger.info(f"PESQ [-0.5 to 4.5]: {avg_metrics['pesq']:.4f}")
        logger.info(f"STOI [0 to 1]: {avg_metrics['stoi']:.4f}")
        logger.info(f"SNR: {avg_metrics['snr']:.4f} dB")
        logger.info(f"MCD (lower is better): {avg_metrics['mcd']:.4f}")
        logger.info(f"Saved {sample_count}/{self.num_samples_to_save} audio samples")

        # Save detailed results to file
        results_file = self.output_dir / "test_results.txt"
        with open(results_file, "w") as f:
            f.write("# Spectrogram Generator Test Results\n\n")
            f.write(f"Sample rate: {self.sr}Hz\n")
            f.write("Vocoder: HiFiGAN (speechbrain/tts-hifigan-ljspeech)\n")
            f.write(f"Audio samples: {sample_count} (saved to {self.samples_dir})\n")
            f.write("\n## Average Metrics\n\n")
            f.write(f"- MSE: {avg_metrics['mse']:.4f}\n")
            f.write(f"- PESQ [-0.5 to 4.5]: {avg_metrics['pesq']:.4f}\n")
            f.write(f"- STOI [0 to 1]: {avg_metrics['stoi']:.4f}\n")
            f.write(f"- SNR: {avg_metrics['snr']:.4f} dB\n")
            f.write(f"- MCD (lower is better): {avg_metrics['mcd']:.4f}\n")

            # Write histogram data
            f.write("\n## Metric Distributions\n\n")
            for metric_name in metrics:
                f.write(f"### {metric_name.upper()} Distribution\n")
                for value in metrics[metric_name]:
                    f.write(f"{value:.6f}\n")
                f.write("\n")

        return avg_metrics


if __name__ == "__main__":
    config = load_config("Testing spectrogram generator")

    model = SpectrogramGenerator()

    load_model_weights(
        model,
        Path(config.transformer.checkpoints) / "checkpoint_best_lib.pt",
        torch.device("cuda:1"),
    )

    audio_generator = AudioGenerator()

    tester = Tester(
        config.data.processed.vid_features.test,
        config.data.processed.specs,
        model,
        audio_generator,
        "experiments/audio_test",
        num_samples_to_save=10,
    )

    metrics = tester.test()
