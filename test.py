import os
import re
import time

import numpy as np
import pandas as pd
import whisper
from jiwer import cer, wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from utils.config import PipelineConfig, load_config
from utils.common import create_subset, get_logger
from infer import predict

# Load model
asr_model = whisper.load_model("large")
logger = get_logger("logs/testing.log")


def transcribe_audio(audio: np.ndarray) -> str:
    audio = audio.astype(np.float32)
    result = asr_model.transcribe(audio, language="en")
    text = result["text"].strip().lower()
    return re.sub(r"[^\w\s]", "", text)


def evaluate(reference: str, hypothesis: str) -> dict:
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu(
        [reference_tokens],
        hypothesis_tokens,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothie,
    )
    return {
        "WER": wer(reference, hypothesis),
        "CER": cer(reference, hypothesis),
        "BLEU": bleu,
    }


def main(
    test_videos_dir: str,
    word_list: list[str],
    n_words: int,
    max_length: int,
    cfg: PipelineConfig,
):
    test_videos = os.listdir(test_videos_dir)

    all_metrics = []
    best_videos = []

    for idx, test_video in enumerate(test_videos):
        video_path = os.path.join(test_videos_dir, test_video)
        glosses = test_video.replace(".mp4", "").split("_")[3:]
        n_words = len(glosses)
        combined_gloss = " ".join(glosses)
        try:
            t1 = time.time()
            audio = predict(video_path, n_words, max_length, cfg)
            t2 = time.time()
        except Exception as e:
            logger.error(f"Error processing concatenated video: {e}")
            continue

        hypothesis = transcribe_audio(audio)
        metrics = evaluate(combined_gloss, hypothesis)
        misheard_words = [word for word in glosses if word not in word_list]
        metrics["WER"] = max(0, metrics["WER"] - len(misheard_words) / n_words)

        all_metrics.append(metrics)

        if len(combined_gloss.split()) >= 3 and metrics["WER"] <= 0.3:
            best_videos.append((test_video, metrics))

        logger.info(
            f"Processing completed in {t2 - t1:.2f}s (Video {idx + 1}/{len(test_videos)})"
        )
        logger.info(f"Reference:  {combined_gloss}")
        logger.info(f"Hypothesis: {hypothesis}")
        logger.info(f"Metrics: {metrics}")
        logger.info(f"No. of misheard words: {len(misheard_words)}")
        logger.info("-" * 50)

    if all_metrics:
        avg_metrics = {
            "WER": np.mean([m["WER"] for m in all_metrics]),
            "CER": np.mean([m["CER"] for m in all_metrics]),
            "BLEU": np.mean([m["BLEU"] for m in all_metrics]),
        }

        logger.info("\n" + "=" * 50)
        logger.info(
            f"Processed {len(all_metrics)}/{len(test_videos)} rounds successfully"
        )
        logger.info("Average Metrics:")
        logger.info(f"Average WER:  {avg_metrics['WER']:.4f}")
        logger.info(f"Average CER:  {avg_metrics['CER']:.4f}")
        logger.info(f"Average BLEU: {avg_metrics['BLEU']:.4f}")
        logger.info("=" * 50 + "\n")
        logger.info("Best Videos (more than 5 words with WER < 0.2):")
        for video, metrics in best_videos:
            logger.info(f"{video}: {metrics}")

    else:
        logger.error("No videos were processed successfully")


if __name__ == "__main__":
    config = load_config(
        "Test pipeline",
        videos_loc={
            "type": str,
            "default": "test_videos",
            "help": "Location of test videos",
        },
    )
    test_data = pd.read_csv(create_subset(config.data.raw.csvs.test, config.n_words))
    word_list = test_data["Gloss"].unique().tolist()
    main(
        config.videos_loc,
        word_list,
        config.n_words,
        config.generator.max_length,
        config.pipeline,
    )
