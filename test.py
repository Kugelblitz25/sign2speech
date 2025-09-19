import os
import re
import time

import numpy as np
import pandas as pd
import whisper
from jiwer import cer, wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from utils.config import load_config
from infer import predict

# Load model
config = load_config("Testing...")
asr_model = whisper.load_model("large")


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


def main(test_videos_dir: str, word_list: list[str]):
    test_videos = os.listdir(test_videos_dir)

    all_metrics = []
    best_videos = []

    for idx, test_video in enumerate(test_videos):
        video_path = os.path.join(test_videos_dir, test_video)
        glosses = test_video.replace(".mp4", "").split("_")[3:]
        combined_gloss = " ".join(glosses)
        try:
            t1 = time.time()
            audio = predict(
                video_path,
                config.n_words,
                config.generator.max_length,
                config.pipeline,
            )
            t2 = time.time()
        except Exception as e:
            print(f"Error processing concatenated video: {e}")

        hypothesis = transcribe_audio(audio)
        metrics = evaluate(combined_gloss, hypothesis)
        misheard_words = [word for word in hypothesis.split() if word not in word_list]

        all_metrics.append(metrics)

        if len(combined_gloss.split()) > 5 and metrics["WER"] <= 0.3:
            best_videos.append((test_video, metrics))

        print(
            f"Processing completed in {t2 - t1:.2f}s (Video {idx + 1}/{len(test_videos)})"
        )
        print(f"Reference:  {combined_gloss}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Metrics: {metrics}")
        print(f"No. of misheard words: {len(misheard_words)}")
        print("-" * 50)

    if all_metrics:
        avg_metrics = {
            "WER": np.mean([m["WER"] for m in all_metrics]),
            "CER": np.mean([m["CER"] for m in all_metrics]),
            "BLEU": np.mean([m["BLEU"] for m in all_metrics]),
        }

        print("\n" + "=" * 50)
        print(f"Processed {len(all_metrics)}/{len(test_videos)} rounds successfully")
        print("Average Metrics:")
        print(f"Average WER:  {avg_metrics['WER']:.4f}")
        print(f"Average CER:  {avg_metrics['CER']:.4f}")
        print(f"Average BLEU: {avg_metrics['BLEU']:.4f}")
        print("=" * 50 + "\n")
        print("Best Videos (more than 5 words with WER < 0.2):")
        for video, metrics in best_videos:
            print(f"{video}: {metrics}")

    else:
        print("No videos were processed successfully")


if __name__ == "__main__":
    test_data = pd.read_csv("data/wlasl/raw/test.csv")
    word_list = test_data["Gloss"].unique().tolist()
    main("test_videos", word_list)
