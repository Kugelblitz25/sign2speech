import os
import random
import re
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import whisper
from jiwer import cer, wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm

from models import Sign2Speech
from utils.common import create_subset
from utils.config import load_config

# Load model
config = load_config("Generate Audio")
asr_model = whisper.load_model("large")  # or "tiny", "small", etc.


def concatenate_videos(video_paths, output_path):
    if not video_paths:
        return False

    # Get information from first video
    first_video = cv2.VideoCapture(video_paths[0])
    if not first_video.isOpened():
        return False

    frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = first_video.get(cv2.CAP_PROP_FPS)
    first_video.release()

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Concatenate videos
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    return True


def predict(file: str, temp_dir: str):
    video = cv2.VideoCapture(file)
    model = Sign2Speech(
        num_words=config.n_words,
        spec_len=config.generator.max_length,
        config=config.pipeline,
    )

    audio_complete = np.zeros((0,))

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            ret, audio = model.process_frame(frame)
            if ret:
                audio_complete = np.concatenate((audio_complete, audio))
            pbar.update(1)

    ret, audio = model.close_stream()
    if ret:
        audio_complete = np.concatenate((audio_complete, audio))
    video.release()

    audio_path = os.path.join(temp_dir, "temp_audio.wav")
    sf.write(audio_path, audio_complete, 24000)
    return audio_path


def transcribe_audio(audio_path: str) -> str:
    result = asr_model.transcribe(audio_path, language="en")
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


def main(csv_file, n, k, video_base_dir):
    # Load and process CSV file
    df = pd.read_csv(create_subset(csv_file, 100))
    print(list(df.Gloss.unique()))

    # Check if required columns exist
    required_columns = ["Participant ID", "Video file", "Gloss"]
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV file")
            return

    # Get all unique participant IDs (we'll select randomly from these each round)
    all_participant_ids = df["Participant ID"].unique()

    if len(all_participant_ids) == 0:
        print("Error: No participant IDs found in CSV file")
        return

    all_metrics = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for round_num in range(n):
            # Randomly select a participant for this round
            participant_id = random.choice(all_participant_ids)
            print(f"\nRound {round_num + 1}/{n}: Selected participant {participant_id}")

            # Get all videos for this participant
            participant_videos = df[df["Participant ID"] == participant_id]

            if len(participant_videos) < k:
                print(
                    f"Warning: Participant {participant_id} only has {len(participant_videos)} videos, using all of them"
                )
                selected_videos = participant_videos
            else:
                selected_videos = participant_videos.sample(k)

            video_paths = [
                os.path.join(video_base_dir, row["Video file"])
                for _, row in selected_videos.iterrows()
            ]
            words = selected_videos["Gloss"].tolist()
            words = [word.lower() for word in words]
            combined_gloss = " ".join(words)

            # Concatenate videos
            concat_video_path = os.path.join(
                temp_dir, f"concat_video_round_{round_num}.mp4"
            )
            print(f"Concatenating {len(video_paths)} videos...")
            success = concatenate_videos(video_paths, concat_video_path)

            if not success:
                print(f"Error: Failed to concatenate videos for round {round_num + 1}")
                continue

            # Process the concatenated video
            try:
                t1 = time.time()
                audio_path = predict(concat_video_path, temp_dir)
                t2 = time.time()
            except Exception as e:
                print(f"Error processing concatenated video: {e}")

            hypothesis = transcribe_audio(audio_path)
            metrics = evaluate(combined_gloss, hypothesis)

            all_metrics.append(metrics)

            print(f"Processing completed in {t2 - t1:.2f}s")
            print(f"Reference:  {combined_gloss}")
            print(f"Hypothesis: {hypothesis}")
            print(f"Metrics: {metrics}")

    if all_metrics:
        avg_metrics = {
            "WER": np.mean([m["WER"] for m in all_metrics]),
            "CER": np.mean([m["CER"] for m in all_metrics]),
            "BLEU": np.mean([m["BLEU"] for m in all_metrics]),
        }

        print("\n" + "=" * 50)
        print(f"Processed {len(all_metrics)}/{n} rounds successfully")
        print("Average Metrics:")
        print(f"Average WER:  {avg_metrics['WER']:.4f}")
        print(f"Average CER:  {avg_metrics['CER']:.4f}")
        print(f"Average BLEU: {avg_metrics['BLEU']:.4f}")
    else:
        print("No videos were processed successfully")


if __name__ == "__main__":
    main("data/wlasl/raw/test.csv", 100, 20, "data/wlasl/raw/videos")
