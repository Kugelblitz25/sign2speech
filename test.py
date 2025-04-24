import re
import time
from pathlib import Path

import soundfile as sf
import whisper
from jiwer import cer, wer
from moviepy.editor import AudioFileClip, VideoFileClip
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from models import Sign2Speech
from utils.config import load_config

# Load model
config = load_config("Generate Audio")
model = Sign2Speech(num_words=config.n_words, spec_len=config.generator.max_length ,config=config.pipeline)
asr_model = whisper.load_model("medium")  # or "tiny", "small", etc.


def predict(video_path: Path, output_path: Path, audio_out_path: Path) -> None:
    audio = model(str(video_path))
    sf.write(audio_out_path, audio, 24000)

    video_clip = VideoFileClip(str(video_path))
    audio_clip = AudioFileClip(str(audio_out_path))
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")


def extract_words(video_path: str, word_list_file: str) -> str:
    name = Path(video_path).stem
    parts = name.split("_")
    word_indices = [int(p) for p in parts[:-1]]
    with open(word_list_file, "r") as f:
        words = [re.sub(r"\d", "", line.strip().lower()).split("/")[0] for line in f]
    return " ".join([words[i] for i in word_indices])


def transcribe_audio(audio_path: str) -> str:
    result = asr_model.transcribe(audio_path)
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


def main():
    video_path = (
        "data/asl-citizen/processed/continuous_videos/6_0_7_4_8_2_3_1_9_5_P22.mp4"
    )
    output_video = "output.mp4"
    output_audio = "output.wav"
    class_file = "data/asl-citizen/processed/classes.txt"

    print("Starting Prediction...")
    t1 = time.time()
    predict(video_path, output_video, output_audio)
    t2 = time.time()
    print(f"Completed in {t2 - t1:.2f}s")

    reference = extract_words(video_path, class_file)
    hypothesis = transcribe_audio(output_audio)
    metrics = evaluate(reference, hypothesis)

    print(f"\nReference:  {reference}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
