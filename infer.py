import numpy as np
from tqdm import tqdm
from models import Sign2Speech
import soundfile as sf

from utils.config import load_config, PipelineConfig
from utils.common import create_path, Video, source


def predict(
    source: source, n_words: int, spec_len: int, config: PipelineConfig
) -> np.ndarray:
    video = Video(source)
    model = Sign2Speech(
        num_words=n_words,
        spec_len=spec_len,
        config=config,
    )

    total_frames = video.n_frames
    audio_length_per_frame = 24000 // video.fps
    total_audio_length = (total_frames + config.nms.win_size) * audio_length_per_frame
    audio_complete = np.zeros((total_audio_length,))
    idx = 0
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        for frame in video:
            ret, audio = model.process_frame(frame)
            if ret:
                audio_idx = idx * audio_length_per_frame
                audio_complete[audio_idx : audio_idx + len(audio)] += audio
            idx += 1
            pbar.update(1)

    ret, audio = model.close_stream()
    audio_idx = idx * audio_length_per_frame
    if ret:
        audio_complete[audio_idx : audio_idx + len(audio)] += audio
    video.release()
    return audio_complete


if __name__ == "__main__":
    config = load_config(
        "Generate Audio",
        test_video_path={
            "type": str,
            "required": True,
            "help": "Path to the test video file",
        },
        output_audio_path={
            "type": str,
            "default": "outputs/audio.wav",
            "help": "Path to save the output audio file",
        },
    )

    audio_output = predict(
        config.test_video_path,
        config.n_words,
        config.generator.max_length,
        config.pipeline,
    )

    output_path = create_path(config.output_audio_path)
    sf.write(output_path, audio_output, 24000)
