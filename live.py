import multiprocessing as mp
import tempfile

import cv2
import gradio as gr
import soundfile as sf

from models import Sign2Speech
from utils.config import load_config

config = load_config("Generate Audio")


def acquire_frames(video_path, frame_queue, stop_event):
    video = cv2.VideoCapture(video_path)

    while not stop_event.is_set():
        ret, frame = video.read()
        if not ret:
            break
        frame_queue.put(frame)

    frame_queue.put(None)
    video.release()


def process_frames(frame_queue, audio_queue, config_dict, stop_event):
    model = Sign2Speech(
        num_words=config_dict["n_words"],
        spec_len=config_dict["spec_len"],
        config=config_dict["pipeline"],
    )

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
            if frame is None:
                break

            ret, audio = model.process_frame(frame)
            if ret:
                audio_queue.put(audio)
        except mp.queues.Empty:
            continue
        except Exception as e:
            print(f"Error in process_frames: {e}")

    ret, audio = model.close_stream()
    if ret:
        audio_queue.put(audio)

    audio_queue.put(None)


def temp_audio_file(audio, sr):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file_path = temp_file.name
        sf.write(temp_file_path, audio, sr)
    return temp_file_path


def predict(file: str):
    frame_queue = mp.Queue(maxsize=40)
    audio_queue = mp.Queue()
    stop_event = mp.Event()

    config_dict = {
        "n_words": config.n_words,
        "spec_len": config.generator.max_length,
        "pipeline": config.pipeline,
    }

    acquire_process = mp.Process(
        target=acquire_frames, args=(file, frame_queue, stop_event)
    )

    process_process = mp.Process(
        target=process_frames,
        args=(frame_queue, audio_queue, config_dict, stop_event),
    )

    acquire_process.start()
    process_process.start()

    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break
        yield temp_audio_file(audio_chunk, config.generator.sr)

    acquire_process.join()
    process_process.join()


interface = gr.Interface(
    fn=predict,
    inputs=gr.Video(label="Input Video"),
    outputs=gr.Audio(label="Generated Audio", streaming=True, autoplay=True),
    allow_flagging="never",
    title="Sign2Speech",
)

if __name__ == "__main__":
    mp.freeze_support()
    interface.launch(share=True)
