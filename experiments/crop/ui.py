import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import gradio as gr
import mediapipe as mp
import pandas as pd


@dataclass
class Stats:
    total_seen: int = 0
    accepted: int = 0
    min_frames: Optional[int] = None
    max_frames: Optional[int] = None
    total_frames: int = 0

    @property
    def percent_accepted(self) -> float:
        return (self.accepted / self.total_seen * 100) if self.total_seen > 0 else 0

    @property
    def avg_frames(self) -> float:
        return (self.total_frames / self.accepted) if self.accepted > 0 else 0

    def update_frames(self, frame_count: int) -> None:
        if self.min_frames is None or frame_count < self.min_frames:
            self.min_frames = frame_count
        if self.max_frames is None or frame_count > self.max_frames:
            self.max_frames = frame_count
        self.total_frames += frame_count


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results.multi_hand_landmarks is not None


class VideoCroppingApp:
    def __init__(self, csv_path: str, video_root: str):
        self.csv_path = csv_path
        self.video_root = video_root
        self.df = pd.read_csv(self.csv_path)
        self.current_video_index = None
        self.stats = Stats()
        self.hand_detector = HandDetector()

    def get_random_video(self):
        return random.randint(0, len(self.df) - 1)

    def get_video_path(self, index: int) -> str:
        video_file = self.df.iloc[index]["Video file"]
        return os.path.join(self.video_root, video_file)

    def get_gloss(self, index: int) -> str:
        return self.df.iloc[index]["Gloss"]

    def crop_video(self, video_path: str) -> Tuple[str, int]:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        hand_detection_status = []

        # Read all frames from the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            has_hands = self.hand_detector.detect_hands(frame)
            hand_detection_status.append(has_hands)

        cap.release()

        print(hand_detection_status)

        # Find start index (when hands are visible for 5 consecutive frames)
        start_idx = None
        consecutive_visible = 0
        for i, has_hands in enumerate(hand_detection_status):
            if has_hands:
                consecutive_visible += 1
                if consecutive_visible >= 5 and start_idx is None:
                    start_idx = max(0, i - 4)  # Go back to first frame of the 5
                    break
            else:
                consecutive_visible = 0

        # Find end index (starting from the end and looking for the last hand appearance)
        end_idx = None
        if start_idx is not None:
            consecutive_visible = 0
            for i in range(len(hand_detection_status) - 1, -1, -1):
                if hand_detection_status[i]:
                    consecutive_visible += 1
                    if consecutive_visible >= 5 and end_idx is None:
                        end_idx = i + 5  # 5 frames buffer after last hand appearance
                        break
                else:
                    consecutive_visible = 0

        if start_idx is None:
            start_idx = 0
        if end_idx is None or end_idx <= start_idx:
            end_idx = len(frames) - 1

        # Create cropped video
        cropped_path = f"{os.path.splitext(video_path)[0]}_cropped.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cropped_path, fourcc, fps, (width, height))

        for i in range(start_idx, end_idx + 1):
            out.write(frames[i])

        out.release()

        return cropped_path, end_idx - start_idx + 1

    def load_video(self):
        index = self.get_random_video()
        self.current_video_index = index
        video_path = self.get_video_path(index)
        gloss = self.get_gloss(index)

        cropped_path, frame_count = self.crop_video(video_path)
        self.current_frame_count = frame_count

        return video_path, cropped_path, gloss, self.generate_stats_html()

    def accept_video(self):
        self.stats.total_seen += 1
        self.stats.accepted += 1
        self.stats.update_frames(self.current_frame_count)
        return self.load_video()[:-1] + (self.generate_stats_html(),)

    def reject_video(self):
        self.stats.total_seen += 1
        return self.load_video()[:-1] + (self.generate_stats_html(),)

    def generate_stats_html(self) -> str:
        if self.stats.total_seen == 0:
            return "<h3>No videos reviewed yet</h3>"

        html = f"""
        <h3>Statistics</h3>
        <p>Total videos seen: {self.stats.total_seen}</p>
        <p>Accepted: {self.stats.accepted} ({self.stats.percent_accepted:.1f}%)</p>
        """

        if self.stats.accepted > 0:
            html += f"""
            <p>Frames in accepted videos:</p>
            <ul>
                <li>Min: {self.stats.min_frames}</li>
                <li>Max: {self.stats.max_frames}</li>
                <li>Avg: {self.stats.avg_frames:.1f}</li>
            </ul>
            """

        return html


def create_interface(csv_path, video_root):
    app = VideoCroppingApp(csv_path, video_root)

    with gr.Blocks() as interface:
        with gr.Row():
            original_video = gr.Video(label="Original Video")
            cropped_video = gr.Video(label="Cropped Video")

        gloss = gr.Textbox(label="Gloss")
        stats = gr.HTML(label="Statistics")

        with gr.Row():
            accept_btn = gr.Button("Accept", variant="primary")
            reject_btn = gr.Button("Reject", variant="secondary")
            next_btn = gr.Button("Next Video")

        # Initialize with first video
        original, cropped, current_gloss, initial_stats = app.load_video()
        original_video.value = original
        cropped_video.value = cropped
        gloss.value = current_gloss
        stats.value = initial_stats

        # Button events
        accept_btn.click(
            fn=app.accept_video, outputs=[original_video, cropped_video, gloss, stats]
        )

        reject_btn.click(
            fn=app.reject_video, outputs=[original_video, cropped_video, gloss, stats]
        )

        next_btn.click(
            fn=app.load_video, outputs=[original_video, cropped_video, gloss, stats]
        )

    return interface


if __name__ == "__main__":
    # Replace with your CSV and video root paths
    CSV_PATH = "data/asl-citizen/raw/train_10.csv"
    VIDEO_ROOT = "data/asl-citizen/raw/videos"

    interface = create_interface(CSV_PATH, VIDEO_ROOT)
    interface.launch(share=True)
