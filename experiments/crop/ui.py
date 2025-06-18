import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import pandas as pd

from utils.common import create_subset


@dataclass
class ComparisonStats:
    total_comparisons: int = 0
    hand_crop_selected: int = 0
    motion_crop_selected: int = 0
    none_selected: int = 0

    @property
    def hand_crop_percent(self) -> float:
        return (
            (self.hand_crop_selected / self.total_comparisons * 100)
            if self.total_comparisons > 0
            else 0
        )

    @property
    def motion_crop_percent(self) -> float:
        return (
            (self.motion_crop_selected / self.total_comparisons * 100)
            if self.total_comparisons > 0
            else 0
        )

    @property
    def none_percent(self) -> float:
        return (
            (self.none_selected / self.total_comparisons * 100)
            if self.total_comparisons > 0
            else 0
        )


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


class BackgroundRemover:
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )

    def remove_background(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(frame_rgb)

        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_frame = np.zeros(frame.shape, dtype=np.uint8)
        bg_frame[:] = (0, 0, 0)

        output_frame = np.where(condition, frame, bg_frame)
        return output_frame


class MotionDetector:
    def __init__(self):
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def detect_motion_regions(self, frames: List[np.ndarray]) -> List[float]:
        if len(frames) < 2:
            return [0.0] * len(frames)

        motion_scores = [0.0]

        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            p0 = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
            )

            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, p0, None, **self.lk_params
                )

                good_new = p1[st == 1]
                good_old = p0[st == 1]

                if len(good_new) > 0 and len(good_old) > 0:
                    motion_magnitude = np.mean(
                        np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
                    )
                    motion_scores.append(motion_magnitude)
                else:
                    motion_scores.append(0.0)
            else:
                motion_scores.append(0.0)

        return motion_scores


class VideoCroppingApp:
    def __init__(self, csv_path: str, video_root: str):
        self.csv_path = csv_path
        self.video_root = video_root
        self.df = pd.read_csv(self.csv_path)
        self.current_video_index = None
        self.stats = ComparisonStats()
        self.hand_detector = HandDetector()
        self.background_remover = BackgroundRemover()
        self.motion_detector = MotionDetector()

    def get_random_video(self):
        return random.randint(0, len(self.df) - 1)

    def get_video_path(self, index: int) -> str:
        video_file = self.df.iloc[index]["Video file"]
        return os.path.join(self.video_root, video_file)

    def get_gloss(self, index: int) -> str:
        return self.df.iloc[index]["Gloss"]

    def crop_video_by_hands(self, video_path: str) -> Tuple[str, int]:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        hand_detection_status = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            has_hands = self.hand_detector.detect_hands(frame)
            hand_detection_status.append(has_hands)

        cap.release()

        start_idx = None
        consecutive_visible = 0
        for i, has_hands in enumerate(hand_detection_status):
            if has_hands:
                consecutive_visible += 1
                if consecutive_visible >= 5 and start_idx is None:
                    start_idx = max(0, i - 4)
                    break
            else:
                consecutive_visible = 0

        end_idx = None
        if start_idx is not None:
            consecutive_visible = 0
            for i in range(len(hand_detection_status) - 1, -1, -1):
                if hand_detection_status[i]:
                    consecutive_visible += 1
                    if consecutive_visible >= 5 and end_idx is None:
                        end_idx = i + 5
                        break
                else:
                    consecutive_visible = 0

        if start_idx is None:
            start_idx = 0
        if end_idx is None or end_idx <= start_idx:
            end_idx = len(frames) - 1

        cropped_path = "hand_cropped.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cropped_path, fourcc, fps, (width, height))

        for i in range(start_idx, end_idx + 1):
            bg_removed_frame = self.background_remover.remove_background(frames[i])
            out.write(bg_removed_frame)

        out.release()
        return cropped_path, end_idx - start_idx + 1

    def crop_video_by_motion(self, video_path: str) -> Tuple[str, int]:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        motion_scores = self.motion_detector.detect_motion_regions(frames)

        motion_threshold = np.mean(motion_scores) + np.std(motion_scores) * 0.5
        high_motion_indices = [
            i for i, score in enumerate(motion_scores) if score > motion_threshold
        ]

        if not high_motion_indices:
            start_idx, end_idx = 0, len(frames) - 1
        else:
            start_idx = max(0, min(high_motion_indices) - 5)
            end_idx = min(len(frames) - 1, max(high_motion_indices) + 5)

        cropped_path = "cropped.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cropped_path, fourcc, fps, (width, height))

        for i in range(start_idx, end_idx + 1):
            bg_removed_frame = self.background_remover.remove_background(frames[i])
            out.write(bg_removed_frame)

        out.release()
        return cropped_path, end_idx - start_idx + 1

    def load_video(self):
        index = self.get_random_video()
        self.current_video_index = index
        video_path = self.get_video_path(index)
        gloss = self.get_gloss(index)

        hand_cropped_path, hand_frame_count = self.crop_video_by_hands(video_path)
        motion_cropped_path, motion_frame_count = self.crop_video_by_motion(video_path)

        self.current_hand_frame_count = hand_frame_count
        self.current_motion_frame_count = motion_frame_count

        return (
            video_path,
            hand_cropped_path,
            motion_cropped_path,
            gloss,
            self.generate_stats_html(),
        )

    def select_hand_crop(self):
        self.stats.total_comparisons += 1
        self.stats.hand_crop_selected += 1
        return self.load_video()[:-1] + (self.generate_stats_html(),)

    def select_motion_crop(self):
        self.stats.total_comparisons += 1
        self.stats.motion_crop_selected += 1
        return self.load_video()[:-1] + (self.generate_stats_html(),)

    def select_none(self):
        self.stats.total_comparisons += 1
        self.stats.none_selected += 1
        return self.load_video()[:-1] + (self.generate_stats_html(),)

    def generate_stats_html(self) -> str:
        if self.stats.total_comparisons == 0:
            return "<h3>No comparisons made yet</h3>"

        html = f"""
        <h3>Comparison Statistics</h3>
        <p>Total comparisons: {self.stats.total_comparisons}</p>
        <div style="margin: 10px 0;">
            <p><strong>Hand-based cropping:</strong> {self.stats.hand_crop_selected} ({self.stats.hand_crop_percent:.1f}%)</p>
            <p><strong>Motion-based cropping:</strong> {self.stats.motion_crop_selected} ({self.stats.motion_crop_percent:.1f}%)</p>
            <p><strong>Neither option good:</strong> {self.stats.none_selected} ({self.stats.none_percent:.1f}%)</p>
        </div>
        """

        return html


def create_interface(csv_path, video_root):
    app = VideoCroppingApp(csv_path, video_root)

    with gr.Blocks(title="Video Cropping Comparison") as interface:
        gr.Markdown("# Video Cropping Comparison Tool")
        gr.Markdown(
            "Compare hand-based cropping vs motion-based cropping with background removal"
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Original Video")
                original_video = gr.Video(label="Original")
            with gr.Column():
                gr.Markdown("### Hand-based Crop (with bg removal)")
                hand_cropped_video = gr.Video(label="Hand Cropped")
            with gr.Column():
                gr.Markdown("### Motion-based Crop (with bg removal)")
                motion_cropped_video = gr.Video(label="Motion Cropped")

        gloss = gr.Textbox(label="Gloss", interactive=False)
        stats = gr.HTML(label="Statistics")

        with gr.Row():
            hand_btn = gr.Button("Select Hand-based Crop", variant="primary", size="lg")
            motion_btn = gr.Button(
                "Select Motion-based Crop", variant="primary", size="lg"
            )
            none_btn = gr.Button("Neither is Good", variant="secondary", size="lg")

        with gr.Row():
            next_btn = gr.Button("Skip to Next Video", variant="secondary")

        original, hand_cropped, motion_cropped, current_gloss, initial_stats = (
            app.load_video()
        )
        original_video.value = original
        hand_cropped_video.value = hand_cropped
        motion_cropped_video.value = motion_cropped
        gloss.value = current_gloss
        stats.value = initial_stats

        hand_btn.click(
            fn=app.select_hand_crop,
            outputs=[
                original_video,
                hand_cropped_video,
                motion_cropped_video,
                gloss,
                stats,
            ],
        )

        motion_btn.click(
            fn=app.select_motion_crop,
            outputs=[
                original_video,
                hand_cropped_video,
                motion_cropped_video,
                gloss,
                stats,
            ],
        )

        none_btn.click(
            fn=app.select_none,
            outputs=[
                original_video,
                hand_cropped_video,
                motion_cropped_video,
                gloss,
                stats,
            ],
        )

        next_btn.click(
            fn=app.load_video,
            outputs=[
                original_video,
                hand_cropped_video,
                motion_cropped_video,
                gloss,
                stats,
            ],
        )

    return interface


if __name__ == "__main__":
    CSV_PATH = create_subset("data/asl-citizen/raw/train.csv", 10)
    VIDEO_ROOT = "data/asl-citizen/raw/videos"

    interface = create_interface(CSV_PATH, VIDEO_ROOT)
    interface.launch(share=True)
