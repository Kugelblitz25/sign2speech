import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
)


def pose_features(path, clip_duration):
    data = []
    pose = mp.solutions.pose
    hands = mp.solutions.hands
    pose_detector = pose.Pose()
    hands_detector = hands.Hands()

    video = EncodedVideo.from_path(path)
    video_data = video.get_clip(start_sec=0, end_sec=clip_duration)
    sampled_video = transform(video_data)["video"].permute(1, 2, 3, 0).numpy()

    for frame in sampled_video:
        frame_rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
        pose_results = pose_detector.process(frame_rgb)
        pose_data = [0] * 44

        if pose_results.pose_landmarks:
            pose = []
            for lm in pose_results.pose_landmarks.landmark:
                pose.append(lm.x)
                pose.append(lm.y)
            pose_data = pose[0:44]

        hand_results = hands_detector.process(frame_rgb)
        left_hand_data = [0] * 42
        right_hand_data = [0] * 42

        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.append(lm.x)
                    hand_data.append(lm.y)

                if i == 0:
                    left_hand_data = hand_data
                else:
                    right_hand_data = hand_data

        data.extend(pose_data + left_hand_data + right_hand_data)

    return data


num_frames = 64
sampling_rate = 8
frames_per_second = 25
side_size = 256
crop_size = 256
clip_duration = (num_frames * sampling_rate) / frames_per_second

data = []
source = "data/asl-citizen/processed/videos"
output = "experiments/pose_features_train_10.csv"
transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size=(crop_size, crop_size)),
        ]
    ),
)

train = pd.read_csv("data/asl-citizen/processed/train_10.csv")

for file in train["Video file"]:
    path = os.path.join(source, file)
    features = pose_features(path, clip_duration)
    data.append([file] + features)

num_features = len(data[0]) - 1
columns = ["video"] + [f"feature_{i}" for i in range(num_features)]
pd.DataFrame(data, columns=columns).to_csv(output, index=False)
