import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


def init_mediapipe():
    """Initialize MediaPipe solutions for holistic detection"""
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1,
        static_image_mode=False,
    )
    return holistic


def get_frame_keypoints(frame, holistic):
    # Process the frame
    results = holistic.process(frame)

    left_hand_points = []
    right_hand_points = []

    # Extract left hand landmarks (21 points)
    if results.left_hand_landmarks:
        left_hand_points = [
            [landmark.x, landmark.y, landmark.z * 0]
            for landmark in results.left_hand_landmarks.landmark
        ]
    else:
        left_hand_points = [[0, 0, 0]] * 21

    # Extract right hand landmarks (21 points)
    if results.right_hand_landmarks:
        right_hand_points = [
            [landmark.x, landmark.y, landmark.z * 0]
            for landmark in results.right_hand_landmarks.landmark
        ]
    else:
        right_hand_points = [[0, 0, 0]] * 21

    # Combine all points
    all_points = np.array(left_hand_points + right_hand_points)
    return all_points


def get_err_and_range(tot_frames, start_frame, end_frame):
    start_frame = max(0, start_frame - 10)
    end_frame = min(end_frame + 10, tot_frames)

    cropped_range = end_frame - start_frame
    cropped_mid = (end_frame + start_frame) / 2
    act_mid = tot_frames / 2
    modified_range = 2 * max(act_mid - start_frame, end_frame - act_mid)
    return abs(cropped_mid - act_mid) * 100 / cropped_range, modified_range


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    holistic = init_mediapipe()

    all_frames_keypoints = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_keypoints = get_frame_keypoints(frame, holistic)
        all_frames_keypoints.append(frame_keypoints)

    cap.release()
    holistic.close()

    keypoints_array = np.array(all_frames_keypoints)
    frames, points, channels = keypoints_array.shape
    normalized = np.zeros((points, frames, channels))

    # Normalize each channel independently
    for c in range(channels):
        channel_data = keypoints_array[:, :, c].T
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)

        # Avoid division by zero
        if max_val - min_val != 0:
            normalized[:, :, c] = (channel_data - min_val) / (max_val - min_val)
        else:
            normalized[:, :, c] = channel_data

    return (255 * normalized).astype("uint8")


def get_min_max(path):
    img = process_video(path)
    diff_image = np.abs(np.diff(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=1))

    x_activity = np.sum(diff_image, axis=0) / diff_image.shape[0]
    abv_thresh = np.argwhere(x_activity >= 20)
    if len(abv_thresh) == 0:
        return 0, img.shape[1], img.shape[1]
    x_loc_min = abv_thresh.min() + 1
    x_loc_max = abv_thresh.max() + 1
    return x_loc_min, x_loc_max, img.shape[1]


if __name__ == "__main__":
    df = pd.read_csv("data/asl-citizen/raw/test_10.csv")
    video_files = df["Video file"].to_list()

    data = []

    for video in video_files:
        video_path = f"data/asl-citizen/raw/videos/{video}"
        f_min, f_max, f_tot = get_min_max(video_path)
        err, rng = get_err_and_range(f_tot, f_min, f_max)

        data.append([f_min, f_max, f_tot, round(err, 2), rng])

        new_df = pd.DataFrame(
            data,
            columns=["start_frame", "end_frame", "total_frames", "rel_err", "range"],
        )
        new_df["Video file"] = df["Video file"]
        new_df.to_csv("experiments/vid_length.csv", index=False)
