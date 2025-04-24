import os

import cv2
import pandas as pd

# Load class-to-index mapping
with open("data/asl-citizen/processed/classes.txt", "r") as f:
    class_to_index = {line.strip(): str(i) for i, line in enumerate(f)}

# Read CSV and group by participant
df = pd.read_csv("data/asl-citizen/raw/test_10.csv")
grouped = df.groupby("Participant ID")
os.makedirs("data/asl-citizen/processed/continuous_videos", exist_ok=True)

for id, group in grouped:
    print("Processing Participant", id)
    video_files = group["Video file"].tolist()
    glosses = group["Gloss"].tolist()

    # Convert glosses to index form
    indexed_glosses = [class_to_index.get(gloss, "UNK") for gloss in glosses]
    output_filename = (
        "data/asl-citizen/processed/continuous_videos/"
        + "_".join(indexed_glosses)
        + f"_{id}.mp4"
    )

    if len(video_files) == 0:
        print(f"No videos found for {id}")
        continue

    # Initialize video writer with properties of the first video
    first_video_path = f"data/asl-citizen/raw/videos/{video_files[0]}"
    if not os.path.exists(first_video_path):
        print(f"Missing file: {first_video_path}")
        continue

    cap = cv2.VideoCapture(first_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for video_file in video_files:
        video_path = f"data/asl-citizen/raw/videos/{video_file}"
        if not os.path.exists(video_path):
            print(f"Missing video: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"Successfully created {output_filename}")
