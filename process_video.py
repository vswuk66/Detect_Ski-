import os

import cv2
import argparse
import numpy as np
from sort.sort import Sort
from collections import deque
from tqdm import tqdm
from ultralytics import YOLO

# Function to calculate speed


def calculate_speed(displacement_pixels, time_interval_s, scale):
    speed_m_s = (displacement_pixels * scale) / time_interval_s
    return speed_m_s * 3.6  # Convert speed to km/h

# Command-line arguments


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Jet Ski tracking and speed estimation.")
    parser.add_argument(
        "--video",
        type=str,
        default="test_jetski.mp4",
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tracked_output.mp4",
        help="Path for saving the output video."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="runs/detect/train11/weights/best.onnx",
        help="Path to the YOLO ONNX model weights.",
    )
    return parser.parse_args()

# Main function


def process_video(args):
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(
            f"YOLO model file not found: {
                args.model_path}")

    JET_SKI_LENGTH = 3.0
    SMOOTHING_WINDOW = 10

    model = YOLO(args.model_path)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}, Resolution: {frame_width}x{frame_height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps,
                          (frame_width, frame_height))

    tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)
    scale = None
    object_data = {}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections_raw = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        conf_threshold = 0.5
        detections = detections_raw[confidences > conf_threshold]

        if detections.shape[0] > 0:
            detections = np.hstack((
                detections,
                np.zeros((detections.shape[0], 1)),  # Dummy class id column
                confidences[confidences > conf_threshold].reshape(-1, 1)
            ))
        else:
            detections = np.empty((0, 6))

        # Calculate scale if not already set
        if scale is None and len(detections) > 0:
            first_detection = detections[0]
            x1, y1, x2, y2 = first_detection[:4]
            width = x2 - x1
            if width > 0:
                scale = JET_SKI_LENGTH / width

        if detections is not None and len(detections) > 0:
            tracked_objects = tracker.update(detections)
        else:
            tracked_objects = []

        for track in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if track_id not in object_data:
                object_data[track_id] = {
                    "last_position": (cx, cy),
                    "speeds": deque(maxlen=SMOOTHING_WINDOW),
                }

            last_cx, last_cy = object_data[track_id]["last_position"]
            displacement_pixels = np.sqrt(
                (cx - last_cx) ** 2 + (cy - last_cy) ** 2)
            object_data[track_id]["last_position"] = (cx, cy)

            if scale is not None and fps > 0:
                speed_kmh = calculate_speed(
                    displacement_pixels, 1 / fps, scale)
                object_data[track_id]["speeds"].append(speed_kmh)
                smoothed_speed = np.mean(object_data[track_id]["speeds"])
            else:
                smoothed_speed = 0.0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track_id} Speed: {smoothed_speed:.2f} km/h",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")


if __name__ == "__main__":
    args = parse_arguments()
    try:
        process_video(args)
    except Exception as e:
        print(f"An error occurred: {e}")
