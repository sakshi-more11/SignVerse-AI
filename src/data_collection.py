"""
Step 1 of the pipeline: capture your own labeled hand-landmark data.

First time only:
    python src/download_model.py

Then for every gesture in config.GESTURES:
    python src/data_collection.py --gesture A
    python src/data_collection.py --gesture B
    ... repeat for every letter, plus SPACE and DEL.

Controls while the window is focused:
    SPACE bar -> start/stop capturing samples for the current gesture
    q         -> quit

Each run appends rows to data/<GESTURE>.csv. You can re-run to add more
samples (e.g. from different angles/lighting) at any time before training.
"""
import argparse
import csv
import os

import cv2

import config
from hand_detector import HandDetector, draw_landmarks
from utils import extract_landmark_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gesture", required=True, choices=config.GESTURES,
        help="Which gesture label you are about to record.",
    )
    parser.add_argument(
        "--samples", type=int, default=config.SAMPLES_PER_GESTURE,
        help="How many frames to capture.",
    )
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)
    out_path = os.path.join(config.DATA_DIR, f"{args.gesture}.csv")
    file_exists = os.path.exists(out_path)

    detector = HandDetector()
    cap = cv2.VideoCapture(config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    recording = False
    collected = 0

    print(f"\n=== Collecting data for gesture: '{args.gesture}' ===")
    print("Position your hand, then press SPACE to start/stop recording.")
    print("Press 'q' to quit.\n")

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [f"f{i}" for i in range(63)] + ["label"]
            writer.writerow(header)

        while cap.isOpened() and collected < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            hands = detector.detect(frame)
            if hands:
                landmarks = hands[0]
                draw_landmarks(frame, landmarks)

                if recording:
                    vec = extract_landmark_vector(landmarks)
                    writer.writerow(list(vec) + [args.gesture])
                    collected += 1

            status = "RECORDING" if recording else "PAUSED"
            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(frame, f"Gesture: {args.gesture}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"[{status}] {collected}/{args.samples}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, "SPACE: toggle recording | q: quit", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Data Collection - SignVerseAI", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = not recording
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"\nSaved {collected} samples to {out_path}")


if __name__ == "__main__":
    main()
