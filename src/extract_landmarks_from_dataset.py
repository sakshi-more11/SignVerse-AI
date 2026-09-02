"""
Converts a folder-per-letter ASL image dataset into the same
normalized-landmark CSV format that data_collection.py produces — so
train_model.py works completely unchanged, regardless of which data
source you used.

Works with EITHER:
  - The Kaggle "ASL Alphabet" dataset (after download_kaggle_dataset.py),
    layout: <root>/asl_alphabet_train/asl_alphabet_train/<LABEL>/*.jpg
    (includes 'space' and 'del' folders)
  - The khansa3999/ASL-Alphabet-Dataset GitHub repo
    (https://github.com/khansa3999/ASL-Alphabet-Dataset), layout:
    <cloned-repo>/Data/train/<LETTER>/*.jpg
    (26 letters only — no space/del folders, so you'll still need to
    record those two yourself with data_collection.py --gesture SPACE
    and --gesture DEL, ~2 minutes total)

Run (after download_model.py):
    python src/extract_landmarks_from_dataset.py --raw-dir <path-to-letter-folders>

Options:
    --raw-dir            path to the folder whose subfolders are per-letter
                          image directories (default: Kaggle layout, see above)
    --max-per-class      how many images to sample per class (default 300 —
                         plenty for this landmark-based model; using the
                         full 4000/class would just make extraction slower
                         for no real accuracy gain)

What it does:
    - Maps folder names to this project's gesture labels:
        'A'..'Y' (skips J and Z — dynamic gestures, can't be captured
                  in a single still frame, same reason src/config.py
                  excludes them)
        'space'  -> 'SPACE'   (only present in the Kaggle layout)
        'del'    -> 'DEL'     (only present in the Kaggle layout)
        'nothing' is skipped (no equivalent gesture in this project)
    - Runs MediaPipe hand-landmark detection on each sampled image
    - Normalizes landmarks with the exact same function used by
      data_collection.py (utils.extract_landmark_vector), so vectors
      from dataset images and vectors from your own webcam are on
      identical footing
    - Appends rows to data/<LABEL>.csv, same file train_model.py reads

Images where MediaPipe can't find a hand (occasional, especially near
frame edges or unusual crops) are silently skipped and counted.

IMPORTANT — tight-crop datasets: some public ASL datasets (including the
Kaggle "asl_alphabet" set and its GitHub mirrors) crop the hand to fill
almost the entire frame. MediaPipe's palm detector is trained on natural
photos where a hand occupies a smaller portion of a larger scene, so it
frequently fails on these edge-to-edge crops. To compensate, this script
pads every image with a replicated border before detection (effectively
shrinking the hand's footprint in the frame) and retries with a
progressively larger pad and a lower confidence threshold if the first
attempt fails.
"""
import argparse
import csv
import os
import random

import cv2
import numpy as np

import config
from hand_detector import HandDetector
from utils import extract_landmark_vector

DEFAULT_RAW_DIR = os.path.join(config.BASE_DIR, "data_raw", "asl_alphabet_train", "asl_alphabet_train")

# Kaggle folder name (case-insensitive) -> this project's gesture label.
# Letters map to themselves; J and Z are intentionally left unmapped (skipped).
LABEL_MAP = {letter: letter for letter in config.LETTERS}
LABEL_MAP.update({"space": "SPACE", "del": "DEL", "delete": "DEL"})

# Increasing pad fractions to retry with if detection fails on the raw image.
# 0.0 = try the image as-is first (cheap, works fine for naturally-framed photos).
PAD_ATTEMPTS = [0.0, 0.5, 1.0, 1.5]


def pad_image(frame, pad_fraction):
    if pad_fraction <= 0:
        return frame
    h, w = frame.shape[:2]
    pad_h, pad_w = int(h * pad_fraction), int(w * pad_fraction)
    return cv2.copyMakeBorder(
        frame, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_REPLICATE
    )


def detect_with_retry(detector, frame):
    """Try detection on the raw image, then progressively larger padding."""
    for pad_fraction in PAD_ATTEMPTS:
        candidate = pad_image(frame, pad_fraction)
        hands = detector.detect(candidate)
        if hands:
            return hands
    return []


def resolve_label(folder_name):
    if folder_name in LABEL_MAP:
        return LABEL_MAP[folder_name]
    if folder_name.lower() in LABEL_MAP:
        return LABEL_MAP[folder_name.lower()]
    return None  # covers 'nothing', 'J', 'Z', and anything unexpected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isdir(args.raw_dir):
        raise SystemExit(
            f"Raw dataset folder not found at:\n  {args.raw_dir}\n"
            "Run `python src/download_kaggle_dataset.py` first, or pass "
            "--raw-dir pointing at wherever you extracted it."
        )

    random.seed(args.seed)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    # Lower confidence than the live-webcam default (config.MIN_DETECTION_CONFIDENCE)
    # since these are static, sometimes tightly-cropped dataset photos, not a
    # live tracked video stream — a little more leniency here is a reasonable
    # trade for coverage, and bad detections are a small minority diluted by
    # hundreds of good samples per class.
    detector = HandDetector(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    class_folders = sorted(os.listdir(args.raw_dir))
    if not class_folders:
        raise SystemExit(f"No class subfolders found in {args.raw_dir}")

    grand_total_written = 0
    grand_total_skipped = 0

    for folder_name in class_folders:
        folder_path = os.path.join(args.raw_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label = resolve_label(folder_name)
        if label is None:
            print(f"Skipping '{folder_name}' (no matching gesture in this project)")
            continue

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(image_files)
        sample = image_files[: args.max_per_class]

        out_path = os.path.join(config.DATA_DIR, f"{label}.csv")
        file_exists = os.path.exists(out_path)

        written, skipped = 0, 0
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([f"f{i}" for i in range(63)] + ["label"])

            for fname in sample:
                img_path = os.path.join(folder_path, fname)
                frame = cv2.imread(img_path)
                if frame is None:
                    skipped += 1
                    continue

                hands = detect_with_retry(detector, frame)
                if not hands:
                    skipped += 1
                    continue

                vec = extract_landmark_vector(hands[0])
                writer.writerow(list(vec) + [label])
                written += 1

        print(f"{folder_name:>8} -> {label:<6} | wrote {written:4d}, skipped {skipped:4d} "
              f"(no hand detected) of {len(sample)} sampled")
        grand_total_written += written
        grand_total_skipped += skipped

    detector.close()
    print(f"\nDone. {grand_total_written} landmark rows written across data/*.csv "
          f"({grand_total_skipped} images skipped — no hand detected).")
    print("Next step: python src/train_model.py")


if __name__ == "__main__":
    main()