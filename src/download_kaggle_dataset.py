"""
One-time: download the public "ASL Alphabet" dataset from Kaggle
(87,000 images, 29 classes: A-Z + space/del/nothing) as an alternative to
recording your own gestures with data_collection.py.

Setup (one-time, ~2 minutes):
  1. Create a free Kaggle account: https://www.kaggle.com
  2. Go to https://www.kaggle.com/settings -> "API" section -> "Create New Token"
     This downloads a file called kaggle.json.
  3. Place it at:
       Linux/Mac: ~/.kaggle/kaggle.json
       Windows:   C:\\Users\\<you>\\.kaggle\\kaggle.json
     (On Linux/Mac, also run: chmod 600 ~/.kaggle/kaggle.json)
  4. pip install kaggle

Run:
    python src/download_kaggle_dataset.py

Downloads and unzips into data_raw/asl_alphabet_train/asl_alphabet_train/,
with one subfolder per class (A, B, C, ..., space, del, nothing).

This is a ~1GB download and may take a few minutes depending on your
connection.
"""
import os
import zipfile

import config

DATASET_SLUG = "grassknoted/asl-alphabet"
RAW_DIR = os.path.join(config.BASE_DIR, "data_raw")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    expected_dir = os.path.join(RAW_DIR, "asl_alphabet_train", "asl_alphabet_train")
    if os.path.isdir(expected_dir) and os.listdir(expected_dir):
        print(f"Dataset already present at {expected_dir}")
        return

    try:
        import kaggle
    except ImportError:
        raise SystemExit(
            "The 'kaggle' package isn't installed. Run: pip install kaggle\n"
            "Then make sure ~/.kaggle/kaggle.json exists (see docstring at the top of this file)."
        )

    print(f"Downloading '{DATASET_SLUG}' from Kaggle into {RAW_DIR} ...")
    print("(This is ~1GB and may take a few minutes.)")

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET_SLUG, path=RAW_DIR, unzip=True, quiet=False)

    if os.path.isdir(expected_dir):
        print(f"\nDone. Dataset extracted to: {expected_dir}")
        print("Next step: python src/extract_landmarks_from_dataset.py")
    else:
        print(
            "\nDownload finished, but the expected folder structure wasn't found.\n"
            f"Check the contents of {RAW_DIR} and adjust --raw-dir when running "
            "extract_landmarks_from_dataset.py if the folder layout differs."
        )


if __name__ == "__main__":
    main()
