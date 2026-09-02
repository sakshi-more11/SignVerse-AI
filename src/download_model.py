"""
One-time setup: downloads Google's pretrained hand-landmark detector model
(hand_landmarker.task, ~7.5MB) that HandDetector needs for the DESKTOP
scripts (data_collection.py, recognize_desktop.py).

Run this once before data_collection.py or recognize_desktop.py:
    python src/download_model.py

(The web app does NOT need this — the browser frontend loads the same
model directly from Google's CDN via JavaScript.)
"""
import os
import urllib.request

import config


def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    if os.path.exists(config.HAND_LANDMARKER_MODEL_PATH):
        print(f"Model already present at {config.HAND_LANDMARKER_MODEL_PATH}")
        return

    print(f"Downloading hand landmark model from:\n  {config.HAND_LANDMARKER_MODEL_URL}")
    urllib.request.urlretrieve(config.HAND_LANDMARKER_MODEL_URL, config.HAND_LANDMARKER_MODEL_PATH)
    print(f"Saved to {config.HAND_LANDMARKER_MODEL_PATH}")


if __name__ == "__main__":
    main()
