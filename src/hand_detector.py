"""
Thin wrapper around MediaPipe's modern Tasks API (HandLandmarker), used by
the desktop data-collection and demo scripts.

Note: mediapipe >= 0.10.x removed the old `mp.solutions.hands` API that most
older tutorials/repos use. This project uses the current, actively
maintained `mediapipe.tasks.python.vision.HandLandmarker` API instead, so
it keeps working with `pip install mediapipe` going forward.

Requires the hand_landmarker.task model file — run download_model.py once
before using this (see README).
"""
import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

import config

HAND_CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]


class HandDetector:
    """Wraps HandLandmarker for simple per-frame detection on a webcam stream."""

    def __init__(self, model_path=None, min_detection_confidence=None, min_tracking_confidence=None):
        model_path = model_path or config.HAND_LANDMARKER_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmark model not found at {model_path}.\n"
                "Run `python src/download_model.py` first."
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=min_detection_confidence or config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=min_tracking_confidence or config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, bgr_frame):
        """
        Run detection on a single BGR OpenCV frame.
        Returns a list of landmarks (each a list of 21 points with .x/.y/.z)
        for every detected hand, or an empty list if none found.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33  # ~30fps virtual timestamp, must be monotonically increasing
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        return result.hand_landmarks  # list of lists of NormalizedLandmark

    def close(self):
        self._landmarker.close()


def draw_landmarks(frame, landmarks):
    """Draw hand skeleton on a BGR frame given a list of NormalizedLandmark points."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 100, 255), -1)