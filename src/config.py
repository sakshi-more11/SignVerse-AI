"""
Central configuration for gesture labels, camera settings, and file paths.
"""
import os

# ---- Gesture classes ----
# Static ASL alphabet letters (J and Z excluded — they require motion,
# which a single-frame landmark classifier can't capture).
LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
]

# Custom control gestures (not part of the ASL alphabet itself,
# used the same way the original inspiration repo used "thumbs up" / "five fingers").
CONTROL_GESTURES = {
    "SPACE": "Thumbs up (👍) — inserts a space",
    "DEL": "Open palm, fingers spread (🖐) — deletes the last character",
}

GESTURES = LETTERS + list(CONTROL_GESTURES.keys())

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "asl_model.pth")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
HAND_LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# ---- Data collection ----
SAMPLES_PER_GESTURE = 200   # how many landmark frames to capture per gesture
CAM_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

# ---- MediaPipe ----
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# ---- Recognition / inference ----
PREDICTION_CONFIDENCE_THRESHOLD = 0.75
STABILITY_FRAMES = 12       # consecutive identical predictions needed to "lock in" a letter
COOLDOWN_FRAMES = 15        # frames to wait after locking a letter before accepting a new one
