"""
Helper functions to turn a MediaPipe hand-landmark detection into a
fixed-length, scale/position-invariant feature vector.
"""
import numpy as np


def extract_landmark_vector(landmarks):
    """
    Convert a list of 21 MediaPipe landmark points (each with .x/.y/.z) into
    a normalized, flattened 63-dim numpy feature vector.

    Normalization steps:
      1. Translate so the wrist (landmark 0) is the origin.
      2. Scale by the maximum distance from the wrist to any other
         landmark, so the feature vector is invariant to how close the
         hand is to the camera.

    This is what replaces the old skin-mask-and-HOG approach — it's far
    more robust because it depends on hand geometry, not on lighting or
    skin-color thresholds. The SAME normalization is reimplemented in
    JavaScript in webapp/frontend/app.js so the browser and the trained
    model agree on feature representation.
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    )

    wrist = coords[0].copy()
    coords -= wrist

    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 1e-6:
        coords /= max_dist

    return coords.flatten()  # shape (63,)


def landmarks_to_pixel_points(landmarks, frame_width, frame_height):
    """Convert normalized MediaPipe landmarks to pixel coordinates for drawing."""
    return [
        (int(lm.x * frame_width), int(lm.y * frame_height))
        for lm in landmarks
    ]
