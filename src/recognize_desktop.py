"""
Standalone desktop demo (no browser/backend needed): real-time webcam
inference + sentence building, all in one Python process.

Run:
    python src/recognize_desktop.py

Controls:
    c -> clear the sentence
    q -> quit

For the web app version, see webapp/ instead — same trained model, browser
UI, FastAPI backend.
"""
import json
import os
from collections import deque

import cv2
import torch
import torch.nn.functional as F

import config
from hand_detector import HandDetector, draw_landmarks
from model import ASLClassifier
from utils import extract_landmark_vector


def load_model():
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.LABELS_PATH):
        raise FileNotFoundError(
            "No trained model found. Run data_collection.py for each gesture, "
            "then train_model.py, before running recognize_desktop.py."
        )

    checkpoint = torch.load(config.MODEL_PATH, map_location="cpu")
    model = ASLClassifier(
        input_dim=checkpoint["input_dim"], num_classes=checkpoint["num_classes"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with open(config.LABELS_PATH) as f:
        labels = json.load(f)

    return model, labels


def main():
    model, labels = load_model()
    detector = HandDetector()

    cap = cv2.VideoCapture(config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    sentence = ""
    recent_preds = deque(maxlen=config.STABILITY_FRAMES)
    locked_label = None
    cooldown = 0

    print("Running. Press 'c' to clear sentence, 'q' to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        hands = detector.detect(frame)
        current_label, current_conf = None, 0.0

        if hands:
            landmarks = hands[0]
            draw_landmarks(frame, landmarks)

            vec = extract_landmark_vector(landmarks)
            x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1).squeeze(0)
                conf, idx = torch.max(probs, dim=0)

            current_conf = conf.item()
            if current_conf >= config.PREDICTION_CONFIDENCE_THRESHOLD:
                current_label = labels[idx.item()]

        recent_preds.append(current_label)

        if cooldown > 0:
            cooldown -= 1
        else:
            if (
                len(recent_preds) == config.STABILITY_FRAMES
                and current_label is not None
                and all(p == current_label for p in recent_preds)
                and current_label != locked_label
            ):
                if current_label == "SPACE":
                    sentence += " "
                elif current_label == "DEL":
                    sentence = sentence[:-1]
                else:
                    sentence += current_label

                locked_label = current_label
                cooldown = config.COOLDOWN_FRAMES
                recent_preds.clear()

        if current_label is None:
            locked_label = None

        # --- UI overlay ---
        overlay_h = 90
        cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_h), (30, 30, 30), -1)
        cv2.putText(frame, sentence[-45:], (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        pred_text = f"{current_label or '-'} ({current_conf*100:.0f}%)"
        cv2.putText(frame, pred_text, (15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "c: clear | q: quit", (frame.shape[1] - 260, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("SignVerseAI - Desktop Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            sentence = ""
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
