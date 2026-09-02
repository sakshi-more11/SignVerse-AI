"""
SignVerseAI backend — FastAPI + WebSocket.

Architecture (Option A — browser does hand detection, backend only classifies):

    Browser (MediaPipe HandLandmarker, JS/WASM)
        --> extracts 21 hand landmarks per frame, normalizes them
        --> sends the 63-float feature vector over a WebSocket
    FastAPI backend
        --> runs the trained PyTorch model on the vector
        --> returns {label, confidence} JSON back over the same socket

This keeps video frames entirely on the client (fast, private, low
bandwidth) — only ~63 numbers per frame cross the network, not video.

Run:
    cd webapp/backend
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in a browser.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Reuse the model/config defined in ../../src so there is a single source
# of truth shared between the desktop pipeline and this backend.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

import config  # noqa: E402
from model import ASLClassifier  # noqa: E402

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

app = FastAPI(title="SignVerseAI API")

_model = None
_labels = None


def get_model():
    """Lazy-load the trained model once, on first use."""
    global _model, _labels
    if _model is None:
        if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.LABELS_PATH):
            return None, None
        checkpoint = torch.load(config.MODEL_PATH, map_location="cpu")
        model = ASLClassifier(
            input_dim=checkpoint["input_dim"], num_classes=checkpoint["num_classes"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with open(config.LABELS_PATH) as f:
            labels = json.load(f)
        _model, _labels = model, labels
    return _model, _labels


class PredictRequest(BaseModel):
    landmarks: list[float]  # 63-dim normalized landmark feature vector


@app.get("/api/health")
def health():
    model, labels = get_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "num_classes": len(labels) if labels else 0,
        "labels": labels or [],
    }


@app.post("/api/predict")
def predict_http(req: PredictRequest):
    """One-shot HTTP prediction endpoint (fallback if WebSocket isn't used)."""
    model, labels = get_model()
    if model is None:
        return {"error": "Model not trained yet. Run src/train_model.py first."}

    if len(req.landmarks) != model.net[0].in_features:
        return {"error": f"Expected {model.net[0].in_features} values, got {len(req.landmarks)}."}

    x = torch.tensor(req.landmarks, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)

    return {"label": labels[idx.item()], "confidence": conf.item()}


@app.websocket("/ws/predict")
async def predict_ws(websocket: WebSocket):
    """
    Streaming prediction endpoint. Client sends:
        {"landmarks": [63 floats]}   -> when a hand is detected
        {"landmarks": null}          -> when no hand is detected this frame

    Server replies:
        {"label": "A", "confidence": 0.94}
        {"label": None, "confidence": 0.0}    -> no hand / low confidence
        {"error": "..."}                       -> model not trained yet
    """
    await websocket.accept()
    model, labels = get_model()

    if model is None:
        await websocket.send_json(
            {"error": "Model not trained yet. Run src/train_model.py first, then restart the server."}
        )

    try:
        while True:
            data = await websocket.receive_json()
            landmarks = data.get("landmarks")

            if model is None:
                model, labels = get_model()
                if model is None:
                    await websocket.send_json({"error": "Model still not trained."})
                    continue

            if not landmarks:
                await websocket.send_json({"label": None, "confidence": 0.0})
                continue

            x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1).squeeze(0)
                conf, idx = torch.max(probs, dim=0)

            label = labels[idx.item()] if conf.item() >= config.PREDICTION_CONFIDENCE_THRESHOLD else None
            await websocket.send_json({"label": label, "confidence": conf.item()})

    except WebSocketDisconnect:
        pass


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# Serve style.css, app.js, and any other static frontend assets
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
