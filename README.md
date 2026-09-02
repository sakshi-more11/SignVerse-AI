# 🤟 SignVerseAI — Real-Time Sign Language Translator

**A full-stack, real-time American Sign Language (ASL) alphabet-to-text translator.** A webcam feed is processed with MediaPipe hand-landmark detection, then classified letter-by-letter by a PyTorch neural network — building a live sentence with dedicated gestures for space and delete.

Two interfaces, one shared model: a polished animated **web app** (FastAPI + WebSocket + vanilla JS) and a lightweight **desktop demo** (pure OpenCV).

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat&logo=google&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![WebSocket](https://img.shields.io/badge/WebSocket-realtime-blue)
![License](https://img.shields.io/badge/License-MIT-green)


---

## ✨ Features

- 🖐️ **Real-time hand tracking** via MediaPipe's `HandLandmarker` (21 3D keypoints) — robust to lighting and background, unlike classical skin-color-segmentation approaches
- 🧠 **PyTorch neural network classifier** trained on normalized hand-landmark geometry, not raw pixels — lightweight (~50KB), trains in seconds
- 🔤 **24 static ASL alphabet letters** (A–Y, excluding J/Z, which require motion a single-frame classifier can't capture)
- 📝 **Live sentence building** — 👍 thumbs-up for space, 🖐️ open palm for delete
- 🎯 **98% test accuracy** across 26 gesture classes, trained on 4,969 labeled samples
- 🌐 **Full-stack web app** — browser does hand detection (MediaPipe WASM), FastAPI backend classifies over WebSocket, animated "signal scanner" UI with a live captioning strip
- 🖥️ **Standalone desktop mode** — single-command OpenCV demo, no server required
- 🎚️ **Stability-based debouncing** — a letter only locks in after ~12 consecutive stable frames, eliminating jittery misfires

---

## 🎬 Demo

| Desktop Mode | Web App |
|---|---|
| Single-window OpenCV feed with live sentence overlay | Animated browser UI with live confidence meter & caption strip |

Sign Symbols are:
<p align="center">
  <img width="1126" height="1500" alt="sign_images" src="https://github.com/user-attachments/assets/55cfa769-9b02-4416-85ac-b34fb5b9099b" />

</p>

---

## 🏗️ Architecture

```
Webcam → MediaPipe (21 hand landmarks) → normalize → PyTorch MLP → letter
                                                                       ↓
                                                    sentence with SPACE / DEL gestures
```

**Web app data flow** (browser handles perception, server handles classification):

```
Browser: MediaPipe HandLandmarker (WASM)
    → extracts 21 landmarks, normalizes to a 63-float vector
    → streams vector over WebSocket (~12/sec)
FastAPI backend
    → runs the trained PyTorch model
    → returns {label, confidence} JSON
```

This keeps raw video entirely client-side — only ~63 numbers per frame cross the network, not video frames.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Hand landmark detection** | MediaPipe Tasks API (`HandLandmarker`) — Python & JavaScript |
| **Classification model** | PyTorch (custom MLP, landmark-geometry input) |
| **Desktop interface** | OpenCV |
| **Backend API** | FastAPI + WebSocket |
| **Frontend** | Vanilla HTML / CSS / JavaScript (no framework) |
| **Data pipeline** | NumPy, Pandas, scikit-learn |
| **Language** | Python 3.10+, JavaScript (ES modules) |

---

## 🚀 Setup

```bash
git clone https://github.com/sakshi-more11/SignVerse-AI.git
cd SignVerse-AI
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Download the hand-landmark model
```bash
python src/download_model.py
```

### 2. Collect training data
```bash
python src/data_collection.py --gesture A
# ...repeat for every letter A-Y (skip J, Z), plus SPACE and DEL
```

### 3. Train the classifier
```bash
python src/train_model.py
```
This project's own training run: **4,969 samples across 26 classes, 98.0% test accuracy.**

### 4. Run it

**Desktop demo:**
```bash
python src/recognize_desktop.py
```
Controls: hold a gesture steadily to type it · `c` clears the sentence · `q` quits.

**Web app:**
```bash
cd webapp/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Open `http://localhost:8000` in your browser and allow camera access.

---

## 📊 Dataset

26 gesture classes — the 24 static ASL alphabet letters (A–Y, excluding J/Z) plus custom **SPACE** and **DEL** control gestures — trained on 4,969 hand-landmark samples, achieving 98.0% held-out test accuracy.

---

## 🎯 Design Decisions

- **Landmark geometry over raw pixels** — classifying normalized hand-landmark coordinates (translation + scale invariant) instead of image pixels makes the model robust to lighting, background, and distance from camera, unlike classical skin-color-segmentation + HOG approaches.
- **MediaPipe Tasks API, not legacy `solutions`** — current `mediapipe` releases removed the old `mp.solutions.hands` API most tutorials use; this project is built on the actively maintained `HandLandmarker` Tasks API instead, in both Python and JavaScript.
- **Browser-side perception, server-side classification** — the web app's WebSocket protocol streams only a 63-number feature vector, not video, keeping bandwidth minimal and raw video entirely client-side.
- **Stability-based debouncing** — a sliding window of consecutive predictions must agree before a letter locks in, filtering out frame-to-frame prediction noise.

---
<p align="center">👩‍💻Developed by Sakshi More</p>
