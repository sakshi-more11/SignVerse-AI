# SignVerseAI: Real-Time Sign Language Translator

A real-time American Sign Language (ASL) alphabet-to-text translator with
two interfaces:

1. **Web app** — browser-based, with a live camera feed, animated caption
   strip, and a FastAPI backend (recommended — this is the polished demo).
2. **Desktop script** — a single-file OpenCV window, no server needed.

Both share the same trained PyTorch model. Hand detection uses MediaPipe's
21-point landmark tracker; classification is a lightweight PyTorch MLP
trained on your own recorded gestures.

---

## How it works

```
Webcam → MediaPipe (21 hand landmarks) → normalize → PyTorch MLP → letter
                                                                       ↓
                                                        sentence with SPACE / DEL gestures
```

- **SPACE** gesture: 👍 thumbs up
- **DEL** gesture: 🖐 open palm, fingers spread
- 24 static alphabet letters (A–Y, excluding J and Z, which require motion
  a single-frame classifier can't capture)

A prediction only "locks in" once your hand holds the pose steadily for
~12 consecutive frames, preventing jittery misfires.

---

## Project Structure

```
SignVerseAI/
├── requirements.txt          # desktop/training pipeline deps
├── data/                      # your captured / extracted training CSVs go here
├── data_raw/                  # (Option B only) downloaded Kaggle images, gitignored
├── models/                    # trained model + downloaded MediaPipe model
├── src/                       # shared ML pipeline (training + desktop demo)
│   ├── config.py
│   ├── hand_detector.py
│   ├── utils.py
│   ├── model.py                          # PyTorch classifier (used by src/ AND webapp/backend)
│   ├── download_model.py                 # one-time: fetch MediaPipe's hand model (desktop only)
│   ├── data_collection.py                # Option A: capture your own labeled gesture data
│   ├── download_kaggle_dataset.py        # Option B: fetch the public Kaggle ASL image dataset
│   ├── extract_landmarks_from_dataset.py # Option B: images -> landmark CSVs
│   ├── train_model.py                    # train the classifier on data/*.csv (either option)
│   └── recognize_desktop.py              # standalone OpenCV window demo
└── webapp/
    ├── backend/
    │   ├── main.py             # FastAPI + WebSocket classification endpoint
    │   └── requirements.txt
    └── frontend/
        ├── index.html
        ├── style.css
        └── app.js              # browser-side MediaPipe + WebSocket client
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Download the hand-landmark model (desktop pipeline only)

```bash
python src/download_model.py
```

Fetches Google's pretrained `hand_landmarker.task` (~7.5MB) into `models/`.
(The web app doesn't need this step — the browser loads its own copy
directly from Google's CDN via JavaScript.)

### 2. Get training data — three options

**Option A: Record your own gestures (strongest resume story, ~15–20 min)**

A model trained on someone else's hand/lighting/camera generalizes poorly,
so recording yourself gives the most defensible "I collected and trained
on my own dataset" story. For every letter A–Y (skip J, Z) plus `SPACE`
and `DEL`:

```bash
python src/data_collection.py --gesture A
python src/data_collection.py --gesture B
# ...repeat for every gesture in src/config.py -> GESTURES
python src/data_collection.py --gesture SPACE
python src/data_collection.py --gesture DEL
```

Hold the pose, press **SPACE** to start/stop recording ~150–200 frames per
gesture, moving your hand slightly (angle/distance) for variety.

**Option B: Clone a ready-made GitHub image dataset (fastest, no signup)**

[khansa3999/ASL-Alphabet-Dataset](https://github.com/khansa3999/ASL-Alphabet-Dataset)
hosts 4,000 images per letter directly on GitHub — no Kaggle account or API
key needed:

```bash
git clone https://github.com/khansa3999/ASL-Alphabet-Dataset.git data_raw_github

python src/extract_landmarks_from_dataset.py \
    --raw-dir data_raw_github/Data/train \
    --max-per-class 300
```

**Note:** this dataset only has the 26 letters — no SPACE/DEL folders — so
you'll still need to record those two gestures yourself (~2 minutes):

```bash
python src/data_collection.py --gesture SPACE
python src/data_collection.py --gesture DEL
```

**Option C: Kaggle's "ASL Alphabet" dataset (87K images, needs a free API key)**

Same underlying dataset as Option B but larger (includes space/del/nothing
folders natively), via Kaggle's official API:

```bash
# One-time: get a free Kaggle API token
#   1. Create an account at kaggle.com
#   2. kaggle.com/settings -> API -> "Create New Token" -> downloads kaggle.json
#   3. Place it at ~/.kaggle/kaggle.json (chmod 600 on Linux/Mac)
pip install kaggle

python src/download_kaggle_dataset.py          # ~1GB download
python src/extract_landmarks_from_dataset.py   # converts images -> data/*.csv
```

---

Options B and C both run through the same `extract_landmarks_from_dataset.py`
script — it just walks whatever `--raw-dir` you point it at, maps folder
names to gesture labels, runs MediaPipe on each sampled image *offline*,
and writes the exact same landmark-CSV format `data_collection.py` produces.
`train_model.py` doesn't care which option(s) you used — CSVs from multiple
sources in `data/` simply combine.

By default it samples 300 images per class (adjust with `--max-per-class`)
and automatically skips J, Z, and "nothing" (no equivalent gesture here).
Images where MediaPipe can't find a hand are skipped and counted, not
silently dropped.

> **Trade-off to know:** Public datasets are typically one or a few signers,
> fairly uniform lighting/background. The resulting model may not
> generalize as well to *your* webcam as Option A would. For the best of
> both, use Option B or C for bulk data, then layer a smaller amount of
> your own Option A recordings on top — `data_collection.py` appends to
> the same CSVs, so mixing is automatic.

### 3. Train the classifier


```bash
python src/train_model.py
```

Saves `models/asl_model.pth` + `models/labels.json` — used by **both** the
desktop demo and the web app. Prints a full per-class accuracy report; note
the final test accuracy for your resume.

### 4a. Run the desktop demo

```bash
python src/recognize_desktop.py
```

### 4b. Run the web app

```bash
cd webapp/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser and allow camera access.

---

## Architecture notes (web app)

The web app splits work between browser and server deliberately:

- **Browser** runs MediaPipe's `HandLandmarker` (WASM) directly on the
  video feed — this keeps raw video entirely client-side (fast, private,
  zero video bandwidth to the server).
- Only the **63-number normalized landmark vector** is sent to the backend
  over a WebSocket, roughly 12 times/second.
- **FastAPI backend** runs the trained PyTorch model on that vector and
  returns `{label, confidence}` — a few KB per message, real-time feel
  with minimal server load.

This is the same "Option A" split used in production real-time CV web
apps: heavy perception on-device, lightweight classification server-side.

---

## Design notes — landmarks over pixels

Older sign-language projects (skin-color segmentation + HOG + SVM on raw
pixels) are fragile: they break under different lighting, backgrounds, or
skin tones, and require manual calibration per session.

This project classifies **normalized hand geometry** instead — landmark
coordinates translated to the wrist and scaled by hand size — which is
invariant to lighting, background, and how close the hand is to the
camera. Both the desktop pipeline (Python) and the web frontend
(JavaScript) implement the identical normalization so the trained model
behaves consistently in both places.

---

## Resume Bullet Points

**SignVerseAI: Real-Time Sign Language Translator**
*Python, OpenCV, MediaPipe, PyTorch, FastAPI, WebSocket*

- Built a full-stack real-time ASL translator with a browser-based MediaPipe
  hand-tracking frontend and a FastAPI/WebSocket backend, streaming
  normalized 3D hand-landmark vectors for classification by a PyTorch
  neural network across 24 letter classes plus space/delete control
  gestures.
- Designed a stability-based debouncing system for reliable gesture-to-letter
  locking, and trained the classifier end-to-end on a self-collected
  labeled dataset, achieving [X]% test accuracy.

*(Fill in [X]% with your actual `train_model.py` output once you've
collected and trained on your own data.)*
