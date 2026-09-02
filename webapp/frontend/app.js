/* ============================================================
   SignVerseAI frontend logic
   - Captures webcam video
   - Runs MediaPipe HandLandmarker (WASM, in-browser) per frame
   - Normalizes landmarks the SAME way as src/utils.py (Python)
   - Streams the 63-float vector to the FastAPI backend over WebSocket
   - Applies stability/debounce logic and builds the caption sentence
   ============================================================ */

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ---- Config (mirrors src/config.py) ----
const LETTERS = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"];
const CONTROL_GESTURES = { SPACE: "👍 space", DEL: "🖐 delete" };
const GESTURES = [...LETTERS, ...Object.keys(CONTROL_GESTURES)];

const STABILITY_FRAMES = 12;
const COOLDOWN_FRAMES = 15;
const SEND_INTERVAL_MS = 80; // ~12.5 predictions/sec over the socket

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

// ---- DOM refs ----
const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const videoEmpty = document.getElementById("videoEmpty");
const statusPill = document.getElementById("statusPill");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const currentGestureEl = document.getElementById("currentGesture");
const confidenceValueEl = document.getElementById("confidenceValue");
const meterFillEl = document.getElementById("meterFill");
const captionTextEl = document.getElementById("captionText");
const captionPlaceholderEl = document.getElementById("captionPlaceholder");
const legendEl = document.getElementById("legend");
const clearBtn = document.getElementById("clearBtn");
const setupBanner = document.getElementById("setupBanner");
const scanIndicator = document.getElementById("scanIndicator");

// ---- State ----
let sentence = "";
let recentPreds = [];
let lockedLabel = null;
let cooldown = 0;
let lastSendTime = 0;
let ws = null;
let handLandmarker = null;
let legendChips = {};

// ---- Build gesture legend ----
function buildLegend() {
  legendEl.innerHTML = "";
  legendChips = {};
  GESTURES.forEach((g) => {
    const chip = document.createElement("div");
    chip.className = "legend-chip" + (CONTROL_GESTURES[g] ? " control" : "");
    chip.textContent = CONTROL_GESTURES[g] ? CONTROL_GESTURES[g] : g;
    legendEl.appendChild(chip);
    legendChips[g] = chip;
  });
}
buildLegend();

function highlightLegend(label) {
  Object.entries(legendChips).forEach(([key, chip]) => {
    chip.classList.toggle("active", key === label);
  });
}

// ---- Status pill ----
function setStatus(state, text) {
  statusPill.classList.remove("live", "error");
  if (state) statusPill.classList.add(state);
  statusText.textContent = text;
}

// ---- WebSocket ----
function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/predict`);

  ws.onopen = () => setStatus("live", "Live");
  ws.onclose = () => {
    setStatus("error", "Disconnected — retrying…");
    setTimeout(connectWebSocket, 1500);
  };
  ws.onerror = () => setStatus("error", "Connection error");

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.error) {
      setupBanner.hidden = false;
      setStatus("error", "Model not trained");
      return;
    }
    setupBanner.hidden = true;

    handlePrediction(data.label, data.confidence || 0);
  };
}

// ---- Prediction handling: stability + sentence building ----
function handlePrediction(label, confidence) {
  currentGestureEl.textContent = label || "—";
  currentGestureEl.classList.toggle("active", !!label);
  const pct = Math.round(confidence * 100);
  confidenceValueEl.textContent = `${pct}%`;
  meterFillEl.style.width = `${pct}%`;
  highlightLegend(label);

  recentPreds.push(label);
  if (recentPreds.length > STABILITY_FRAMES) recentPreds.shift();

  if (cooldown > 0) {
    cooldown -= 1;
  } else if (
    recentPreds.length === STABILITY_FRAMES &&
    label !== null &&
    recentPreds.every((p) => p === label) &&
    label !== lockedLabel
  ) {
    if (label === "SPACE") {
      sentence += " ";
      appendCharAnimation(" ");
    } else if (label === "DEL") {
      sentence = sentence.slice(0, -1);
      captionTextEl.textContent = sentence; // full re-render only on delete
      captionPlaceholderEl.classList.toggle("hidden", sentence.length > 0);
    } else {
      sentence += label;
      appendCharAnimation(label);
    }
    lockedLabel = label;
    cooldown = COOLDOWN_FRAMES;
    recentPreds = [];
  }

  if (label === null) lockedLabel = null;
}

function appendCharAnimation(char) {
  const span = document.createElement("span");
  span.className = "letter-in";
  span.textContent = char;
  captionTextEl.appendChild(span);
  captionPlaceholderEl.classList.add("hidden");
}

clearBtn.addEventListener("click", () => {
  sentence = "";
  captionTextEl.innerHTML = "";
  captionPlaceholderEl.classList.remove("hidden");
});

// ---- Landmark normalization (mirrors src/utils.py extract_landmark_vector) ----
function extractLandmarkVector(landmarks) {
  const wrist = landmarks[0];
  const coords = landmarks.map((lm) => [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]);

  let maxDist = 0;
  for (const [x, y, z] of coords) {
    const dist = Math.sqrt(x * x + y * y + z * z);
    if (dist > maxDist) maxDist = dist;
  }
  if (maxDist < 1e-6) maxDist = 1;

  const vec = [];
  for (const [x, y, z] of coords) {
    vec.push(x / maxDist, y / maxDist, z / maxDist);
  }
  return vec;
}

// ---- Drawing ----
function drawLandmarks(landmarks) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  const w = overlay.width;
  const h = overlay.height;

  overlayCtx.lineWidth = 2;
  overlayCtx.strokeStyle = "rgba(94, 234, 212, 0.85)";
  HAND_CONNECTIONS.forEach(([a, b]) => {
    overlayCtx.beginPath();
    overlayCtx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    overlayCtx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    overlayCtx.stroke();
  });

  overlayCtx.fillStyle = "rgba(167, 139, 250, 0.95)";
  landmarks.forEach((lm) => {
    overlayCtx.beginPath();
    overlayCtx.arc(lm.x * w, lm.y * h, 3.5, 0, Math.PI * 2);
    overlayCtx.fill();
  });
}

// ---- Main detection loop ----
function detectionLoop() {
  if (video.readyState >= 2 && handLandmarker) {
    const result = handLandmarker.detectForVideo(video, performance.now());

    if (result.landmarks && result.landmarks.length > 0) {
      const landmarks = result.landmarks[0];
      drawLandmarks(landmarks);
      scanIndicator.textContent = "hand detected";

      const now = performance.now();
      if (ws && ws.readyState === WebSocket.OPEN && now - lastSendTime >= SEND_INTERVAL_MS) {
        lastSendTime = now;
        const vec = extractLandmarkVector(landmarks);
        ws.send(JSON.stringify({ landmarks: vec }));
      }
    } else {
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      scanIndicator.textContent = "scanning";
      const now = performance.now();
      if (ws && ws.readyState === WebSocket.OPEN && now - lastSendTime >= SEND_INTERVAL_MS) {
        lastSendTime = now;
        ws.send(JSON.stringify({ landmarks: null }));
      }
    }
  }
  requestAnimationFrame(detectionLoop);
}

// ---- Setup ----
async function setupHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });
}

async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 960, height: 720 },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      videoEmpty.classList.add("hidden");
      resolve();
    };
  });
}

async function init() {
  buildLegend();
  connectWebSocket();

  try {
    await Promise.all([setupHandLandmarker(), setupWebcam()]);
    requestAnimationFrame(detectionLoop);
  } catch (err) {
    console.error(err);
    videoEmpty.querySelector("span").textContent =
      "Camera access denied or unavailable — check browser permissions.";
    setStatus("error", "Camera error");
  }
}

init();
