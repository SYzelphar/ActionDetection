"""FastAPI backend that serves the action-detection UI.

Each WebSocket connection gets its own MediaPipe Holistic tracker and a rolling
30-frame keypoint buffer; once the buffer is full, every frame runs the LSTM.

Run from the project root:
    .venv\\Scripts\\python -m uvicorn webapp.server:app --port 8000
"""
import asyncio
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("ACTION_MODEL", ROOT.parent / "action.keras"))
LABELS = json.loads((ROOT / "labels.json").read_text(encoding="utf-8"))

mp_holistic = mp.solutions.holistic

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
SEQUENCE_LENGTH, NUM_KEYPOINTS = model.input_shape[1:]
NUM_CLASSES = model.output_shape[-1]
if len(LABELS) != NUM_CLASSES:
    raise RuntimeError(f"labels.json has {len(LABELS)} labels but the model outputs {NUM_CLASSES} classes")


@tf.autograph.experimental.do_not_convert
def _forward(x):
    return model(x, training=False)


# Compiled graph: ~20x faster than eager calls for this LSTM.
predict = tf.function(_forward, input_signature=[tf.TensorSpec([1, SEQUENCE_LENGTH, NUM_KEYPOINTS], tf.float32)])
predict(np.zeros((1, SEQUENCE_LENGTH, NUM_KEYPOINTS), np.float32))
model_lock = threading.Lock()


def extract_keypoints(results) -> np.ndarray:
    """Same 1662-value layout as the training notebook: pose, face, left hand, right hand."""
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def points(landmarks, with_visibility=False):
    """Landmarks as compact [x, y(, visibility)] lists for drawing in the browser."""
    if not landmarks:
        return None
    if with_visibility:
        return [[round(p.x, 4), round(p.y, 4), round(p.visibility, 2)] for p in landmarks.landmark]
    return [[round(p.x, 4), round(p.y, 4)] for p in landmarks.landmark]


class Session:
    """Per-connection tracker state, mirroring the notebook's real-time loop."""

    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)

    def reset(self):
        self.sequence.clear()

    def close(self):
        self.holistic.close()

    def process(self, image_bytes: bytes) -> dict:
        bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode frame")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        results = self.holistic.process(rgb)
        holistic_ms = (time.perf_counter() - t0) * 1000

        self.sequence.append(extract_keypoints(results))
        probs, model_ms = None, None
        if len(self.sequence) == SEQUENCE_LENGTH:
            x = np.expand_dims(np.array(self.sequence, dtype=np.float32), 0)
            t1 = time.perf_counter()
            with model_lock:
                probs = predict(x).numpy()[0]
            model_ms = (time.perf_counter() - t1) * 1000

        return {
            "landmarks": {
                "pose": points(results.pose_landmarks, with_visibility=True),
                "face": points(results.face_landmarks),
                "left_hand": points(results.left_hand_landmarks),
                "right_hand": points(results.right_hand_landmarks),
            },
            "buffered": len(self.sequence),
            "probs": [float(p) for p in probs] if probs is not None else None,
            "holistic_ms": round(holistic_ms, 1),
            "model_ms": round(model_ms, 1) if model_ms is not None else None,
        }


app = FastAPI(title="Action Detection")


@app.middleware("http")
async def no_cache(request, call_next):
    # Always revalidate, so edits (or another app on the same port) never serve stale files.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/info")
def info():
    return {
        "model": MODEL_PATH.name,
        "labels": LABELS,
        "sequence_length": SEQUENCE_LENGTH,
        "keypoints": NUM_KEYPOINTS,
        "parameters": int(model.count_params()),
        "connections": {
            "pose": sorted(mp_holistic.POSE_CONNECTIONS),
            "hand": sorted(mp_holistic.HAND_CONNECTIONS),
            "face": sorted(mp.solutions.face_mesh.FACEMESH_CONTOURS),
        },
    }


@app.websocket("/ws")
async def stream(ws: WebSocket):
    """Binary messages are JPEG frames; the text message "reset" clears the buffer."""
    await ws.accept()
    session = Session()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") == "reset":
                session.reset()
                continue
            if msg.get("bytes") is None:
                continue
            try:
                result = await asyncio.to_thread(session.process, msg["bytes"])
            except ValueError as e:
                result = {"error": str(e)}
            await ws.send_json(result)
    except WebSocketDisconnect:
        pass
    finally:
        session.close()
