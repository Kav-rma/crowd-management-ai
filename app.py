# ──────────────────────────────────────────────
# AI Detection Service — Flask + YOLOv8
# ──────────────────────────────────────────────
#
# Endpoints:
#   GET /detect      — run YOLO on latest frame, return risk assessment
#   GET /history     — return recent crowd log entries (for trend chart)
#   GET /video_feed  — MJPEG stream of the live camera feed

import threading
import time
from datetime import datetime, timezone

import cv2
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

import config
import database
from risk_engine import RiskEngine

# ── Flask app ─────────────────────────────────
app = Flask(__name__)

# ── YOLO model (loaded once) ─────────────────
model = YOLO(config.YOLO_MODEL_PATH)

# ── Risk engine (stateful singleton) ─────────
risk_engine = RiskEngine()

# ── Persistent camera capture thread ─────────
_latest_frame = None
_frame_lock = threading.Lock()
_camera_ready = threading.Event()


def _camera_loop():
    """Background thread that continuously grabs frames from the webcam."""
    global _latest_frame
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {config.CAMERA_INDEX}")
        return

    _camera_ready.set()
    print(f"[INFO] Camera {config.CAMERA_INDEX} opened — capture thread running")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        with _frame_lock:
            _latest_frame = frame


def _start_camera():
    t = threading.Thread(target=_camera_loop, daemon=True)
    t.start()
    _camera_ready.wait(timeout=5)


# ── Endpoints ─────────────────────────────────

@app.route("/detect")
def detect():
    with _frame_lock:
        frame = _latest_frame

    if frame is None:
        return jsonify({"error": "Camera not ready"}), 503

    # Run YOLO inference
    results = model(frame, conf=config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)

    # Count persons (COCO class 0)
    raw_count = 0
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                raw_count += 1

    # Evaluate risk
    assessment = risk_engine.evaluate(raw_count)
    data = assessment.to_dict()

    # Convert timestamp to ISO string for JSON
    data["timestamp"] = datetime.fromtimestamp(
        data["timestamp"], tz=timezone.utc
    ).isoformat()

    # Log to database (fire-and-forget on a thread to avoid blocking)
    threading.Thread(target=database.log_reading, args=(data,), daemon=True).start()

    return jsonify(data)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream — each frame is JPEG-encoded from the latest capture."""

    def generate():
        while True:
            with _frame_lock:
                frame = _latest_frame
            if frame is None:
                time.sleep(0.1)
                continue
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
            time.sleep(0.05)  # ~20 fps cap

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/history")
def history():
    minutes = request.args.get("minutes", 2, type=int)
    minutes = max(1, min(minutes, 60))  # clamp 1–60
    rows = database.get_history(minutes)
    return jsonify(rows)


# ── Startup ───────────────────────────────────

if __name__ == "__main__":
    database.init_db()
    _start_camera()
    app.run(port=config.FLASK_PORT)
