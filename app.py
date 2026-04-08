"""
Cortexa Flask Backend — Feature-aware detection
Features: emotion | drowsy | headpose
"""
import os, time, threading, json
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, Response, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Model ─────────────────────────────────────────────────────────────────────
print("[Cortexa] Loading model...")
try:
    model = tf.keras.models.load_model("cnn_model.h5", compile=False)
    print("[Cortexa] Model loaded OK.")
except Exception as e:
    print(f"[Cortexa] Model load failed -> using mock predictions. ({e.__class__.__name__})")
    model = None

emotion_dict = {0:"Fear",1:"Neutral",2:"Sad",3:"Angry",4:"Happy",5:"Disgust",6:"Surprise"}

face_cas = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
eye_cas  = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml"))

# ── Shared state ──────────────────────────────────────────────────────────────
state = dict(
    running=False, mode="emotion",
    emotion="Neutral", confidence=0,
    ear=0.31, earStatus="OK",
    focus="High", drowsiness=False,
    head_pose="Centered",
    face_detected=False,
    look_towards_center=False,
    history=[],
    faces=[],           # multi-face emotion list [{id, emotion, confidence}]
    away_count=0,       # how many times user looked away
    pose_counts={"Looking Left":0,"Looking Right":0,"Looking Up":0,"Looking Down":0,"Centered":0},
)

_output_frame = None
_frame_lock   = threading.Lock()
_closed_start = None
_prev_emotion = None
_drowsy_alerted = False   # prevent repeated history spam
_prev_pose    = "Centered"  # track transitions for away_count

def _add_history(kind, label):
    state["history"].insert(0, {"type": kind, "label": label, "ts": time.time()})
    state["history"] = state["history"][:30]

def _mock_emotion():
    import random
    idx = random.randint(0, len(emotion_dict)-1)
    return emotion_dict[idx], random.randint(55, 92)

def _estimate_head_pose(face_rect, frame_shape):
    x, y, w, h = face_rect
    fx = (x + w / 2) / frame_shape[1]   # 0-1, horizontal center
    fy = (y + h / 2) / frame_shape[0]   # 0-1, vertical center

    if fx < 0.40:   return "Looking Right"
    if fx > 0.60:   return "Looking Left"
    if fy < 0.38:   return "Looking Up"
    if fy > 0.62:   return "Looking Down"
    return "Centered"

# ── Processing loop ───────────────────────────────────────────────────────────
def processing_loop():
    global _output_frame, _closed_start, _prev_emotion, _drowsy_alerted

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        state["running"] = False
        print("[Cortexa] Cannot open camera.")
        return

    print(f"[Cortexa] Camera opened.")
    consecutive_failures = 0

    while state["running"]:
        mode = state["mode"] # Read mode dynamically
        ret, frame = cap.read()
        
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 5: # allow some glitches
                print("[Cortexa] Camera stream lost.")
                break
            time.sleep(0.05)
            continue
        
        consecutive_failures = 0
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.3, 5)
        state["face_detected"] = len(faces) > 0
        state["faces"] = []  # reset per frame before classifying

        if len(faces) == 0:
            _closed_start = None
            # If in headpose mode, losing the face should trigger the 'Look towards center' alert
            state["look_towards_center"] = (mode == "headpose")
            _drawHUD(frame, mode)
            with _frame_lock:
                _output_frame = frame.copy()
            time.sleep(0.033)
            continue

        for i, (x, y, w, h) in enumerate(faces):
            # ── EMOTION mode ──────────────────────────────────────────────
            if mode == "emotion":
                roi = gray[y: y+h, x: x+w]
                roi = cv2.resize(roi, (48, 48)).astype("float32") / 255.0
                roi = np.expand_dims(roi, (0, -1))

                if model is not None:
                    pred  = model.predict(roi, verbose=0)
                    idx   = int(np.argmax(pred))
                    conf  = int(round(pred[0][idx] * 100))
                    label = emotion_dict[idx]
                else:
                    label, conf = _mock_emotion()

                # Update primary emotion from the first (largest) face
                if i == 0:
                    state.update(emotion=label, confidence=conf)
                    if label != _prev_emotion:
                        _add_history("emotion", f"Emotion: {label}")
                        _prev_emotion = label

                # Accumulate multi-face list (reset before the loop in a separate pass)
                state["faces"].append({"id": i+1, "emotion": label, "confidence": conf})

                # Draw face box + label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (80, 180, 140), 2)
                cv2.putText(frame, f"User {i+1}: {label}  {conf}%",
                            (x, y-10), cv2.FONT_HERSHEY_DUPLEX,
                            0.65, (255, 230, 200), 1, cv2.LINE_AA)

            # ── DROWSY (AlertGuard) mode ──────────────────────────────────
            elif mode == "drowsy":
                roi_eye   = gray[y: y + int(h * 0.6), x: x + w]
                eyes      = eye_cas.detectMultiScale(roi_eye, 1.2, 10, minSize=(20,20))
                eyes_open = len(eyes) > 0  # Changed from >= 2 to > 0 for much higher stability

                if not eyes_open:
                    if _closed_start is None:
                        _closed_start = time.time()
                    drowsy = (time.time() - _closed_start) >= 2.0  # Faster alert
                else:
                    _closed_start = None
                    drowsy = False

                ear_val = round(0.20 + len(eyes) * 0.06, 2)
                state.update(ear=ear_val, earStatus="OK" if eyes_open else "LOW",
                             drowsiness=drowsy)

                if drowsy and not _drowsy_alerted:
                    _add_history("drowsy", "Drowsiness Alert!")
                    _drowsy_alerted = True
                if not drowsy:
                    _drowsy_alerted = False

                # Draw face + eye boxes
                cv2.rectangle(frame, (x, y), (x+w, y+h), (60, 120, 220), 2)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (200, 200, 60), 1)

                # Drowsiness banner
                if drowsy:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 160), -1)
                    cv2.putText(frame, "!! DROWSINESS DETECTED !!",
                                (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                                (255, 80, 80), 2, cv2.LINE_AA)
                else:
                    status_txt = "Eyes Open — OK" if eyes_open else "Eyes Closing..."
                    color      = (80, 200, 100) if eyes_open else (60, 160, 220)
                    cv2.putText(frame, status_txt, (x, y-10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)

            # ── HEAD POSE (Attention Matrix) mode ─────────────────────────
            elif mode == "headpose":
                pose = _estimate_head_pose((x, y, w, h), frame.shape)
                state.update(head_pose=pose)

                # Track gaze direction frequency
                global _prev_pose
                if pose in state["pose_counts"]:
                    state["pose_counts"][pose] += 1

                # Count distraction events (transitions to non-Centered)
                if _prev_pose == "Centered" and pose != "Centered":
                    state["away_count"] += 1
                    _add_history("focus", f"Looked Away: {pose}")
                elif _prev_pose != "Centered" and pose == "Centered":
                    _add_history("focus", f"Returned: Centered")
                _prev_pose = pose

                # Draw box + orientation
                pose_colors = {
                    "Centered":       (80, 200, 80),
                    "Looking Left":   (220, 140, 60),
                    "Looking Right":  (220, 140, 60),
                    "Looking Up":     (60, 180, 220),
                    "Looking Down":   (60, 120, 220),
                }
                col = pose_colors.get(pose, (200, 200, 200))
                cv2.rectangle(frame, (x, y), (x+w, y+h), col, 2)
                cv2.putText(frame, pose, (x, y-10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 230, 200), 1, cv2.LINE_AA)

                # Draw face center crosshair
                cx, cy = x + w//2, y + h//2
                cv2.line(frame, (cx-15, cy), (cx+15, cy), (200, 200, 60), 1)
                cv2.line(frame, (cx, cy-15), (cx, cy+15), (200, 200, 60), 1)


            # Extra: Head pose alert for individual 'headpose' mode as well
            if mode == "headpose":
                state["look_towards_center"] = (state["head_pose"] != "Centered")

        _drawHUD(frame, mode)
        with _frame_lock:
            _output_frame = frame.copy()

        time.sleep(0.033)

    cap.release()
    state["running"] = False
    state["face_detected"] = False
    print("[Cortexa] Camera released.")


def _drawHUD(frame, mode):
    """Bottom status bar overlaid on frame."""
    h = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-44), (frame.shape[1], h), (15, 28, 48), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if mode == "emotion":
        hud = f"EMOTION MODE  |  {state['emotion']}  {state['confidence']}%"
    elif mode == "drowsy":
        hud = f"ALERTGUARD  |  EAR: {state['ear']:.2f} {state['earStatus']}  |  {'** DROWSY **' if state['drowsiness'] else 'Awake'}"
    else:
        hud = f"ATTENTION MATRIX  |  Head: {state['head_pose']}"

    cv2.putText(frame, hud, (10, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 185, 160), 1, cv2.LINE_AA)

    mode_label = {"emotion":"Emotion Insight","drowsy":"AlertGuard","headpose":"Attention Matrix"}
    cv2.putText(frame, mode_label.get(mode,""), (frame.shape[1]-220, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 160, 201), 1, cv2.LINE_AA)


# ── MJPEG generator ───────────────────────────────────────────────────────────
def _mjpeg_gen():
    while True:
        with _frame_lock:
            frame = _output_frame

        if frame is None:
            ph = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "Session not started", (160, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 140, 110), 2)
            _, buf = cv2.imencode(".jpg", ph, [cv2.IMWRITE_JPEG_QUALITY, 70])
        else:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/api/video_feed")
def video_feed():
    return Response(_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/start", methods=["POST", "OPTIONS"])
def start():
    global _closed_start, _prev_emotion, _drowsy_alerted
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "emotion")
    
    # Update mode even if already running
    state["mode"] = mode
    _closed_start   = None
    _prev_emotion   = None
    _drowsy_alerted = False

    # Reset mode-irrelevant fields
    if mode != "emotion":  state.update(emotion="Neutral", confidence=0)
    if mode != "drowsy":   state.update(drowsiness=False, ear=0.0, earStatus="OK")
    if mode != "headpose": state.update(head_pose="Centered")
    state["look_towards_center"] = False
    state["faces"]      = []
    state["away_count"] = 0
    state["pose_counts"] = {"Looking Left":0,"Looking Right":0,"Looking Up":0,"Looking Down":0,"Centered":0}

    if not state["running"]:
        print(f"[Cortexa] Starting new session thread | feature={mode}")
        state["running"] = True
        threading.Thread(target=processing_loop, daemon=True).start()
    else:
        print(f"[Cortexa] Session already active | switching to feature={mode}")

    return jsonify({"ok": True, "mode": mode})


@app.route("/api/stop", methods=["POST", "OPTIONS"])
def stop():
    global _output_frame
    state["running"] = False
    with _frame_lock:
        _output_frame = None
    return jsonify({"ok": True})


@app.route("/api/status")
def status():
    return jsonify({
        "running":      state["running"],
        "mode":         state["mode"],
        "faceDetected": state["face_detected"],
        "emotion":      state["emotion"],
        "confidence":   state["confidence"],
        "ear":          state["ear"],
        "earStatus":    state["earStatus"],
        "focus":        state["focus"],
        "drowsiness":   state["drowsiness"],
        "headPose":     state["head_pose"],
        "lookTowardsCenter": state["look_towards_center"],
        "history":      state["history"],
        "faces":        state["faces"],
        "awayCount":    state["away_count"],
        "poseCounts":   state["pose_counts"],
    })


@app.route("/api/history/clear", methods=["POST", "OPTIONS"])
def clear_history():
    state["history"] = []
    return jsonify({"ok": True})


@app.route("/")
def index():
    return "Cortexa API :5001 | /api/start (POST, body:{mode}) | /api/stop | /api/status | /api/video_feed"


if __name__ == "__main__":
    print("[Cortexa] Starting on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False,
            use_reloader=False, threaded=True)
