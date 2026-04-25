"""
Cortexa Flask Backend — Feature-aware detection
Features: emotion | drowsy | headpose
"""
import os, time, threading, json
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, Response, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#initalization of the model and classifiers
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

# ── YOLO phone detection setup
YOLO_CFG     = "yolov4-tiny.cfg" # it is the blueprint of the model , builds the archeitecture
YOLO_WEIGHTS = "yolov4-tiny.weights" # it is the trained model that has learned to deetct models
COCO_NAMES   = "coco.names"
PHONE_CLASS  = 67   # "cell phone" index in COCO (0-indexed)
YOLO_CONF    = 0.35 # threshold frequency
YOLO_NMS     = 0.45 

yolo_net    = None
yolo_output_layers = []

try:
    if os.path.exists(YOLO_WEIGHTS) and os.path.exists(YOLO_CFG):
        yolo_net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
        yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names = yolo_net.getLayerNames()
        yolo_output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers().flatten()]
        print("[Cortexa] YOLO model loaded OK.")
    else:
        print("[Cortexa] YOLO weights not found, phone detection disabled.")
except Exception as e:
    print(f"[Cortexa] YOLO load failed: {e}")
    yolo_net = None

# ── Shared state
state = dict(
    running=False, mode="emotion",
    emotion="Neutral", confidence=0,
    ear=0.31, earStatus="OK",
    focus="High", drowsiness=False,
    head_pose="Centered",
    face_detected=False,
    look_towards_center=False,
    phone_detected=False,
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
_phone_alerted = False     # debounce phone alert

def _add_history(kind, label):
    state["history"].insert(0, {"type": kind, "label": label, "ts": time.time()})
    state["history"] = state["history"][:30]

def _mock_emotion():
    import random
    idx = random.randint(0, len(emotion_dict)-1)
    return emotion_dict[idx], random.randint(55, 92)

def _detect_phone(frame):
    """Run YOLO on the frame; return True and bboxes if a cell phone is found."""
    if yolo_net is None:
        return False, []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(yolo_output_layers)
    boxes, confidences = [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cls    = int(np.argmax(scores))
            conf   = float(scores[cls])
            if cls == PHONE_CLASS and conf >= YOLO_CONF:
                cx, cy, bw, bh = det[:4]
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                boxes.append([x1, y1, int(bw*w), int(bh*h)])
                confidences.append(conf)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_CONF, YOLO_NMS)
    result = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            result.append((boxes[i], confidences[i]))
    return len(result) > 0, result

def _estimate_head_pose(face_rect, frame_shape):
    x, y, w, h = face_rect
    fx = (x + w / 2) / frame_shape[1]   # 0-1, horizontal center
    fy = (y + h / 2) / frame_shape[0]   # 0-1, vertical center

    if fx < 0.40:   return "Looking Right"
    if fx > 0.60:   return "Looking Left"
    if fy < 0.38:   return "Looking Up"
    if fy > 0.62:   return "Looking Down"
    return "Centered"

# ── Processing loop
def processing_loop():
    global _output_frame, _closed_start, _prev_emotion, _drowsy_alerted, _phone_alerted

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
            # ── EMOTION mode 
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

            # ── DROWSY (AlertGuard) mode
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

                # ── YOLO phone detection (every frame, throttled by _phone_alerted)
                phone_found, phone_boxes = _detect_phone(frame)
                state["phone_detected"] = phone_found
                if phone_found and not _phone_alerted:
                    _add_history("drowsy", "⚠ Phone Detected while driving!")
                    _phone_alerted = True
                if not phone_found:
                    _phone_alerted = False

                # Draw phone bounding boxes
                for (bx, by, bw, bh), conf in phone_boxes:
                    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 60, 255), 2)
                    cv2.putText(frame, f"PHONE! {int(conf*100)}%",
                                (bx, by-10), cv2.FONT_HERSHEY_DUPLEX,
                                0.7, (0, 60, 255), 2, cv2.LINE_AA)

                # Phone banner
                if phone_found:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 200), -1)
                    cv2.putText(frame, "!! PHONE DETECTED — PUT IT DOWN !!",
                                (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                                (255, 100, 100), 2, cv2.LINE_AA)

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

            # ── HEAD POSE (Attention Matrix) mode
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


# ── MJPEG generator 
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


# ── Routes 
@app.route("/api/video_feed")
def video_feed():
    return Response(_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Camera Preview (raw feed, no ML) ─────────────────────────────────────────
_preview_running = False
_preview_cap     = None
_preview_frame   = None
_preview_lock    = threading.Lock()

def _preview_loop():
    global _preview_running, _preview_cap, _preview_frame
    _preview_cap = cv2.VideoCapture(0)
    if not _preview_cap.isOpened():
        _preview_running = False
        print("[Cortexa] Preview camera failed to open.")
        return
    print("[Cortexa] Camera preview started.")
    while _preview_running:
        ret, frame = _preview_cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        # Overlay a subtle "PREVIEW" watermark
        cv2.putText(frame, "PREVIEW — Select a feature and Start Session",
                    (10, frame.shape[0]-14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 160, 130), 1, cv2.LINE_AA)
        with _preview_lock:
            _preview_frame = frame.copy()
        time.sleep(0.033)
    if _preview_cap:
        _preview_cap.release()
        _preview_cap = None
    print("[Cortexa] Camera preview stopped.")

def _preview_mjpeg():
    while True:
        with _preview_lock:
            frame = _preview_frame
        if frame is None:
            ph = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "Starting camera...", (200, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 140, 110), 2)
            _, buf = cv2.imencode(".jpg", ph, [cv2.IMWRITE_JPEG_QUALITY, 70])
        else:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)

@app.route("/api/preview_feed")
def preview_feed():
    return Response(_preview_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/preview/start", methods=["POST", "OPTIONS"])
def preview_start():
    global _preview_running
    if not _preview_running and not state["running"]:
        _preview_running = True
        threading.Thread(target=_preview_loop, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/preview/stop", methods=["POST", "OPTIONS"])
def preview_stop():
    global _preview_running
    _preview_running = False
    return jsonify({"ok": True})


@app.route("/api/start", methods=["POST", "OPTIONS"])
def start():
    global _closed_start, _prev_emotion, _drowsy_alerted, _phone_alerted
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "emotion")

    # Update mode even if already running
    state["mode"] = mode
    _closed_start   = None
    _prev_emotion   = None
    _drowsy_alerted = False
    _phone_alerted  = False

    # Reset mode-irrelevant fields
    if mode != "emotion":  state.update(emotion="Neutral", confidence=0)
    if mode != "drowsy":   state.update(drowsiness=False, ear=0.0, earStatus="OK")
    if mode != "headpose": state.update(head_pose="Centered")
    state["look_towards_center"] = False
    state["phone_detected"] = False
    state["faces"]      = []
    state["away_count"] = 0
    state["pose_counts"] = {"Looking Left":0,"Looking Right":0,"Looking Up":0,"Looking Down":0,"Centered":0}

    # Stop preview if it was running (camera will be taken over by processing_loop)
    global _preview_running
    _preview_running = False
    time.sleep(0.15)  # brief wait for preview thread to release camera

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
        "phoneDetected": state["phone_detected"],
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
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def styles():
    return send_from_directory(".", "style.css")

if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    port = int(os.environ.get("PORT", 5001))

    # Construct the absolute path to index.html and open it in the default web browser
    html_file_path = f"http://localhost:{port}"
    
    def open_browser():
        print(f"[Cortexa] Opening UI in browser: {html_file_path}")
        webbrowser.open(html_file_path)
        
    # Open the browser 1 second after starting the server
    Timer(1.0, open_browser).start()

    print(f"[Cortexa] Starting on http://localhost:{port} (local device only)")
    app.run(host="0.0.0.0", port=port, debug=False,
            use_reloader=False, threaded=True)
