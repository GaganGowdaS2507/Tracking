from contextlib import contextmanager
from collections import deque
from datetime import datetime
from pathlib import Path
import csv
import io
import json
import math
import os
import sqlite3
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "bicep.db"
CONFIG_PATH = DATA_DIR / "config.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

DEFAULT_CONFIG = {
    "operating_start": "08:30",
    "operating_end": "16:30",
    "recognition_threshold": 0.37,
    "unknown_alerts": True,
    "recognition_interval": 0.25,
    "cameras": {
        "entry": {"name": "Entry Camera", "camera_type": "usb", "source": "0", "enabled": True},
        "exit":  {"name": "Exit Camera",  "camera_type": "usb", "source": "1", "enabled": True},
    },
    # ── motion / tracking / recognition knobs (tuned for moving faces) ──
    "motion_sensitivity": 25,       # pixel-delta threshold for motion detection
    "motion_min_area": 800,         # min motion blob area (full-res pixels)
    "idle_interval": 0.8,           # seconds between AI passes when the scene is empty/still
    "detect_size": 640,             # 640x640 detector input for early distance detection of walking people
    "min_face_size": 35,            # allow faces from 35px upwards
    "min_det_score": 0.40,          # detector confidence floor
    "min_sharpness": 20.0,          # soft threshold: faces below this are treated as moving (soft), not discarded
    "max_yaw": 0.85,                # allows natural head turning while walking towards door
    "track_gate": 3.5,              # max jump between detections (handles brisk walking steps)
    "track_max_age": 1.5,           # seconds a track survives without a detection
    "embedding_queue_size": 6,      # embeddings averaged per track (quality-weighted)
    "max_embeds_per_frame": 3,      # recognition budget per AI pass
    "vote_window": 4,               # last N recognition votes considered
    "vote_needed": 1,               # fast lock on confident match; accumulates if lower
    "match_margin": 0.03,           # best member must beat runner-up by 0.03
    "swap_similarity": 0.20,        # embedding vs track average below this = a different person took the track
    "verify_every": 4.0,            # seconds between re-checks of an already identified track
}


# ═══════════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════════

def connect_db():
    @contextmanager
    def _connection():
        conn = sqlite3.connect(DB_PATH, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    return _connection()


def init_db():
    with connect_db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            embedding TEXT DEFAULT '',
            photo_path TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER,
            member_name TEXT NOT NULL,
            member_code TEXT DEFAULT '',
            event_type TEXT NOT NULL,
            event_time TEXT NOT NULL,
            confidence REAL DEFAULT 0,
            source TEXT DEFAULT '',
            reviewed INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'inside'
        );
        CREATE INDEX IF NOT EXISTS idx_visits_member_status ON visits(member_id, status);
        CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time);
        """)


# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════

def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            stored = json.load(f)
    except Exception:
        stored = {}

    cfg = {**DEFAULT_CONFIG, **stored}
    cfg["cameras"] = {k: dict(v) for k, v in DEFAULT_CONFIG["cameras"].items()}
    saved_cameras = stored.get("cameras", {})
    if isinstance(saved_cameras, dict):
        for role in ("entry", "exit"):
            if isinstance(saved_cameras.get(role), dict):
                cfg["cameras"][role].update(saved_cameras[role])

    if not stored.get("cameras") and "camera_source" in stored:
        cfg["cameras"]["entry"]["camera_type"] = stored.get("camera_type", "usb")
        cfg["cameras"]["entry"]["source"] = str(stored.get("camera_source", "0"))

    cfg["recognition_threshold"] = float(cfg.get("recognition_threshold", 0.37))
    cfg["recognition_interval"] = max(0.10, min(2.0, float(cfg.get("recognition_interval", 0.25))))
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


if not CONFIG_PATH.exists():
    save_config(DEFAULT_CONFIG)
init_db()


# ═══════════════════════════════════════════════════════════════════════
#  FACE MODEL + EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════

MODEL_LOCK = threading.RLock()
INFERENCE_LOCK = threading.Lock()
EMBEDDINGS_LOCK = threading.RLock()
FACE_MODEL = None
FACE_MODEL_ERROR = ""
FACE_MODEL_ATTEMPTED = False
EMBEDDINGS_CACHE = {"at": 0.0, "rows": []}


def get_face_model():
    global FACE_MODEL, FACE_MODEL_ERROR, FACE_MODEL_ATTEMPTED
    with MODEL_LOCK:
        if FACE_MODEL is not None:
            return FACE_MODEL
        if FACE_MODEL_ATTEMPTED:
            return None
        FACE_MODEL_ATTEMPTED = True
        try:
            from insightface.app import FaceAnalysis
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
            except Exception:
                available = ["CPUExecutionProvider"]
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
            size = int(load_config().get("detect_size", 480))
            # Only load what we use: detector + recogniser (skips 3D/2D landmarks and gender/age).
            FACE_MODEL = FaceAnalysis(
                name="buffalo_l", providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            FACE_MODEL.prepare(ctx_id=ctx_id, det_size=(size, size))
            FACE_MODEL_ERROR = ""
        except Exception as exc:
            FACE_MODEL_ERROR = str(exc)
            FACE_MODEL = None
        return FACE_MODEL


class Det:
    """One raw face detection (no embedding yet — recognition is run only when needed)."""
    __slots__ = ("bbox", "kps", "score", "quality", "reason")

    def __init__(self, bbox, kps, score):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.kps = None if kps is None else np.asarray(kps, dtype=np.float32)
        self.score = float(score)
        self.quality = 0.0
        self.reason = ""


def detect_faces(model, frame):
    """Detector only (fast). Returns a list of Det."""
    with INFERENCE_LOCK:
        bboxes, kpss = model.models["detection"].detect(frame, max_num=0, metric="default")
    return [
        Det(bboxes[i][:4], None if kpss is None else kpss[i], bboxes[i][4])
        for i in range(len(bboxes))
    ]


def embed_face(model, frame, det):
    """Recogniser for ONE detected face -> L2-normalised 512-d vector (or None)."""
    if det.kps is None:
        return None
    from insightface.app.common import Face
    face = Face(bbox=det.bbox, kps=det.kps, det_score=det.score)
    with INFERENCE_LOCK:
        emb = model.models["recognition"].get(frame, face)
    emb = np.asarray(emb, dtype=np.float32).flatten()
    n = float(np.linalg.norm(emb))
    return None if n < 1e-6 else emb / n


def get_member_embeddings():
    """Each member can now hold SEVERAL templates (matrix k x 512). Old single-vector rows still load."""
    now = time.time()
    with EMBEDDINGS_LOCK:
        if now - EMBEDDINGS_CACHE["at"] < 4:
            return EMBEDDINGS_CACHE["rows"]
        with connect_db() as c:
            rows = c.execute(
                "SELECT id, name, member_code, embedding FROM members WHERE embedding != ''"
            ).fetchall()
        parsed = []
        for row in rows:
            try:
                arr = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.ndim != 2 or arr.shape[0] == 0:
                    continue
                arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-6)
                parsed.append({
                    "id": row["id"], "name": row["name"],
                    "member_code": row["member_code"], "templates": arr,
                })
            except Exception:
                continue
        EMBEDDINGS_CACHE["at"] = now
        EMBEDDINGS_CACHE["rows"] = parsed
        return parsed


def identify_embedding(embedding, threshold):
    """
    Returns (member | None, best_score, margin).
    Score per member = best match over that member's templates.
    margin = best score minus the runner-up member's score (ambiguity check).
    """
    rows = [r for r in get_member_embeddings() if r["templates"].shape[1] == embedding.shape[0]]
    if not rows:
        return None, 0.0, 0.0
    scores = np.array([float(np.max(r["templates"] @ embedding)) for r in rows])
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    top = float(scores[best])
    second = float(scores[order[1]]) if len(order) > 1 else 0.0
    if top < threshold:
        return None, top, top - second
    return rows[best], top, top - second

# ═══════════════════════════════════════════════════════════════════════
#  VISIT / EVENT HELPERS
# ═══════════════════════════════════════════════════════════════════════

def find_open_visit(c, member_id):
    return c.execute(
        "SELECT id, entry_time FROM visits "
        "WHERE member_id=? AND status='inside' ORDER BY id DESC LIMIT 1",
        (member_id,),
    ).fetchone()


def record_detection(role, member, confidence, source):
    now = datetime.now().isoformat(timespec="seconds")
    event_type = "ENTRY" if role == "entry" else "EXIT"
    with connect_db() as c:
        open_visit = find_open_visit(c, member["id"])
        if role == "entry":
            if open_visit:
                return False, "Already inside"
            c.execute(
                "INSERT INTO visits(member_id, entry_time, status) VALUES(?, ?, 'inside')",
                (member["id"], now),
            )
        else:
            if not open_visit:
                return False, "No open entry visit"
            c.execute(
                "UPDATE visits SET exit_time=?, status='outside' WHERE id=?",
                (now, open_visit["id"]),
            )
        c.execute(
            """INSERT INTO events(member_id, member_name, member_code,
                                  event_type, event_time, confidence, source)
               VALUES(?, ?, ?, ?, ?, ?, ?)""",
            (member["id"], member["name"], member["member_code"],
             event_type, now, float(confidence), source),
        )
    return True, event_type


def record_unknown(role, source, confidence=0.0):
    now = datetime.now().isoformat(timespec="seconds")
    with connect_db() as c:
        c.execute(
            """INSERT INTO events(member_id, member_name, member_code,
                                  event_type, event_time, confidence, source)
               VALUES(NULL, 'Unknown person', '', 'UNKNOWN', ?, ?, ?)""",
            (now, float(confidence), source),
        )


def parse_source(camera_cfg):
    source = str(camera_cfg.get("source", "0")).strip()
    if camera_cfg.get("camera_type", "usb") == "usb":
        try:
            return int(source)
        except ValueError:
            return 0
    return source


def resize_width(frame, max_width=1280):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    nh = max(1, int(h * (max_width / w)))
    return cv2.resize(frame, (max_width, nh), interpolation=cv2.INTER_AREA)




# ═══════════════════════════════════════════════════════════════════════
#  MOTION DETECTOR  –  cheap gate (runs on a 1/4-size copy)
# ═══════════════════════════════════════════════════════════════════════

class MotionDetector:
    def __init__(self, scale=0.25):
        self.prev_gray = None
        self.scale = scale

    def detect(self, frame, sensitivity=25, min_area=800):
        """Return (has_motion, [(x, y, w, h), ...]) in full-resolution coordinates."""
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return True, []
        delta = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        thresh = cv2.dilate(cv2.threshold(delta, sensitivity, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inv = 1.0 / self.scale
        floor = min_area * self.scale * self.scale
        rects = [
            tuple(int(v * inv) for v in cv2.boundingRect(c))
            for c in contours if cv2.contourArea(c) >= floor
        ]
        return bool(rects), rects


# ═══════════════════════════════════════════════════════════════════════
#  FACE QUALITY  –  decide which detections are worth recognising
# ═══════════════════════════════════════════════════════════════════════

def face_quality(frame, det, cfg):
    """
    Sets det.quality (0.15..1.0) and det.reason.
    For moving faces: quality serves as a confidence/importance weight for template
    averaging rather than a hard rejection gate. Faces with moderate motion blur or
    head turns are still evaluated by ArcFace to ensure fast moving recognition.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in det.bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    fw, fh = x2 - x1, y2 - y1
    min_face = int(cfg.get("min_face_size", 35))
    max_yaw = float(cfg.get("max_yaw", 0.85))
    min_sharp = float(cfg.get("min_sharpness", 20.0))

    det.quality, det.reason = 0.0, "ok"
    if min(fw, fh) < min_face:
        det.reason = "too small"
        return
    if det.score < float(cfg.get("min_det_score", 0.40)):
        det.reason = "low confidence"
        return
    yaw = 0.0
    if det.kps is not None:                       # order: eyeL, eyeR, nose, mouthL, mouthR
        eye_mid = (det.kps[0] + det.kps[1]) / 2.0
        eye_dist = max(1e-6, float(np.linalg.norm(det.kps[1] - det.kps[0])))
        yaw = float((det.kps[2][0] - eye_mid[0]) / eye_dist)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        det.reason = "invalid crop"
        return
    gray = cv2.cvtColor(cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Continuous quality weight (0.20 to 1.0) - never drops to 0.0 on moving/angled faces:
    sharp = min(1.0, max(0.15, lap / 100.0))
    size = min(1.0, max(0.25, min(fw, fh) / 100.0))
    front = max(0.20, 1.0 - min(1.0, abs(yaw) / max(0.1, max_yaw)))
    det.quality = round(0.40 * sharp + 0.30 * size + 0.30 * front, 3)

    if lap < min_sharp:
        det.reason = "moving (soft)"
    elif abs(yaw) > 0.60:
        det.reason = "angled"
    else:
        det.reason = "ok"


# ═══════════════════════════════════════════════════════════════════════
#  FACE TRACKER  –  motion-predicted, Hungarian-matched, time-based
# ═══════════════════════════════════════════════════════════════════════

def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class TrackedFace:
    _next_id = 0
    _id_lock = threading.Lock()

    def __init__(self, det, now, queue_size=6, vote_window=4):
        with TrackedFace._id_lock:
            TrackedFace._next_id += 1
            self.track_id = TrackedFace._next_id
        self.bbox = det.bbox.copy()
        self.vel = np.zeros(4, dtype=np.float32)          # px/second for x1,y1,x2,y2
        self.born = self.last_seen = now
        self.hits = 1
        self.det = det
        self.samples = deque(maxlen=queue_size)           # (embedding, quality)
        self.votes = deque(maxlen=vote_window)            # None | (member_id, member, score)
        self.member = None
        self.score = 0.0
        self.locked = False
        self.strikes = 0
        self.last_verify = 0.0
        self.unknown_fired = False

    @property
    def size(self):
        return max(1.0, float(max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])))

    def predict(self, now, max_dt=0.75):
        dt = min(max_dt, max(0.0, now - self.last_seen))
        return self.bbox + self.vel * dt

    def update(self, det, now):
        dt = max(1e-3, now - self.last_seen)
        inst = (det.bbox - self.bbox) / dt
        limit = 8.0 * self.size                           # ignore absurd velocities
        inst = np.clip(inst, -limit, limit)
        self.vel = inst if self.hits < 2 else 0.5 * self.vel + 0.5 * inst
        self.bbox = det.bbox.copy()
        self.det = det
        self.last_seen = now
        self.hits += 1

    def mean_embedding(self):
        """Quality-weighted, L2-normalised average (sharp frontal frames count most)."""
        embs = np.stack([e for e, _ in self.samples])
        w = np.array([max(q, 0.05) for _, q in self.samples], dtype=np.float32)
        mean = (embs * w[:, None]).sum(axis=0)
        n = float(np.linalg.norm(mean))
        return mean if n < 1e-6 else (mean / n).astype(np.float32)

    def reset_identity(self):
        self.samples.clear()
        self.votes.clear()
        self.member, self.score, self.locked, self.strikes = None, 0.0, False, 0
        self.unknown_fired = False


class FaceTracker:
    BIG = 1e6

    def __init__(self, gate=3.0, max_age=1.2, queue_size=6, vote_window=4):
        self.tracks = []
        self.gate, self.max_age = gate, max_age
        self.queue_size, self.vote_window = queue_size, vote_window

    def update(self, dets, now):
        tracks, matched_t, matched_d = self.tracks, set(), set()
        if tracks and dets:
            cost = np.full((len(tracks), len(dets)), self.BIG)
            for i, t in enumerate(tracks):
                pb = t.predict(now)
                pw = max(1.0, pb[2] - pb[0])
                pc = ((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)
                for j, d in enumerate(dets):
                    dc = ((d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2)
                    dist = math.hypot(pc[0] - dc[0], pc[1] - dc[1]) / t.size
                    if dist > self.gate:
                        continue
                    ratio = abs(math.log(max(1.0, d.bbox[2] - d.bbox[0]) / pw))
                    cost[i, j] = dist + 0.5 * (1.0 - _iou(pb, d.bbox)) + 0.3 * ratio
            try:
                from scipy.optimize import linear_sum_assignment
                pairs = list(zip(*linear_sum_assignment(cost)))
            except Exception:                                  # greedy fallback if scipy is missing
                pairs = []
                for idx in np.argsort(cost, axis=None):
                    i, j = divmod(int(idx), len(dets))
                    if i not in {p[0] for p in pairs} and j not in {p[1] for p in pairs}:
                        pairs.append((i, j))
            for i, j in pairs:
                if cost[i, j] < self.BIG:
                    tracks[i].update(dets[j], now)
                    matched_t.add(i)
                    matched_d.add(j)
        for j, d in enumerate(dets):
            if j not in matched_d:
                tracks.append(TrackedFace(d, now, self.queue_size, self.vote_window))
        self.tracks = [t for t in tracks if now - t.last_seen <= self.max_age]
        return self.tracks


# ═══════════════════════════════════════════════════════════════════════
#  CAMERA WORKER  –  capture thread + recognition thread + smooth overlay
# ═══════════════════════════════════════════════════════════════════════

class CameraWorker:
    """One capture thread + one recognition thread per camera."""

    def __init__(self, role):
        self.role = role
        self.lock = threading.RLock()
        self.cap = None
        self.running = False
        self.capture_thread = None
        self.inference_thread = None
        self.latest_raw = None
        self.latest_seq = 0
        self.latest_ts = 0.0
        self.display_frame = None
        self.status = "Not connected"
        self.last_error = ""
        self.label = "Entry Camera" if role == "entry" else "Exit Camera"
        self.source_label = ""
        self.last_recognition = 0.0
        self.last_event_by_member = {}
        self.last_unknown_event = 0.0
        self.fps = 0.0
        self.frames_in_second = 0
        self.fps_epoch = time.time()
        self.motion_detector = MotionDetector()
        self.face_tracker = FaceTracker()
        self.motion_detected = False
        self.motion_rects = []
        self.overlay = []                 # drawn on every live frame, not only on AI frames
        self.last_motion_time = 0.0
        self.inference_fps = 0.0
        self._inf_count = 0
        self._inf_epoch = time.time()
        self._cfg = (0.0, None)

    def _config(self):
        at, cfg = self._cfg
        if cfg is None or time.time() - at > 1.0:
            cfg = load_config()
            self._cfg = (time.time(), cfg)
        return cfg

    def _open_capture(self):
        cfg = load_config()["cameras"][self.role]
        source = parse_source(cfg)
        if cfg.get("camera_type", "usb") == "usb":
            if os.name == "nt":
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, source

    def start(self):
        self.stop()
        cfg = load_config()["cameras"][self.role]
        self.label = str(
            cfg.get("name") or ("Entry Camera" if self.role == "entry" else "Exit Camera")
        )
        cap, source = self._open_capture()
        self.source_label = f'{cfg.get("camera_type", "usb").upper()} · {source}'
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            self.status = "Could not open camera"
            self.last_error = (
                f"Could not open source {source}. "
                "Check camera assignment and Windows camera permissions."
            )
            return False
        with self.lock:
            self.cap = cap
            self.running = True
            self.latest_raw = None
            self.display_frame = None
            self.latest_seq = 0
            self.overlay = []
            self.motion_rects = []
            self.last_error = ""
            self.status = "Connected · waiting for frames"
            c = load_config()
            self.motion_detector = MotionDetector()
            self.face_tracker = FaceTracker(
                float(c.get("track_gate", 3.0)), float(c.get("track_max_age", 1.2)),
                int(c.get("embedding_queue_size", 6)), int(c.get("vote_window", 4)),
            )
        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name=f"{self.role}-capture"
        )
        self.inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name=f"{self.role}-inference"
        )
        self.capture_thread.start()
        self.inference_thread.start()
        return True

    def stop(self):
        with self.lock:
            self.running = False
            cap = self.cap
            self.cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        for thread in (self.capture_thread, self.inference_thread):
            if thread and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self.status = "Stopped"
        self.display_frame = None
        self.latest_raw = None
        self.overlay = []

    def _capture_loop(self):
        while self.running:
            with self.lock:
                cap = self.cap
            if cap is None:
                time.sleep(0.05)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                self.last_error = (
                    "Camera stopped returning frames. "
                    "Try reconnecting or another camera index."
                )
                self.status = "No frames"
                time.sleep(0.1)
                continue
            frame = resize_width(frame, 1280)
            now = time.time()
            self.frames_in_second += 1
            if now - self.fps_epoch >= 1.0:
                self.fps = self.frames_in_second / max(0.01, now - self.fps_epoch)
                self.frames_in_second = 0
                self.fps_epoch = now
            with self.lock:
                self.latest_raw = frame
                self.latest_ts = now          # capture time: used for tracking + latency compensation
                self.latest_seq += 1

    def _inference_loop(self):
        last_seq = -1
        MOTION_GRACE_FRAMES = 5
        no_motion_count = 0
        while self.running:
            cfg = self._config()
            with self.lock:
                seq, ts = self.latest_seq, self.latest_ts
                raw = None if (self.latest_raw is None or seq == last_seq) else self.latest_raw.copy()
            if raw is None:
                time.sleep(0.01)
                continue
            last_seq = seq                    # every frame is looked at once, motion included

            has_motion, rects = self.motion_detector.detect(
                raw, int(cfg.get("motion_sensitivity", 25)), int(cfg.get("motion_min_area", 800))
            )
            self.motion_rects = rects
            if has_motion:
                no_motion_count = 0
                self.motion_detected = True
                self.last_motion_time = time.time()
            else:
                no_motion_count += 1
                if no_motion_count > MOTION_GRACE_FRAMES:
                    self.motion_detected = False

            # Active = motion OR somebody is already being tracked.
            active = self.motion_detected or bool(self.face_tracker.tracks)
            # Low latency detection: 0.05s while active, 0.15s while waiting (never sleep 1.0s)
            base = float(cfg.get("recognition_interval", 0.10))
            interval = max(0.04, base) if active else float(cfg.get("idle_interval", 0.15))
            if time.time() - self.last_recognition < interval:
                continue

            try:
                self._process(raw, cfg, ts)
                self.last_recognition = time.time()
                with self.lock:
                    if self.status not in ("No frames", "Could not open camera"):
                        self.status = "Live · AI active" if FACE_MODEL is not None else "Live · loading AI"
                self._inf_count += 1
                if time.time() - self._inf_epoch >= 2.0:
                    self.inference_fps = self._inf_count / max(0.01, time.time() - self._inf_epoch)
                    self._inf_count, self._inf_epoch = 0, time.time()
            except Exception as exc:
                self.last_error = str(exc)
                self.last_recognition = time.time()
                with self.lock:
                    self.status = "Camera live · AI issue"

    # ── one AI pass: detect → track → (selectively) recognise → vote ──

    def _process(self, frame, cfg, now):
        model = get_face_model()
        if model is None:
            self.overlay = []
            return
        tr = self.face_tracker
        tr.gate = float(cfg.get("track_gate", 3.0))
        tr.max_age = float(cfg.get("track_max_age", 1.2))
        tr.queue_size = int(cfg.get("embedding_queue_size", 6))
        tr.vote_window = int(cfg.get("vote_window", 4))

        min_face = int(cfg.get("min_face_size", 40))
        dets = []
        for d in detect_faces(model, frame):
            if min(d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]) < min_face * 0.6:
                continue                                   # far too small to matter
            face_quality(frame, d, cfg)
            dets.append(d)

        tracks = tr.update(dets, now)

        # Recognition is the expensive step, so spend it only where it helps:
        #   unidentified tracks first (best quality first); identified tracks only re-verified occasionally.
        verify_every = float(cfg.get("verify_every", 4.0))
        todo = [
            t for t in tracks
            if t.last_seen == now and t.det.quality > 0
            and (not t.locked or now - t.last_verify >= verify_every)
        ]
        todo.sort(key=lambda t: (t.locked, -t.det.quality))
        for t in todo[: int(cfg.get("max_embeds_per_frame", 2))]:
            self._identify(model, frame, t, now, cfg)

        need = int(cfg.get("vote_needed", 1))
        overlay = []
        now_ts = time.time()
        for t in tracks:
            # Immediate feedback: show bounding box on the very first detection frame
            if t.locked:
                color, text = (40, 200, 105), f'{t.member["name"]} ({t.score:.2f})'
            elif t.unknown_fired or (
                len(t.votes) >= need and all(v is None for v in list(t.votes)[-need:])
            ):
                color, text = (40, 60, 235), "Unknown Person"
            elif t.det.quality == 0 and not t.samples:
                color, text = (150, 150, 150), t.det.reason
            else:
                color, text = (0, 190, 255), "Scanning..."
            overlay.append({
                "bbox": t.bbox.copy(), "t": now_ts,
                "color": color, "text": text, "id": t.track_id,
            })
        self.overlay = overlay

    def _identify(self, model, frame, t, now, cfg):
        emb = embed_face(model, frame, t.det)
        if emb is None:
            return
        t.last_verify = now
        threshold = float(cfg.get("recognition_threshold", 0.37))

        # Someone else took over this track (people crossing): don't pollute the average.
        if len(t.samples) >= 2 and float(emb @ t.mean_embedding()) < float(cfg.get("swap_similarity", 0.20)):
            t.reset_identity()
        t.samples.append((emb, t.det.quality))

        member, score, margin = identify_embedding(emb, threshold)
        min_margin = float(cfg.get("match_margin", 0.03))
        if member is not None and margin < min_margin:
            member = None                                  # two members look alike → don't guess

        if t.locked:                                       # already identified: just re-verify
            same = member is not None and member["id"] == t.member["id"]
            t.strikes = 0 if same else t.strikes + 1
            if t.strikes >= 2:
                t.reset_identity()
            return

        t.votes.append(None if member is None else (member["id"], member, score))
        need = int(cfg.get("vote_needed", 1))

        # FAST-PATH FOR MOVING FACES:
        # If a walking person matches with confident score (>= 0.39 or threshold + 0.02) and safe margin,
        # lock immediately! Waiting multiple frames on moving targets causes missed registrations.
        is_fast_hit = (
            member is not None
            and score >= max(0.39, threshold + 0.02)
            and margin >= min_margin
        )

        tally = {}
        for v in t.votes:
            if v:
                tally[v[0]] = tally.get(v[0], 0) + 1

        if is_fast_hit or (tally and max(tally.values()) >= need):
            mid = member["id"] if is_fast_hit else max(tally.items(), key=lambda kv: kv[1])[0]
            # Verify with mean embedding if we have multiple samples and not a fast hit
            m2, _, _ = identify_embedding(t.mean_embedding(), threshold - 0.02)
            if is_fast_hit or (m2 is not None and m2["id"] == mid):
                hits = [v for v in t.votes if v and v[0] == mid]
                t.member = member if is_fast_hit else (hits[-1][1] if hits else m2)
                t.score = float(score if is_fast_hit else (np.mean([v[2] for v in hits]) if hits else score))
                t.locked, t.strikes = True, 0
                key = str(mid)
                if now - self.last_event_by_member.get(key, -1e9) >= 10:
                    record_detection(self.role, t.member, t.score, self.source_label)
                    self.last_event_by_member[key] = now
                return

        if (
            cfg.get("unknown_alerts", True) and not t.unknown_fired
            and len(t.votes) >= max(2, need) and all(v is None for v in list(t.votes)[-need:])
            and now - self.last_unknown_event >= 15
        ):
            _, best, _ = identify_embedding(t.mean_embedding(), 0.0)
            record_unknown(self.role, self.source_label, best)
            t.unknown_fired = True
            self.last_unknown_event = now

    # ── live rendering: boxes are drawn on EVERY frame, extrapolated by velocity ──

    def _render(self, frame):
        now = time.time()
        h, w = frame.shape[:2]
        cv2.putText(frame, self.label.upper(), (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255), 2)
        for (mx, my, mw, mh) in list(self.motion_rects):
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 120, 0), 1)
        for o in list(self.overlay):
            age = now - o["t"]
            if age > 3.0:
                continue
            x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            color, text = o["color"], o["text"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 24)), (min(w, x1 + max(160, len(text) * 10)), y1), color, -1)
            cv2.putText(frame, text, (x1 + 4, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f'ID:{o["id"]}', (x1 + 4, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        label = "MOTION" if self.motion_detected else "still"
        cv2.putText(frame, label, (w - 90, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 255) if self.motion_detected else (120, 120, 120), 2)
        cv2.putText(frame, f"{self.fps:.0f} FPS cam | {self.inference_fps:.1f} AI/s | "
                    f"{len(self.face_tracker.tracks)} tracks", (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        if FACE_MODEL is None and FACE_MODEL_ATTEMPTED:
            cv2.putText(frame, "AI model unavailable - check install/model download",
                        (14, max(45, h - 34)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        return frame

    def mjpeg(self):
        while True:
            with self.lock:
                frame = None if self.latest_raw is None else self.latest_raw.copy()
                running = self.running
                status = self.status
            if frame is None:
                frame = np.zeros((480, 800, 3), dtype=np.uint8)
                cv2.putText(frame, f"{self.label}: {status}"[:70], (25, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2)
                cv2.putText(frame, "Open Cameras and connect this camera", (25, 265),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 180, 220), 1)
            else:
                frame = self._render(frame)
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")
            time.sleep(0.04 if running else 0.25)

# ═══════════════════════════════════════════════════════════════════════
#  CAMERA MANAGER
# ═══════════════════════════════════════════════════════════════════════

class CameraManager:
    def __init__(self):
        self.workers = {
            "entry": CameraWorker("entry"),
            "exit": CameraWorker("exit"),
        }
        self.discovery_lock = threading.Lock()
        self.discovery_cache = {"at": 0.0, "devices": []}

    def get(self, role):
        return self.workers[role]

    def start(self, role):
        if role not in self.workers:
            return False, "Unknown camera role"
        cfg = load_config()["cameras"]
        other = "exit" if role == "entry" else "entry"
        this_cfg, other_cfg = cfg[role], cfg[other]
        if this_cfg.get("enabled") is False:
            return False, (
                f"{this_cfg.get('name', role.title())} is disabled. "
                "Enable it in Cameras first."
            )
        if (
            this_cfg.get("camera_type") == "usb"
            and other_cfg.get("camera_type") == "usb"
            and str(this_cfg.get("source")) == str(other_cfg.get("source"))
            and self.workers[other].running
        ):
            return False, (
                "Both roles are assigned to the same USB camera index. "
                "Assign different cameras."
            )
        ok = self.workers[role].start()
        if ok:
            self.discovery_cache["at"] = 0
            return True, self.workers[role].status
        return False, self.workers[role].last_error

    def statuses(self):
        result = {}
        cfg = load_config()["cameras"]
        for role, worker in self.workers.items():
            result[role] = {
                "name": cfg[role].get("name", worker.label),
                "configured_type": cfg[role].get("camera_type", "usb"),
                "source": str(cfg[role].get("source", "")),
                "enabled": bool(cfg[role].get("enabled", True)),
                "running": worker.running,
                "status": worker.status,
                "error": worker.last_error,
                "label": worker.source_label,
                "fps": round(worker.fps, 1),
                "inference_fps": round(worker.inference_fps, 1),
                "motion": worker.motion_detected,
                "tracks": len(worker.face_tracker.tracks),
            }
        return result

    def discover(self, force=False):
        with self.discovery_lock:
            if not force and time.time() - self.discovery_cache["at"] < 10:
                return self.discovery_cache["devices"]
            active_sources = set()
            cfg = load_config()["cameras"]
            for role, worker in self.workers.items():
                if worker.running and cfg[role].get("camera_type") == "usb":
                    try:
                        active_sources.add(int(cfg[role].get("source", "0")))
                    except Exception:
                        pass
            devices = []
            for index in range(6):
                if index in active_sources:
                    devices.append({
                        "index": index, "source": str(index),
                        "label": f"USB camera {index} (connected)",
                        "available": True, "active": True,
                    })
                    continue
                cap = None
                try:
                    cap = (
                        cv2.VideoCapture(index, cv2.CAP_DSHOW)
                        if os.name == "nt"
                        else cv2.VideoCapture(index)
                    )
                    if cap is not None and cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                        ok, _ = cap.read()
                        if ok:
                            devices.append({
                                "index": index, "source": str(index),
                                "label": f"USB camera {index}",
                                "available": True, "active": False,
                            })
                except Exception:
                    pass
                finally:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
            self.discovery_cache = {"at": time.time(), "devices": devices}
            return devices

    def stop_all(self):
        for worker in self.workers.values():
            worker.stop()


cameras = CameraManager()


# ═══════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed_legacy():
    return Response(
        cameras.get("entry").mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video_feed/<role>")
def video_feed(role):
    if role not in ("entry", "exit"):
        return "Unknown camera role", 404
    return Response(
        cameras.get(role).mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
def api_status():
    with connect_db() as c:
        members = c.execute("SELECT COUNT(*) n FROM members").fetchone()["n"]
        inside = c.execute(
            "SELECT COUNT(*) n FROM visits WHERE status='inside'"
        ).fetchone()["n"]
        today = datetime.now().strftime("%Y-%m-%d")
        entries = c.execute(
            "SELECT COUNT(*) n FROM events "
            "WHERE event_type='ENTRY' AND event_time LIKE ?",
            (today + "%",),
        ).fetchone()["n"]
        exits = c.execute(
            "SELECT COUNT(*) n FROM events "
            "WHERE event_type='EXIT' AND event_time LIKE ?",
            (today + "%",),
        ).fetchone()["n"]
        unknown = c.execute(
            "SELECT COUNT(*) n FROM events "
            "WHERE event_type='UNKNOWN' AND event_time LIKE ?",
            (today + "%",),
        ).fetchone()["n"]
    statuses = cameras.statuses()
    any_running = any(s["running"] for s in statuses.values())
    status_text = "Monitoring active" if any_running else "Cameras not connected"
    return jsonify({
        "status": status_text, "last_error": "",
        "members": members, "inside": inside,
        "entries_today": entries, "exits_today": exits,
        "unknown_today": unknown, "camera_running": any_running,
        "cameras": statuses,
    })


@app.get("/api/members")
def get_members():
    with connect_db() as c:
        rows = c.execute(
            """SELECT id, member_code, name, email, notes, created_at,
                      photo_path, embedding
               FROM members ORDER BY name"""
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        emb_str = d.pop("embedding", "")
        count = 0
        if emb_str:
            try:
                parsed = json.loads(emb_str)
                count = len(parsed) if isinstance(parsed, list) else 1
            except Exception:
                count = 1
        d["has_face"] = 1 if count > 0 else 0
        d["template_count"] = count
        result.append(d)
    return jsonify(result)


@app.post("/api/members")
def add_member():
    data = request.get_json(force=True)
    code = str(data.get("member_code", "")).strip()
    name = str(data.get("name", "")).strip()
    if not code or not name:
        return jsonify({"error": "Member code and name are required."}), 400
    with connect_db() as c:
        try:
            cur = c.execute(
                "INSERT INTO members(member_code,name,email,notes,created_at) "
                "VALUES(?,?,?,?,?)",
                (
                    code, name,
                    str(data.get("email", "")).strip(),
                    str(data.get("notes", "")).strip(),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            mid = cur.lastrowid
        except sqlite3.IntegrityError:
            return jsonify({"error": "That member code already exists."}), 409
    with EMBEDDINGS_LOCK:
        EMBEDDINGS_CACHE["at"] = 0
    return jsonify({"ok": True, "id": mid})


@app.delete("/api/members/<int:member_id>")
def delete_member(member_id):
    with connect_db() as c:
        c.execute("DELETE FROM members WHERE id=?", (member_id,))
        c.execute("DELETE FROM visits WHERE member_id=?", (member_id,))
    with EMBEDDINGS_LOCK:
        EMBEDDINGS_CACHE["at"] = 0
    return jsonify({"ok": True})


@app.post("/api/members/<int:member_id>/enroll-multi")
def enroll_multi(member_id):
    """
    3-angle enrollment: accepts 'photo_front', 'photo_left', 'photo_right'
    (or any multi-part files under these keys or 'photos').
    Computes 512-d normalized ArcFace embeddings for each valid face.
    Saves:
      data/member_photos/member_{member_id}_front.jpg
      data/member_photos/member_{member_id}_left.jpg
      data/member_photos/member_{member_id}_right.jpg
    Stores all 3 templates in SQLite members.embedding.
    """
    with connect_db() as c:
        member = c.execute("SELECT id, name FROM members WHERE id=?", (member_id,)).fetchone()
    if not member:
        return jsonify({"error": "Member not found."}), 404

    model = get_face_model()
    if model is None:
        return jsonify({
            "error": "Face AI failed to initialize: " + (FACE_MODEL_ERROR or "unknown error")
        }), 503

    photo_dir = DATA_DIR / "member_photos"
    photo_dir.mkdir(parents=True, exist_ok=True)

    slots = [("front", "Front View"), ("left", "Left Angle"), ("right", "Right Angle")]
    templates = []
    saved_photos = {}
    errors = []

    # Check for designated slots first
    for slot_key, slot_label in slots:
        file_obj = request.files.get(f"photo_{slot_key}")
        if not file_obj:
            continue
        raw_bytes = file_obj.read()
        if not raw_bytes:
            continue
        image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            errors.append(f"{slot_label}: could not decode image format.")
            continue

        with INFERENCE_LOCK:
            faces = model.get(resize_width(image, 1280))
        if not faces:
            errors.append(f"{slot_label}: no face detected in photo.")
            continue

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.normed_embedding.astype(np.float32).tolist()
        templates.append(emb)

        slot_path = photo_dir / f"member_{member_id}_{slot_key}.jpg"
        cv2.imwrite(str(slot_path), image)
        saved_photos[slot_key] = str(slot_path.relative_to(ROOT))

    # Also handle generic files or photos list if slots weren't provided directly
    if not templates and "photos" in request.files:
        for idx, file_obj in enumerate(request.files.getlist("photos")[:3]):
            raw_bytes = file_obj.read()
            if not raw_bytes:
                continue
            image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            with INFERENCE_LOCK:
                faces = model.get(resize_width(image, 1280))
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = face.normed_embedding.astype(np.float32).tolist()
            templates.append(emb)
            slot_name = slots[idx][0] if idx < len(slots) else f"angle_{idx+1}"
            slot_path = photo_dir / f"member_{member_id}_{slot_name}.jpg"
            cv2.imwrite(str(slot_path), image)
            saved_photos[slot_name] = str(slot_path.relative_to(ROOT))

    if not templates:
        return jsonify({
            "error": "No valid faces were detected in the provided photos: " + "; ".join(errors)
            if errors else "Please select at least 1 photo with a clear face."
        }), 400

    primary_path = saved_photos.get("front", next(iter(saved_photos.values())))
    # Keep primary thumbnail as member_{id}.jpg as well
    front_img_path = photo_dir / f"member_{member_id}_{'front' if 'front' in saved_photos else list(saved_photos.keys())[0]}.jpg"
    if front_img_path.exists():
        import shutil
        shutil.copyfile(str(front_img_path), str(photo_dir / f"member_{member_id}.jpg"))

    with connect_db() as c:
        c.execute(
            "UPDATE members SET embedding=?, photo_path=? WHERE id=?",
            (json.dumps(templates), primary_path, member_id),
        )

    with EMBEDDINGS_LOCK:
        EMBEDDINGS_CACHE["at"] = 0

    return jsonify({
        "ok": True,
        "count": len(templates),
        "message": f"Successfully enrolled {len(templates)} face template(s) for {member['name']} (Front/Left/Right).",
        "photos": saved_photos,
        "warnings": errors if errors else None
    })


@app.post("/api/members/<int:member_id>/enroll-face")
def enroll_face(member_id):
    # Delegate to multi-enroll if 3-angle fields are provided
    if (
        "photo_front" in request.files
        or "photo_left" in request.files
        or "photo_right" in request.files
        or len(request.files.getlist("photos")) > 1
    ):
        return enroll_multi(member_id)

    if "photo" not in request.files:
        return jsonify({"error": "Choose a face photo first."}), 400
    image = cv2.imdecode(
        np.frombuffer(request.files["photo"].read(), np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        return jsonify({"error": "The uploaded file is not a readable image."}), 400
    try:
        model = get_face_model()
        if model is None:
            return jsonify({
                "error": "Face AI failed to initialize: "
                + (FACE_MODEL_ERROR or "unknown error")
            }), 503
        with INFERENCE_LOCK:
            faces = model.get(resize_width(image, 1280))
        if not faces:
            return jsonify({
                "error": "No face found. Use a clear, front-facing photo."
            }), 400
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        emb = face.normed_embedding.astype(np.float32).tolist()
        photo_dir = DATA_DIR / "member_photos"
        photo_dir.mkdir(parents=True, exist_ok=True)
        photo_path = photo_dir / f"member_{member_id}.jpg"
        cv2.imwrite(str(photo_path), image)

        with connect_db() as c:
            row = c.execute("SELECT embedding FROM members WHERE id=?", (member_id,)).fetchone()
            if not row:
                return jsonify({"error": "Member not found."}), 404
            templates = []
            if row["embedding"] and request.args.get("replace") != "1":
                try:
                    old = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
                    templates = (old[None, :] if old.ndim == 1 else old).tolist()
                except Exception:
                    templates = []
            templates = (templates + [emb])[-8:]          # keep up to 8 most recent photos
            c.execute(
                "UPDATE members SET embedding=?,photo_path=? WHERE id=?",
                (json.dumps(templates), str(photo_path.relative_to(ROOT)), member_id),
            )
        with EMBEDDINGS_LOCK:
            EMBEDDINGS_CACHE["at"] = 0
        return jsonify({"ok": True, "count": len(templates), "message": f"Face template saved locally ({len(templates)} of 8 on file)."})
    except Exception as exc:
        return jsonify({"error": "Face enrollment failed: " + str(exc)}), 500


@app.get("/api/events")
def get_events():
    limit = min(500, max(1, int(request.args.get("limit", 100))))
    with connect_db() as c:
        rows = c.execute(
            """SELECT id, member_name, member_code, event_type, event_time,
                      confidence, source, reviewed
               FROM events ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.get("/api/inside")
def get_inside():
    with connect_db() as c:
        rows = c.execute(
            """SELECT v.id, v.entry_time, m.id member_id, m.member_code, m.name
               FROM visits v JOIN members m ON m.id = v.member_id
               WHERE v.status='inside'
               ORDER BY v.entry_time DESC"""
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.post("/api/events/<int:event_id>/review")
def review_event(event_id):
    with connect_db() as c:
        c.execute("UPDATE events SET reviewed=1 WHERE id=?", (event_id,))
    return jsonify({"ok": True})


@app.get("/api/config")
def get_config():
    cfg = load_config()
    return jsonify({k: v for k, v in cfg.items() if k != "cameras"})


@app.post("/api/config")
def update_config():
    data = request.get_json(force=True)
    cfg = load_config()
    for key in (
        "operating_start", "operating_end", "recognition_threshold",
        "unknown_alerts", "recognition_interval",
                "motion_sensitivity", "motion_min_area", "idle_interval", "detect_size",
        "min_face_size", "min_det_score", "min_sharpness", "max_yaw",
        "track_gate", "track_max_age", "embedding_queue_size", "max_embeds_per_frame",
        "vote_window", "vote_needed", "match_margin", "swap_similarity", "verify_every",
    ):
        if key in data:
            cfg[key] = data[key]
    cfg["recognition_threshold"] = max(
        0.1, min(0.95, float(cfg.get("recognition_threshold", 0.37)))
    )
    cfg["recognition_interval"] = max(
        0.10, min(2.0, float(cfg.get("recognition_interval", 0.25)))
    )
    save_config(cfg)
    return jsonify({
        "ok": True,
        "config": {k: v for k, v in cfg.items() if k != "cameras"},
    })


@app.get("/api/cameras")
def get_cameras():
    cfg = load_config()["cameras"]
    return jsonify({"config": cfg, "statuses": cameras.statuses()})


@app.get("/api/cameras/discover")
def discover_cameras():
    force = request.args.get("refresh", "0") == "1"
    devices = cameras.discover(force=force)
    return jsonify({
        "devices": devices,
        "message": f"Found {len(devices)} available USB camera(s).",
    })


@app.post("/api/cameras/<role>")
def save_camera(role):
    if role not in ("entry", "exit"):
        return jsonify({"error": "Camera role must be entry or exit."}), 400
    data = request.get_json(force=True)
    cfg = load_config()
    item = cfg["cameras"][role]
    item["name"] = str(
        data.get("name", item.get("name", role.title() + " Camera"))
    ).strip()[:80]
    item["camera_type"] = data.get("camera_type", item.get("camera_type", "usb"))
    item["source"] = str(data.get("source", item.get("source", "0"))).strip()
    item["enabled"] = bool(data.get("enabled", True))
    if item["camera_type"] not in ("usb", "rtsp"):
        return jsonify({"error": "Camera type must be USB or RTSP."}), 400
    if item["camera_type"] == "usb":
        try:
            item["source"] = str(int(item["source"]))
        except ValueError:
            return jsonify({"error": "Choose a valid USB camera index."}), 400
    elif not item["source"].startswith(
        ("rtsp://", "rtsps://", "http://", "https://")
    ):
        return jsonify({
            "error": "Enter a valid RTSP/IP-camera stream URL."
        }), 400
    save_config(cfg)
    return jsonify({"ok": True, "camera": item, "statuses": cameras.statuses()})


@app.post("/api/cameras/<role>/start")
def start_camera(role):
    ok, message = cameras.start(role)
    if not ok:
        return jsonify({
            "ok": False, "error": message, "statuses": cameras.statuses()
        }), 400
    return jsonify({
        "ok": True, "message": message, "statuses": cameras.statuses()
    })


@app.post("/api/cameras/<role>/stop")
def stop_camera(role):
    if role not in ("entry", "exit"):
        return jsonify({"error": "Camera role must be entry or exit."}), 400
    cameras.get(role).stop()
    return jsonify({"ok": True, "statuses": cameras.statuses()})


@app.post("/api/cameras/<role>/test")
def test_camera(role):
    if role not in ("entry", "exit"):
        return jsonify({"error": "Camera role must be entry or exit."}), 400
    cfg = load_config()["cameras"][role]
    cap = None
    try:
        source = parse_source(cfg)
        if cfg.get("camera_type") == "usb" and os.name == "nt":
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        elif cfg.get("camera_type") == "usb":
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if cap is None or not cap.isOpened():
            return jsonify({"ok": False, "error": f"Could not open {source}."}), 400
        ok, frame = cap.read()
        if not ok or frame is None:
            return jsonify({
                "ok": False,
                "error": "Camera opened but did not return a frame.",
            }), 400
        h, w = frame.shape[:2]
        return jsonify({
            "ok": True,
            "message": f"Camera test successful: {w} × {h} frame.",
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


@app.post("/api/camera/start")
def start_entry_legacy():
    ok, message = cameras.start("entry")
    if not ok:
        return jsonify({"ok": False, "error": message}), 400
    return jsonify({"ok": True, "status": message})


@app.post("/api/camera/stop")
def stop_entry_legacy():
    cameras.get("entry").stop()
    return jsonify({"ok": True})


@app.get("/api/export.csv")
def export_csv():
    with connect_db() as c:
        rows = c.execute(
            """SELECT event_time, member_code, member_name, event_type,
                      confidence, source
               FROM events ORDER BY id DESC"""
        ).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Timestamp", "Member Code", "Name", "Event", "Confidence", "Camera Source",
    ])
    for row in rows:
        writer.writerow(list(row))
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=entry_exit_access_logs.csv"
        },
    )


@app.post("/api/manual-event")
def manual_event():
    data = request.get_json(force=True)
    try:
        member_id = int(data.get("member_id", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid member ID."}), 400
    role = str(data.get("event_type", "")).upper()
    if role not in ("ENTRY", "EXIT"):
        return jsonify({"error": "Event must be ENTRY or EXIT."}), 400
    with connect_db() as c:
        row = c.execute(
            "SELECT id, name, member_code FROM members WHERE id=?",
            (member_id,),
        ).fetchone()
    if not row:
        return jsonify({"error": "Member not found."}), 404
    member = dict(row)
    changed, message = record_detection(
        "entry" if role == "ENTRY" else "exit",
        member, 1.0, "Manual admin entry",
    )
    if not changed:
        return jsonify({"error": message}), 409
    return jsonify({"ok": True, "message": message})


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "python": os.sys.version.split()[0],
        "face_model_loaded": FACE_MODEL is not None,
        "face_model_error": FACE_MODEL_ERROR,
    })


if __name__ == "__main__":
    print("Entry/Exit Tracking running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)