from flask import Flask, render_template, request, jsonify, Response, send_file
from pathlib import Path
import sqlite3, threading, time, cv2, json, os, csv, io
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "bicep.db"
CONFIG_PATH = ROOT / "data" / "config.json"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

DEFAULT_CONFIG = {
    "camera_type": "usb",
    "camera_source": "0",
    "line_x_percent": 50,
    "operating_start": "08:30",
    "operating_end": "16:30",
    "recognition_threshold": 0.42,
    "unknown_alerts": True
}

def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

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
            reviewed INTEGER DEFAULT 0,
            FOREIGN KEY(member_id) REFERENCES members(id)
        );
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'inside',
            FOREIGN KEY(member_id) REFERENCES members(id)
        );
        """)
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)

def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {**DEFAULT_CONFIG, **cfg}
    except Exception:
        return DEFAULT_CONFIG.copy()

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

init_db()

class CameraService:
    def __init__(self):
        self.lock = threading.RLock()
        self.capture = None
        self.thread = None
        self.running = False
        self.frame = None
        self.status = "Camera not started"
        self.last_error = ""
        self.frame_count = 0
        self.model = None
        self.model_attempted = False
        self.last_member_event = {}
        self.tracks = {}
        self.next_track_id = 1
        self.last_centers = {}
        self.unknown_seen_at = 0
        self.camera_label = ""
        self.current_frame_width = 1280

    def load_model(self):
        if self.model is not None:
            return self.model
        if self.model_attempted:
            return None
        self.model_attempted = True
        try:
            from insightface.app import FaceAnalysis
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.model = FaceAnalysis(name="buffalo_l", providers=providers)
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            self.status = "AI model ready"
        except Exception as e:
            self.last_error = "AI model unavailable: " + str(e)
            self.status = "Camera running; AI recognition unavailable"
        return self.model

    def stop(self):
        with self.lock:
            self.running = False
            cap = self.capture
            self.capture = None
        if cap is not None:
            try: cap.release()
            except Exception: pass
        self.status = "Camera stopped"

    def start(self):
        self.stop()
        cfg = load_config()
        source = cfg.get("camera_source", "0")
        if cfg.get("camera_type") == "usb":
            try:
                source = int(source)
            except Exception:
                source = 0
        self.camera_label = f'{cfg.get("camera_type", "usb").upper()} · {source}'
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.status = "Could not open camera"
            self.last_error = f"Could not open source: {source}. Check USB camera index or RTSP URL."
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        with self.lock:
            self.capture = cap
            self.running = True
            self.last_error = ""
            self.status = "Camera connected"
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def _loop(self):
        while self.running:
            cap = self.capture
            if cap is None:
                time.sleep(.2)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                self.status = "Camera feed lost; reconnecting..."
                time.sleep(1)
                cfg = load_config()
                source = cfg.get("camera_source", "0")
                if cfg.get("camera_type") == "usb":
                    try: source = int(source)
                    except Exception: source = 0
                try:
                    cap.release()
                    newcap = cv2.VideoCapture(source)
                    if newcap.isOpened():
                        with self.lock: self.capture = newcap
                    else:
                        newcap.release()
                except Exception:
                    pass
                continue
            self.frame_count += 1
            if self.frame_count % 3 == 0:
                try:
                    frame = self.process_frame(frame)
                except Exception as e:
                    self.last_error = str(e)
            with self.lock:
                self.frame = frame.copy()
                if self.status not in ("AI model unavailable",):
                    self.status = "Camera connected" if self.model is None else "Live monitoring"
            time.sleep(.01)

    def process_frame(self, frame):
        cfg = load_config()
        h, w = frame.shape[:2]
        self.current_frame_width = w
        line_x = int(w * float(cfg.get("line_x_percent", 50)) / 100)
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 200, 255), 2)
        cv2.putText(frame, "ENTRY  |  EXIT", (max(10, line_x-90), 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 255), 2)
        model = self.load_model()
        if model is None:
            cv2.putText(frame, "AI unavailable - install requirements and download model on first run",
                        (15, h-20), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 255), 1)
            return frame
        faces = model.get(frame)
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            emb = face.normed_embedding.astype(np.float32)
            name, code, member_id, score = self.identify(emb, cfg)
            center_x = int((x1+x2)/2)
            track_key = self.assign_track(center_x, int((y1+y2)/2))
            color = (40, 210, 80) if member_id else (40, 40, 230)
            label = f"{name} {score:.2f}" if member_id else "Unknown person"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, max(22,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            self.check_crossing(track_key, center_x, member_id, name, code, score, cfg)
        return frame

    def identify(self, emb, cfg):
        best = None
        best_score = -1
        with connect_db() as c:
            rows = c.execute("SELECT id, name, member_code, embedding FROM members WHERE embedding != ''").fetchall()
        for row in rows:
            try:
                stored = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
                if stored.shape != emb.shape: continue
                score = float(np.dot(stored, emb))
                if score > best_score:
                    best_score, best = score, row
            except Exception:
                continue
        threshold = float(cfg.get("recognition_threshold", .42))
        if best is not None and best_score >= threshold:
            return best["name"], best["member_code"], best["id"], best_score
        return "Unknown", "", None, max(0, best_score)

    def assign_track(self, x, y):
        # Lightweight centroid association for this MVP; not a persistent identity tracker.
        for tid, pos in list(self.tracks.items()):
            if abs(pos[0]-x) < 90 and abs(pos[1]-y) < 120:
                self.tracks[tid] = (x, y)
                return tid
        tid = self.next_track_id
        self.next_track_id += 1
        self.tracks[tid] = (x, y)
        if len(self.tracks) > 80:
            self.tracks = dict(list(self.tracks.items())[-40:])
        return tid

    def check_crossing(self, track_key, x, member_id, name, code, score, cfg):
        # Crossing direction is inferred from center movement over consecutive detections.
        prev = self.last_centers.get(track_key)
        self.last_centers[track_key] = x
        if prev is None or abs(x-prev) < 35:
            return
        line = self.current_frame_width * float(cfg.get("line_x_percent", 50)) / 100
        if prev < line <= x:
            kind = "ENTRY"
        elif prev > line >= x:
            kind = "EXIT"
        else:
            return
        now = time.time()
        key = str(member_id) if member_id else "unknown"
        if now - self.last_member_event.get(key, 0) < 8:
            return
        self.last_member_event[key] = now
        if member_id:
            self.record_member_event(member_id, name, code, kind, score, self.camera_label)
        elif cfg.get("unknown_alerts", True):
            with connect_db() as c:
                c.execute("INSERT INTO events(member_id,member_name,member_code,event_type,event_time,confidence,source) VALUES(NULL,?,?,?,?,?,?)",
                          ("Unknown person", "", "UNKNOWN", datetime.now().isoformat(timespec="seconds"), float(score), self.camera_label))

    def record_member_event(self, member_id, name, code, kind, score, source):
        now = datetime.now().isoformat(timespec="seconds")
        with connect_db() as c:
            open_visit = c.execute("SELECT id FROM visits WHERE member_id=? AND status='inside' ORDER BY id DESC LIMIT 1",
                                   (member_id,)).fetchone()
            if kind == "ENTRY":
                if open_visit:
                    # Avoid creating duplicate open visits for the same member.
                    return
                c.execute("INSERT INTO visits(member_id,entry_time,status) VALUES(?,?,'inside')", (member_id, now))
            else:
                if open_visit:
                    c.execute("UPDATE visits SET exit_time=?,status='outside' WHERE id=?", (now, open_visit["id"]))
                else:
                    # Keep the observed exit event even if the entry was missed.
                    pass
            c.execute("INSERT INTO events(member_id,member_name,member_code,event_type,event_time,confidence,source) VALUES(?,?,?,?,?,?,?)",
                      (member_id, name, code, kind, now, float(score), source))

    def mjpeg(self):
        while True:
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
            if frame is None:
                frame = np.zeros((480, 800, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera not started. Configure camera and click Start.", (25,240),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (220,220,220), 2)
            ok, encoded = cv2.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")
            time.sleep(.05)

camera = CameraService()

def serialize_rows(rows):
    return [dict(r) for r in rows]

@app.route("/")
def index():
    return render_template("index.html")

@app.get("/video_feed")
def video_feed():
    return Response(camera.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/status")
def api_status():
    with connect_db() as c:
        members = c.execute("SELECT COUNT(*) n FROM members").fetchone()["n"]
        inside = c.execute("SELECT COUNT(*) n FROM visits WHERE status='inside'").fetchone()["n"]
        today = datetime.now().strftime("%Y-%m-%d")
        entries = c.execute("SELECT COUNT(*) n FROM events WHERE event_type='ENTRY' AND event_time LIKE ?", (today+"%",)).fetchone()["n"]
        exits = c.execute("SELECT COUNT(*) n FROM events WHERE event_type='EXIT' AND event_time LIKE ?", (today+"%",)).fetchone()["n"]
        unknown = c.execute("SELECT COUNT(*) n FROM events WHERE event_type='UNKNOWN' AND event_time LIKE ?", (today+"%",)).fetchone()["n"]
    return jsonify({"status": camera.status, "last_error": camera.last_error, "camera_label": camera.camera_label,
                    "members": members, "inside": inside, "entries_today": entries, "exits_today": exits,
                    "unknown_today": unknown, "camera_running": camera.running})

@app.get("/api/members")
def get_members():
    with connect_db() as c:
        rows = c.execute("SELECT id,member_code,name,email,notes,created_at,photo_path,CASE WHEN embedding!='' THEN 1 ELSE 0 END has_face FROM members ORDER BY name").fetchall()
    return jsonify(serialize_rows(rows))

@app.post("/api/members")
def add_member():
    data = request.get_json(force=True)
    code = str(data.get("member_code","")).strip()
    name = str(data.get("name","")).strip()
    if not code or not name:
        return jsonify({"error":"Member code and name are required."}), 400
    with connect_db() as c:
        try:
            cur = c.execute("INSERT INTO members(member_code,name,email,notes,created_at) VALUES(?,?,?,?,?)",
                            (code,name,str(data.get("email","")).strip(),str(data.get("notes","")).strip(),datetime.now().isoformat(timespec="seconds")))
            mid = cur.lastrowid
        except sqlite3.IntegrityError:
            return jsonify({"error":"That member code already exists."}), 409
    return jsonify({"ok":True,"id":mid})

@app.delete("/api/members/<int:member_id>")
def delete_member(member_id):
    with connect_db() as c:
        c.execute("DELETE FROM members WHERE id=?", (member_id,))
        # Keep historical events, but remove active visits so dashboard occupancy cannot get stuck.
        c.execute("DELETE FROM visits WHERE member_id=?", (member_id,))
    return jsonify({"ok":True})

@app.post("/api/members/<int:member_id>/enroll-face")
def enroll_face(member_id):
    if "photo" not in request.files:
        return jsonify({"error":"Choose a face photo first."}), 400
    file = request.files["photo"]
    raw = file.read()
    arr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error":"The uploaded file is not a readable image."}), 400
    try:
        from insightface.app import FaceAnalysis
        model = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        model.prepare(ctx_id=0, det_size=(640,640))
        faces = model.get(image)
        if not faces:
            return jsonify({"error":"No face found. Use a clear, front-facing photo."}), 400
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = face.normed_embedding.astype(np.float32).tolist()
        photo_dir = ROOT / "data" / "member_photos"
        photo_dir.mkdir(parents=True, exist_ok=True)
        photo_path = photo_dir / f"member_{member_id}.jpg"
        cv2.imwrite(str(photo_path), image)
        with connect_db() as c:
            c.execute("UPDATE members SET embedding=?,photo_path=? WHERE id=?",
                      (json.dumps(emb), str(photo_path.relative_to(ROOT)), member_id))
        return jsonify({"ok":True,"message":"Face template saved locally."})
    except ImportError:
        return jsonify({"error":"InsightFace is not installed. Run install.bat, then restart the app."}), 503
    except Exception as e:
        return jsonify({"error":"Face enrollment failed: "+str(e)}), 500

@app.get("/api/events")
def get_events():
    limit = min(500, max(1, int(request.args.get("limit", 100))))
    with connect_db() as c:
        rows = c.execute("SELECT id,member_name,member_code,event_type,event_time,confidence,source,reviewed FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return jsonify(serialize_rows(rows))

@app.get("/api/inside")
def get_inside():
    with connect_db() as c:
        rows = c.execute("""SELECT v.id,v.entry_time,m.id member_id,m.member_code,m.name
            FROM visits v JOIN members m ON m.id=v.member_id
            WHERE v.status='inside' ORDER BY v.entry_time DESC""").fetchall()
    return jsonify(serialize_rows(rows))

@app.post("/api/events/<int:event_id>/review")
def review_event(event_id):
    with connect_db() as c:
        c.execute("UPDATE events SET reviewed=1 WHERE id=?", (event_id,))
    return jsonify({"ok":True})

@app.get("/api/config")
def get_config():
    return jsonify(load_config())

@app.post("/api/config")
def update_config():
    data = request.get_json(force=True)
    cfg = {**load_config(), **data}
    if cfg.get("camera_type") not in ("usb","rtsp"):
        return jsonify({"error":"Camera type must be USB or RTSP."}), 400
    if cfg.get("camera_type") == "usb":
        try: cfg["camera_source"] = str(int(cfg.get("camera_source","0")))
        except Exception: cfg["camera_source"] = "0"
    cfg["line_x_percent"] = max(10, min(90, int(cfg.get("line_x_percent",50))))
    save_config(cfg)
    return jsonify({"ok":True,"config":cfg})

@app.post("/api/camera/start")
def start_camera():
    ok = camera.start()
    if not ok:
        return jsonify({"ok":False,"error":camera.last_error or camera.status}), 400
    return jsonify({"ok":True,"status":camera.status})

@app.post("/api/camera/stop")
def stop_camera():
    camera.stop()
    return jsonify({"ok":True})

@app.get("/api/export.csv")
def export_csv():
    with connect_db() as c:
        rows = c.execute("SELECT event_time,member_code,member_name,event_type,confidence,source FROM events ORDER BY id DESC").fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp","Member Code","Name","Event","Confidence","Camera Source"])
    for r in rows:
        writer.writerow(list(r))
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=bicep_access_logs.csv"})

@app.post("/api/manual-event")
def manual_event():
    data = request.get_json(force=True)
    member_id = int(data.get("member_id",0))
    kind = str(data.get("event_type","")).upper()
    if kind not in ("ENTRY","EXIT"):
        return jsonify({"error":"Event must be ENTRY or EXIT."}), 400
    with connect_db() as c:
        row = c.execute("SELECT id,name,member_code FROM members WHERE id=?", (member_id,)).fetchone()
    if not row:
        return jsonify({"error":"Member not found."}), 404
    camera.record_member_event(row["id"],row["name"],row["member_code"],kind,1.0,"Manual admin entry")
    return jsonify({"ok":True})

if __name__ == "__main__":
    print("BICEP AI Room Management running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
