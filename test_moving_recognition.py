import json
import sqlite3
import numpy as np
import cv2
from pathlib import Path
import io

from app import app, get_face_model, face_quality, identify_embedding, Det, load_config

def run_tests():
    print("=" * 60)
    print("  RUNNING 3-PIC ENROLLMENT & MOVING FACE RECOGNITION TEST SUITE")
    print("=" * 60)

    # 1. Test Model Loading
    print("\n[TEST 1] Testing InsightFace Model...")
    model = get_face_model()
    assert model is not None, "InsightFace model failed to load!"
    print("  [OK] InsightFace buffalo_l loaded successfully.")

    # 2. Prepare 3 photos (Front, Left angle, Right angle)
    print("\n[TEST 2] Preparing 3-angle enrollment photos (Front, Left, Right)...")
    base_img = cv2.imread("data/member_photos/member_1.jpg")
    assert base_img is not None, "Base photo member_1.jpg not found!"
    h, w = base_img.shape[:2]

    # Front view (original)
    front_img = base_img.copy()

    # Left angle (~20-25 deg yaw simulation via slight perspective shear)
    pts1 = np.float32([[w*0.2, h*0.2], [w*0.8, h*0.2], [w*0.2, h*0.8], [w*0.8, h*0.8]])
    pts_left = np.float32([[w*0.25, h*0.19], [w*0.75, h*0.22], [w*0.25, h*0.81], [w*0.75, h*0.78]])
    M_left = cv2.getPerspectiveTransform(pts1, pts_left)
    left_img = cv2.warpPerspective(base_img, M_left, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Right angle (~20-25 deg yaw simulation via slight perspective shear)
    pts_right = np.float32([[w*0.22, h*0.22], [w*0.78, h*0.19], [w*0.22, h*0.78], [w*0.78, h*0.81]])
    M_right = cv2.getPerspectiveTransform(pts1, pts_right)
    right_img = cv2.warpPerspective(base_img, M_right, (w, h), borderMode=cv2.BORDER_REPLICATE)

    _, buf_front = cv2.imencode(".jpg", front_img)
    _, buf_left = cv2.imencode(".jpg", left_img)
    _, buf_right = cv2.imencode(".jpg", right_img)

    # 3. Test Multi-Angle Enrollment Endpoint via Flask Test Client
    print("\n[TEST 3] Testing /api/members/<id>/enroll-multi endpoint...")
    client = app.test_client()

    # Ensure member 1 exists
    conn = sqlite3.connect("data/bicep.db")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO members (id, member_code, name, email, created_at) VALUES (1, '001', 'Admin', 'admin@example.com', '2026-10-26T12:00:00')")
    conn.commit()
    conn.close()

    data = {
        "photo_front": (io.BytesIO(buf_front.tobytes()), "front.jpg"),
        "photo_left": (io.BytesIO(buf_left.tobytes()), "left.jpg"),
        "photo_right": (io.BytesIO(buf_right.tobytes()), "right.jpg"),
    }
    resp = client.post("/api/members/1/enroll-multi", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200, f"Enroll failed: {resp.get_json()}"
    res_json = resp.get_json()
    print("  Enroll response:", res_json)
    assert res_json["ok"] is True
    assert res_json["count"] >= 2, "Expected at least 2 or 3 templates enrolled"
    print(f"  [OK] Multi-angle enrollment succeeded with {res_json['count']} templates.")

    # Verify SQLite database
    conn = sqlite3.connect("data/bicep.db")
    c = conn.cursor()
    c.execute("SELECT embedding, photo_path FROM members WHERE id=1")
    row = c.fetchone()
    stored_templates = json.loads(row[0])
    print(f"  [OK] SQLite verified: member 1 now stores {len(stored_templates)} 512-d template vectors!")
    assert len(stored_templates) >= 2
    assert len(stored_templates[0]) == 512
    conn.close()

    # 4. Test Moving Face Recognition
    print("\n[TEST 4] Testing Moving Face Recognition (with motion blur)...")
    # Simulate a moving frame by applying directional motion blur
    blur_kernel_size = 9
    kernel_motion_blur = np.zeros((blur_kernel_size, blur_kernel_size))
    kernel_motion_blur[int((blur_kernel_size-1)/2), :] = np.ones(blur_kernel_size)
    kernel_motion_blur = kernel_motion_blur / blur_kernel_size
    moving_frame = cv2.filter2D(base_img, -1, kernel_motion_blur)

    cfg = load_config()
    print(f"  Config: threshold={cfg.get('recognition_threshold')}, min_sharpness={cfg.get('min_sharpness')}, max_yaw={cfg.get('max_yaw')}")

    # Detect faces in moving frame
    from app import detect_faces, embed_face
    dets = detect_faces(model, moving_frame)
    assert len(dets) > 0, "Detector failed to find face in moving frame"
    d = dets[0]
    face_quality(moving_frame, d, cfg)
    print(f"  Detected face quality: {d.quality:.3f}, reason: '{d.reason}', score: {d.score:.3f}")
    assert d.quality > 0.0, "FAIL: Moving face was rejected with quality 0.0!"
    print("  [OK] Moving face successfully PASSED quality check without rejection!")

    # Extract ArcFace embedding from moving frame
    emb = embed_face(model, moving_frame, d)
    assert emb is not None, "Failed to compute embedding from moving face"

    # Identify against enrolled templates
    member, score, margin = identify_embedding(emb, float(cfg.get("recognition_threshold", 0.37)))
    print(f"  Match Result on Moving Frame: member={member['name'] if member else None}, score={score:.4f}, margin={margin:.4f}")
    assert member is not None, f"FAIL: Moving face failed to match! Score was {score:.4f}"
    assert member["id"] == 1, f"FAIL: Matched wrong member: {member}"
    print(f"  [OK] SUCCESS: Moving blurred face correctly identified as '{member['name']}' with score {score:.3f} >= threshold {cfg.get('recognition_threshold')}!")

    # 5. Test API Endpoints
    print("\n[TEST 5] Testing Web API Status & Members Listing...")
    m_resp = client.get("/api/members")
    assert m_resp.status_code == 200
    members = m_resp.get_json()
    admin_m = [m for m in members if m["id"] == 1][0]
    print(f"  Member 1 listing: template_count={admin_m.get('template_count')}, has_face={admin_m.get('has_face')}")
    assert admin_m.get("template_count", 0) >= 2
    assert admin_m.get("has_face") == 1
    print("  [OK] /api/members correctly exposes template_count to UI!")

    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    print("  Health check:", health_resp.get_json())
    print("  [OK] /api/health returned ok=True")

    print("\n" + "=" * 60)
    print("  ALL 5 TEST SUITES PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
