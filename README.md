# Entry / Exit Tracking

Local Flask application for a two-camera entry/exit prototype. Each camera is configured independently as a USB webcam or RTSP stream.

## Windows setup
1. Use Python 3.10 or 3.11 (64-bit).
2. Run `install.bat`, then `start.bat`.
3. Open `http://127.0.0.1:5000`.
4. Open **Camera management**, detect available cameras, test each source, assign separate entry and exit sources, save, then connect each camera.
5. Register a member and enroll 3–5 clear face images from slightly different angles/lighting. Enrollment accepts up to 8 templates; use `?replace=1` on the enrollment endpoint to replace previous templates.

## Recognition improvements
- Persistent embeddings remain in `data/bicep.db`; prior one-vector templates continue to load.
- The cache stores multiple normalized templates per member and matches them with cosine similarity.
- The tracker uses motion-predicted association, blur/size/confidence/pose quality checks, a per-pass embedding budget, identity voting, ambiguity margin, and an identity-swap guard.
- Each camera has separate capture and inference threads. Shared inference locking prevents both cameras from running model calls simultaneously on a small GPU.
- The local database path is unchanged; do not delete `data/bicep.db` when updating the code.

## GPU
The normal installer uses CPU ONNX Runtime. For GPU mode, create a fresh virtual environment and install dependencies from `requirements-gpu.txt` instead of `requirements.txt`. CUDA provider support depends on the installed ONNX Runtime and NVIDIA driver stack. Check `/api/health` after start.

## Data
- `data/bicep.db`: members, embeddings, visits and events
- `data/config.json`: camera and recognition settings
- `data/member_photos/`: enrollment photos

This is a prototype, not certified access-control or attendance software. Validate live camera performance, false matches, false rejects, liveness and event correctness before operational use. Obtain appropriate notice/consent for biometric processing, restrict access to this PC, protect backups, define retention/deletion policies, and add authentication before exposing the app beyond localhost. Check current licensing terms for InsightFace code and pretrained weights before commercial use.
