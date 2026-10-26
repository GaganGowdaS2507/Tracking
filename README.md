# Entry / Exit Tracking — local two-camera MVP

A local Flask application for a room entry/exit prototype. It supports two independently configured camera inputs: an **Entry Camera** and an **Exit Camera**. Each can use a USB webcam or an RTSP stream.

## Windows quick start

1. Use Python 3.10 or 3.11 (64-bit).
2. Run `install.bat`.
3. Run `start.bat`.
4. Open `http://127.0.0.1:5000`.
5. Open **Camera management**, click **Detect available cameras**, then use **Test source** on each detected camera.
6. Assign different USB indexes to Entry and Exit (usually index `0` and `1`, but confirm with tests).
7. Save each camera and press **Connect entry** / **Connect exit**.
8. Register members under **Members** and enroll clear, front-facing images.

## Why this version should feel smoother

- Each camera has a capture thread and a separate AI inference thread.
- The capture thread keeps reading frames instead of waiting for recognition to finish.
- Frames are capped at 1280px wide for local processing.
- Face detection runs at a configurable interval and a shared inference lock prevents both cameras from loading/using the face model concurrently on a small GPU.
- CPU/GPU selection is automatic based on ONNX Runtime providers installed in the environment; CPU fallback is supported.

## Entry/exit behavior

- A recognized member seen by **Entry Camera** is recorded as ENTRY if they do not already have an open visit.
- A recognized member seen by **Exit Camera** is recorded as EXIT if they have an open visit.
- Unmatched face observations can create throttled UNKNOWN events.
- Cameras should be framed tightly on the doorway. If a camera continuously sees people sitting in the room, that can cause incorrect presence-based events. Test carefully with controlled participants.

## USB cameras

OpenCV exposes numeric camera indexes and does not reliably return device model names. In many Windows setups the built-in camera is `0` and an external USB camera is `1`, but this is not guaranteed. Use **Detect available cameras** and **Test source** to map them. If camera discovery reports no device, try closing apps that may be using the webcam, replugging the USB webcam, checking Windows Settings → Privacy & security → Camera, and clicking Detect again.

## RTSP cameras

Select RTSP / IP camera and provide the stream URL. Keep embedded credentials private. The workstation must have network access to the camera/NVR.

## Data location

- `data/bicep.db` — member records, visits and events (keeps compatibility with the earlier app database)
- `data/config.json` — camera and app settings
- `data/member_photos/` — uploaded face photos

## Limitations and safe use

This is an MVP, not production security or attendance software. Presence-based camera rules can produce incorrect events if framing is poor, lighting is weak, a face is not visible, or several people overlap. Verify logs and do not use these outputs as the only basis for access or disciplinary decisions. Add authentication before exposing the app beyond localhost. Obtain appropriate notice/consent for biometric enrollment, set retention/deletion policies, and review applicable law and club policy.

Check current InsightFace code and pretrained-weight licenses before commercial use. The standard installer uses the CPU ONNX Runtime dependency; GPU mode is used only when a compatible ONNX Runtime GPU package and runtime are installed.
