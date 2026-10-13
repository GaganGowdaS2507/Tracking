# BICEP AI Room Management — Local MVP

A local-first prototype for managing room entry/exit activity. It includes a dashboard, member directory, face-template enrollment, camera input settings (USB or RTSP), entry/exit event logs, occupancy, unknown-event review and CSV export.

## Requirements
- Windows 10/11
- Python 3.10 or 3.11 recommended (64-bit)
- Internet connection for installing packages and the first InsightFace model download
- USB webcam for initial testing; an RTSP URL can be configured for an IP camera

## Start
1. Extract this folder to a normal writable location, e.g. `C:\BICEP_AI_Room_Management`.
2. Install Python 3.10 or 3.11 from python.org. Enable **Add Python to PATH**.
3. Double-click `install.bat` and wait for package installation.
4. Double-click `start.bat`.
5. Open `http://127.0.0.1:5000` in your browser.
6. Open **Camera & settings**, select USB webcam and use camera index `0`, save settings, then click **Save & connect** or go to Live monitoring and click Start camera.
7. Add a member, then choose **Enroll face** and upload a clear front-facing photo.
8. Test with only consenting participants and verify all logs manually.

## RTSP camera
In Camera & settings, select RTSP / IP camera and enter the RTSP URL supplied by the camera/NVR administrator. Keep the URL private because it can contain credentials. This workstation must be able to reach the camera on the network.

## What is included
- Flask web server and responsive dashboard
- SQLite database created automatically in `data/bicep.db`
- Member management, face embedding enrollment, entry/exit logs and current open visits
- USB camera and RTSP source selection
- Basic experimental line-crossing events and unknown-event review
- CSV export
- Local storage of member records and face embeddings

## Important limitations — do not skip
This is an **MVP/prototype, not production-ready attendance/security software**.
- Face recognition uses InsightFace `buffalo_l` embeddings. The app attempts to use ONNX Runtime GPU and falls back to CPU if available. First run may download pretrained model files. Verify the current license terms for both InsightFace code and pretrained weights before any commercial deployment.
- The current line-crossing tracker is intentionally lightweight and experimental; it can misclassify direction, lose tracks, double-count or miss events, especially with multiple people, occlusion, rapid movement, camera reconnects or a different camera resolution. Do not rely on its automatic records without testing and review. It needs a more robust tracker and calibrated crossing line before deployment.
- The configured operating hours are saved in settings but are not yet enforced as a hard processing schedule.
- Unknown detection only records a crossing event when face AI is working and an unmatched face crosses the line. It is not a person identity or proof of unauthorized access.
- No login/authentication is implemented yet. Keep the app bound to `127.0.0.1`; do not expose it to a network or the internet until authentication, authorization, HTTPS and security hardening are added.
- Back up and protect `data/`. Get informed consent before biometric enrollment, restrict access and define deletion/retention procedures. Review applicable Indian privacy law and club policy.
- Do not use this MVP as the sole basis for disciplinary, access-denial or safety decisions.

## Troubleshooting
- **Camera cannot open:** close other camera apps; try camera index `1`; confirm Windows camera privacy permissions; check RTSP network access.
- **AI unavailable:** run `install.bat` again and check installation output. Model files may download on the first recognition/enrollment. Internet access may be required.
- **No face found:** use a clear, well-lit, front-facing photo.
- **GPU issues:** this default setup is CPU-friendly. For a compatible NVIDIA CUDA environment, follow ONNX Runtime GPU compatibility guidance and install a matching package in the virtual environment. Do not install both CPU and GPU runtime packages blindly.
- **App data reset:** stop the app and back up the `data` folder before making changes.

## Data layout
- `data/bicep.db` — local SQLite database
- `data/config.json` — camera and app settings
- `data/member_photos/` — uploaded member photos
