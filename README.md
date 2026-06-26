# 🚗 Driver Drowsiness Detection

Real-time computer-vision system that watches a driver's eyes through a
webcam and raises an alert (on-screen + sound) when they blink
abnormally — i.e. close their eyes for far longer than a normal blink,
a well-known early sign of drowsiness.

Built with **OpenCV** + **MediaPipe Face Mesh** (468-point facial
landmarks). No GPU required — runs comfortably on CPU in real time.

---

## How it works

1. **MediaPipe Face Mesh** locates 468 landmarks on the driver's face,
   every frame.
2. From those, the 6 landmarks around each eye are used to compute the
   **Eye Aspect Ratio (EAR)**:

   ```
   EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 · ‖p1-p4‖)
   ```

   An open eye has a high EAR (eye is "tall"); a closed eye's EAR drops
   close to zero (eye is "flat").
3. A small state machine watches the EAR over time:
   - A *brief* dip below the threshold (a few frames) = a **normal blink**.
   - EAR staying below the threshold for many **consecutive frames** =
     **drowsiness** → on-screen red banner + audio alarm.
4. (Bonus) The same landmarks give a **Mouth Aspect Ratio** for basic
   yawn detection, shown as a secondary signal.

This is the same EAR-based approach from Soukupová & Čech's 2016 paper
*"Real-Time Eye Blink Detection using Facial Landmarks,"* adapted to
MediaPipe's landmark model instead of dlib's.

---

## Project structure

```
driver-drowsiness-detection/
├── main.py                     # Desktop app — opens your webcam directly (OpenCV GUI)
├── app.py                      # Browser app — deploy to the web (Streamlit + WebRTC)
├── src/
│   ├── ear_utils.py            # EAR / MAR math + MediaPipe landmark indices
│   ├── detector.py             # DrowsinessDetector: FaceMesh wrapper + state machine
│   └── alarm.py                # Generates + plays the alarm beep (no audio assets needed)
├── tests/
│   └── test_ear.py             # Unit tests for the EAR/MAR math
├── assets/                     # alarm.wav is auto-generated here on first run
├── requirements.txt            # Deps for the desktop app
├── requirements-streamlit.txt  # Deps for the web app / cloud deployment
├── Dockerfile                  # Containerized deployment of the web app
└── README.md
```

---

## Option A — Run locally on your own machine (desktop app)

Best for: testing with your own webcam, highest performance, real audio alarm.

```bash
git clone <this-repo>
cd driver-drowsiness-detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

Controls: press **`q`** to quit, **`r`** to reset the blink counter.

Useful flags:

```bash
python main.py --camera 1                 # use a different webcam
python main.py --ear-threshold 0.22       # more/less sensitive eye-closed detection
python main.py --drowsy-frames 15         # alert sooner/later
python main.py --no-sound                 # visual alert only
```

> 🔊 Audio uses your OS's built-in player (`winsound` on Windows, `afplay`
> on macOS, `aplay`/`paplay` on Linux) — there is nothing extra to
> install for sound to work on a normal desktop/laptop. If no audio
> device is found, it silently falls back to visual-only alerts.

---

## Option B — Run / deploy as a web app (works in the browser)

This version (`app.py`) uses [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc)
so the camera runs **in the visitor's browser** — meaning you can deploy
it to a server that has no webcam at all, and each visitor still uses
their own camera.

### Run locally

```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) and
click **Start** to allow camera access.

### Deploy instantly — Streamlit Community Cloud (free)

1. Push this folder to a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**.
3. Pick the repo, set **Main file path** to `app.py`.
4. Under **Advanced settings → Python dependencies**, point it at
   `requirements-streamlit.txt` (or rename it to `requirements.txt` for
   that deployment if the platform doesn't let you choose a filename).
5. Deploy. You'll get a public URL (`your-app.streamlit.app`) within a
   few minutes.

> ⚠️ **If the video feed gets stuck on "Connecting..." once deployed:**
> this is a known [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)
> limitation, not a bug in this project — some cloud platforms (including
> Streamlit Community Cloud, on some networks) sit behind a proxy that
> blocks the raw WebRTC packets unless a **TURN server** is configured.
> `app.py` already includes a public STUN server, which is enough on most
> home networks. If it still won't connect, add a free-trial
> [Twilio TURN server](https://www.twilio.com/docs/stun-turn) — a few
> lines in `RTC_CONFIGURATION` at the top of `app.py` — or run the
> desktop app (`main.py`) instead, which has no networking step at all.

### Deploy with Docker (any cloud: Render, Fly.io, AWS, GCP, your own VPS)

```bash
docker build -t drowsiness-app .
docker run -p 8501:8501 drowsiness-app
```

Then open `http://localhost:8501` (or your server's address/port).

### Deploy on Hugging Face Spaces

Create a new Space → SDK: **Streamlit** → push this repo's contents
(use `requirements-streamlit.txt` as `requirements.txt` in the Space).

---

## Tuning parameters

| Parameter             | Default | What it does                                                        |
|------------------------|---------|----------------------------------------------------------------------|
| `ear_threshold`        | `0.25`  | EAR below this = "eyes closed". Lower it if it triggers too easily, raise it if it misses real closures. |
| `drowsy_consec_frames` | `20`    | How many consecutive closed-eye frames before alerting. At ~20–30 FPS this is roughly 0.7–1 second. |
| `blink_max_frames`     | `3`     | A closure shorter than this counts as a normal blink, not drowsiness. |
| `yawn_threshold`       | `0.6`   | MAR above this counts as a yawn (bonus signal, shown but not alarmed on by default). |

Lighting and camera angle affect raw EAR values more than the threshold
default accounts for — if you get false alerts, try sitting in better
light and/or nudging `ear_threshold` down a little first.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

These test the EAR/MAR geometry directly (no camera or model needed),
so they run instantly anywhere, including CI.

---

## Limitations & possible extensions

- Works best with a clear, front-facing view of the eyes; sunglasses,
  extreme head turns, or very poor lighting reduce landmark accuracy
  (this is a MediaPipe limitation, not specific to this project).
- Single-driver detection (`max_faces=1` by default) — bump
  `max_num_faces` in `DrowsinessDetector` for multi-person monitoring.
- Ideas to extend this further:
  - Head-pose / nodding detection as a second drowsiness signal.
  - Log drowsiness events with timestamps for a post-drive report.
  - Mobile deployment via MediaPipe Tasks for Android/iOS.
  - Combine the existing yawn (MAR) signal into the alert logic itself.

---

## License

MIT — see `LICENSE`. Use it, modify it, ship it.
