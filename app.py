"""
Driver Drowsiness Detection — Streamlit web app.

This version runs the camera *in the visitor's browser* via WebRTC, so it
can be deployed to a server with no physical webcam (Streamlit Community
Cloud, Hugging Face Spaces, Render, your own Docker host, etc.) and still
access each visitor's own camera.

Run locally:
    pip install -r requirements-streamlit.txt
    streamlit run app.py

Deploy:
    See README.md for Streamlit Cloud / Docker / Hugging Face instructions.
"""

import base64
import os
import threading
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.alarm import generate_alarm_wav
from src.detector import DrowsinessDetector

ALARM_PATH = os.path.join("assets", "alarm.wav")
generate_alarm_wav(ALARM_PATH)

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="wide")


@st.cache_resource
def _load_alarm_base64() -> str:
    with open(ALARM_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


_ALARM_B64 = _load_alarm_base64()


def _autoplay_audio_html() -> str:
    return f"""
    <audio autoplay>
        <source src="data:audio/wav;base64,{_ALARM_B64}" type="audio/wav">
    </audio>
    """


class DrowsinessVideoProcessor(VideoProcessorBase):
    """Runs once per browser-submitted video frame, on a background thread."""

    def __init__(self) -> None:
        self.detector = DrowsinessDetector(
            ear_threshold=st.session_state.get("ear_threshold", 0.25),
            drowsy_consec_frames=st.session_state.get("drowsy_frames", 20),
        )
        self.lock = threading.Lock()
        self.latest = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror, like a selfie camera

        result = self.detector.process(img)

        if result.face_found:
            color = (0, 0, 255) if result.is_drowsy else (0, 255, 0)
            for x, y in result.landmarks_left_eye + result.landmarks_right_eye:
                cv2.circle(img, (int(x), int(y)), 1, color, -1)

            cv2.putText(img, f"EAR: {result.ear:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if result.is_drowsy:
                h, w = img.shape[:2]
                cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 12)
                text = "DROWSINESS ALERT!"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                cv2.putText(img, text, (max((w - tw) // 2, 10), (h + th) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        with self.lock:
            self.latest = result

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("🚗 Driver Drowsiness Detection")
    st.caption(
        "Real-time facial-landmark eye tracking that flags abnormally "
        "prolonged eye closure — a common sign of driver fatigue."
    )

    with st.sidebar:
        st.header("Settings")
        st.session_state["ear_threshold"] = st.slider(
            "Eye-closed sensitivity (EAR threshold)", 0.15, 0.35, 0.25, 0.01,
            help="Lower = eyes must close more before it counts as 'closed'.",
        )
        st.session_state["drowsy_frames"] = st.slider(
            "Frames closed before alert", 5, 60, 20, 1,
            help="How many consecutive closed-eye frames trigger the alarm.",
        )
        sound_on = st.checkbox("Play audio alarm", value=True)
        st.markdown("---")
        st.markdown(
            "**How it works:** MediaPipe Face Mesh locates 468 facial "
            "landmarks per frame. The Eye Aspect Ratio (EAR) — the ratio "
            "of eye height to eye width — drops sharply when the eyes "
            "close. If it stays low for too many consecutive frames, the "
            "app raises a drowsiness alert."
        )

    col_video, col_stats = st.columns([3, 1])

    with col_video:
        ctx = webrtc_streamer(
            key="drowsiness-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=DrowsinessVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_stats:
        ear_metric = st.empty()
        blink_metric = st.empty()
        status_box = st.empty()
        audio_box = st.empty()

    last_alarm_time = 0.0
    while ctx.state.playing:
        if ctx.video_processor:
            with ctx.video_processor.lock:
                result = ctx.video_processor.latest

            if result is not None and result.face_found:
                ear_metric.metric("EAR", f"{result.ear:.2f}")
                blink_metric.metric("Blinks", result.blink_count)
                if result.is_drowsy:
                    status_box.error("⚠️ DROWSINESS DETECTED — Pull over and rest!")
                    if sound_on and (time.time() - last_alarm_time) > 2.0:
                        audio_box.markdown(_autoplay_audio_html(), unsafe_allow_html=True)
                        last_alarm_time = time.time()
                else:
                    status_box.success("✅ Alert and awake")
            else:
                status_box.warning("No face detected — adjust your position or lighting.")
        time.sleep(0.3)


if __name__ == "__main__":
    main()
