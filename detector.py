"""
Core drowsiness-detection logic.

Wraps MediaPipe's FaceMesh to get 468 facial landmarks per frame, derives
the Eye Aspect Ratio (EAR) each frame, and runs a small state machine that
tells a normal blink apart from a prolonged "drowsy" eye closure.

This module has no UI/display code, so it can be reused by both the
desktop OpenCV app (main.py) and the browser-based Streamlit app (app.py).
"""

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import mediapipe as mp

from .ear_utils import (
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    MOUTH_IDX,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    extract_points,
)


@dataclass
class DetectionResult:
    face_found: bool
    ear: float = 0.0
    mar: float = 0.0
    is_drowsy: bool = False
    is_blinking: bool = False
    is_yawning: bool = False
    blink_count: int = 0
    closed_frames: int = 0
    fps: float = 0.0
    landmarks_left_eye: List[Tuple[float, float]] = field(default_factory=list)
    landmarks_right_eye: List[Tuple[float, float]] = field(default_factory=list)
    landmarks_mouth: List[Tuple[float, float]] = field(default_factory=list)


class DrowsinessDetector:
    """
    Stateful, frame-by-frame drowsiness detector.

    Usage:
        detector = DrowsinessDetector()
        result = detector.process(bgr_frame)   # call once per video frame
        ...
        detector.close()
    """

    def __init__(
        self,
        ear_threshold: float = 0.25,
        drowsy_consec_frames: int = 20,
        blink_max_frames: int = 3,
        yawn_threshold: float = 0.6,
        max_faces: int = 1,
    ):
        """
        Args:
            ear_threshold: EAR below this value counts as "eyes closed".
            drowsy_consec_frames: consecutive closed-eye frames required
                before raising a drowsiness alert. Roughly
                (frames / camera_fps) seconds of continuous closure.
            blink_max_frames: a closure lasting at most this many frames
                is counted as a normal blink rather than drowsiness.
            yawn_threshold: MAR above this value counts as a yawn.
            max_faces: max number of faces MediaPipe will track at once.
        """
        self.ear_threshold = ear_threshold
        self.drowsy_consec_frames = drowsy_consec_frames
        self.blink_max_frames = blink_max_frames
        self.yawn_threshold = yawn_threshold

        self._mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._closed_frames = 0
        self._was_closed = False
        self._blink_count = 0
        self._last_time = time.time()

    def process(self, frame_bgr) -> DetectionResult:
        """Run detection on a single BGR frame (as returned by cv2.VideoCapture)."""
        now = time.time()
        fps = 1.0 / max(now - self._last_time, 1e-6)
        self._last_time = now

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._closed_frames = 0
            self._was_closed = False
            return DetectionResult(face_found=False, fps=fps, blink_count=self._blink_count)

        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = extract_points(landmarks, LEFT_EYE_IDX, w, h)
        right_eye = extract_points(landmarks, RIGHT_EYE_IDX, w, h)
        mouth = extract_points(landmarks, MOUTH_IDX, w, h)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        eyes_closed = ear < self.ear_threshold
        is_drowsy = False
        is_blinking = False

        if eyes_closed:
            self._closed_frames += 1
            self._was_closed = True
        else:
            if self._was_closed and 0 < self._closed_frames <= self.blink_max_frames:
                self._blink_count += 1
                is_blinking = True
            self._closed_frames = 0
            self._was_closed = False

        if self._closed_frames >= self.drowsy_consec_frames:
            is_drowsy = True

        return DetectionResult(
            face_found=True,
            ear=ear,
            mar=mar,
            is_drowsy=is_drowsy,
            is_blinking=is_blinking,
            is_yawning=mar > self.yawn_threshold,
            blink_count=self._blink_count,
            closed_frames=self._closed_frames,
            fps=fps,
            landmarks_left_eye=left_eye,
            landmarks_right_eye=right_eye,
            landmarks_mouth=mouth,
        )

    def reset_blink_count(self) -> None:
        self._blink_count = 0

    def close(self) -> None:
        self.face_mesh.close()
