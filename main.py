"""
Driver Drowsiness Detection — desktop application.

Opens your webcam, tracks facial landmarks in real time, and sounds an
alarm + shows an on-screen warning when your eyes stay closed too long.

Usage:
    python main.py
    python main.py --camera 1 --ear-threshold 0.22 --drowsy-frames 15
    python main.py --no-sound

Controls:
    q  - quit
    r  - reset the blink counter
"""

import argparse
import os

import cv2

from src.alarm import generate_alarm_wav, play_alarm_async, is_audio_available
from src.detector import DrowsinessDetector

ALARM_PATH = os.path.join("assets", "alarm.wav")


def draw_eye_points(frame, points, color=(0, 255, 0)):
    for x, y in points:
        cv2.circle(frame, (int(x), int(y)), 1, color, -1)


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time driver drowsiness detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--ear-threshold", type=float, default=0.25,
                         help="EAR below this value counts as eyes closed (default: 0.25)")
    parser.add_argument("--drowsy-frames", type=int, default=20,
                         help="Consecutive closed-eye frames before alerting (default: 20)")
    parser.add_argument("--no-sound", action="store_true", help="Disable the audio alarm")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_alarm_wav(ALARM_PATH)

    if not args.no_sound and not is_audio_available():
        print("[info] 'simpleaudio' not installed — running with visual alerts only.")
        print("       Install it with: pip install simpleaudio")

    detector = DrowsinessDetector(
        ear_threshold=args.ear_threshold,
        drowsy_consec_frames=args.drowsy_frames,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Check that a webcam is connected and not already in use."
        )

    print("Driver Drowsiness Detection running — press 'q' to quit, 'r' to reset blink count.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the camera. Exiting.")
                break

            frame = cv2.flip(frame, 1)  # mirror, like a selfie camera
            result = detector.process(frame)

            if result.face_found:
                color = (0, 0, 255) if result.is_drowsy else (0, 255, 0)
                draw_eye_points(frame, result.landmarks_left_eye, color)
                draw_eye_points(frame, result.landmarks_right_eye, color)

                cv2.putText(frame, f"EAR: {result.ear:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Blinks: {result.blink_count}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"FPS: {result.fps:.1f}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                if result.is_yawning:
                    cv2.putText(frame, "Yawn detected", (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if result.is_drowsy:
                    h, w = frame.shape[:2]
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 12)
                    text = "DROWSINESS ALERT!"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if not args.no_sound:
                        play_alarm_async(ALARM_PATH)
            else:
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            cv2.imshow("Driver Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                detector.reset_blink_count()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
