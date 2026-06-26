"""
Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) utilities.

EAR is the core signal used to detect eye closure from facial landmarks.
It is the ratio of eye "height" to eye "width" — a wide-open eye has a
high EAR, a closed eye's EAR drops close to zero.

Reference:
    Soukupova, T., & Cech, J. (2016). "Real-Time Eye Blink Detection
    using Facial Landmarks." 21st Computer Vision Winter Workshop.
"""

from typing import List, Tuple
import numpy as np

# MediaPipe Face Mesh has 468 landmarks. These index subsets correspond to
# the classic 6-point eye model used in the EAR formula:
#   [outer_corner, upper_lid_1, upper_lid_2, inner_corner, lower_lid_1, lower_lid_2]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Used for an optional Mouth Aspect Ratio (yawn) signal.
MOUTH_IDX = [61, 81, 311, 291, 178, 402]

Point = Tuple[float, float]


def _euclidean(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def eye_aspect_ratio(eye_points: List[Point]) -> float:
    """
    Compute the Eye Aspect Ratio for one eye.

    Args:
        eye_points: exactly 6 (x, y) points ordered as
            [outer_corner, upper_1, upper_2, inner_corner, lower_1, lower_2]

    Returns:
        EAR value. Typically ~0.30-0.40 for an open eye and drops below
        ~0.20 when the eye is closed (exact values vary per face/camera).
    """
    if len(eye_points) != 6:
        raise ValueError("eye_points must contain exactly 6 (x, y) points")

    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth_points: List[Point]) -> float:
    """
    Compute the Mouth Aspect Ratio (optional yawn-detection signal).
    Same 6-point convention as eye_aspect_ratio.
    """
    if len(mouth_points) != 6:
        raise ValueError("mouth_points must contain exactly 6 (x, y) points")

    p1, p2, p3, p4, p5, p6 = mouth_points
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    return vertical / (2.0 * horizontal)


def extract_points(landmarks, indices: List[int], width: int, height: int) -> List[Point]:
    """
    Convert MediaPipe's normalized (0-1) landmark coordinates into pixel
    (x, y) coordinates for a given list of landmark indices.
    """
    return [(landmarks[i].x * width, landmarks[i].y * height) for i in indices]
