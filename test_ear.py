"""
Unit tests for the EAR / MAR math (no camera or MediaPipe model needed).

Run with:
    pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.ear_utils import eye_aspect_ratio, mouth_aspect_ratio


def test_ear_open_eye_is_high():
    eye = [(0, 5), (2, 0), (4, 0), (8, 5), (4, 10), (2, 10)]
    assert eye_aspect_ratio(eye) > 0.5


def test_ear_closed_eye_is_low():
    eye = [(0, 5), (2, 4.9), (4, 4.9), (8, 5), (4, 5.1), (2, 5.1)]
    assert eye_aspect_ratio(eye) < 0.1


def test_ear_open_greater_than_closed():
    open_eye = [(0, 5), (2, 0), (4, 0), (8, 5), (4, 10), (2, 10)]
    closed_eye = [(0, 5), (2, 4.9), (4, 4.9), (8, 5), (4, 5.1), (2, 5.1)]
    assert eye_aspect_ratio(open_eye) > eye_aspect_ratio(closed_eye)


def test_ear_invalid_input_raises():
    with pytest.raises(ValueError):
        eye_aspect_ratio([(0, 0), (1, 1)])  # only 2 points, needs 6


def test_ear_zero_width_is_safe():
    eye = [(5, 5), (2, 0), (4, 0), (5, 5), (4, 10), (2, 10)]
    assert eye_aspect_ratio(eye) == 0.0


def test_mar_basic_shape():
    mouth = [(0, 5), (2, 0), (4, 0), (8, 5), (4, 10), (2, 10)]
    assert mouth_aspect_ratio(mouth) > 0


def test_mar_invalid_input_raises():
    with pytest.raises(ValueError):
        mouth_aspect_ratio([(0, 0)])
