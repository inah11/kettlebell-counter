"""Tests for geometry.py — pure math, no MediaPipe."""

import math
from collections import deque

import pytest

from geometry import angle_at_joint, smooth, get_landmark_y, normalise_wrist


# ── angle_at_joint ────────────────────────────────────────────────────────────

def test_straight_arm_is_180():
    # A-B-C collinear → angle at B = 180°
    angle = angle_at_joint(0, 0, 1, 0, 2, 0)
    assert abs(angle - 180.0) < 1e-6


def test_right_angle_is_90():
    # A = (0,1), B = (0,0), C = (1,0) → 90°
    angle = angle_at_joint(0, 1, 0, 0, 1, 0)
    assert abs(angle - 90.0) < 1e-6


def test_degenerate_joint_returns_180():
    # B coincides with A — magnitude is zero
    angle = angle_at_joint(1, 1, 1, 1, 2, 2)
    assert angle == 180.0


def test_angle_is_symmetric():
    # Reversing A and C should give the same angle at B
    a1 = angle_at_joint(0, 1, 0, 0, 1, 0)
    a2 = angle_at_joint(1, 0, 0, 0, 0, 1)
    assert abs(a1 - a2) < 1e-9


# ── smooth ────────────────────────────────────────────────────────────────────

def test_smooth_single_value():
    buf = deque(maxlen=5)
    result = smooth(buf, 0.8)
    assert result == pytest.approx(0.8)


def test_smooth_averages_window():
    buf = deque(maxlen=3)
    smooth(buf, 0.0)
    smooth(buf, 1.0)
    result = smooth(buf, 0.5)
    assert result == pytest.approx(0.5)


def test_smooth_evicts_old_values():
    buf = deque(maxlen=2)
    smooth(buf, 100.0)  # will be evicted after two more appends
    smooth(buf, 1.0)
    result = smooth(buf, 3.0)
    assert result == pytest.approx(2.0)  # (1+3)/2, 100 evicted


# ── get_landmark_y ────────────────────────────────────────────────────────────

class _FakeLM:
    def __init__(self, y): self.y = y; self.x = 0.5; self.z = 0.0


def test_get_landmark_y_scales_by_frame_h():
    lms = [_FakeLM(0.0), _FakeLM(0.5), _FakeLM(1.0)]
    assert get_landmark_y(lms, 1, 480) == pytest.approx(240.0)


def test_get_landmark_y_top_of_frame():
    lms = [_FakeLM(0.0)]
    assert get_landmark_y(lms, 0, 720) == pytest.approx(0.0)


# ── normalise_wrist ───────────────────────────────────────────────────────────

def test_normalise_at_shoulder_is_zero():
    # wrist_y == shoulder_y → norm_y = 0
    assert normalise_wrist(100.0, 100.0, 300.0) == pytest.approx(0.0)


def test_normalise_at_hip_is_one():
    assert normalise_wrist(300.0, 100.0, 300.0) == pytest.approx(1.0)


def test_normalise_above_shoulder_is_negative():
    assert normalise_wrist(50.0, 100.0, 300.0) < 0


def test_normalise_below_hip_exceeds_one():
    assert normalise_wrist(400.0, 100.0, 300.0) > 1.0


def test_normalise_degenerate_torso_returns_half():
    # shoulder_y == hip_y → guard returns 0.5
    assert normalise_wrist(100.0, 200.0, 200.0) == pytest.approx(0.5)
