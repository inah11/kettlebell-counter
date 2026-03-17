"""Shared test fixtures and landmark helpers."""

import sys
import pathlib
import pytest

# Ensure project root is on the path so modules resolve without install
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from constants import (
    BOTTOM, RISING, TOP, FALLING,
    RISE_THRESHOLD, DROP_THRESHOLD,
    SMOOTH_WINDOW, MIN_TOP_FRAMES,
)
from tracker import SideTracker


# ── Fake MediaPipe landmark helpers ───────────────────────────────────────────

class _FakeLM:
    """Minimal stand-in for a MediaPipe NormalizedLandmark."""
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def make_landmark_list(
    wrist_y_norm: float,
    shoulder_y_norm: float = 0.3,
    hip_y_norm: float = 0.6,
    elbow_y_norm: float = 0.4,
    frame_h: int = 480,
) -> list:
    """
    Return a 33-element list of _FakeLM objects.

    The RIGHT_* slots (indices 12, 14, 16, 24) are set from the supplied
    normalised-Y values; all other landmarks are neutral stubs.
    """
    lms = [_FakeLM() for _ in range(33)]

    # MediaPipe RIGHT landmarks (mirrored in video: right shoulder is index 12 etc.)
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW    = 14
    RIGHT_WRIST    = 16
    RIGHT_HIP      = 24

    lms[RIGHT_SHOULDER].y = shoulder_y_norm
    lms[RIGHT_ELBOW].y    = elbow_y_norm
    lms[RIGHT_WRIST].y    = wrist_y_norm
    lms[RIGHT_HIP].y      = hip_y_norm
    return lms


@pytest.fixture
def tracker():
    """Fresh SideTracker with MIN_TOP_FRAMES=2 for fast test cycles."""
    return SideTracker(
        rise_threshold=RISE_THRESHOLD,
        drop_threshold=DROP_THRESHOLD,
        min_top_frames=2,
    )


def drive_to_bottom(t: SideTracker, n: int = SMOOTH_WINDOW) -> None:
    """Feed n frames of deep-swing norm_y to push the tracker solidly to BOTTOM."""
    for _ in range(n):
        t.transition(DROP_THRESHOLD + 0.2)


def drive_to_top(t: SideTracker, n: int = SMOOTH_WINDOW) -> None:
    """Feed n frames of overhead norm_y (no elbow/knee gating)."""
    for _ in range(n):
        t.transition(RISE_THRESHOLD - 0.3)
