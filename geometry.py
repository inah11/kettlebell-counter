"""Pure geometry helpers — no CV or MediaPipe dependencies."""

import math
from collections import deque


def angle_at_joint(ax, ay, bx, by, cx, cy) -> float:
    """Angle in degrees at B in triangle A-B-C (e.g. A=shoulder, B=elbow, C=wrist)."""
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag = (bax**2 + bay**2)**0.5 * (bcx**2 + bcy**2)**0.5
    if mag < 1e-6:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def smooth(window: deque, value: float) -> float:
    """Append value to a rolling window and return the current average."""
    window.append(value)
    return sum(window) / len(window)


def get_landmark_y(landmark_list, landmark_idx, frame_h: int) -> float:
    """Return pixel-space Y for a MediaPipe landmark index."""
    return landmark_list[landmark_idx].y * frame_h


def normalise_wrist(wrist_y: float, shoulder_y: float, hip_y: float) -> float:
    """
    Map wrist Y to a torso-relative scale:
      0   = at shoulder level
      1   = at hip level
      >1  = below hip (deep swing bottom)
      <0  = above shoulder (overhead)

    Lower pixel-Y means higher in the frame (pixel origin is top-left).
    """
    torso_height_px = hip_y - shoulder_y
    if torso_height_px < 1e-6:          # guard against degenerate poses
        return 0.5
    return (wrist_y - shoulder_y) / torso_height_px
