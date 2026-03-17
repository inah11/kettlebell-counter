"""
Kettlebell Snatch Rep Counter
Usage:
  python snatch_counter.py <video_file>                # single KB (right hand)
  python snatch_counter.py <video_file> --mode double  # double KB (both hands)
  python snatch_counter.py <video_file> --mode switch  # hand-to-hand KB set
  python snatch_counter.py <video_file> --elbow-lock   # require elbow lockout at top
  python snatch_counter.py <video_file> --calibrate    # dedicated calibration pass
  python snatch_counter.py <video_file> --no-auto-calib  # skip auto-calibration
"""

import json
import math
import pathlib
import sys
import argparse
from collections import deque

import cv2
import mediapipe as mp


# ── State machine ────────────────────────────────────────────────────────────
BOTTOM  = "BOTTOM"
RISING  = "RISING"
TOP     = "TOP"
FALLING = "FALLING"

# Thresholds (in normalised units).
# Normalised wrist Y = (wrist_y - shoulder_y) / (hip_y - shoulder_y)
# 0 → at shoulder level, 1 → at hip level.
# Values > 1 mean the wrist is below the hip (deep swing).
RISE_THRESHOLD  = 0.4   # wrist must reach above this to be considered "high"
DROP_THRESHOLD  = 0.85  # wrist must fall below this to be considered "low"
SMOOTH_WINDOW   = 5
MIN_TOP_FRAMES  = 4     # legacy fallback; real value is FPS-derived per tracker
                        # (prevents brief swing-through from counting)
MIN_REP_LOCKOUT = 15    # frames locked after a rep is counted (~0.5 s at 30 fps)
                        # (prevents backswing from immediately starting a new cycle)
ELBOW_LOCK_ANGLE = 160  # degrees; elbow ≥ this → arm is locked out

# ── Calibration constants ─────────────────────────────────────────────────────
CALIB_FRAMES    = 150   # ~5 s at 30 fps
CALIB_MIN_RANGE = 0.5   # narrower range → fall back to defaults
CALIB_RISE_FRAC = 0.35
CALIB_DROP_FRAC = 0.75
CALIB_FILE      = pathlib.Path("calibration.json")

STATE_COLORS = {
    BOTTOM:  (200, 200, 200),
    RISING:  (0,   220, 100),
    TOP:     (0,   200, 255),
    FALLING: (80,  80,  255),
}


class SideTracker:
    """Holds state-machine + smoothing buffer for one wrist."""
    def __init__(self,
                 rise_threshold: float = RISE_THRESHOLD,
                 drop_threshold: float = DROP_THRESHOLD,
                 min_top_frames: int = MIN_TOP_FRAMES):
        self.state          = BOTTOM
        self.rep_count      = 0
        self.smooth_buf     = deque(maxlen=SMOOTH_WINDOW)
        self.norm_y         = 0.5
        self.top_frames     = 0     # frames spent continuously in TOP with locked elbow
        self.lockout        = 0     # frames remaining before next rep cycle can start
        self.needs_init     = True  # wait for wrist to reach bottom before counting
        self.rise_threshold = rise_threshold
        self.drop_threshold = drop_threshold
        self.min_top_frames = min_top_frames
        self.elbow_locked   = False  # current lock status (for HUD)
        self.elbow_angle    = 180.0  # current angle in degrees (for HUD)


class Calibrator:
    """Collects norm_y samples over a fixed window and derives per-session thresholds."""
    def __init__(self, n_frames: int = CALIB_FRAMES):
        self._n_frames = n_frames
        self._samples  = []
        self.done      = False

    def update(self, norm_y: float) -> None:
        """Append one norm_y sample; mark done when n_frames reached."""
        if self.done:
            return
        self._samples.append(norm_y)
        if len(self._samples) >= self._n_frames:
            self.done = True

    @property
    def n_frames(self) -> int:
        """Total frames to collect before calibration is complete."""
        return self._n_frames

    @property
    def samples(self):
        """Read-only view of collected norm_y samples."""
        return self._samples

    @property
    def progress(self) -> float:
        """Fraction of calibration frames collected (0.0–1.0)."""
        return len(self._samples) / self._n_frames

    @property
    def frames_collected(self) -> int:
        """Number of norm_y samples collected so far."""
        return len(self._samples)

    def derive_thresholds(self):
        """Derive rise/drop thresholds from collected samples."""
        return derive_thresholds_from_samples(self._samples)


def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Kettlebell snatch rep counter")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--mode", choices=["single", "double", "switch"], default="single",
                        help="'single' tracks right wrist only (default); "
                             "'double' tracks both wrists independently; "
                             "'switch' tracks both wrists, combined count (for hand-to-hand sets)")
    parser.add_argument("--min-top-time", type=float, default=0.5, metavar="SECS",
        help="Seconds wrist must hold overhead to confirm top position (default: 0.5)")
    parser.add_argument("--rise-threshold", type=float, default=None, metavar="FLOAT",
        help=f"norm_y below which wrist is 'high' (default: {RISE_THRESHOLD}). "
             "For jerk/press where wrist doesn't swing low, try 0.6")
    parser.add_argument("--drop-threshold", type=float, default=None, metavar="FLOAT",
        help=f"norm_y above which wrist is 'low' (default: {DROP_THRESHOLD}). "
             "For jerk/press where wrist stays at rack, try 0.5")
    parser.add_argument("--elbow-lock", action="store_true", default=False,
        help="Require elbow lockout at the top of each rep (default: off)")
    parser.add_argument("--mirror", action="store_true", default=False,
        help="Flip video horizontally before processing (use for selfie/front-facing camera)")
    parser.add_argument("--log", metavar="FILE",
        help="Write per-frame CSV log to FILE for debugging (e.g. --log debug.csv)")
    calib_group = parser.add_mutually_exclusive_group()
    calib_group.add_argument("--calibrate", action="store_true",
        help="Dedicated calibration pass: swing KB, save thresholds to calibration.json")
    calib_group.add_argument("--no-auto-calib", action="store_true",
        help="Skip auto-calibration window; load calibration.json if present")
    return parser.parse_args()


def angle_at_joint(ax, ay, bx, by, cx, cy) -> float:
    """Angle in degrees at B in triangle A-B-C (A=shoulder, B=elbow, C=wrist)."""
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax*bcx + bay*bcy
    mag = (bax**2+bay**2)**0.5 * (bcx**2+bcy**2)**0.5
    if mag < 1e-6:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))


def smooth(window: deque, value: float) -> float:
    """Append value to the rolling window and return the current average."""
    window.append(value)
    return sum(window) / len(window)


def get_landmark_y(landmark_list, landmark_idx, frame_h: int) -> float:
    """Return pixel-space Y for a MediaPipe landmark index."""
    return landmark_list[landmark_idx].y * frame_h


def normalise_wrist(wrist_y: float, shoulder_y: float, hip_y: float) -> float:
    """
    Map wrist Y into a scale where:
      0  = at shoulder level
      1  = at hip level
      >1 = below hip (swing bottom)

    Lower pixel-Y means higher in the frame, so we negate the direction.
    """
    torso_height_px = hip_y - shoulder_y
    if torso_height_px < 1e-6:          # guard against degenerate poses
        return 0.5
    return (wrist_y - shoulder_y) / torso_height_px


def derive_thresholds_from_samples(samples,
                                   calib_min_range: float = CALIB_MIN_RANGE,
                                   calib_rise_frac: float = CALIB_RISE_FRAC,
                                   calib_drop_frac: float = CALIB_DROP_FRAC):
    """
    Derive rise/drop thresholds from observed norm_y samples.

    Returns (rise_threshold, drop_threshold).
    Falls back to module defaults if the observed range is too narrow.
    Raises ValueError on empty samples.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    obs_min   = min(samples)
    obs_max   = max(samples)
    obs_range = obs_max - obs_min
    if obs_range < calib_min_range:
        return RISE_THRESHOLD, DROP_THRESHOLD
    rise = obs_min + obs_range * calib_rise_frac
    drop = obs_min + obs_range * calib_drop_frac
    if drop <= rise:
        return RISE_THRESHOLD, DROP_THRESHOLD
    if rise > 0.1:
        print(f"Warning: calibrated rise={rise:.3f} > 0.1; "
              "wrist may not have cleared shoulder during calibration")
    return rise, drop


def transition(tracker: SideTracker, norm_y: float, count_top_frame: bool = True) -> None:
    """
    Drive the state machine one step, mutating tracker in-place.

    norm_y convention (increases downward in pixel space):
      small / negative  → wrist is HIGH (above shoulder)
      large (> 1)       → wrist is LOW  (below hip / deep swing)

    count_top_frame: when False the current frame at the top is not counted toward
    min_top_frames (used by update_tracker to gate on elbow lock).

    Two guards against double-counting (common in half-snatch):
      - min_top_frames: wrist must stay overhead for several frames before TOP is
        confirmed, preventing a swing-through from counting.
      - lockout: after each rep is counted, a cooldown blocks BOTTOM→RISING so
        the backswing can't immediately start a new cycle.
    """
    if tracker.lockout > 0:
        tracker.lockout -= 1

    if tracker.state == BOTTOM:
        if tracker.needs_init:
            if norm_y > tracker.drop_threshold:
                tracker.needs_init = False   # anchored at bottom, ready to count
        elif tracker.lockout == 0 and norm_y < tracker.rise_threshold:
            tracker.state = RISING

    elif tracker.state == RISING:
        if norm_y < tracker.rise_threshold:
            tracker.state = TOP
            tracker.top_frames = 0
        elif norm_y > tracker.drop_threshold:
            # wrist dropped back to hip without reaching overhead — reset cleanly
            tracker.state = BOTTOM

    elif tracker.state == TOP:
        if norm_y < tracker.rise_threshold:
            if count_top_frame:
                tracker.top_frames += 1
        else:
            # wrist has started descending
            if tracker.top_frames >= tracker.min_top_frames:
                tracker.state = FALLING
            else:
                # didn't stay overhead long enough — treat as a swing-through
                tracker.state = BOTTOM
            tracker.top_frames = 0

    elif tracker.state == FALLING:
        if norm_y > tracker.drop_threshold:
            tracker.state = BOTTOM
            tracker.rep_count += 1
            tracker.lockout = MIN_REP_LOCKOUT


def update_display_only(tracker: SideTracker, landmark_list,
                        wrist_idx, shoulder_idx, hip_idx, elbow_idx,
                        frame_w: int, frame_h: int) -> None:
    """Update norm_y and elbow display fields without running the state machine.
    Used for the inactive side in switch mode so the HUD stays current."""
    wrist_y    = get_landmark_y(landmark_list, wrist_idx,    frame_h)
    shoulder_y = get_landmark_y(landmark_list, shoulder_idx, frame_h)
    hip_y      = get_landmark_y(landmark_list, hip_idx,      frame_h)
    tracker.norm_y = smooth(tracker.smooth_buf,
                            normalise_wrist(wrist_y, shoulder_y, hip_y))
    sx = landmark_list[shoulder_idx].x * frame_w
    sy = landmark_list[shoulder_idx].y * frame_h
    ex = landmark_list[elbow_idx].x    * frame_w
    ey = landmark_list[elbow_idx].y    * frame_h
    wx = landmark_list[wrist_idx].x    * frame_w
    wy = landmark_list[wrist_idx].y    * frame_h
    tracker.elbow_angle  = angle_at_joint(sx, sy, ex, ey, wx, wy)
    tracker.elbow_locked = tracker.elbow_angle >= ELBOW_LOCK_ANGLE


def update_tracker(tracker: SideTracker, landmark_list,
                   wrist_idx, shoulder_idx, hip_idx, elbow_idx,
                   frame_w: int, frame_h: int,
                   require_elbow_lock: bool = False):
    """Run one frame of smoothing + state-machine update for a single side.

    require_elbow_lock: when True the elbow must be locked out (≥ ELBOW_LOCK_ANGLE)
    for top_frames to accumulate.  When False (default) the elbow check is
    bypassed for counting; the measured angle is still shown in the HUD.
    """
    wrist_y    = get_landmark_y(landmark_list, wrist_idx,    frame_h)
    shoulder_y = get_landmark_y(landmark_list, shoulder_idx, frame_h)
    hip_y      = get_landmark_y(landmark_list, hip_idx,      frame_h)

    raw_norm_y      = normalise_wrist(wrist_y, shoulder_y, hip_y)
    tracker.norm_y  = smooth(tracker.smooth_buf, raw_norm_y)

    # Compute elbow angle (shoulder → elbow → wrist) in pixel space
    sx = landmark_list[shoulder_idx].x * frame_w
    sy = landmark_list[shoulder_idx].y * frame_h
    ex = landmark_list[elbow_idx].x    * frame_w
    ey = landmark_list[elbow_idx].y    * frame_h
    wx = landmark_list[wrist_idx].x    * frame_w
    wy = landmark_list[wrist_idx].y    * frame_h
    tracker.elbow_angle  = angle_at_joint(sx, sy, ex, ey, wx, wy)
    tracker.elbow_locked = tracker.elbow_angle >= ELBOW_LOCK_ANGLE

    # count_top_frame: True always (elbow lock off) or only when elbow is locked
    count_top_frame = (not require_elbow_lock) or tracker.elbow_locked
    transition(tracker, tracker.norm_y, count_top_frame)


def save_calibration(rise: float, drop: float,
                     path: pathlib.Path = CALIB_FILE) -> None:
    """Persist rise/drop thresholds to a JSON file at path."""
    data = {"rise_threshold": rise, "drop_threshold": drop}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: pathlib.Path = CALIB_FILE):
    """Return (rise, drop) tuple, or None on any error (missing/malformed file)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rise = data["rise_threshold"]
        drop = data["drop_threshold"]
        return (rise, drop)
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None


def draw_wrist_dot(frame, landmark_list, wrist_idx, frame_w: int, frame_h: int,
                   color):
    """Draw a filled circle on frame at the wrist landmark position."""
    wrist   = landmark_list[wrist_idx]
    pixel_x = int(wrist.x * frame_w)
    pixel_y = int(wrist.y * frame_h)
    cv2.circle(frame, (pixel_x, pixel_y), 10, color, -1)


def draw_skeleton(frame, pose_landmarks, mp_pose):
    """Render the MediaPipe pose skeleton and landmark dots onto frame."""
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    mp_draw.draw_landmarks(
        frame,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_style.get_default_pose_landmarks_style(),
    )


def draw_calibration_hud(frame, calibrator: Calibrator,
                         dedicated_mode: bool = False) -> None:
    """Overlay calibration progress info on the frame."""
    h, w = frame.shape[:2]
    title = "CALIBRATION PASS" if dedicated_mode else "Calibrating..."
    cv2.putText(frame, title, (w // 2 - 180, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Frames: {calibrator.frames_collected}/{calibrator.n_frames}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    if calibrator.samples:
        obs_min = min(calibrator.samples)
        obs_max = max(calibrator.samples)
        cv2.putText(frame, f"obs_min={obs_min:.2f}  obs_max={obs_max:.2f}",
                    (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    # Progress bar at bottom of frame
    bar_h    = 20
    bar_y    = h - bar_h - 10
    bar_fill = int(w * calibrator.progress)
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (0, bar_y), (bar_fill, bar_y + bar_h), (0, 220, 255), -1)


def _elbow_status_text(tracker: SideTracker) -> tuple:
    if tracker.elbow_locked:
        return f"Elbow LOCKED ({tracker.elbow_angle:.0f}\xb0)", (0, 220, 100)
    return f"Elbow bent ({tracker.elbow_angle:.0f}\xb0)", (80, 80, 255)


def draw_overlay_single(frame, pose_landmarks, frame_w: int, frame_h: int,
                        tracker: SideTracker, mp_pose):
    """HUD for single-KB mode (right wrist only)."""
    landmark_list = pose_landmarks.landmark
    draw_skeleton(frame, pose_landmarks, mp_pose)
    draw_wrist_dot(frame, landmark_list, mp_pose.PoseLandmark.RIGHT_WRIST,
                   frame_w, frame_h, (0, 255, 255))

    state_color = STATE_COLORS.get(tracker.state, (255, 255, 255))
    cv2.putText(frame, f"Reps: {tracker.rep_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"State: {tracker.state}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"norm_y: {tracker.norm_y:.2f}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
    elbow_text, elbow_color = _elbow_status_text(tracker)
    cv2.putText(frame, elbow_text, (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, elbow_color, 2, cv2.LINE_AA)
    if tracker.state == TOP:
        cv2.putText(frame, f"Top hold: {tracker.top_frames}/{tracker.min_top_frames}",
                    (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
    if tracker.rise_threshold != RISE_THRESHOLD:
        cv2.putText(frame,
                    f"calib rise={tracker.rise_threshold:.2f} drop={tracker.drop_threshold:.2f}",
                    (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 255), 1, cv2.LINE_AA)


def draw_overlay_double(frame, pose_landmarks, frame_w: int, frame_h: int,
                        right_tracker: SideTracker, left_tracker: SideTracker,
                        mp_pose):
    """HUD for double-KB mode (both wrists)."""
    landmark_list = pose_landmarks.landmark
    draw_skeleton(frame, pose_landmarks, mp_pose)

    # Right wrist = yellow, left wrist = cyan
    draw_wrist_dot(frame, landmark_list, mp_pose.PoseLandmark.RIGHT_WRIST,
                   frame_w, frame_h, (0, 255, 255))
    draw_wrist_dot(frame, landmark_list, mp_pose.PoseLandmark.LEFT_WRIST,
                   frame_w, frame_h, (255, 255, 0))

    combined_rep_count = right_tracker.rep_count + left_tracker.rep_count

    right_state_color = STATE_COLORS.get(right_tracker.state, (255, 255, 255))
    cv2.putText(frame, f"R Reps: {right_tracker.rep_count}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"R State: {right_tracker.state}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, right_state_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"R norm_y: {right_tracker.norm_y:.2f}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    r_elbow_text, r_elbow_color = _elbow_status_text(right_tracker)
    cv2.putText(frame, f"R {r_elbow_text}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, r_elbow_color, 1, cv2.LINE_AA)

    left_state_color = STATE_COLORS.get(left_tracker.state, (255, 255, 255))
    cv2.putText(frame, f"L Reps: {left_tracker.rep_count}", (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"L State: {left_tracker.state}", (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, left_state_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"L norm_y: {left_tracker.norm_y:.2f}", (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    l_elbow_text, l_elbow_color = _elbow_status_text(left_tracker)
    cv2.putText(frame, f"L {l_elbow_text}", (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, l_elbow_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Total: {combined_rep_count}", (20, 305),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    if right_tracker.rise_threshold != RISE_THRESHOLD:
        calib_text = (f"calib rise={right_tracker.rise_threshold:.2f}"
                      f" drop={right_tracker.drop_threshold:.2f}")
        cv2.putText(frame, calib_text,
                    (20, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 255), 1, cv2.LINE_AA)


def draw_overlay_switch(frame, pose_landmarks, frame_w: int, frame_h: int,
                        right_tracker: SideTracker, left_tracker: SideTracker,
                        mp_pose, switch_side: str = 'right', switch_lockout: int = 0):
    """HUD for switch (hand-to-hand) mode: combined rep count, active-hand indicator."""
    landmark_list = pose_landmarks.landmark
    draw_skeleton(frame, pose_landmarks, mp_pose)

    # Active wrist: large bright dot. Inactive: small grey dot.
    PL = mp_pose.PoseLandmark
    active_wrist_idx   = PL.RIGHT_WRIST if switch_side == 'right' else PL.LEFT_WRIST
    inactive_wrist_idx = PL.LEFT_WRIST  if switch_side == 'right' else PL.RIGHT_WRIST

    inactive_wrist = landmark_list[inactive_wrist_idx]
    cv2.circle(frame,
               (int(inactive_wrist.x * frame_w), int(inactive_wrist.y * frame_h)),
               6, (100, 100, 100), -1)
    active_wrist = landmark_list[active_wrist_idx]
    cv2.circle(frame,
               (int(active_wrist.x * frame_w), int(active_wrist.y * frame_h)),
               14, (0, 220, 255), -1)

    active_label   = switch_side[0].upper()  # 'R' or 'L'
    active_tracker = right_tracker if switch_side == 'right' else left_tracker

    combined = right_tracker.rep_count + left_tracker.rep_count
    cv2.putText(frame, f"Reps: {combined}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

    # Active label + init status
    init_tag = " INIT" if active_tracker.needs_init else ""
    lock_tag = f" lock:{switch_lockout}" if switch_lockout > 0 else ""
    cv2.putText(frame, f"Active: {active_label}{init_tag}{lock_tag}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

    r_state_color = STATE_COLORS.get(right_tracker.state, (255, 255, 255))
    l_state_color = STATE_COLORS.get(left_tracker.state,  (255, 255, 255))
    r_init = " INIT" if right_tracker.needs_init else ""
    l_init = " INIT" if left_tracker.needs_init  else ""
    cv2.putText(frame, f"R{r_init}: {right_tracker.state} ({right_tracker.rep_count})", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, r_state_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"L{l_init}: {left_tracker.state} ({left_tracker.rep_count})", (20, 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, l_state_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"R ny:{right_tracker.norm_y:.2f}  L ny:{left_tracker.norm_y:.2f}"
                       f"  top:{active_tracker.top_frames}/{active_tracker.min_top_frames}",
                (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    elbow_text, elbow_color = _elbow_status_text(active_tracker)
    cv2.putText(frame, elbow_text, (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, elbow_color, 2, cv2.LINE_AA)


def _save_partial_calibration(calibrator: Calibrator, label: str) -> None:
    """Save thresholds from whatever samples were collected; skip if too few."""
    samples = calibrator.samples
    if len(samples) >= 10:
        rise, drop = derive_thresholds_from_samples(samples)
        save_calibration(rise, drop)
        print(f"Calibration saved ({label}, {len(samples)} frames): "
              f"rise={rise:.3f}, drop={drop:.3f}")


def main():
    args = parse_args()
    double_mode        = args.mode == "double"
    switch_mode        = args.mode == "switch"
    dedicated_calib    = args.calibrate
    require_elbow_lock = args.elbow_lock

    # ── Optional per-frame CSV log ─────────────────────────────────────────────
    _log_file = None
    if args.log:
        import csv as _csv
        _log_file = open(args.log, "w", newline="", encoding="utf-8")
        _log_writer = _csv.writer(_log_file)
        _log_writer.writerow([
            "frame", "pose_detected",
            "active_side", "active_state", "active_norm_y",
            "active_needs_init", "active_lockout", "active_top_frames", "active_rep_count",
            "inactive_norm_y",
            "switch_lockout",
        ])

    video_capture = cv2.VideoCapture(args.video)
    if not video_capture.isOpened():
        sys.exit(f"Error: cannot open '{args.video}'")

    frame_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = video_capture.get(cv2.CAP_PROP_FPS) or 30.0

    # ── Auto-correct video rotation from metadata ──────────────────────────────
    # Mobile cameras store frames in landscape with a rotation tag; OpenCV
    # ignores that tag and returns raw frames, so we rotate manually.
    _ORIENT_TO_ROTATE = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    try:
        _raw_orient = int(video_capture.get(cv2.CAP_PROP_ORIENTATION_META))
    except AttributeError:
        _raw_orient = 0
    _rotation_fix = _ORIENT_TO_ROTATE.get(_raw_orient)
    if _raw_orient in (90, 270):
        frame_w, frame_h = frame_h, frame_w  # dimensions swap after 90°/270° rotation
    min_top_frames = max(1, round(fps * args.min_top_time))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    PoseLandmark = mp_pose.PoseLandmark

    right_tracker = SideTracker(min_top_frames=min_top_frames)
    left_tracker  = (SideTracker(min_top_frames=min_top_frames)
                     if (double_mode or switch_mode) else None)

    # ── Determine calibration mode ─────────────────────────────────────────────
    # Priority: --calibrate > --no-auto-calib / existing file > auto-calibrate
    existing_calib = load_calibration()

    if dedicated_calib:
        calibrator = Calibrator()
    elif args.no_auto_calib or existing_calib is not None:
        calibrator = None
        if existing_calib is not None:
            rise, drop = existing_calib
            right_tracker.rise_threshold = rise
            right_tracker.drop_threshold = drop
            if (double_mode or switch_mode) and left_tracker:
                left_tracker.rise_threshold = rise
                left_tracker.drop_threshold = drop
            print(f"Loaded calibration: rise={rise:.3f}, drop={drop:.3f}")
    else:
        calibrator = None   # no calibration.json and no --calibrate flag: use defaults

    # ── CLI threshold overrides (applied after calibration so they take priority) ─
    for tracker in filter(None, [right_tracker, left_tracker]):
        if args.rise_threshold is not None:
            tracker.rise_threshold = args.rise_threshold
        if args.drop_threshold is not None:
            tracker.drop_threshold = args.drop_threshold
    if args.rise_threshold is not None or args.drop_threshold is not None:
        print(f"Threshold overrides: rise={right_tracker.rise_threshold:.3f}  "
              f"drop={right_tracker.drop_threshold:.3f}")

    calib_done_and_saved = False   # True only in dedicated mode after save
    # switch mode: track which wrist currently holds the KB
    switch_side    = 'right'   # 'right' | 'left'
    switch_lockout = 0         # frames before another switch is allowed
    frame_num      = 0

    while True:
        frame_read_ok, frame = video_capture.read()
        frame_num += 1
        if not frame_read_ok:
            # Video ended mid-calibration in dedicated mode → save partial data
            if dedicated_calib and calibrator is not None and not calibrator.done:
                _save_partial_calibration(calibrator, "partial")
            break

        if _rotation_fix is not None:
            frame = cv2.rotate(frame, _rotation_fix)
        if args.mirror:
            frame = cv2.flip(frame, 1)

        rgb_frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb_frame)

        if pose_result.pose_landmarks:
            landmark_list = pose_result.pose_landmarks.landmark

            if calibrator is not None and not calibrator.done:
                # ── Calibration phase: collect samples, do not count reps ──────
                wrist_y    = get_landmark_y(landmark_list, PoseLandmark.RIGHT_WRIST,    frame_h)
                shoulder_y = get_landmark_y(landmark_list, PoseLandmark.RIGHT_SHOULDER, frame_h)
                hip_y      = get_landmark_y(landmark_list, PoseLandmark.RIGHT_HIP,      frame_h)
                raw_norm_y = normalise_wrist(wrist_y, shoulder_y, hip_y)
                smoothed   = smooth(right_tracker.smooth_buf, raw_norm_y)
                calibrator.update(smoothed)

                if calibrator.done:
                    rise, drop = calibrator.derive_thresholds()
                    right_tracker.rise_threshold = rise
                    right_tracker.drop_threshold = drop
                    if (double_mode or switch_mode) and left_tracker:
                        left_tracker.rise_threshold = rise
                        left_tracker.drop_threshold = drop
                    if dedicated_calib:
                        save_calibration(rise, drop)
                        print(f"Calibration saved: rise={rise:.3f}, drop={drop:.3f}")
                        calib_done_and_saved = True
                    else:
                        calibrator = None   # switch to normal counting next frame

            elif calibrator is None:
                # ── Normal counting phase ─────────────────────────────────────
                if switch_mode:
                    # ── Switch (hand-to-hand) mode ────────────────────────────
                    # Only the ACTIVE side drives the state machine.
                    # The INACTIVE side gets display-only updates (norm_y, elbow)
                    # so the HUD stays current and switch detection is accurate.
                    PL = PoseLandmark
                    R_lm = (PL.RIGHT_WRIST, PL.RIGHT_SHOULDER,
                            PL.RIGHT_HIP,   PL.RIGHT_ELBOW)
                    L_lm = (PL.LEFT_WRIST,  PL.LEFT_SHOULDER,
                            PL.LEFT_HIP,    PL.LEFT_ELBOW)

                    active_t   = right_tracker if switch_side == 'right' else left_tracker
                    inactive_t = left_tracker  if switch_side == 'right' else right_tracker
                    active_lm  = R_lm          if switch_side == 'right' else L_lm
                    inactive_lm = L_lm         if switch_side == 'right' else R_lm

                    # Update inactive wrist position for display + switch detection
                    update_display_only(inactive_t, landmark_list,
                                        *inactive_lm, frame_w, frame_h)

                    # ── Switch detection ─────────────────────────────────────
                    # Fires when:
                    #   - active tracker is between reps (BOTTOM)
                    #   - active wrist is physically LOW (> drop_threshold) — confirms
                    #     the KB has been released (prevents firing while wrist is still
                    #     overhead during needs_init phase → no ping-pong)
                    #   - inactive wrist is overhead (< rise_threshold) — has the KB
                    # Also handles initial-hand auto-detect: if KB starts in the wrong
                    # hand the same conditions apply (active at hip, inactive overhead).
                    if switch_lockout > 0:
                        switch_lockout -= 1
                    elif (active_t.state == BOTTOM
                            and active_t.norm_y > active_t.drop_threshold
                            and inactive_t.norm_y < active_t.rise_threshold):
                        # Incoming wrist is confirmed overhead; recognise it as the top
                        # of the first rep so the descent to rack counts immediately.
                        inactive_t.needs_init = False
                        inactive_t.state      = TOP
                        inactive_t.top_frames = inactive_t.min_top_frames  # pre-filled
                        inactive_t.lockout    = 0
                        inactive_t.smooth_buf.clear()
                        active_t.state        = BOTTOM
                        active_t.top_frames   = 0
                        switch_side    = 'left' if switch_side == 'right' else 'right'
                        switch_lockout = 30   # ~1 s at 30 fps; lets old wrist descend
                        active_t, active_lm = inactive_t, inactive_lm

                    # Run full state machine for the active side only
                    update_tracker(active_t, landmark_list,
                                   *active_lm, frame_w, frame_h,
                                   require_elbow_lock=require_elbow_lock)

                    draw_overlay_switch(frame, pose_result.pose_landmarks,
                                        frame_w, frame_h,
                                        right_tracker, left_tracker, mp_pose,
                                        switch_side=switch_side,
                                        switch_lockout=switch_lockout)
                else:
                    update_tracker(right_tracker, landmark_list,
                                   PoseLandmark.RIGHT_WRIST,
                                   PoseLandmark.RIGHT_SHOULDER,
                                   PoseLandmark.RIGHT_HIP,
                                   PoseLandmark.RIGHT_ELBOW,
                                   frame_w, frame_h,
                                   require_elbow_lock=require_elbow_lock)
                    if double_mode:
                        update_tracker(left_tracker, landmark_list,
                                       PoseLandmark.LEFT_WRIST,
                                       PoseLandmark.LEFT_SHOULDER,
                                       PoseLandmark.LEFT_HIP,
                                       PoseLandmark.LEFT_ELBOW,
                                       frame_w, frame_h,
                                       require_elbow_lock=require_elbow_lock)
                        draw_overlay_double(frame, pose_result.pose_landmarks,
                                            frame_w, frame_h,
                                            right_tracker, left_tracker, mp_pose)
                    else:
                        draw_overlay_single(frame, pose_result.pose_landmarks,
                                            frame_w, frame_h, right_tracker, mp_pose)

        # Draw calibration HUD (even when pose not detected, so progress is visible)
        if calibrator is not None and not calib_done_and_saved:
            draw_calibration_hud(frame, calibrator, dedicated_calib)

        # ── Per-frame CSV log ──────────────────────────────────────────────────
        if _log_file is not None and switch_mode:
            pose_ok = pose_result.pose_landmarks is not None
            active_t   = right_tracker if switch_side == 'right' else left_tracker
            inactive_t = left_tracker  if switch_side == 'right' else right_tracker
            _log_writer.writerow([
                frame_num, int(pose_ok),
                switch_side, active_t.state, f"{active_t.norm_y:.3f}",
                int(active_t.needs_init), active_t.lockout,
                active_t.top_frames, active_t.rep_count,
                f"{inactive_t.norm_y:.3f}",
                switch_lockout,
            ])

        cv2.imshow("Kettlebell Snatch Counter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if dedicated_calib and calibrator is not None and not calibrator.done:
                _save_partial_calibration(calibrator, "early exit")
            break
        if calib_done_and_saved:
            break

    video_capture.release()
    pose.close()
    cv2.destroyAllWindows()
    if _log_file is not None:
        _log_file.close()
        print(f"Log written to {args.log}")

    if dedicated_calib:
        return   # no rep count to print

    if double_mode:
        combined_rep_count = right_tracker.rep_count + left_tracker.rep_count
        print(f"\nFinal rep count — R: {right_tracker.rep_count}  "
              f"L: {left_tracker.rep_count}  Total: {combined_rep_count}")
    elif switch_mode:
        combined_rep_count = right_tracker.rep_count + left_tracker.rep_count
        print(f"\nFinal rep count (hand-to-hand) — R: {right_tracker.rep_count}  "
              f"L: {left_tracker.rep_count}  Total: {combined_rep_count}")
    else:
        print(f"\nFinal rep count: {right_tracker.rep_count}")


if __name__ == "__main__":
    main()
