"""
Kettlebell Rep Counter
Usage:
  python kb_counter.py <video_file>                   # single KB (right hand)
  python kb_counter.py <video_file> --mode double     # double KB (both hands)
  python kb_counter.py <video_file> --mode switch     # hand-to-hand KB set
  python kb_counter.py <video_file> --elbow-lock      # require elbow lockout at top
  python kb_counter.py <video_file> --knee-lock       # require knee lockout at bottom before rep
  python kb_counter.py <video_file> --ankle-grounded  # warn if heel rises during rep
  python kb_counter.py <video_file> --movement "Single Arm Jerk"  # label on screen
  python kb_counter.py <video_file> --calibrate       # dedicated calibration pass
  python kb_counter.py <video_file> --no-auto-calib   # skip auto-calibration
"""

import json
import math
import pathlib
import sys
import argparse
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


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
KNEE_LOCK_ANGLE  = 160  # degrees; knee  ≥ this → leg is locked out
ANKLE_RISE_PX_FRAC = 0.03  # fraction of frame height; heel rise greater than this is flagged

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

# ── HUD style constants ───────────────────────────────────────────────────────
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_AA         = cv2.LINE_AA
_PANEL_W    = 268
_PANEL_X    = 10
_PANEL_BG   = 0.68   # opacity of the dark panel background (0 = transparent, 1 = black)
_C_WHITE    = (255, 255, 255)
_C_GREY     = (160, 160, 160)
_C_DIM      = (80,  80,  80)
_C_GOLD     = (40,  210, 255)   # BGR: warm gold for movement tag
_C_OK       = (50,  200, 80)    # green status dot
_C_WARN     = (40,  60,  230)   # red status dot


def _panel_bg(frame, x: int, y: int, w: int, h: int) -> None:
    """Blend a near-black rectangle over a frame region for panel background."""
    roi = frame[y:y + h, x:x + w]
    frame[y:y + h, x:x + w] = (roi * (1 - _PANEL_BG)).astype(np.uint8)


def _divider(frame, x: int, y: int, w: int) -> None:
    cv2.line(frame, (x + 6, y), (x + w - 6, y), _C_DIM, 1, _AA)


def _dot_row(frame, x: int, y: int, label: str, ok: bool, scale: float = 0.6) -> None:
    """Colored circle + label on a single row."""
    color = _C_OK if ok else _C_WARN
    cv2.circle(frame, (x + 7, y - 4), 5, color, -1)
    cv2.putText(frame, label, (x + 20, y), _FONT, scale, color, 1, _AA)


def _movement_tag(frame, text: str, frame_w: int) -> None:
    """Floating label in the top-right corner with a properly fitted dark chip."""
    if not text:
        return
    (tw, th), baseline = cv2.getTextSize(text, _FONT, 0.72, 2)
    # Full text height = ascent (th) + descent (baseline) + 1px rounding
    full_h = th + baseline + 1
    pad_x, pad_y = 10, 8
    chip_w = tw + pad_x * 2
    chip_h = full_h + pad_y * 2
    chip_x = frame_w - chip_w - 8
    chip_y = 8
    # Clamp to frame bounds before slicing
    x1 = max(0, chip_x)
    y1 = max(0, chip_y)
    x2 = min(frame_w, chip_x + chip_w)
    y2 = min(frame.shape[0], chip_y + chip_h)
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        frame[y1:y2, x1:x2] = (roi * 0.22).astype(np.uint8)
    # Text baseline sits pad_y + th pixels below the chip top
    text_x = chip_x + pad_x
    text_y = chip_y + pad_y + th
    cv2.putText(frame, text, (text_x, text_y), _FONT, 0.72, _C_GOLD, 2, _AA)


def _top_hold_bar(frame, x: int, y: int, w: int,
                  top_frames: int, min_top_frames: int) -> None:
    """Thin progress bar showing overhead fixation progress."""
    fill = int(w * min(top_frames / max(min_top_frames, 1), 1.0))
    cv2.rectangle(frame, (x, y),     (x + w,    y + 6), _C_DIM, -1)
    cv2.rectangle(frame, (x, y),     (x + fill, y + 6), (0, 200, 255), -1)
    pct_text = f"{top_frames}/{min_top_frames}"
    cv2.putText(frame, pct_text, (x + w + 8, y + 6), _FONT, 0.48, _C_GREY, 1, _AA)


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
        self.knee_locked    = False  # current knee lock status (for HUD)
        self.knee_angle     = 180.0  # current knee angle in degrees (for HUD)
        self.ankle_baseline_y = None  # pixel Y of ankle recorded at BOTTOM (heel-rise reference)
        self.ankle_raised   = False   # True if heel rose above baseline during current rep
        self._counted_this_cycle = False  # True once rep counted; drives lockout at BOTTOM

    def transition(self, norm_y: float, count_top_frame: bool = True,
                   allow_rise: bool = True) -> None:
        """
        Drive the state machine one step, mutating self in-place.

        norm_y convention (increases downward in pixel space):
          small / negative  → wrist is HIGH (above shoulder)
          large (> 1)       → wrist is LOW  (below hip / deep swing)

        count_top_frame: when False the current frame at the top is not counted toward
        min_top_frames (used by update to gate on elbow lock).

        allow_rise: when False the BOTTOM→RISING transition is blocked regardless of
        norm_y (used by update to gate on knee lock — knees must be locked before
        starting the next rep).

        Two guards against double-counting:
          - min_top_frames: wrist must stay overhead for this many frames (≈ min_top_time
            seconds) before the rep is counted — prevents swing-throughs.
          - lockout: after each rep a cooldown is applied once the wrist returns to
            the bottom, blocking BOTTOM→RISING so the backswing cannot start a new cycle.
        """
        if self.lockout > 0:
            self.lockout -= 1

        if self.state == BOTTOM:
            if self.needs_init:
                if norm_y > self.drop_threshold:
                    self.needs_init = False   # anchored at bottom, ready to count
            elif self.lockout == 0 and norm_y < self.rise_threshold and allow_rise:
                self.state = RISING

        elif self.state == RISING:
            if norm_y < self.rise_threshold:
                self.state = TOP
                self.top_frames = 0
            elif norm_y > self.drop_threshold:
                # wrist dropped back to hip without reaching overhead — reset cleanly
                self.state = BOTTOM

        elif self.state == TOP:
            if norm_y < self.rise_threshold:
                if count_top_frame:
                    self.top_frames += 1
                    if (self.top_frames >= self.min_top_frames
                            and not self._counted_this_cycle):
                        self.rep_count += 1
                        self._counted_this_cycle = True
            else:
                # wrist has started descending
                if self.top_frames >= self.min_top_frames:
                    self.state = FALLING
                else:
                    # didn't stay overhead long enough — treat as a swing-through
                    self.state = BOTTOM
                    self._counted_this_cycle = False
                self.top_frames = 0

        elif self.state == FALLING:
            if norm_y > self.drop_threshold:
                self.state = BOTTOM
                # Apply post-rep lockout now that the wrist has returned to bottom;
                # starting it here (rather than at the top) guarantees the full
                # cooldown is available before the next rep can begin.
                if self._counted_this_cycle:
                    self.lockout = MIN_REP_LOCKOUT
                    self._counted_this_cycle = False

    def update(self, landmark_list,
               wrist_idx, shoulder_idx, hip_idx, elbow_idx,
               frame_w: int, frame_h: int,
               require_elbow_lock: bool = False,
               knee_idx=None, ankle_idx=None,
               require_knee_lock: bool = False,
               check_ankle_grounded: bool = False):
        """Run one frame of smoothing + state-machine update for a single side.

        require_elbow_lock: when True the elbow must be locked out (≥ ELBOW_LOCK_ANGLE)
        for top_frames to accumulate.  When False (default) the elbow check is
        bypassed for counting; the measured angle is still shown in the HUD.

        knee_idx / ankle_idx: MediaPipe landmark indices for the same side's knee and
        ankle.  Required for knee-lock and ankle-grounded features; ignored when None.

        require_knee_lock: when True the knee must be locked (≥ KNEE_LOCK_ANGLE) at
        BOTTOM for the BOTTOM→RISING transition to fire (i.e. rep start is gated on
        locked knees).  Angle is computed and shown in HUD even when False.

        check_ankle_grounded: when True, flags self.ankle_raised if the heel rises more
        than ANKLE_RISE_PX_FRAC of frame height above its BOTTOM-state baseline.
        """
        wrist_y    = get_landmark_y(landmark_list, wrist_idx,    frame_h)
        shoulder_y = get_landmark_y(landmark_list, shoulder_idx, frame_h)
        hip_y      = get_landmark_y(landmark_list, hip_idx,      frame_h)

        raw_norm_y   = normalise_wrist(wrist_y, shoulder_y, hip_y)
        self.norm_y  = smooth(self.smooth_buf, raw_norm_y)

        # Compute elbow angle (shoulder → elbow → wrist) in pixel space
        sx = landmark_list[shoulder_idx].x * frame_w
        sy = landmark_list[shoulder_idx].y * frame_h
        ex = landmark_list[elbow_idx].x    * frame_w
        ey = landmark_list[elbow_idx].y    * frame_h
        wx = landmark_list[wrist_idx].x    * frame_w
        wy = landmark_list[wrist_idx].y    * frame_h
        self.elbow_angle  = angle_at_joint(sx, sy, ex, ey, wx, wy)
        self.elbow_locked = self.elbow_angle >= ELBOW_LOCK_ANGLE

        # Compute knee angle (hip → knee → ankle) in pixel space
        if knee_idx is not None and ankle_idx is not None:
            hkx = landmark_list[hip_idx].x   * frame_w
            hky = landmark_list[hip_idx].y   * frame_h
            knx = landmark_list[knee_idx].x  * frame_w
            kny = landmark_list[knee_idx].y  * frame_h
            akx = landmark_list[ankle_idx].x * frame_w
            aky = landmark_list[ankle_idx].y * frame_h
            self.knee_angle  = angle_at_joint(hkx, hky, knx, kny, akx, aky)
            self.knee_locked = self.knee_angle >= KNEE_LOCK_ANGLE

        # count_top_frame: True always (elbow lock off) or only when elbow is locked
        count_top_frame = (not require_elbow_lock) or self.elbow_locked
        # allow_rise: True always (knee lock off) or only when knee is locked
        allow_rise = (not require_knee_lock) or self.knee_locked
        self.transition(self.norm_y, count_top_frame, allow_rise)

        # Ankle grounding check: flag heel rise > threshold during a rep
        if check_ankle_grounded and ankle_idx is not None:
            ankle_y = get_landmark_y(landmark_list, ankle_idx, frame_h)
            if self.state == BOTTOM:
                self.ankle_baseline_y = ankle_y
                self.ankle_raised = False
            elif self.ankle_baseline_y is not None:
                threshold_px = ANKLE_RISE_PX_FRAC * frame_h
                if ankle_y < self.ankle_baseline_y - threshold_px:
                    self.ankle_raised = True

    def update_display_only(self, landmark_list,
                            wrist_idx, shoulder_idx, hip_idx, elbow_idx,
                            frame_w: int, frame_h: int) -> None:
        """Update norm_y and elbow display fields without running the state machine.
        Used for the inactive side in switch mode so the HUD stays current."""
        wrist_y    = get_landmark_y(landmark_list, wrist_idx,    frame_h)
        shoulder_y = get_landmark_y(landmark_list, shoulder_idx, frame_h)
        hip_y      = get_landmark_y(landmark_list, hip_idx,      frame_h)
        self.norm_y = smooth(self.smooth_buf,
                             normalise_wrist(wrist_y, shoulder_y, hip_y))
        sx = landmark_list[shoulder_idx].x * frame_w
        sy = landmark_list[shoulder_idx].y * frame_h
        ex = landmark_list[elbow_idx].x    * frame_w
        ey = landmark_list[elbow_idx].y    * frame_h
        wx = landmark_list[wrist_idx].x    * frame_w
        wy = landmark_list[wrist_idx].y    * frame_h
        self.elbow_angle  = angle_at_joint(sx, sy, ex, ey, wx, wy)
        self.elbow_locked = self.elbow_angle >= ELBOW_LOCK_ANGLE

    def elbow_status_text(self) -> tuple:
        if self.elbow_locked:
            return f"Elbow LOCKED ({self.elbow_angle:.0f}\xb0)", (0, 220, 100)
        return f"Elbow bent ({self.elbow_angle:.0f}\xb0)", (80, 80, 255)

    def knee_status_text(self) -> tuple:
        if self.knee_locked:
            return f"Knees LOCKED ({self.knee_angle:.0f}\xb0)", (0, 220, 100)
        return f"Knees bent ({self.knee_angle:.0f}\xb0)", (80, 80, 255)

    def ankle_status_text(self) -> tuple:
        if self.ankle_raised:
            return "HEEL RAISED!", (0, 60, 255)
        return "Ankle grounded", (0, 220, 100)


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
    parser = argparse.ArgumentParser(description="Kettlebell rep counter")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--mode", choices=["single", "double", "switch"], default="single",
                        help="'single' tracks right wrist only (default); "
                             "'double' tracks both wrists independently; "
                             "'switch' tracks both wrists, combined count (for hand-to-hand sets)")
    parser.add_argument("--min-top-time", type=float, default=0.5, metavar="SECS",
        help="Seconds wrist must hold overhead before the rep is counted (default: 0.5)")
    parser.add_argument("--rise-threshold", type=float, default=None, metavar="FLOAT",
        help=f"norm_y below which wrist is 'high' (default: {RISE_THRESHOLD}). "
             "For jerk/press where wrist doesn't swing low, try 0.6")
    parser.add_argument("--drop-threshold", type=float, default=None, metavar="FLOAT",
        help=f"norm_y above which wrist is 'low' (default: {DROP_THRESHOLD}). "
             "For jerk/press where wrist stays at rack, try 0.5")
    parser.add_argument("--elbow-lock", action="store_true", default=False,
        help="Require elbow lockout at the top of each rep (default: off)")
    parser.add_argument("--knee-lock", action="store_true", default=False,
        help="Require knee lockout at the bottom before each rep starts (default: off)")
    parser.add_argument("--ankle-grounded", action="store_true", default=False,
        help="Warn on screen if heel rises during a rep (for push press; default: off)")
    parser.add_argument("--movement", default="", metavar="TEXT",
        help="Movement name to display on screen, e.g. 'Single Arm Jerk' (default: none)")
    parser.add_argument("--no-skeleton", action="store_true", default=False,
        help="Hide the MediaPipe pose skeleton overlay (default: skeleton shown)")
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
    fh, fw = frame.shape[:2]
    title = "CALIBRATING" if dedicated_mode else "Calibrating..."

    # Centered banner
    (tw, _), _ = cv2.getTextSize(title, _FONT, 1.1, 2)
    tx = (fw - tw) // 2
    _panel_bg(frame, tx - 16, 14, tw + 32, 44)
    cv2.putText(frame, title, (tx, 48), _FONT, 1.1, (0, 220, 255), 2, _AA)

    # Side panel with stats
    ph = 80 if calibrator.samples else 50
    _panel_bg(frame, _PANEL_X, 68, _PANEL_W, ph)
    cv2.putText(frame,
                f"{calibrator.frames_collected} / {calibrator.n_frames}  frames",
                (_PANEL_X + 12, 100), _FONT, 0.72, _C_GREY, 1, _AA)
    if calibrator.samples:
        obs_min = min(calibrator.samples)
        obs_max = max(calibrator.samples)
        cv2.putText(frame, f"ny  min {obs_min:.2f}   max {obs_max:.2f}",
                    (_PANEL_X + 12, 128), _FONT, 0.6, _C_GREY, 1, _AA)

    # Progress bar at bottom
    bar_h  = 14
    bar_y  = fh - bar_h - 8
    fill_w = int(fw * calibrator.progress)
    cv2.rectangle(frame, (0, bar_y), (fw, bar_y + bar_h), _C_DIM, -1)
    cv2.rectangle(frame, (0, bar_y), (fill_w, bar_y + bar_h), (0, 220, 255), -1)


def draw_overlay_single(frame, pose_landmarks, frame_w: int, frame_h: int,
                        tracker: SideTracker, mp_pose,
                        show_knee: bool = False, show_ankle: bool = False,
                        movement_name: str = "", show_skeleton: bool = True):
    """HUD for single-KB mode (right wrist only)."""
    landmark_list = pose_landmarks.landmark
    if show_skeleton:
        draw_skeleton(frame, pose_landmarks, mp_pose)
    draw_wrist_dot(frame, landmark_list, mp_pose.PoseLandmark.RIGHT_WRIST,
                   frame_w, frame_h, (0, 255, 255))

    # ── compute panel height ──────────────────────────────────────────────────
    ph = 190  # REPS label + count + bar slot (always reserved) + state + norm_y + divider + elbow
    if show_knee:
        ph += 26
    if show_ankle:
        ph += 26
    if tracker.rise_threshold != RISE_THRESHOLD:
        ph += 22
    ph = min(ph, frame_h - 20)

    # ── panel background ──────────────────────────────────────────────────────
    _panel_bg(frame, _PANEL_X, 10, _PANEL_W, ph)

    px = _PANEL_X + 12   # inner left edge
    y  = 29              # first text baseline

    # "REPS" label
    cv2.putText(frame, "REPS", (px, y), _FONT, 0.5, _C_GREY, 1, _AA)
    y += 46

    # Large rep count
    cv2.putText(frame, str(tracker.rep_count), (px, y), _FONT, 2.2, _C_WHITE, 3, _AA)
    y += 10

    # Top-hold progress bar slot — always reserved so layout stays fixed
    if tracker.state == TOP:
        _top_hold_bar(frame, px, y,
                      _PANEL_W - 44, tracker.top_frames, tracker.min_top_frames)
    y += 22   # advance regardless of state
    y += 20

    # State indicator dot + label
    sc = STATE_COLORS.get(tracker.state, _C_WHITE)
    cv2.circle(frame, (px + 7, y - 5), 7, sc, -1)
    cv2.putText(frame, tracker.state, (px + 22, y), _FONT, 0.78, sc, 2, _AA)
    y += 22

    # norm_y  (compact)
    cv2.putText(frame, f"ny {tracker.norm_y:+.2f}", (px, y), _FONT, 0.55, _C_GREY, 1, _AA)
    y += 18

    _divider(frame, _PANEL_X, y, _PANEL_W)
    y += 15

    # Elbow status
    _dot_row(frame, px, y,
             f"Elbow  {tracker.elbow_angle:.0f}\xb0", tracker.elbow_locked)
    y += 26

    if show_knee:
        _dot_row(frame, px, y,
                 f"Knees  {tracker.knee_angle:.0f}\xb0", tracker.knee_locked)
        y += 26

    if show_ankle:
        _dot_row(frame, px, y,
                 "Ankle  OK" if not tracker.ankle_raised else "HEEL RAISED!",
                 not tracker.ankle_raised)
        y += 26

    if tracker.rise_threshold != RISE_THRESHOLD:
        cv2.putText(frame,
                    f"rise {tracker.rise_threshold:.2f}  drop {tracker.drop_threshold:.2f}",
                    (px, y), _FONT, 0.48, (100, 220, 255), 1, _AA)

    _movement_tag(frame, movement_name, frame_w)


def draw_overlay_double(frame, pose_landmarks, frame_w: int, frame_h: int,
                        right_tracker: SideTracker, left_tracker: SideTracker,
                        mp_pose, show_knee: bool = False, show_ankle: bool = False,
                        movement_name: str = "", show_skeleton: bool = True):
    """HUD for double-KB mode (both wrists): total + per-side rows."""
    landmark_list = pose_landmarks.landmark
    if show_skeleton:
        draw_skeleton(frame, pose_landmarks, mp_pose)
    draw_wrist_dot(frame, landmark_list, mp_pose.PoseLandmark.RIGHT_WRIST,
                   frame_w, frame_h, (0, 255, 255))
    draw_wrist_dot(frame, landmark_list, mp_pose.PoseLandmark.LEFT_WRIST,
                   frame_w, frame_h, (255, 255, 0))

    # ── helper: draw one side's sub-section ──────────────────────────────────
    def _side_section(tracker, label, wrist_color, y_start):
        y = y_start
        sc = STATE_COLORS.get(tracker.state, _C_WHITE)
        # "R  3  ● TOP"
        cv2.putText(frame, label, (px, y), _FONT, 0.62, wrist_color, 2, _AA)
        cv2.putText(frame, str(tracker.rep_count),
                    (px + 22, y), _FONT, 0.95, wrist_color, 2, _AA)
        cv2.circle(frame, (px + 65, y - 5), 6, sc, -1)
        cv2.putText(frame, tracker.state, (px + 78, y), _FONT, 0.62, sc, 1, _AA)
        y += 22
        cv2.putText(frame, f"ny {tracker.norm_y:+.2f}", (px + 8, y),
                    _FONT, 0.52, _C_GREY, 1, _AA)
        y += 20
        _dot_row(frame, px + 4, y,
                 f"Elbow  {tracker.elbow_angle:.0f}\xb0", tracker.elbow_locked, 0.58)
        y += 22
        if show_knee:
            _dot_row(frame, px + 4, y,
                     f"Knees  {tracker.knee_angle:.0f}\xb0", tracker.knee_locked, 0.58)
            y += 22
        if show_ankle:
            _dot_row(frame, px + 4, y,
                     "Ankle  OK" if not tracker.ankle_raised else "HEEL RAISED!",
                     not tracker.ankle_raised, 0.58)
            y += 22
        return y

    # ── compute panel height ──────────────────────────────────────────────────
    per_side = 64 + (22 if show_knee else 0) + (22 if show_ankle else 0)
    ph = 80 + per_side * 2 + 16   # header + 2 sides + dividers
    if right_tracker.rise_threshold != RISE_THRESHOLD:
        ph += 22
    ph = min(ph, frame_h - 20)

    _panel_bg(frame, _PANEL_X, 10, _PANEL_W, ph)

    px = _PANEL_X + 12
    y  = 29

    # Total count header
    combined = right_tracker.rep_count + left_tracker.rep_count
    cv2.putText(frame, "TOTAL", (px, y), _FONT, 0.5, _C_GREY, 1, _AA)
    y += 40
    cv2.putText(frame, str(combined), (px, y), _FONT, 1.8, _C_WHITE, 3, _AA)
    y += 16

    _divider(frame, _PANEL_X, y, _PANEL_W)
    y += 14

    y = _side_section(right_tracker, "R", (0, 255, 255), y)

    _divider(frame, _PANEL_X, y, _PANEL_W)
    y += 14

    y = _side_section(left_tracker,  "L", (255, 255, 0), y)

    if right_tracker.rise_threshold != RISE_THRESHOLD:
        _divider(frame, _PANEL_X, y, _PANEL_W)
        y += 14
        cv2.putText(frame,
                    f"rise {right_tracker.rise_threshold:.2f}"
                    f"  drop {right_tracker.drop_threshold:.2f}",
                    (px, y), _FONT, 0.48, (100, 220, 255), 1, _AA)

    _movement_tag(frame, movement_name, frame_w)


def draw_overlay_switch(frame, pose_landmarks, frame_w: int, frame_h: int,
                        right_tracker: SideTracker, left_tracker: SideTracker,
                        mp_pose, switch_side: str = 'right', switch_lockout: int = 0,
                        show_knee: bool = False, show_ankle: bool = False,
                        movement_name: str = "", show_skeleton: bool = True):
    """HUD for switch mode: combined rep count, active-hand badge, per-side states."""
    landmark_list  = pose_landmarks.landmark
    active_tracker = right_tracker if switch_side == 'right' else left_tracker
    if show_skeleton:
        draw_skeleton(frame, pose_landmarks, mp_pose)

    # Wrist dots: active = bright large, inactive = small grey
    PL = mp_pose.PoseLandmark
    active_idx   = PL.RIGHT_WRIST if switch_side == 'right' else PL.LEFT_WRIST
    inactive_idx = PL.LEFT_WRIST  if switch_side == 'right' else PL.RIGHT_WRIST
    ia_lm = landmark_list[inactive_idx]
    cv2.circle(frame,
               (int(ia_lm.x * frame_w), int(ia_lm.y * frame_h)),
               6, (90, 90, 90), -1)
    ac_lm = landmark_list[active_idx]
    cv2.circle(frame,
               (int(ac_lm.x * frame_w), int(ac_lm.y * frame_h)),
               14, (0, 220, 255), -1)

    # ── panel height ──────────────────────────────────────────────────────────
    ph = 220  # REPS + top hold + active badge + state rows + norm_y + div + elbow
    if show_knee:
        ph += 26
    if show_ankle:
        ph += 26
    ph = min(ph, frame_h - 20)

    _panel_bg(frame, _PANEL_X, 10, _PANEL_W, ph)

    px = _PANEL_X + 12
    y  = 29

    # "REPS" label + large combined count
    cv2.putText(frame, "REPS", (px, y), _FONT, 0.5, _C_GREY, 1, _AA)
    y += 46
    combined = right_tracker.rep_count + left_tracker.rep_count
    cv2.putText(frame, str(combined), (px, y), _FONT, 2.2, _C_WHITE, 3, _AA)
    y += 10

    # Top-hold bar slot — always reserved so layout stays fixed
    if active_tracker.state == TOP:
        _top_hold_bar(frame, px, y,
                      _PANEL_W - 44, active_tracker.top_frames, active_tracker.min_top_frames)
    y += 22   # advance regardless of state
    y += 14

    # Active-hand badge  ── "[R]" / "[L]" ─────────────────────────────────────
    badge_label = switch_side[0].upper()
    badge_color = (0, 220, 255)
    init_tag = "  INIT" if active_tracker.needs_init else ""
    lock_tag = f"  lock {switch_lockout}" if switch_lockout > 0 else ""
    cv2.putText(frame, f"ACTIVE", (px, y), _FONT, 0.52, _C_GREY, 1, _AA)
    cv2.putText(frame, badge_label, (px + 62, y), _FONT, 0.75, badge_color, 2, _AA)
    if init_tag or lock_tag:
        cv2.putText(frame, (init_tag + lock_tag).strip(),
                    (px + 82, y), _FONT, 0.45, _C_GREY, 1, _AA)
    y += 22

    _divider(frame, _PANEL_X, y, _PANEL_W)
    y += 14

    # Per-side state rows: "R  3  ● TOP"
    for t, lbl, wc in ((right_tracker, "R", (0, 255, 255)),
                        (left_tracker,  "L", (255, 255, 0))):
        sc = STATE_COLORS.get(t.state, _C_WHITE)
        cv2.putText(frame, lbl, (px, y), _FONT, 0.62, wc, 2, _AA)
        cv2.putText(frame, str(t.rep_count), (px + 18, y), _FONT, 0.72, wc, 2, _AA)
        cv2.circle(frame, (px + 58, y - 5), 6, sc, -1)
        cv2.putText(frame, t.state, (px + 72, y), _FONT, 0.6, sc, 1, _AA)
        y += 24

    # norm_y compact row
    cv2.putText(frame,
                f"R ny {right_tracker.norm_y:+.2f}   L ny {left_tracker.norm_y:+.2f}",
                (px, y), _FONT, 0.52, _C_GREY, 1, _AA)
    y += 18

    _divider(frame, _PANEL_X, y, _PANEL_W)
    y += 15

    # Active-side status indicators
    _dot_row(frame, px, y,
             f"Elbow  {active_tracker.elbow_angle:.0f}\xb0",
             active_tracker.elbow_locked)
    y += 26

    if show_knee:
        _dot_row(frame, px, y,
                 f"Knees  {active_tracker.knee_angle:.0f}\xb0",
                 active_tracker.knee_locked)
        y += 26

    if show_ankle:
        _dot_row(frame, px, y,
                 "Ankle  OK" if not active_tracker.ankle_raised else "HEEL RAISED!",
                 not active_tracker.ankle_raised)

    _movement_tag(frame, movement_name, frame_w)


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
    double_mode           = args.mode == "double"
    switch_mode           = args.mode == "switch"
    dedicated_calib       = args.calibrate
    require_elbow_lock    = args.elbow_lock
    require_knee_lock     = args.knee_lock
    check_ankle_grounded  = args.ankle_grounded
    movement_name         = args.movement.strip()
    show_skeleton         = not args.no_skeleton

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
                    R_knee_ankle = (PL.RIGHT_KNEE, PL.RIGHT_ANKLE)
                    L_knee_ankle = (PL.LEFT_KNEE,  PL.LEFT_ANKLE)

                    active_t        = right_tracker   if switch_side == 'right' else left_tracker
                    inactive_t      = left_tracker    if switch_side == 'right' else right_tracker
                    active_lm       = R_lm            if switch_side == 'right' else L_lm
                    inactive_lm     = L_lm            if switch_side == 'right' else R_lm
                    active_ka       = R_knee_ankle    if switch_side == 'right' else L_knee_ankle

                    # Update inactive wrist position for display + switch detection
                    inactive_t.update_display_only(landmark_list,
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
                        inactive_t.needs_init          = False
                        inactive_t.state               = TOP
                        inactive_t.top_frames          = inactive_t.min_top_frames  # pre-filled
                        inactive_t.lockout             = 0
                        inactive_t._counted_this_cycle = False
                        inactive_t.smooth_buf.clear()
                        active_t.state        = BOTTOM
                        active_t.top_frames   = 0
                        switch_side    = 'left' if switch_side == 'right' else 'right'
                        switch_lockout = 30   # ~1 s at 30 fps; lets old wrist descend
                        active_t, active_lm = inactive_t, inactive_lm

                    # Run full state machine for the active side only
                    active_t.update(landmark_list,
                                    *active_lm, frame_w, frame_h,
                                    require_elbow_lock=require_elbow_lock,
                                    knee_idx=active_ka[0], ankle_idx=active_ka[1],
                                    require_knee_lock=require_knee_lock,
                                    check_ankle_grounded=check_ankle_grounded)

                    draw_overlay_switch(frame, pose_result.pose_landmarks,
                                        frame_w, frame_h,
                                        right_tracker, left_tracker, mp_pose,
                                        switch_side=switch_side,
                                        switch_lockout=switch_lockout,
                                        show_knee=require_knee_lock,
                                        show_ankle=check_ankle_grounded,
                                        movement_name=movement_name,
                                        show_skeleton=show_skeleton)
                else:
                    PL = PoseLandmark
                    right_tracker.update(landmark_list,
                                         PL.RIGHT_WRIST, PL.RIGHT_SHOULDER,
                                         PL.RIGHT_HIP,   PL.RIGHT_ELBOW,
                                         frame_w, frame_h,
                                         require_elbow_lock=require_elbow_lock,
                                         knee_idx=PL.RIGHT_KNEE,
                                         ankle_idx=PL.RIGHT_ANKLE,
                                         require_knee_lock=require_knee_lock,
                                         check_ankle_grounded=check_ankle_grounded)
                    if double_mode:
                        left_tracker.update(landmark_list,
                                            PL.LEFT_WRIST, PL.LEFT_SHOULDER,
                                            PL.LEFT_HIP,   PL.LEFT_ELBOW,
                                            frame_w, frame_h,
                                            require_elbow_lock=require_elbow_lock,
                                            knee_idx=PL.LEFT_KNEE,
                                            ankle_idx=PL.LEFT_ANKLE,
                                            require_knee_lock=require_knee_lock,
                                            check_ankle_grounded=check_ankle_grounded)
                        draw_overlay_double(frame, pose_result.pose_landmarks,
                                            frame_w, frame_h,
                                            right_tracker, left_tracker, mp_pose,
                                            show_knee=require_knee_lock,
                                            show_ankle=check_ankle_grounded,
                                            movement_name=movement_name,
                                            show_skeleton=show_skeleton)
                    else:
                        draw_overlay_single(frame, pose_result.pose_landmarks,
                                            frame_w, frame_h, right_tracker, mp_pose,
                                            show_knee=require_knee_lock,
                                            show_ankle=check_ankle_grounded,
                                            movement_name=movement_name,
                                            show_skeleton=show_skeleton)

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

        cv2.imshow("Kettlebell Rep Counter", frame)
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
