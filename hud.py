"""OpenCV overlay drawing functions for the kettlebell rep counter HUD."""

import cv2
import numpy as np

from constants import STATE_COLORS, RISE_THRESHOLD
from calibration import Calibrator
from tracker import SideTracker

# ── Style constants ───────────────────────────────────────────────────────────
_FONT     = cv2.FONT_HERSHEY_SIMPLEX
_AA       = cv2.LINE_AA
_PANEL_W  = 268
_PANEL_X  = 10
_PANEL_BG = 0.68    # dark panel opacity (0 = transparent, 1 = black)
_C_WHITE  = (255, 255, 255)
_C_GREY   = (160, 160, 160)
_C_DIM    = (80,  80,  80)
_C_GOLD   = (40,  210, 255)   # BGR warm gold
_C_OK     = (50,  200, 80)    # green
_C_WARN   = (40,  60,  230)   # red


# ── Primitive helpers ─────────────────────────────────────────────────────────

def _panel_bg(frame, x: int, y: int, w: int, h: int) -> None:
    """Blend a near-black rectangle over a frame region."""
    roi = frame[y:y + h, x:x + w]
    frame[y:y + h, x:x + w] = (roi * (1 - _PANEL_BG)).astype(np.uint8)


def _divider(frame, x: int, y: int, w: int) -> None:
    cv2.line(frame, (x + 6, y), (x + w - 6, y), _C_DIM, 1, _AA)


def _dot_row(frame, x: int, y: int, label: str, ok: bool, scale: float = 0.6) -> None:
    """Colored status dot + label on a single row."""
    color = _C_OK if ok else _C_WARN
    cv2.circle(frame, (x + 7, y - 4), 5, color, -1)
    cv2.putText(frame, label, (x + 20, y), _FONT, scale, color, 1, _AA)


def _movement_tag(frame, text: str, frame_w: int) -> None:
    """Floating label chip in the top-right corner."""
    if not text:
        return
    (tw, th), baseline = cv2.getTextSize(text, _FONT, 0.72, 2)
    full_h = th + baseline + 1
    pad_x, pad_y = 10, 8
    chip_w = tw + pad_x * 2
    chip_h = full_h + pad_y * 2
    chip_x = frame_w - chip_w - 8
    chip_y = 8
    x1 = max(0, chip_x);  y1 = max(0, chip_y)
    x2 = min(frame_w, chip_x + chip_w);  y2 = min(frame.shape[0], chip_y + chip_h)
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        frame[y1:y2, x1:x2] = (roi * 0.22).astype(np.uint8)
    cv2.putText(frame, text, (chip_x + pad_x, chip_y + pad_y + th),
                _FONT, 0.72, _C_GOLD, 2, _AA)


def _top_hold_bar(frame, x: int, y: int, w: int,
                  top_frames: int, min_top_frames: int) -> None:
    """Thin progress bar showing overhead hold progress."""
    fill = int(w * min(top_frames / max(min_top_frames, 1), 1.0))
    cv2.rectangle(frame, (x, y), (x + w, y + 6), _C_DIM, -1)
    cv2.rectangle(frame, (x, y), (x + fill, y + 6), (0, 200, 255), -1)
    cv2.putText(frame, f"{top_frames}/{min_top_frames}",
                (x + w + 8, y + 6), _FONT, 0.48, _C_GREY, 1, _AA)


# ── Skeleton / wrist dots ─────────────────────────────────────────────────────

def draw_wrist_dot(frame, landmark_list, wrist_idx, frame_w: int, frame_h: int,
                   color) -> None:
    wrist = landmark_list[wrist_idx]
    cv2.circle(frame,
               (int(wrist.x * frame_w), int(wrist.y * frame_h)),
               10, color, -1)


def draw_skeleton(frame, pose_landmarks, mp_pose) -> None:
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    mp_draw.draw_landmarks(
        frame,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_style.get_default_pose_landmarks_style(),
    )


# ── Calibration HUD ───────────────────────────────────────────────────────────

def draw_calibration_hud(frame, calibrator: Calibrator,
                         dedicated_mode: bool = False) -> None:
    """Overlay calibration progress.  Panel height is fixed to prevent layout jitter."""
    fh, fw = frame.shape[:2]
    title = "CALIBRATING" if dedicated_mode else "Calibrating..."

    # Centered banner
    (tw, _), _ = cv2.getTextSize(title, _FONT, 1.1, 2)
    tx = (fw - tw) // 2
    _panel_bg(frame, tx - 16, 14, tw + 32, 44)
    cv2.putText(frame, title, (tx, 48), _FONT, 1.1, (0, 220, 255), 2, _AA)

    # Stats panel — fixed height 80 so layout never shifts
    _panel_bg(frame, _PANEL_X, 68, _PANEL_W, 80)
    cv2.putText(frame,
                f"{calibrator.frames_collected} / {calibrator.n_frames}  frames",
                (_PANEL_X + 12, 100), _FONT, 0.72, _C_GREY, 1, _AA)
    if calibrator.samples:
        obs_min = min(calibrator.samples)
        obs_max = max(calibrator.samples)
        cv2.putText(frame, f"ny  min {obs_min:.2f}   max {obs_max:.2f}",
                    (_PANEL_X + 12, 128), _FONT, 0.6, _C_GREY, 1, _AA)

    # Progress bar — fixed at bottom of frame
    bar_h  = 14
    bar_y  = fh - bar_h - 8
    fill_w = int(fw * calibrator.progress)
    cv2.rectangle(frame, (0, bar_y), (fw, bar_y + bar_h), _C_DIM, -1)
    cv2.rectangle(frame, (0, bar_y), (fill_w, bar_y + bar_h), (0, 220, 255), -1)


# ── Overlay: single mode ──────────────────────────────────────────────────────

def draw_overlay_single(frame, pose_landmarks, frame_w: int, frame_h: int,
                        tracker: SideTracker, mp_pose,
                        show_knee: bool = False, show_ankle: bool = False,
                        movement_name: str = "", show_skeleton: bool = True) -> None:
    """HUD for single-KB mode (right wrist only)."""
    landmark_list = pose_landmarks.landmark
    if show_skeleton:
        draw_skeleton(frame, pose_landmarks, mp_pose)
    draw_wrist_dot(frame, landmark_list, mp_pose.PoseLandmark.RIGHT_WRIST,
                   frame_w, frame_h, (0, 255, 255))

    # Panel height — top-hold bar slot is always reserved so layout stays fixed
    ph = 190
    if show_knee:  ph += 26
    if show_ankle: ph += 26
    if tracker.rise_threshold != RISE_THRESHOLD: ph += 22
    ph = min(ph, frame_h - 20)

    _panel_bg(frame, _PANEL_X, 10, _PANEL_W, ph)
    px = _PANEL_X + 12
    y  = 29

    cv2.putText(frame, "REPS", (px, y), _FONT, 0.5, _C_GREY, 1, _AA)
    y += 46
    cv2.putText(frame, str(tracker.rep_count), (px, y), _FONT, 2.2, _C_WHITE, 3, _AA)
    y += 10

    # Top-hold bar slot — always advance y so lower elements never shift
    if tracker.state == "TOP":
        _top_hold_bar(frame, px, y, _PANEL_W - 44,
                      tracker.top_frames, tracker.min_top_frames)
    y += 22
    y += 20

    sc = STATE_COLORS.get(tracker.state, _C_WHITE)
    cv2.circle(frame, (px + 7, y - 5), 7, sc, -1)
    cv2.putText(frame, tracker.state, (px + 22, y), _FONT, 0.78, sc, 2, _AA)
    y += 22

    cv2.putText(frame, f"ny {tracker.norm_y:+.2f}", (px, y), _FONT, 0.55, _C_GREY, 1, _AA)
    y += 18

    _divider(frame, _PANEL_X, y, _PANEL_W)
    y += 15

    _dot_row(frame, px, y, f"Elbow  {tracker.elbow_angle:.0f}\xb0", tracker.elbow_locked)
    y += 26

    if show_knee:
        _dot_row(frame, px, y, f"Knees  {tracker.knee_angle:.0f}\xb0", tracker.knee_locked)
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


# ── Overlay: double mode ──────────────────────────────────────────────────────

def draw_overlay_double(frame, pose_landmarks, frame_w: int, frame_h: int,
                        right_tracker: SideTracker, left_tracker: SideTracker,
                        mp_pose, show_knee: bool = False, show_ankle: bool = False,
                        movement_name: str = "", show_skeleton: bool = True) -> None:
    """HUD for double-KB mode (both wrists, independent counts)."""
    landmark_list = pose_landmarks.landmark
    if show_skeleton:
        draw_skeleton(frame, pose_landmarks, mp_pose)
    draw_wrist_dot(frame, landmark_list, mp_pose.PoseLandmark.RIGHT_WRIST,
                   frame_w, frame_h, (0, 255, 255))
    draw_wrist_dot(frame, landmark_list, mp_pose.PoseLandmark.LEFT_WRIST,
                   frame_w, frame_h, (255, 255, 0))

    def _side_section(tracker, label, wrist_color, y_start):
        y  = y_start
        sc = STATE_COLORS.get(tracker.state, _C_WHITE)
        cv2.putText(frame, label,            (px, y),      _FONT, 0.62, wrist_color, 2, _AA)
        cv2.putText(frame, str(tracker.rep_count), (px + 22, y), _FONT, 0.95, wrist_color, 2, _AA)
        cv2.circle(frame, (px + 65, y - 5), 6, sc, -1)
        cv2.putText(frame, tracker.state,    (px + 78, y), _FONT, 0.62, sc, 1, _AA)
        y += 22
        cv2.putText(frame, f"ny {tracker.norm_y:+.2f}", (px + 8, y),
                    _FONT, 0.52, _C_GREY, 1, _AA)
        y += 20
        _dot_row(frame, px + 4, y, f"Elbow  {tracker.elbow_angle:.0f}\xb0",
                 tracker.elbow_locked, 0.58)
        y += 22
        if show_knee:
            _dot_row(frame, px + 4, y, f"Knees  {tracker.knee_angle:.0f}\xb0",
                     tracker.knee_locked, 0.58)
            y += 22
        if show_ankle:
            _dot_row(frame, px + 4, y,
                     "Ankle  OK" if not tracker.ankle_raised else "HEEL RAISED!",
                     not tracker.ankle_raised, 0.58)
            y += 22
        return y

    per_side = 64 + (22 if show_knee else 0) + (22 if show_ankle else 0)
    ph = 80 + per_side * 2 + 16
    if right_tracker.rise_threshold != RISE_THRESHOLD: ph += 22
    ph = min(ph, frame_h - 20)

    _panel_bg(frame, _PANEL_X, 10, _PANEL_W, ph)
    px = _PANEL_X + 12
    y  = 29

    combined = right_tracker.rep_count + left_tracker.rep_count
    cv2.putText(frame, "TOTAL", (px, y), _FONT, 0.5, _C_GREY, 1, _AA)
    y += 40
    cv2.putText(frame, str(combined), (px, y), _FONT, 1.8, _C_WHITE, 3, _AA)
    y += 16

    _divider(frame, _PANEL_X, y, _PANEL_W);  y += 14
    y = _side_section(right_tracker, "R", (0, 255, 255), y)
    _divider(frame, _PANEL_X, y, _PANEL_W);  y += 14
    y = _side_section(left_tracker,  "L", (255, 255, 0), y)

    if right_tracker.rise_threshold != RISE_THRESHOLD:
        _divider(frame, _PANEL_X, y, _PANEL_W);  y += 14
        cv2.putText(frame,
                    f"rise {right_tracker.rise_threshold:.2f}"
                    f"  drop {right_tracker.drop_threshold:.2f}",
                    (px, y), _FONT, 0.48, (100, 220, 255), 1, _AA)

    _movement_tag(frame, movement_name, frame_w)


# ── Overlay: switch mode ──────────────────────────────────────────────────────

def draw_overlay_switch(frame, pose_landmarks, frame_w: int, frame_h: int,
                        right_tracker: SideTracker, left_tracker: SideTracker,
                        mp_pose, switch_side: str = 'right', switch_lockout: int = 0,
                        show_knee: bool = False, show_ankle: bool = False,
                        movement_name: str = "", show_skeleton: bool = True) -> None:
    """HUD for switch mode: combined count, active-hand badge, per-side rows."""
    landmark_list  = pose_landmarks.landmark
    active_tracker = right_tracker if switch_side == 'right' else left_tracker
    if show_skeleton:
        draw_skeleton(frame, pose_landmarks, mp_pose)

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

    ph = 220
    if show_knee:  ph += 26
    if show_ankle: ph += 26
    ph = min(ph, frame_h - 20)

    _panel_bg(frame, _PANEL_X, 10, _PANEL_W, ph)
    px = _PANEL_X + 12
    y  = 29

    cv2.putText(frame, "REPS", (px, y), _FONT, 0.5, _C_GREY, 1, _AA)
    y += 46
    combined = right_tracker.rep_count + left_tracker.rep_count
    cv2.putText(frame, str(combined), (px, y), _FONT, 2.2, _C_WHITE, 3, _AA)
    y += 10

    # Top-hold bar slot — always reserve space
    if active_tracker.state == "TOP":
        _top_hold_bar(frame, px, y, _PANEL_W - 44,
                      active_tracker.top_frames, active_tracker.min_top_frames)
    y += 22
    y += 14

    badge_label = switch_side[0].upper()
    init_tag = "  INIT" if active_tracker.needs_init else ""
    lock_tag = f"  lock {switch_lockout}" if switch_lockout > 0 else ""
    cv2.putText(frame, "ACTIVE", (px, y), _FONT, 0.52, _C_GREY, 1, _AA)
    cv2.putText(frame, badge_label, (px + 62, y), _FONT, 0.75, (0, 220, 255), 2, _AA)
    if init_tag or lock_tag:
        cv2.putText(frame, (init_tag + lock_tag).strip(),
                    (px + 82, y), _FONT, 0.45, _C_GREY, 1, _AA)
    y += 22

    _divider(frame, _PANEL_X, y, _PANEL_W);  y += 14

    for t, lbl, wc in ((right_tracker, "R", (0, 255, 255)),
                        (left_tracker,  "L", (255, 255, 0))):
        sc = STATE_COLORS.get(t.state, _C_WHITE)
        cv2.putText(frame, lbl,           (px,      y), _FONT, 0.62, wc, 2, _AA)
        cv2.putText(frame, str(t.rep_count), (px + 18, y), _FONT, 0.72, wc, 2, _AA)
        cv2.circle(frame, (px + 58, y - 5), 6, sc, -1)
        cv2.putText(frame, t.state,       (px + 72, y), _FONT, 0.60, sc, 1, _AA)
        y += 24

    cv2.putText(frame,
                f"R ny {right_tracker.norm_y:+.2f}   L ny {left_tracker.norm_y:+.2f}",
                (px, y), _FONT, 0.52, _C_GREY, 1, _AA)
    y += 18

    _divider(frame, _PANEL_X, y, _PANEL_W);  y += 15

    _dot_row(frame, px, y, f"Elbow  {active_tracker.elbow_angle:.0f}\xb0",
             active_tracker.elbow_locked)
    y += 26

    if show_knee:
        _dot_row(frame, px, y, f"Knees  {active_tracker.knee_angle:.0f}\xb0",
                 active_tracker.knee_locked)
        y += 26

    if show_ankle:
        _dot_row(frame, px, y,
                 "Ankle  OK" if not active_tracker.ankle_raised else "HEEL RAISED!",
                 not active_tracker.ankle_raised)

    _movement_tag(frame, movement_name, frame_w)


# ── need mp for draw_skeleton — import lazily to avoid circular deps ──────────
import mediapipe as mp  # noqa: E402  (placed here to avoid top-level MediaPipe load in tests)
