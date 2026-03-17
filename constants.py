"""Shared constants for the kettlebell rep counter."""

import pathlib

# ── State machine states ──────────────────────────────────────────────────────
BOTTOM  = "BOTTOM"
RISING  = "RISING"
TOP     = "TOP"
FALLING = "FALLING"

# ── Detection thresholds (normalised wrist Y) ─────────────────────────────────
# norm_y = (wrist_y - shoulder_y) / (hip_y - shoulder_y)
#   0  = at shoulder,  1 = at hip,  >1 = below hip (deep swing)
RISE_THRESHOLD   = 0.4    # wrist must reach above this to be "high"
DROP_THRESHOLD   = 0.85   # wrist must fall below this to be "low"
SMOOTH_WINDOW    = 5
MIN_TOP_FRAMES   = 4      # legacy fallback; real value is FPS-derived per tracker
MIN_REP_LOCKOUT  = 15     # frames locked after a rep returns to bottom (~0.5 s @ 30 fps)
ELBOW_LOCK_ANGLE = 160    # degrees; elbow ≥ this → arm locked out
KNEE_LOCK_ANGLE  = 160    # degrees; knee  ≥ this → leg locked out
ANKLE_RISE_PX_FRAC = 0.03 # fraction of frame height; heel rise above this is flagged

# ── Calibration constants ─────────────────────────────────────────────────────
CALIB_FRAMES    = 150     # ~5 s at 30 fps
CALIB_MIN_RANGE = 0.5     # narrower range → fall back to defaults
CALIB_RISE_FRAC = 0.35
CALIB_DROP_FRAC = 0.75
CALIB_FILE      = pathlib.Path("calibration.json")

STATE_COLORS = {
    BOTTOM:  (200, 200, 200),
    RISING:  (0,   220, 100),
    TOP:     (0,   200, 255),
    FALLING: (80,  80,  255),
}
