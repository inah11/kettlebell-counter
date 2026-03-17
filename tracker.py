"""SideTracker: per-wrist state machine + smoothing."""

from collections import deque

from constants import (
    BOTTOM, RISING, TOP, FALLING,
    RISE_THRESHOLD, DROP_THRESHOLD,
    SMOOTH_WINDOW, MIN_TOP_FRAMES, MIN_REP_LOCKOUT,
    ELBOW_LOCK_ANGLE, KNEE_LOCK_ANGLE, ANKLE_RISE_PX_FRAC,
)
from geometry import angle_at_joint, smooth, get_landmark_y, normalise_wrist


class SideTracker:
    """Holds state-machine + smoothing buffer for one wrist.

    Counting semantics
    ──────────────────
    A rep is counted in two stages:

    1. Validation (TOP state): the wrist must stay overhead for at least
       ``min_top_frames`` frames.  Once that threshold is reached
       ``_counted_this_cycle`` is set to True, but the counter is NOT yet
       incremented.  This lets the bell settle fully at the top before
       committing.

    2. Commit (FALLING → BOTTOM transition): when the wrist returns to hip
       level after a validated top-hold, rep_count is incremented and the
       post-rep lockout is started.  This means the display number ticks up
       only when the movement is fully complete — not while the bell is still
       overhead.

    The two-stage design prevents:
      - Counting a swing-through that never pauses overhead.
      - Counting too early on long overhead holds.
      - Double-counting due to the backswing (lockout guard).
    """

    def __init__(
        self,
        rise_threshold: float = RISE_THRESHOLD,
        drop_threshold: float = DROP_THRESHOLD,
        min_top_frames: int = MIN_TOP_FRAMES,
    ):
        self.state          = BOTTOM
        self.rep_count      = 0
        self.smooth_buf     = deque(maxlen=SMOOTH_WINDOW)
        self.norm_y         = 0.5
        self.top_frames     = 0      # frames wrist has stayed overhead in TOP
        self.lockout        = 0      # frames before next rep cycle can start
        self.needs_init     = True   # must see wrist at bottom before counting
        self.rise_threshold = rise_threshold
        self.drop_threshold = drop_threshold
        self.min_top_frames = min_top_frames
        self.elbow_locked   = False
        self.elbow_angle    = 180.0
        self.knee_locked    = False
        self.knee_angle     = 180.0
        self.ankle_baseline_y = None  # pixel Y of ankle at BOTTOM (heel-rise ref)
        self.ankle_raised   = False
        self._counted_this_cycle = False  # True = top validated; commit on BOTTOM

    # ── State machine ─────────────────────────────────────────────────────────

    def transition(
        self,
        norm_y: float,
        count_top_frame: bool = True,
        allow_rise: bool = True,
    ) -> None:
        """Drive the state machine one step, mutating self in-place.

        norm_y convention (increases downward in pixel space):
          small / negative  → wrist HIGH (above shoulder)
          large (> 1)       → wrist LOW  (below hip / deep swing)

        count_top_frame: when False the current frame is not counted toward
            min_top_frames.  Used to gate top-hold on elbow lock.

        allow_rise: when False the BOTTOM→RISING transition is blocked.
            Used to gate rep start on knee lock.

        Guards against double-counting:
          - min_top_frames: wrist must stay overhead for this many frames
            before the rep is validated.  Prevents swing-throughs.
          - lockout: after each rep a cooldown is applied once the wrist
            returns to BOTTOM, blocking BOTTOM→RISING so the backswing
            cannot start a new cycle immediately.
        """
        if self.lockout > 0:
            self.lockout -= 1

        if self.state == BOTTOM:
            if self.needs_init:
                if norm_y > self.drop_threshold:
                    self.needs_init = False
            elif self.lockout == 0 and norm_y < self.rise_threshold and allow_rise:
                self.state = RISING

        elif self.state == RISING:
            if norm_y < self.rise_threshold:
                self.state = TOP
                self.top_frames = 0
            elif norm_y > self.drop_threshold:
                # wrist dropped back without reaching overhead — abort cleanly
                self.state = BOTTOM

        elif self.state == TOP:
            if norm_y < self.rise_threshold:
                # Wrist still overhead — accumulate hold time
                if count_top_frame:
                    self.top_frames += 1
                    # Validation: bell has been overhead long enough.
                    # Set the flag but do NOT increment rep_count yet —
                    # the counter ticks up only when the bell returns to BOTTOM.
                    if (self.top_frames >= self.min_top_frames
                            and not self._counted_this_cycle):
                        self._counted_this_cycle = True
            else:
                # Wrist has started descending
                if self.top_frames >= self.min_top_frames:
                    self.state = FALLING
                else:
                    # Insufficient hold time — treat as swing-through
                    self.state = BOTTOM
                    self._counted_this_cycle = False
                self.top_frames = 0

        elif self.state == FALLING:
            if norm_y > self.drop_threshold:
                self.state = BOTTOM
                # Rep commit: the full movement is complete.
                # Apply lockout here so the full cooldown is available
                # before the next rep can begin.
                if self._counted_this_cycle:
                    self.rep_count += 1
                    self.lockout = MIN_REP_LOCKOUT
                    self._counted_this_cycle = False

    # ── Full update (landmark coords → state machine) ─────────────────────────

    def update(
        self,
        landmark_list,
        wrist_idx, shoulder_idx, hip_idx, elbow_idx,
        frame_w: int, frame_h: int,
        require_elbow_lock: bool = False,
        knee_idx=None, ankle_idx=None,
        require_knee_lock: bool = False,
        check_ankle_grounded: bool = False,
    ) -> None:
        """Run one frame of smoothing + state-machine update for a single side.

        require_elbow_lock: when True the elbow must be locked (≥ ELBOW_LOCK_ANGLE)
            for top_frames to accumulate.
        knee_idx / ankle_idx: MediaPipe landmark indices for knee/ankle; required
            for knee-lock and ankle-grounded features, ignored when None.
        require_knee_lock: when True the knee must be locked at BOTTOM before the
            BOTTOM→RISING transition fires.
        check_ankle_grounded: when True, flags self.ankle_raised if the heel rises
            more than ANKLE_RISE_PX_FRAC of frame height above its BOTTOM baseline.
        """
        wrist_y    = get_landmark_y(landmark_list, wrist_idx,    frame_h)
        shoulder_y = get_landmark_y(landmark_list, shoulder_idx, frame_h)
        hip_y      = get_landmark_y(landmark_list, hip_idx,      frame_h)

        raw_norm_y  = normalise_wrist(wrist_y, shoulder_y, hip_y)
        self.norm_y = smooth(self.smooth_buf, raw_norm_y)

        # Elbow angle (shoulder → elbow → wrist)
        sx = landmark_list[shoulder_idx].x * frame_w
        sy = landmark_list[shoulder_idx].y * frame_h
        ex = landmark_list[elbow_idx].x    * frame_w
        ey = landmark_list[elbow_idx].y    * frame_h
        wx = landmark_list[wrist_idx].x    * frame_w
        wy = landmark_list[wrist_idx].y    * frame_h
        self.elbow_angle  = angle_at_joint(sx, sy, ex, ey, wx, wy)
        self.elbow_locked = self.elbow_angle >= ELBOW_LOCK_ANGLE

        # Knee angle (hip → knee → ankle)
        if knee_idx is not None and ankle_idx is not None:
            hkx = landmark_list[hip_idx].x   * frame_w
            hky = landmark_list[hip_idx].y   * frame_h
            knx = landmark_list[knee_idx].x  * frame_w
            kny = landmark_list[knee_idx].y  * frame_h
            akx = landmark_list[ankle_idx].x * frame_w
            aky = landmark_list[ankle_idx].y * frame_h
            self.knee_angle  = angle_at_joint(hkx, hky, knx, kny, akx, aky)
            self.knee_locked = self.knee_angle >= KNEE_LOCK_ANGLE

        count_top_frame = (not require_elbow_lock) or self.elbow_locked
        allow_rise      = (not require_knee_lock)  or self.knee_locked
        self.transition(self.norm_y, count_top_frame, allow_rise)

        # Ankle grounding check
        if check_ankle_grounded and ankle_idx is not None:
            ankle_y = get_landmark_y(landmark_list, ankle_idx, frame_h)
            if self.state == BOTTOM:
                self.ankle_baseline_y = ankle_y
                self.ankle_raised = False
            elif self.ankle_baseline_y is not None:
                threshold_px = ANKLE_RISE_PX_FRAC * frame_h
                if ankle_y < self.ankle_baseline_y - threshold_px:
                    self.ankle_raised = True

    # ── Display-only update (inactive side in switch mode) ────────────────────

    def update_display_only(
        self,
        landmark_list,
        wrist_idx, shoulder_idx, hip_idx, elbow_idx,
        frame_w: int, frame_h: int,
    ) -> None:
        """Update norm_y and elbow fields without running the state machine."""
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

    # ── HUD status helpers ────────────────────────────────────────────────────

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
