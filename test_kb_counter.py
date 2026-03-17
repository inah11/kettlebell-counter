"""
Tests for kb_counter.py pure-logic functions.
No CV or MediaPipe required — all heavy deps are avoided.

Run with:  pytest test_kb_counter.py -v
"""

from collections import deque

import pytest

from kb_counter import (
    BOTTOM, RISING, TOP, FALLING,
    RISE_THRESHOLD, DROP_THRESHOLD, SMOOTH_WINDOW,
    MIN_TOP_FRAMES, MIN_REP_LOCKOUT,
    CALIB_MIN_RANGE, CALIB_RISE_FRAC, CALIB_DROP_FRAC,
    smooth, normalise_wrist, transition, SideTracker, update_tracker,
    derive_thresholds_from_samples, Calibrator,
    save_calibration, load_calibration,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

class _MockLandmark:
    """Minimal landmark stub: normalised coords with .x and .y."""
    def __init__(self, y_norm, x_norm=0.5):
        self.y = y_norm
        self.x = x_norm


def make_landmark_list(wrist_y_norm, shoulder_y_norm, hip_y_norm,
                       wrist_idx=0, shoulder_idx=1, hip_idx=2, elbow_idx=3):
    """
    Build a subscriptable list of landmark stubs at the given indices.
    All other slots are filled with a neutral landmark (y=0.5, x=0.5).
    Elbow is placed mid-way between shoulder and wrist y (straight arm).
    """
    landmark_count  = max(wrist_idx, shoulder_idx, hip_idx, elbow_idx) + 1
    landmark_list   = [_MockLandmark(0.5)] * landmark_count
    elbow_y         = (shoulder_y_norm + wrist_y_norm) / 2
    landmark_list[wrist_idx]    = _MockLandmark(wrist_y_norm)
    landmark_list[shoulder_idx] = _MockLandmark(shoulder_y_norm)
    landmark_list[hip_idx]      = _MockLandmark(hip_y_norm)
    landmark_list[elbow_idx]    = _MockLandmark(elbow_y)
    return landmark_list


def drive_state_machine(norm_y_sequence):
    """
    Feed a sequence of norm_y values through the state machine starting at
    BOTTOM with rep_count=0.  Returns (final_state, final_rep_count).
    """
    tracker = SideTracker()
    for norm_y in norm_y_sequence:
        transition(tracker, norm_y)
    return tracker.state, tracker.rep_count


# ── Module-level norm_y values used across transition tests ──────────────────
# These names reflect where the wrist is in the lift, not just magnitude.

OVERHEAD_NORM_Y  = RISE_THRESHOLD - 0.1          # wrist clearly overhead  (0.3)
MID_NORM_Y       = (RISE_THRESHOLD + DROP_THRESHOLD) / 2  # between thresholds (0.625)
BELOW_HIP_NORM_Y = DROP_THRESHOLD + 0.1          # wrist clearly at swing bottom (0.95)

# Minimal frame sequence to drive exactly one rep through the state machine.
# Includes enough overhead frames to satisfy MIN_TOP_FRAMES before descending.
#   1 frame  BELOW_HIP : ensure starting in BOTTOM
#   1 frame  OVERHEAD  : BOTTOM → RISING
#   1 frame  OVERHEAD  : RISING → TOP (top_frames reset to 0)
#   MIN_TOP_FRAMES frames OVERHEAD : increment top_frames to MIN_TOP_FRAMES
#   1 frame  MID       : TOP → FALLING (top_frames satisfied)
#   1 frame  BELOW_HIP : FALLING → BOTTOM (+1 rep, lockout set)
ONE_REP = (
    [BELOW_HIP_NORM_Y]
    + [OVERHEAD_NORM_Y] * (MIN_TOP_FRAMES + 2)
    + [MID_NORM_Y]
    + [BELOW_HIP_NORM_Y]
)

# Extra bottom frames to drain the post-rep lockout so the next rep can start.
LOCKOUT_DRAIN = [BELOW_HIP_NORM_Y] * MIN_REP_LOCKOUT


# ── smooth() ─────────────────────────────────────────────────────────────────

class TestSmooth:
    """Tests for the smooth() rolling-average helper."""

    def test_single_value(self):
        """A single value in an empty window returns that value."""
        smoothing_window = deque(maxlen=SMOOTH_WINDOW)
        assert smooth(smoothing_window, 0.6) == pytest.approx(0.6)

    def test_average_of_two(self):
        """Average of two values is their mean."""
        smoothing_window = deque(maxlen=SMOOTH_WINDOW)
        smooth(smoothing_window, 0.2)
        result = smooth(smoothing_window, 0.8)
        assert result == pytest.approx(0.5)

    def test_window_evicts_oldest(self):
        """Once full, oldest entry is evicted before averaging."""
        smoothing_window = deque(maxlen=3)
        smooth(smoothing_window, 1.0)
        smooth(smoothing_window, 1.0)
        smooth(smoothing_window, 1.0)
        # oldest 1.0 is evicted; window now [1, 1, 0]
        result = smooth(smoothing_window, 0.0)
        assert result == pytest.approx((1.0 + 1.0 + 0.0) / 3)

    def test_converges_to_constant(self):
        """After filling window with a constant, result equals that constant."""
        smoothing_window = deque(maxlen=SMOOTH_WINDOW)
        for _ in range(SMOOTH_WINDOW):
            result = smooth(smoothing_window, 0.7)
        assert result == pytest.approx(0.7)


# ── normalise_wrist() ─────────────────────────────────────────────────────────

class TestNormaliseWrist:
    """Tests for the normalise_wrist() coordinate mapper."""

    # shoulder_y=100px, hip_y=200px → torso_height_px=100
    SHOULDER_Y_PX = 100.0
    HIP_Y_PX      = 200.0

    def test_at_shoulder_level(self):
        """Wrist at shoulder yields norm_y == 0.0."""
        assert normalise_wrist(
            self.SHOULDER_Y_PX, self.SHOULDER_Y_PX, self.HIP_Y_PX
        ) == pytest.approx(0.0)

    def test_at_hip_level(self):
        """Wrist at hip yields norm_y == 1.0."""
        assert normalise_wrist(
            self.HIP_Y_PX, self.SHOULDER_Y_PX, self.HIP_Y_PX
        ) == pytest.approx(1.0)

    def test_overhead(self):
        """Wrist above shoulder yields negative norm_y."""
        # wrist above shoulder → negative result
        assert normalise_wrist(50.0, self.SHOULDER_Y_PX, self.HIP_Y_PX) == pytest.approx(-0.5)

    def test_deep_swing(self):
        """Wrist below hip yields norm_y > 1.0."""
        # wrist well below hip → >1
        assert normalise_wrist(250.0, self.SHOULDER_Y_PX, self.HIP_Y_PX) == pytest.approx(1.5)

    def test_midpoint(self):
        """Wrist at torso midpoint yields norm_y == 0.5."""
        assert normalise_wrist(150.0, self.SHOULDER_Y_PX, self.HIP_Y_PX) == pytest.approx(0.5)

    def test_degenerate_span_returns_half(self):
        """Zero-height torso guard returns 0.5."""
        # shoulder == hip → guard triggers → 0.5
        assert normalise_wrist(100.0, 100.0, 100.0) == pytest.approx(0.5)

    def test_near_zero_span_returns_half(self):
        """Near-zero torso height also triggers the degenerate guard."""
        assert normalise_wrist(100.0, 100.0, 100.0 + 1e-7) == pytest.approx(0.5)


# ── transition() ─────────────────────────────────────────────────────────────

class TestTransitionEdges:
    """Boundary values exactly on the thresholds."""

    def test_bottom_at_threshold_no_transition(self):
        """norm_y exactly equal to RISE_THRESHOLD does not trigger RISING."""
        # norm_y == RISE_THRESHOLD → NOT < threshold, stays BOTTOM
        tracker = SideTracker()
        transition(tracker, RISE_THRESHOLD)
        assert tracker.state == BOTTOM

    def test_bottom_just_below_threshold_rises(self):
        """norm_y just below RISE_THRESHOLD triggers RISING."""
        tracker = SideTracker()
        tracker.needs_init = False
        transition(tracker, RISE_THRESHOLD - 0.001)
        assert tracker.state == RISING

    def test_falling_at_threshold_no_rep(self):
        """norm_y exactly equal to DROP_THRESHOLD does not count a rep."""
        tracker = SideTracker()
        tracker.state = FALLING
        transition(tracker, DROP_THRESHOLD)
        assert tracker.state == FALLING
        assert tracker.rep_count == 0

    def test_falling_just_above_threshold_counts(self):
        """norm_y just above DROP_THRESHOLD completes a rep."""
        tracker = SideTracker()
        tracker.state = FALLING
        transition(tracker, DROP_THRESHOLD + 0.001)
        assert tracker.state == BOTTOM
        assert tracker.rep_count == 1


class TestTransitionStateMachine:
    """Happy-path state progression."""

    def test_bottom_stays_when_wrist_is_low(self):
        """Wrist below hip in BOTTOM stays BOTTOM with no rep."""
        tracker = SideTracker()
        transition(tracker, BELOW_HIP_NORM_Y)
        assert tracker.state == BOTTOM
        assert tracker.rep_count == 0

    def test_bottom_to_rising_when_wrist_overhead(self):
        """Wrist overhead from BOTTOM transitions to RISING."""
        tracker = SideTracker()
        tracker.needs_init = False
        transition(tracker, OVERHEAD_NORM_Y)
        assert tracker.state == RISING

    def test_rising_to_top_when_wrist_stays_overhead(self):
        """Wrist overhead from RISING transitions to TOP."""
        tracker = SideTracker()
        tracker.state = RISING
        transition(tracker, OVERHEAD_NORM_Y)
        assert tracker.state == TOP

    def test_rising_resets_to_bottom_when_wrist_drops_past_hip(self):
        """Wrist dropping past hip while RISING resets to BOTTOM."""
        # Wrist drops below drop_threshold while RISING (no KB at top) → BOTTOM
        tracker = SideTracker()
        tracker.state = RISING
        transition(tracker, BELOW_HIP_NORM_Y)   # norm_y > DROP_THRESHOLD
        assert tracker.state == BOTTOM

    def test_top_to_falling_after_min_overhead_frames(self):
        """After enough overhead frames TOP transitions to FALLING on descent."""
        # Must stay overhead for MIN_TOP_FRAMES frames, then descend → FALLING
        tracker = SideTracker()
        tracker.state = TOP
        for _ in range(MIN_TOP_FRAMES):
            transition(tracker, OVERHEAD_NORM_Y)
        transition(tracker, MID_NORM_Y)
        assert tracker.state == FALLING

    def test_top_aborts_to_bottom_on_brief_overhead(self):
        """Insufficient overhead frames resets to BOTTOM without counting."""
        # Wrist descends before MIN_TOP_FRAMES → swing-through, resets to BOTTOM
        tracker = SideTracker()
        tracker.state = TOP
        tracker.top_frames = MIN_TOP_FRAMES - 1  # one short of the requirement
        transition(tracker, MID_NORM_Y)
        assert tracker.state == BOTTOM
        assert tracker.rep_count == 0

    def test_top_stays_when_wrist_remains_overhead(self):
        """Wrist remaining overhead in TOP stays in TOP."""
        tracker = SideTracker()
        tracker.state = TOP
        transition(tracker, OVERHEAD_NORM_Y)
        assert tracker.state == TOP

    def test_falling_stays_when_wrist_not_yet_low(self):
        """Wrist between thresholds while FALLING stays in FALLING."""
        tracker = SideTracker()
        tracker.state = FALLING
        transition(tracker, MID_NORM_Y)
        assert tracker.state == FALLING
        assert tracker.rep_count == 0

    def test_falling_to_bottom_increments_rep(self):
        """Wrist reaching below hip from FALLING completes the rep."""
        tracker = SideTracker()
        tracker.state = FALLING
        transition(tracker, BELOW_HIP_NORM_Y)
        assert tracker.state == BOTTOM
        assert tracker.rep_count == 1


# ── Lockout (anti-double-count guard) ────────────────────────────────────────

class TestLockout:
    """Post-rep lockout prevents backswing from starting a new cycle."""

    def test_lockout_set_after_rep(self):
        """Completing a rep sets lockout to MIN_REP_LOCKOUT."""
        tracker = SideTracker()
        tracker.state = FALLING
        transition(tracker, BELOW_HIP_NORM_Y)
        assert tracker.lockout == MIN_REP_LOCKOUT

    def test_lockout_blocks_rising_from_bottom(self):
        """Non-zero lockout prevents BOTTOM→RISING transition."""
        tracker = SideTracker()
        tracker.needs_init = False
        tracker.lockout = 5
        transition(tracker, OVERHEAD_NORM_Y)   # would normally trigger RISING
        assert tracker.state == BOTTOM

    def test_lockout_decrements_each_frame(self):
        """Lockout counter decrements by one each frame."""
        tracker = SideTracker()
        tracker.lockout = 3
        transition(tracker, BELOW_HIP_NORM_Y)
        assert tracker.lockout == 2

    def test_lockout_allows_rising_when_zero(self):
        """Lockout reaching zero in the same frame allows RISING."""
        # lockout=1: decrements to 0 in the same frame, then BOTTOM→RISING fires
        tracker = SideTracker()
        tracker.needs_init = False
        tracker.lockout = 1
        transition(tracker, OVERHEAD_NORM_Y)
        assert tracker.state == RISING


# ── Full rep cycles ───────────────────────────────────────────────────────────

class TestFullRepCycle:
    """End-to-end state-machine rep-counting using drive_state_machine."""

    def test_one_complete_rep(self):
        """ONE_REP sequence produces exactly one rep and ends at BOTTOM."""
        state, rep_count = drive_state_machine(ONE_REP)
        assert rep_count == 1
        assert state == BOTTOM

    def test_three_reps(self):
        """Three ONE_REP sequences separated by lockout drains count three reps."""
        seq = (ONE_REP + LOCKOUT_DRAIN) * 2 + ONE_REP
        _, rep_count = drive_state_machine(seq)
        assert rep_count == 3

    def test_no_rep_without_overhead(self):
        """Wrist never going overhead results in zero reps."""
        # Wrist never goes overhead → never leaves BOTTOM → no reps
        norm_y_sequence = [MID_NORM_Y, MID_NORM_Y, BELOW_HIP_NORM_Y,
                           BELOW_HIP_NORM_Y, MID_NORM_Y]
        _, rep_count = drive_state_machine(norm_y_sequence)
        assert rep_count == 0

    def test_no_rep_without_drop(self):
        """Wrist staying overhead without dropping to hip results in zero reps."""
        # Rises and stays overhead long enough, but never drops → stays FALLING
        norm_y_sequence = (
            [BELOW_HIP_NORM_Y]
            + [OVERHEAD_NORM_Y] * (MIN_TOP_FRAMES + 2)
            + [MID_NORM_Y] * 3   # not below DROP_THRESHOLD
        )
        state, rep_count = drive_state_machine(norm_y_sequence)
        assert rep_count == 0
        assert state == FALLING

    def test_rep_count_accumulates(self):
        """Rep counts accumulate correctly across multiple cycles."""
        one_rep_with_drain = ONE_REP + LOCKOUT_DRAIN
        _, count_after_one  = drive_state_machine(ONE_REP)
        _, count_after_five = drive_state_machine(one_rep_with_drain * 4 + ONE_REP)
        assert count_after_one  == 1
        assert count_after_five == 5

    def test_swing_through_does_not_count(self):
        """Swing-through (insufficient overhead frames) does not count as a rep."""
        # Wrist crosses overhead but not long enough (MIN_TOP_FRAMES - 1) → no rep
        swing_through = (
            [BELOW_HIP_NORM_Y]
            + [OVERHEAD_NORM_Y] * (MIN_TOP_FRAMES + 1)  # 1 short: RISING→TOP + (MIN_TOP_FRAMES-1) in TOP
            + [MID_NORM_Y]
            + [BELOW_HIP_NORM_Y]
        )
        _, rep_count = drive_state_machine(swing_through)
        assert rep_count == 0


# ── SideTracker ───────────────────────────────────────────────────────────────

class TestSideTracker:
    """Tests for SideTracker default construction and field values."""

    def test_initial_state(self):
        """Freshly constructed SideTracker has expected default field values."""
        tracker = SideTracker()
        assert tracker.state == BOTTOM
        assert tracker.rep_count == 0
        assert len(tracker.smooth_buf) == 0
        assert tracker.norm_y == 0.5
        assert tracker.top_frames == 0
        assert tracker.lockout == 0
        assert tracker.needs_init is True

    def test_smooth_buf_capacity(self):
        """smooth_buf maxlen matches the SMOOTH_WINDOW constant."""
        tracker = SideTracker()
        assert tracker.smooth_buf.maxlen == SMOOTH_WINDOW


# ── update_tracker() ─────────────────────────────────────────────────────────

class TestUpdateTracker:
    """
    Verifies that update_tracker correctly converts landmark pixel coords into
    norm_y and drives the state machine — without any MediaPipe or CV.

    Layout: shoulder at y=0.25, hip at y=0.75 (normalised coords in 100px frame)
    → pixel shoulder=25, hip=75, torso_height_px=50
    """
    FRAME_W         = 100
    FRAME_H         = 100
    SHOULDER_Y_NORM = 0.25   # shoulder pixel_y = 25
    HIP_Y_NORM      = 0.75   # hip pixel_y = 75
    WRIST_IDX = 0; SHOULDER_IDX = 1; HIP_IDX = 2; ELBOW_IDX = 3

    def _landmarks_with_wrist_at(self, wrist_y_norm):
        """Build a landmark list with the wrist at the given normalised y."""
        return make_landmark_list(
            wrist_y_norm, self.SHOULDER_Y_NORM, self.HIP_Y_NORM,
            self.WRIST_IDX, self.SHOULDER_IDX, self.HIP_IDX, self.ELBOW_IDX,
        )

    def _call(self, tracker, wrist_y_norm, **kwargs):
        """Call update_tracker with a synthetic landmark list."""
        update_tracker(
            tracker, self._landmarks_with_wrist_at(wrist_y_norm),
            self.WRIST_IDX, self.SHOULDER_IDX, self.HIP_IDX, self.ELBOW_IDX,
            self.FRAME_W, self.FRAME_H, **kwargs,
        )

    def test_wrist_overhead_transitions_to_rising(self):
        """Overhead wrist position transitions tracker from BOTTOM to RISING."""
        # wrist at y=0.05 (pixel 5) → norm_y = (5-25)/50 = -0.4 → overhead
        tracker = SideTracker()
        tracker.needs_init = False
        self._call(tracker, 0.05)
        assert tracker.state == RISING

    def test_wrist_at_hip_level_stays_bottom(self):
        """Wrist at hip level keeps tracker in BOTTOM."""
        # wrist at hip level → norm_y ≈ 1.0 → stays BOTTOM
        tracker = SideTracker()
        self._call(tracker, self.HIP_Y_NORM)
        assert tracker.state == BOTTOM

    def test_norm_y_is_zero_when_wrist_at_shoulder(self):
        """Wrist at shoulder level produces norm_y == 0.0."""
        tracker = SideTracker()
        self._call(tracker, self.SHOULDER_Y_NORM)
        assert tracker.norm_y == pytest.approx(0.0, abs=1e-6)

    def test_full_rep_via_update_tracker(self):
        """
        Drive a full rep through update_tracker using landmark positions.

        update_tracker applies a SMOOTH_WINDOW-length rolling average, so raw
        norm_y values are not seen instantly — the buffer must be saturated first.
        The sequence below is long enough that the smoothed value crosses each
        threshold cleanly and the wrist stays overhead long enough to satisfy
        MIN_TOP_FRAMES (verified by hand-tracing the deque averages).

        wrist positions (normalised screen coords) used:
          0.90 → pixel 90, norm_y = (90-25)/50 = 1.3  (below hip)
          0.05 → pixel  5, norm_y = (5-25)/50  = -0.4 (overhead)
          0.55 → pixel 55, norm_y = (55-25)/50 =  0.6 (between thresholds)
        """
        tracker = SideTracker()
        wrist_y_norm_sequence = (
            [0.90] * SMOOTH_WINDOW +        # fill buffer with below-hip → stays BOTTOM
            [0.05] * (SMOOTH_WINDOW + 2) +  # buffer saturates overhead  → RISING → TOP
            [0.55] * (SMOOTH_WINDOW + 1) +  # buffer saturates mid-drop  → FALLING
            [0.90] * (SMOOTH_WINDOW // 2 + 2)  # buffer tips past DROP_THRESHOLD → BOTTOM +1
        )
        for wrist_y_norm in wrist_y_norm_sequence:
            self._call(tracker, wrist_y_norm)

        assert tracker.rep_count == 1
        assert tracker.state == BOTTOM


# ── needs_init guard ─────────────────────────────────────────────────────────

class TestNeedsInit:
    """Tracker must see a confirmed bottom (norm_y > DROP_THRESHOLD) before
    the first rep cycle can start, preventing phantom reps at video start."""

    def test_overhead_at_start_does_not_trigger_rising(self):
        tracker = SideTracker()
        transition(tracker, OVERHEAD_NORM_Y)
        assert tracker.state == BOTTOM

    def test_initialises_on_first_below_hip_frame(self):
        tracker = SideTracker()
        transition(tracker, BELOW_HIP_NORM_Y)
        assert tracker.needs_init == False

    def test_overhead_after_init_triggers_rising(self):
        tracker = SideTracker()
        transition(tracker, BELOW_HIP_NORM_Y)   # clears needs_init
        transition(tracker, OVERHEAD_NORM_Y)
        assert tracker.state == RISING

    def test_mid_position_at_start_does_not_init(self):
        # norm_y between thresholds is not a confirmed bottom
        tracker = SideTracker()
        transition(tracker, MID_NORM_Y)
        assert tracker.needs_init is True
        assert tracker.state == BOTTOM


# ── Independent double-mode trackers ─────────────────────────────────────────

class TestDoubleMode:
    def test_trackers_are_independent(self):
        """Right and left SideTrackers must not share state."""
        right_tracker = SideTracker()
        left_tracker  = SideTracker()

        for norm_y in ONE_REP:
            transition(right_tracker, norm_y)

        assert right_tracker.rep_count == 1
        assert left_tracker.rep_count  == 0
        assert left_tracker.state      == BOTTOM

    def test_different_rep_counts_per_side(self):
        right_tracker = SideTracker()
        left_tracker  = SideTracker()

        one_rep_with_drain = ONE_REP + LOCKOUT_DRAIN

        for norm_y in (one_rep_with_drain * 2 + ONE_REP):
            transition(right_tracker, norm_y)

        for norm_y in (one_rep_with_drain * 4 + ONE_REP):
            transition(left_tracker, norm_y)

        assert right_tracker.rep_count == 3
        assert left_tracker.rep_count  == 5
        assert right_tracker.rep_count + left_tracker.rep_count == 8


# ── derive_thresholds_from_samples() ─────────────────────────────────────────

class TestDeriveThresholds:
    def test_typical_range_derives_correct_fractions(self):
        # obs_min=-0.2, obs_max=1.2 → range=1.4 (> CALIB_MIN_RANGE=0.5)
        samples   = [-0.2, 1.2]
        obs_range = 1.2 - (-0.2)
        rise, drop = derive_thresholds_from_samples(samples)
        assert rise == pytest.approx(-0.2 + obs_range * CALIB_RISE_FRAC)
        assert drop == pytest.approx(-0.2 + obs_range * CALIB_DROP_FRAC)

    def test_narrow_range_returns_defaults(self):
        # range = 0.3 < CALIB_MIN_RANGE → hardcoded defaults
        samples = [0.5, 0.6, 0.7, 0.8]
        rise, drop = derive_thresholds_from_samples(samples)
        assert rise == RISE_THRESHOLD
        assert drop == DROP_THRESHOLD

    def test_empty_samples_raises_value_error(self):
        with pytest.raises(ValueError):
            derive_thresholds_from_samples([])

    def test_exact_boundary_range_derives(self):
        # range == CALIB_MIN_RANGE exactly → NOT < threshold, so derived (guard is <)
        samples = [0.0, CALIB_MIN_RANGE]
        rise, drop = derive_thresholds_from_samples(samples)
        assert rise == pytest.approx(0.0 + CALIB_MIN_RANGE * CALIB_RISE_FRAC)
        assert drop == pytest.approx(0.0 + CALIB_MIN_RANGE * CALIB_DROP_FRAC)

    def test_single_sample_returns_defaults(self):
        # range == 0 < CALIB_MIN_RANGE → defaults
        rise, drop = derive_thresholds_from_samples([0.7])
        assert rise == RISE_THRESHOLD
        assert drop == DROP_THRESHOLD


# ── Calibrator ────────────────────────────────────────────────────────────────

class TestCalibrator:
    def test_progress_at_start(self):
        c = Calibrator(n_frames=10)
        assert c.progress == pytest.approx(0.0)

    def test_progress_at_midpoint(self):
        c = Calibrator(n_frames=10)
        for _ in range(5):
            c.update(0.5)
        assert c.progress == pytest.approx(0.5)

    def test_progress_at_end(self):
        c = Calibrator(n_frames=10)
        for _ in range(10):
            c.update(0.5)
        assert c.progress == pytest.approx(1.0)

    def test_done_flag_after_n_frames(self):
        c = Calibrator(n_frames=5)
        for _ in range(5):
            c.update(0.5)
        assert c.done is True

    def test_extra_update_ignored_after_done(self):
        c = Calibrator(n_frames=3)
        for _ in range(3):
            c.update(0.5)
        c.update(9999.0)   # must be ignored
        assert c.frames_collected == 3

    def test_derive_thresholds_delegates_correctly(self):
        # Feed a wide range so derived thresholds (not defaults) are returned
        samples = [-0.5, -0.3, 0.9, 1.0, 1.2]
        c = Calibrator(n_frames=len(samples))
        for s in samples:
            c.update(s)
        rise, drop = c.derive_thresholds()
        obs_range     = max(samples) - min(samples)
        expected_rise = min(samples) + obs_range * CALIB_RISE_FRAC
        expected_drop = min(samples) + obs_range * CALIB_DROP_FRAC
        assert rise == pytest.approx(expected_rise)
        assert drop == pytest.approx(expected_drop)


# ── save_calibration / load_calibration ───────────────────────────────────────

class TestCalibrationJSON:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "calib.json"
        save_calibration(0.3, 0.8, path=path)
        result = load_calibration(path=path)
        assert result == pytest.approx((0.3, 0.8))

    def test_missing_file_returns_none(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert load_calibration(path=path) is None

    def test_malformed_json_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        assert load_calibration(path=path) is None

    def test_missing_keys_returns_none(self, tmp_path):
        path = tmp_path / "partial.json"
        path.write_text('{"rise_threshold": 0.3}')   # drop_threshold missing
        assert load_calibration(path=path) is None


# ── SideTracker with custom thresholds ────────────────────────────────────────

class TestSideTrackerWithCustomThresholds:
    def test_custom_rise_threshold_fires_at_higher_norm_y(self):
        # rise_threshold=0.6: norm_y=0.5 is < 0.6, so RISING fires
        # (with default 0.4 it would NOT fire since 0.5 > 0.4)
        tracker = SideTracker(rise_threshold=0.6, drop_threshold=0.85)
        tracker.needs_init = False
        transition(tracker, 0.5)
        assert tracker.state == RISING

    def test_default_rise_threshold_does_not_fire_at_0_5(self):
        # Default rise_threshold=0.4: norm_y=0.5 > 0.4, so no transition
        tracker = SideTracker()
        tracker.needs_init = False
        transition(tracker, 0.5)
        assert tracker.state == BOTTOM

    def test_default_tracker_uses_module_constants(self):
        tracker = SideTracker()
        assert tracker.rise_threshold == RISE_THRESHOLD
        assert tracker.drop_threshold == DROP_THRESHOLD


# ── Switch mode simulation + tests ───────────────────────────────────────────

def simulate_switch(norm_y_sequence, min_top_frames=2):
    """
    Replay the switch-mode inner loop without OpenCV/MediaPipe.

    norm_y_sequence: list of (right_norm_y, left_norm_y) per-frame tuples.
    Returns (right_reps, left_reps, final_switch_side).
    """
    right_t = SideTracker(min_top_frames=min_top_frames)
    left_t  = SideTracker(min_top_frames=min_top_frames)
    right_t.needs_init = False   # right side starts ready
    left_t.needs_init  = True    # left must descend to hip before counting

    switch_side    = 'right'
    switch_lockout = 0

    for right_ny, left_ny in norm_y_sequence:
        if switch_side == 'right':
            active_t, inactive_t = right_t, left_t
            active_ny, inactive_ny = right_ny, left_ny
        else:
            active_t, inactive_t = left_t, right_t
            active_ny, inactive_ny = left_ny, right_ny

        inactive_t.norm_y = inactive_ny
        active_t.norm_y   = active_ny   # needed for switch condition

        if switch_lockout > 0:
            switch_lockout -= 1
        elif (active_t.state == BOTTOM
                and active_t.norm_y > active_t.drop_threshold
                and inactive_t.norm_y < active_t.rise_threshold):
            inactive_t.needs_init = False
            inactive_t.state      = TOP
            inactive_t.top_frames = inactive_t.min_top_frames  # pre-filled
            inactive_t.lockout    = 0
            active_t.state        = BOTTOM
            active_t.top_frames   = 0
            switch_side    = 'left' if switch_side == 'right' else 'right'
            switch_lockout = 30
            if switch_side == 'right':
                active_t, active_ny = right_t, right_ny
            else:
                active_t, active_ny = left_t, left_ny

        transition(active_t, active_ny)

    return right_t.rep_count, left_t.rep_count, switch_side


# norm_y shorthands for switch mode tests
_HIP      = DROP_THRESHOLD + 0.3    # 1.15 — wrist clearly at/below hip
_OVERHEAD = RISE_THRESHOLD - 0.25   # 0.15 — wrist clearly overhead
_MID      = (RISE_THRESHOLD + DROP_THRESHOLD) / 2  # between thresholds

_PAUSE = [(_HIP, _HIP)] * (MIN_REP_LOCKOUT + 5)   # inter-rep gap

def _right_rep(n_top=2):
    """Frames for a single right-hand rep (left wrist resting at hip)."""
    return (
        [(_OVERHEAD, _HIP)] +            # BOTTOM → RISING
        [(_OVERHEAD, _HIP)] +            # RISING → TOP
        [(_OVERHEAD, _HIP)] * n_top +    # hold overhead (top_frames)
        [(_MID,      _HIP)] +            # TOP → FALLING
        [(_HIP,      _HIP)]              # FALLING → BOTTOM, rep++
    )

def _left_rep(n_top=2):
    """Frames for a single left-hand rep (right wrist resting at hip)."""
    return (
        [(_HIP, _OVERHEAD)] +
        [(_HIP, _OVERHEAD)] +
        [(_HIP, _OVERHEAD)] * n_top +
        [(_HIP, _MID)] +
        [(_HIP, _HIP)]
    )

def _left_init():
    """Left wrist descends to hip after switch → clears needs_init."""
    return [(_HIP, _HIP)] * 3


class TestSwitchMode:

    def test_single_right_rep_no_switch(self):
        frames = list(_PAUSE) + _right_rep()
        r, l, side = simulate_switch(frames)
        assert r == 1 and l == 0 and side == 'right'

    def test_three_right_reps_no_switch(self):
        frames = (list(_PAUSE) + _right_rep()
                  + list(_PAUSE) + _right_rep()
                  + list(_PAUSE) + _right_rep())
        r, l, side = simulate_switch(frames)
        assert r == 3 and l == 0 and side == 'right'

    def test_switch_moment_itself_does_not_count(self):
        """The switch fires at the transition frame only — it alone must not count."""
        switch_trigger = [(_HIP, _OVERHEAD)]  # active at hip, inactive overhead
        # One switch frame: the switch fires but no descent yet → l still 0
        frames = list(_PAUSE) + _right_rep() + switch_trigger
        r, l, _ = simulate_switch(frames)
        assert l == 0, f"Switch frame alone produced a left rep: l={l}"
        assert r == 1

    def test_first_descent_after_switch_counts_as_rep(self):
        """After a switch the incoming wrist is overhead; lowering to rack = rep."""
        switch_trigger = [(_HIP, _OVERHEAD)]
        frames = list(_PAUSE) + _right_rep() + switch_trigger + _left_init()
        r, l, _ = simulate_switch(frames)
        assert l == 1, f"First descent after switch should count as a rep, got l={l}"
        assert r == 1

    def test_one_right_two_left_after_switch(self):
        """First descent after switch = left rep 1; explicit jerk = left rep 2."""
        switch_trigger = [(_HIP, _OVERHEAD)]
        frames = (
            list(_PAUSE) + _right_rep() +
            switch_trigger + _left_init() +     # first descent = l rep 1
            list(_PAUSE) + _left_rep()          # explicit jerk = l rep 2
        )
        r, l, side = simulate_switch(frames)
        assert r == 1 and l == 2 and side == 'left'

    def test_three_right_two_left(self):
        switch_trigger = [(_HIP, _OVERHEAD)]
        frames = (
            list(_PAUSE) +
            _right_rep() + list(_PAUSE) +
            _right_rep() + list(_PAUSE) +
            _right_rep() +
            switch_trigger + _left_init() +     # first descent = l rep 1
            list(_PAUSE) + _left_rep()          # explicit jerk  = l rep 2
        )
        r, l, _ = simulate_switch(frames)
        assert r == 3 and l == 2

    def test_switch_blocked_during_rising(self):
        """Switch must not fire while active wrist is in RISING state."""
        frames = list(_PAUSE) + [
            (_OVERHEAD, _HIP),        # right: BOTTOM → RISING
            (_OVERHEAD, _OVERHEAD),   # right: RISING, left overhead — no switch
        ]
        _r, _l, side = simulate_switch(frames)
        assert side == 'right', "Switch fired during RISING"

    def test_switch_blocked_during_top(self):
        """Switch must not fire while active wrist is in TOP state."""
        frames = list(_PAUSE) + [
            (_OVERHEAD, _HIP),        # RISING
            (_OVERHEAD, _HIP),        # TOP
            (_OVERHEAD, _OVERHEAD),   # TOP + left overhead — no switch
        ]
        _r, _l, side = simulate_switch(frames)
        assert side == 'right', "Switch fired during TOP"

    def test_switch_blocked_when_active_wrist_mid_air(self):
        """Switch must NOT fire when active wrist is between thresholds."""
        frames = list(_PAUSE) + [(_MID, _OVERHEAD)] * 5
        _r, _l, side = simulate_switch(frames)
        assert side == 'right', "Switch fired while active wrist was not at hip"

    def test_switch_lockout_prevents_ping_pong(self):
        """After a switch, old wrist still overhead must not re-trigger a switch."""
        switch_trigger = [(_HIP, _OVERHEAD)]
        old_wrist_lingers = [(_OVERHEAD, _OVERHEAD)] * 25
        frames = list(_PAUSE) + _right_rep() + switch_trigger + old_wrist_lingers
        _r, _l, side = simulate_switch(frames)
        assert side == 'left', "Ping-pong switch: reverted to right during lockout"

    def test_rep_counted_after_switch_not_before(self):
        """Reps after a switch go to the new active hand's counter, not the old one's."""
        switch_trigger = [(_HIP, _OVERHEAD)]
        frames = (
            list(_PAUSE) + _right_rep() +
            switch_trigger + _left_init()   # first descent = l rep 1
        )
        r, l, _ = simulate_switch(frames)
        assert l == 1           # credited to left, not right
        assert r == 1           # right unchanged

    def test_multiple_switches_n_m_reps(self):
        """N reps right → switch → M reps left → switch back pattern.

        Each switch adds one rep on the incoming side (first descent from overhead).
        2R + switch(→L1) + 1L + switch(→R1) = 3R total, 2L total.
        """
        sw_r_to_l = [(_HIP, _OVERHEAD)]   # right at hip, left overhead → switch to left
        sw_l_to_r = [(_OVERHEAD, _HIP)]   # right overhead, left at hip → switch to right
        frames = (
            list(_PAUSE) +
            _right_rep() + list(_PAUSE) + _right_rep() +        # 2 right reps
            sw_r_to_l + _left_init() + list(_PAUSE) +           # switch; first descent = l rep 1
            _left_rep() + list(_PAUSE) +                        # explicit jerk = l rep 2; drain lockout
            sw_l_to_r + [(_HIP, _HIP)] * 3                     # switch back; first descent = r rep 3
        )
        r, l, _ = simulate_switch(frames)
        assert r == 3 and l == 2
