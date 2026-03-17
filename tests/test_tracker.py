"""Tests for tracker.py — SideTracker state machine and rep counting.

Key invariant (new two-stage design):
  - TOP state: set _counted_this_cycle=True once min_top_frames are accumulated.
    rep_count is NOT incremented yet.
  - FALLING→BOTTOM transition: rep_count is incremented and lockout is applied
    only when _counted_this_cycle is True.
"""

import pytest
from collections import deque

from constants import (
    BOTTOM, RISING, TOP, FALLING,
    RISE_THRESHOLD, DROP_THRESHOLD,
    SMOOTH_WINDOW, MIN_REP_LOCKOUT,
)
from tracker import SideTracker

# Shorthand values
HIGH = RISE_THRESHOLD - 0.3     # clearly overhead
LOW  = DROP_THRESHOLD + 0.2     # clearly at bottom


# ── Helpers ───────────────────────────────────────────────────────────────────

def fresh(min_top_frames: int = 2) -> SideTracker:
    return SideTracker(
        rise_threshold=RISE_THRESHOLD,
        drop_threshold=DROP_THRESHOLD,
        min_top_frames=min_top_frames,
    )


def init_tracker(t: SideTracker) -> None:
    """Send enough LOW frames to satisfy needs_init."""
    for _ in range(SMOOTH_WINDOW):
        t.transition(LOW)
    assert not t.needs_init
    assert t.state == BOTTOM


# ── Initialisation ────────────────────────────────────────────────────────────

def test_needs_init_true_on_creation():
    t = fresh()
    assert t.needs_init is True


def test_needs_init_cleared_after_low_frame():
    t = fresh()
    t.transition(LOW)
    assert not t.needs_init


def test_no_rise_while_needs_init():
    t = fresh()
    # HIGH frame before seeing bottom: must not transition to RISING
    t.transition(HIGH)
    assert t.state == BOTTOM


# ── BOTTOM → RISING ───────────────────────────────────────────────────────────

def test_bottom_to_rising_on_high_frame():
    t = fresh()
    init_tracker(t)
    t.transition(HIGH)
    assert t.state == RISING


def test_rising_to_top_when_still_overhead():
    t = fresh()
    init_tracker(t)
    t.transition(HIGH)   # BOTTOM→RISING
    t.transition(HIGH)   # still overhead → should enter TOP
    assert t.state == TOP


def test_rising_aborts_to_bottom_on_low_frame():
    t = fresh()
    init_tracker(t)
    t.transition(HIGH)   # BOTTOM→RISING
    t.transition(LOW)    # dropped back without reaching overhead
    assert t.state == BOTTOM


# ── TOP / two-stage counting ──────────────────────────────────────────────────

def test_top_flag_set_after_min_top_frames():
    """_counted_this_cycle becomes True after min_top_frames overhead frames."""
    t = fresh(min_top_frames=2)
    init_tracker(t)
    t.transition(HIGH)   # BOTTOM→RISING
    t.transition(HIGH)   # RISING→TOP (top_frames=0)
    # Need min_top_frames=2 overhead frames in TOP
    t.transition(HIGH)   # top_frames=1, not yet validated
    assert not t._counted_this_cycle
    assert t.rep_count == 0
    t.transition(HIGH)   # top_frames=2, validated!
    assert t._counted_this_cycle is True
    assert t.rep_count == 0   # ← count has NOT fired yet


def test_rep_count_zero_while_falling():
    """rep_count stays 0 while state is FALLING (count fires only at BOTTOM)."""
    t = fresh(min_top_frames=2)
    init_tracker(t)
    t.transition(HIGH)   # →RISING
    t.transition(HIGH)   # →TOP
    t.transition(HIGH)   # top_frames=1
    t.transition(HIGH)   # top_frames=2, _counted_this_cycle=True
    t.transition(LOW)    # →FALLING (wrist descending)
    assert t.state == FALLING
    assert t.rep_count == 0


def test_rep_counted_at_falling_to_bottom():
    """rep_count increments exactly when FALLING→BOTTOM transition fires."""
    t = fresh(min_top_frames=2)
    init_tracker(t)
    t.transition(HIGH)
    t.transition(HIGH)   # →TOP
    t.transition(HIGH)   # top_frames=1
    t.transition(HIGH)   # top_frames=2, validated
    t.transition(LOW)    # →FALLING
    assert t.rep_count == 0
    t.transition(LOW)    # FALLING→BOTTOM  ← count fires here
    assert t.rep_count == 1
    assert t.state == BOTTOM


def test_lockout_applied_at_bottom_after_rep():
    """Lockout starts when the wrist returns to BOTTOM, not at TOP."""
    t = fresh(min_top_frames=2)
    init_tracker(t)
    t.transition(HIGH)
    t.transition(HIGH)
    t.transition(HIGH)
    t.transition(HIGH)   # validated
    t.transition(LOW)    # →FALLING
    t.transition(LOW)    # →BOTTOM, rep counted, lockout set
    assert t.lockout == MIN_REP_LOCKOUT


def test_no_lockout_without_validation():
    """If the wrist never validated (swing-through), lockout must not be applied."""
    t = fresh(min_top_frames=5)
    init_tracker(t)
    t.transition(HIGH)   # →RISING
    t.transition(HIGH)   # →TOP (top_frames=0)
    t.transition(HIGH)   # top_frames=1 — not validated yet
    t.transition(LOW)    # not enough frames → swing-through, →BOTTOM
    assert t.state == BOTTOM
    assert t.rep_count == 0
    assert t.lockout == 0


def test_swing_through_resets_counted_flag():
    t = fresh(min_top_frames=5)
    init_tracker(t)
    t.transition(HIGH)
    t.transition(HIGH)   # →TOP
    t.transition(HIGH)   # top_frames=1
    t.transition(LOW)    # swing-through
    assert t._counted_this_cycle is False


# ── Lockout guard ─────────────────────────────────────────────────────────────

def test_lockout_prevents_immediate_new_rep():
    t = fresh(min_top_frames=2)
    init_tracker(t)
    # Full rep cycle
    t.transition(HIGH); t.transition(HIGH)
    t.transition(HIGH); t.transition(HIGH)   # validated
    t.transition(LOW);  t.transition(LOW)    # →BOTTOM, lockout starts
    assert t.lockout > 0
    # Try to start new rep: should be blocked
    t.transition(HIGH)
    assert t.state == BOTTOM


def test_second_rep_counted_after_lockout_expires():
    t = fresh(min_top_frames=2)
    init_tracker(t)
    # First rep
    t.transition(HIGH); t.transition(HIGH)
    t.transition(HIGH); t.transition(HIGH)   # validated
    t.transition(LOW);  t.transition(LOW)    # →BOTTOM, lockout=MIN_REP_LOCKOUT
    # Drain lockout
    for _ in range(MIN_REP_LOCKOUT):
        t.transition(LOW)
    # Second rep
    t.transition(HIGH); t.transition(HIGH)
    t.transition(HIGH); t.transition(HIGH)
    t.transition(LOW);  t.transition(LOW)
    assert t.rep_count == 2


# ── count_top_frame gating (elbow lock) ───────────────────────────────────────

def test_top_frames_not_counted_when_gated():
    """count_top_frame=False must block accumulation of top_frames."""
    t = fresh(min_top_frames=2)
    init_tracker(t)
    t.transition(HIGH)   # →RISING
    t.transition(HIGH)   # →TOP
    t.transition(HIGH, count_top_frame=False)  # overhead but gated
    t.transition(HIGH, count_top_frame=False)
    assert t.top_frames == 0
    assert not t._counted_this_cycle


# ── allow_rise gating (knee lock) ────────────────────────────────────────────

def test_allow_rise_false_blocks_rising():
    t = fresh()
    init_tracker(t)
    t.transition(HIGH, allow_rise=False)
    assert t.state == BOTTOM


def test_allow_rise_true_permits_rising():
    t = fresh()
    init_tracker(t)
    t.transition(HIGH, allow_rise=True)
    assert t.state == RISING


# ── Full rep via SideTracker.update() ────────────────────────────────────────

class _FakeLM:
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x = x; self.y = y; self.z = z


def make_lms(wrist_y: float, shoulder_y: float = 0.3,
             hip_y: float = 0.6, elbow_y: float = 0.4) -> list:
    lms = [_FakeLM() for _ in range(33)]
    lms[12].y = shoulder_y   # RIGHT_SHOULDER
    lms[14].y = elbow_y      # RIGHT_ELBOW
    lms[16].y = wrist_y      # RIGHT_WRIST
    lms[24].y = hip_y        # RIGHT_HIP
    return lms


def test_full_rep_via_update():
    """End-to-end: update() drives a complete rep, count fires at BOTTOM."""
    t = SideTracker(min_top_frames=2)
    frame_w, frame_h = 640, 480
    # shoulder_y=0.3, hip_y=0.6 → torso span = 0.3 in norm coords
    # Overhead: wrist_y=0.15 → norm_y = (0.15-0.3)/(0.6-0.3) = -0.5  (< RISE_THRESHOLD)
    # Low:      wrist_y=0.75 → norm_y = (0.75-0.3)/0.3 = 1.5  (> DROP_THRESHOLD)

    # Initialise: send SMOOTH_WINDOW low frames
    for _ in range(SMOOTH_WINDOW):
        t.update(make_lms(0.75), 16, 12, 24, 14, frame_w, frame_h)
    assert not t.needs_init

    # Rise to overhead until we enter TOP
    for _ in range(SMOOTH_WINDOW):
        t.update(make_lms(0.15), 16, 12, 24, 14, frame_w, frame_h)

    assert t.state == TOP

    # Hold overhead enough to validate (min_top_frames=2, but we overshoot)
    for _ in range(4):
        t.update(make_lms(0.15), 16, 12, 24, 14, frame_w, frame_h)
    assert t._counted_this_cycle is True
    assert t.rep_count == 0   # not yet — bell is still overhead

    # Drop back to low — must go through FALLING then BOTTOM
    for _ in range(SMOOTH_WINDOW):
        t.update(make_lms(0.75), 16, 12, 24, 14, frame_w, frame_h)

    assert t.rep_count == 1
    assert t.state == BOTTOM
