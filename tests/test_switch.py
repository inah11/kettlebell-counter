"""Tests for switch-mode (hand-to-hand) logic.

The switch detection lives in kb_counter.main() and is tested here by
calling the same condition logic directly, so we don't need a video file.
"""

import pytest

from constants import BOTTOM, TOP, RISE_THRESHOLD, DROP_THRESHOLD, MIN_REP_LOCKOUT
from tracker import SideTracker


# ── Helper ────────────────────────────────────────────────────────────────────

def make_tracker(**kwargs) -> SideTracker:
    defaults = dict(rise_threshold=RISE_THRESHOLD, drop_threshold=DROP_THRESHOLD,
                    min_top_frames=2)
    defaults.update(kwargs)
    return SideTracker(**defaults)


def feed(t: SideTracker, norm_y: float, n: int = 1) -> None:
    for _ in range(n):
        t.transition(norm_y)


HIGH = RISE_THRESHOLD - 0.3
LOW  = DROP_THRESHOLD + 0.2


def init(t: SideTracker) -> None:
    feed(t, LOW, 5)
    assert not t.needs_init


# ── Switch condition predicate ────────────────────────────────────────────────

def _should_switch(active: SideTracker, inactive: SideTracker,
                   switch_lockout: int) -> bool:
    """Mirrors the switch-detection condition in kb_counter.main()."""
    return (switch_lockout == 0
            and active.state == BOTTOM
            and active.norm_y > active.drop_threshold
            and inactive.norm_y < active.rise_threshold)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_switch_when_active_not_at_bottom():
    active   = make_tracker(); init(active)
    inactive = make_tracker(); init(inactive)
    # Active in TOP, inactive is overhead
    active.state = TOP
    active.norm_y = HIGH
    inactive.norm_y = HIGH
    assert not _should_switch(active, inactive, 0)


def test_no_switch_when_inactive_not_overhead():
    active   = make_tracker(); init(active)
    inactive = make_tracker(); init(inactive)
    active.state   = BOTTOM
    active.norm_y  = LOW
    inactive.norm_y = LOW   # inactive is also low → no switch
    assert not _should_switch(active, inactive, 0)


def test_switch_fires_when_active_low_and_inactive_overhead():
    active   = make_tracker(); init(active)
    inactive = make_tracker(); init(inactive)
    active.state   = BOTTOM
    active.norm_y  = LOW
    inactive.norm_y = HIGH
    assert _should_switch(active, inactive, 0)


def test_no_switch_during_lockout():
    active   = make_tracker(); init(active)
    inactive = make_tracker(); init(inactive)
    active.state   = BOTTOM
    active.norm_y  = LOW
    inactive.norm_y = HIGH
    assert not _should_switch(active, inactive, switch_lockout=5)


def test_switch_preloads_incoming_tracker():
    """After a switch the incoming tracker must be in TOP with validated top_frames."""
    active   = make_tracker(); init(active)
    inactive = make_tracker(); init(inactive)
    active.state   = BOTTOM
    active.norm_y  = LOW
    inactive.norm_y = HIGH

    # Simulate the switch action from kb_counter.main()
    inactive.needs_init          = False
    inactive.state               = TOP
    inactive.top_frames          = inactive.min_top_frames
    inactive.lockout             = 0
    inactive._counted_this_cycle = False
    inactive.smooth_buf.clear()
    active.state      = BOTTOM
    active.top_frames = 0

    assert inactive.state == TOP
    assert inactive.top_frames == inactive.min_top_frames
    assert not inactive.needs_init
    assert not inactive._counted_this_cycle


def test_switch_preloaded_tracker_can_count_on_descent():
    """After preloading, descending to BOTTOM commits the rep."""
    t = make_tracker(min_top_frames=2)
    # Simulate the preloaded state
    t.needs_init          = False
    t.state               = TOP
    t.top_frames          = t.min_top_frames   # already validated
    t.lockout             = 0
    t._counted_this_cycle = False

    # One overhead frame triggers the validation flag
    t.transition(HIGH)
    assert t._counted_this_cycle is True
    assert t.rep_count == 0

    # Descend to low → count should fire
    t.transition(LOW)   # TOP→FALLING
    t.transition(LOW)   # FALLING→BOTTOM
    assert t.rep_count == 1


def test_initial_hand_auto_detect():
    """
    If the KB starts in the 'wrong' hand (inactive overhead, active at hip),
    the switch fires on the very first frame — same logic as mid-set switch.
    """
    # Imagine switch_side='right' but KB is in left hand
    right = make_tracker(); init(right)
    left  = make_tracker()   # left has not been initialised

    # Right (active) is at bottom, left (inactive) is overhead
    right.state   = BOTTOM
    right.norm_y  = LOW
    left.norm_y   = HIGH

    # The switch condition ignores needs_init on the active side,
    # so it should still fire
    assert _should_switch(right, left, 0)
