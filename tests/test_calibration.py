"""Tests for calibration.py — threshold derivation, Calibrator class, I/O."""

import json
import pathlib
import tempfile

import pytest

from constants import RISE_THRESHOLD, DROP_THRESHOLD, CALIB_FRAMES
from calibration import (
    Calibrator,
    derive_thresholds_from_samples,
    save_calibration,
    load_calibration,
)


# ── derive_thresholds_from_samples ────────────────────────────────────────────

def test_derive_raises_on_empty():
    with pytest.raises(ValueError):
        derive_thresholds_from_samples([])


def test_derive_falls_back_when_range_too_narrow():
    # All samples identical → range = 0 < CALIB_MIN_RANGE → fallback
    samples = [0.5] * 10
    rise, drop = derive_thresholds_from_samples(samples)
    assert rise == RISE_THRESHOLD
    assert drop == DROP_THRESHOLD


def test_derive_produces_ordered_thresholds():
    # Full swing: some below-shoulder, some below-hip
    samples = [i / 100.0 for i in range(-20, 130, 5)]  # -0.20 … 1.25
    rise, drop = derive_thresholds_from_samples(samples)
    assert rise < drop


def test_derive_fallback_when_drop_lte_rise():
    # Force a fringe case: calib_rise_frac > calib_drop_frac
    samples = [0.0, 1.5]
    rise, drop = derive_thresholds_from_samples(
        samples, calib_min_range=0.5, calib_rise_frac=0.8, calib_drop_frac=0.2
    )
    # drop(0.2) < rise(0.8) → fallback
    assert rise == RISE_THRESHOLD
    assert drop == DROP_THRESHOLD


# ── Calibrator class ──────────────────────────────────────────────────────────

def test_calibrator_collects_samples():
    cal = Calibrator(n_frames=5)
    for v in [0.1, 0.2, 0.3]:
        cal.update(v)
    assert cal.frames_collected == 3
    assert not cal.done


def test_calibrator_marks_done_at_n_frames():
    cal = Calibrator(n_frames=3)
    for v in [0.1, 0.9, 0.5]:
        cal.update(v)
    assert cal.done


def test_calibrator_ignores_updates_after_done():
    cal = Calibrator(n_frames=2)
    cal.update(0.1)
    cal.update(0.9)
    assert cal.done
    cal.update(0.5)  # should be ignored
    assert cal.frames_collected == 2


def test_calibrator_progress():
    cal = Calibrator(n_frames=10)
    for _ in range(4):
        cal.update(0.5)
    assert cal.progress == pytest.approx(0.4)


def test_calibrator_derive_thresholds():
    cal = Calibrator(n_frames=10)
    for v in [i / 10 for i in range(10)]:
        cal.update(v)
    rise, drop = cal.derive_thresholds()
    assert rise < drop


# ── save_calibration / load_calibration ───────────────────────────────────────

def test_save_and_load_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = pathlib.Path(f.name)
    try:
        save_calibration(0.12, 0.78, path)
        result = load_calibration(path)
        assert result == pytest.approx((0.12, 0.78))
    finally:
        path.unlink(missing_ok=True)


def test_load_returns_none_on_missing_file():
    result = load_calibration(pathlib.Path("/tmp/nonexistent_kb_calib_xyz.json"))
    assert result is None


def test_load_returns_none_on_malformed_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, encoding="utf-8") as f:
        f.write("not valid json {{{")
        path = pathlib.Path(f.name)
    try:
        assert load_calibration(path) is None
    finally:
        path.unlink(missing_ok=True)


def test_load_returns_none_on_missing_keys():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, encoding="utf-8") as f:
        json.dump({"foo": 1}, f)
        path = pathlib.Path(f.name)
    try:
        assert load_calibration(path) is None
    finally:
        path.unlink(missing_ok=True)
