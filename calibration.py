"""Calibration: threshold derivation, Calibrator class, JSON persistence."""

import json
import pathlib

from constants import (
    RISE_THRESHOLD, DROP_THRESHOLD,
    CALIB_FRAMES, CALIB_MIN_RANGE, CALIB_RISE_FRAC, CALIB_DROP_FRAC,
    CALIB_FILE,
)


class Calibrator:
    """Collects norm_y samples over a fixed window and derives per-session thresholds."""

    def __init__(self, n_frames: int = CALIB_FRAMES):
        self._n_frames = n_frames
        self._samples: list[float] = []
        self.done = False

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
    def samples(self) -> list:
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

    def derive_thresholds(self) -> tuple[float, float]:
        """Derive rise/drop thresholds from collected samples."""
        return derive_thresholds_from_samples(self._samples)


def derive_thresholds_from_samples(
    samples,
    calib_min_range: float = CALIB_MIN_RANGE,
    calib_rise_frac: float = CALIB_RISE_FRAC,
    calib_drop_frac: float = CALIB_DROP_FRAC,
) -> tuple[float, float]:
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
        print(
            f"Warning: calibrated rise={rise:.3f} > 0.1; "
            "wrist may not have cleared shoulder during calibration"
        )
    return rise, drop


def save_calibration(
    rise: float, drop: float, path: pathlib.Path = CALIB_FILE
) -> None:
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
