# Kettlebell Rep Counter

A computer-vision rep counter for kettlebell exercises using [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) and OpenCV. Tracks wrist position relative to the body and counts reps through a state-machine — no wearable required.

Supports **single-hand snatch/jerk**, **double kettlebell**, and **hand-to-hand switch sets**.

---

## Features

- Counts reps in real time from a video file
- Three tracking modes: single wrist, both wrists independently, hand-to-hand pass
- Automatic calibration to your range of motion (or load saved calibration)
- Tunable thresholds for different movements (snatch vs. jerk vs. press)
- Elbow-lockout detection (optional gate for rep validity)
- Per-frame CSV logging for debugging
- HUD overlay: state, rep count, norm_y, elbow angle, active hand indicator

---

## Requirements

```
Python 3.9+
mediapipe
opencv-python
```

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install mediapipe opencv-python
```

---

## Quick Start

```bash
# Single kettlebell, right hand (default)
python snatch_counter.py video.mp4

# Double kettlebell (both hands, independent counts)
python snatch_counter.py video.mp4 --mode double

# Hand-to-hand switch set (combined count, active hand indicator)
python snatch_counter.py video.mp4 --mode switch
```

Press **q** to quit. Final rep count is printed to the terminal.

---

## All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `single` | `single` · `double` · `switch` |
| `--min-top-time SECS` | `0.5` | Seconds wrist must hold overhead to confirm top position |
| `--rise-threshold FLOAT` | `0.4` | norm_y below which wrist is "high". Lower = stricter overhead requirement |
| `--drop-threshold FLOAT` | `0.85` | norm_y above which wrist is "low". Lower = stricter bottom requirement |
| `--elbow-lock` | off | Require elbow lockout (≥ 160°) at the top for top_frames to accumulate |
| `--mirror` | off | Flip video horizontally (for selfie/front-facing camera footage) |
| `--log FILE` | — | Write per-frame CSV debug log to FILE |
| `--calibrate` | — | Dedicated calibration pass: swing KB for ~5 s, saves `calibration.json` |
| `--no-auto-calib` | — | Skip the auto-calibration window; load `calibration.json` if present |

---

## How It Works

### Normalised Wrist Y

Each frame, the wrist position is normalised relative to the torso:

```
norm_y = (wrist_y − shoulder_y) / (hip_y − shoulder_y)
```

| norm_y | Meaning |
|--------|---------|
| < 0 | Wrist above shoulder (overhead) |
| 0 | At shoulder level |
| 1 | At hip level |
| > 1 | Below hip (deep swing bottom) |

A rolling average over 5 frames smooths out jitter before the value enters the state machine.

### State Machine

```
BOTTOM ──(norm_y < rise_threshold)──► RISING
RISING ──(norm_y < rise_threshold)──► TOP
TOP    ──(hold for min_top_frames, then norm_y > rise_threshold)──► FALLING
FALLING──(norm_y > drop_threshold)──► BOTTOM  ← rep counted here
```

Anti-double-count guards:
- **min_top_frames** — wrist must stay overhead for `fps × --min-top-time` frames before the top is confirmed. Prevents swing-throughs from counting.
- **MIN_REP_LOCKOUT (15 frames)** — after each rep, the BOTTOM→RISING transition is blocked for ~0.5 s so the backswing can't immediately start a new cycle.

### Switch Mode (Hand-to-Hand)

Only the **active** wrist runs the full state machine. The **inactive** wrist is tracked for display and switch detection only.

A switch fires when:
1. Active wrist is at **BOTTOM** (between reps, at rack/hip position)
2. Active wrist is physically **low** (`norm_y > drop_threshold`)
3. Inactive wrist is **overhead** (`norm_y < rise_threshold`) — has the KB

On switch, the incoming tracker is placed directly into **TOP** state with `top_frames` pre-filled. This means the very first descent from overhead to rack counts as a rep immediately — no warm-up delay after the hand-off.

A **switch_lockout** of 30 frames (~1 s) prevents ping-pong re-switching.

---

## Tuning for Different Movements

### Snatch (default)
The KB swings from below the hip to overhead.

```bash
python snatch_counter.py video.mp4 --mode switch
```

Defaults (`rise=0.4`, `drop=0.85`) work well.

### Jerk / Press
The KB stays between rack (~shoulder) and overhead. It never reaches hip level, so the default `drop=0.85` is never triggered.

```bash
python snatch_counter.py video.mp4 --mode switch \
  --rise-threshold -0.3 \
  --drop-threshold 0.05 \
  --min-top-time 0.3
```

| Threshold | Jerk value | Why |
|-----------|-----------|-----|
| `--rise-threshold -0.3` | wrist must be well above shoulder | rack position (≈ 0.07) is above −0.3, so it never falsely triggers RISING |
| `--drop-threshold 0.05` | rack position triggers BOTTOM | rack is norm_y ≈ 0.04–0.12, safely above 0.05 |
| `--min-top-time 0.3` | shorter lockout at top | jerk fixation can be brief |

### Calibration

Run a dedicated calibration pass to derive thresholds from your actual range of motion:

```bash
python snatch_counter.py video.mp4 --calibrate
```

Swing the KB through the full range for ~5 seconds. Thresholds are saved to `calibration.json` and loaded automatically on subsequent runs. Override any saved value with `--rise-threshold` / `--drop-threshold`.

---

## Debugging with CSV Logs

```bash
python snatch_counter.py video.mp4 --mode switch --log debug.csv \
  --rise-threshold -0.3 --drop-threshold 0.05 --min-top-time 0.3
```

The CSV contains one row per frame:

| Column | Description |
|--------|-------------|
| `frame` | Frame number |
| `pose_detected` | 1 if MediaPipe found a pose |
| `active_side` | `right` or `left` |
| `active_state` | `BOTTOM` / `RISING` / `TOP` / `FALLING` |
| `active_norm_y` | Smoothed normalised wrist Y |
| `active_needs_init` | 1 if tracker is waiting for first bottom anchor |
| `active_lockout` | Frames remaining in post-rep lockout |
| `active_top_frames` | Frames accumulated at TOP this cycle |
| `active_rep_count` | Running rep total for active side |
| `inactive_norm_y` | Smoothed norm_y of the inactive wrist |
| `switch_lockout` | Frames remaining before next switch is allowed |

---

## Running Tests

```bash
source venv/bin/activate
pytest test_snatch_counter.py -v
```

76 tests covering: smoothing, normalisation, state-machine transitions, lockout guards, calibration, switch-mode detection, and multi-switch N×M rep patterns. No camera or MediaPipe required to run tests.

---

## Project Structure

```
snatch_counter.py        # All logic: state machine, calibration, HUD, main loop
test_snatch_counter.py   # 76 unit tests (pure Python, no CV/MediaPipe)
calibration.json         # Auto-generated after --calibrate (gitignored)
```

---

## Limitations

- Works best with a stationary camera and good lighting
- Requires the full body (shoulder, hip, wrist, elbow) to be visible
- Accuracy depends on MediaPipe pose detection quality; occlusion or fast motion may cause missed frames
- Currently processes video files only (no live webcam mode)
