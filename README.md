# Kettlebell Rep Counter

A computer-vision rep counter for kettlebell exercises using [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) and OpenCV. Tracks wrist position relative to the body and counts reps through a state-machine — no wearable required.

Supports **single-hand**, **double kettlebell**, and **hand-to-hand switch sets** for competition movements including snatch, jerk, long cycle, and press.

---

## Features

- Counts reps in real time from a video file
- Three tracking modes: single wrist, both wrists independently, hand-to-hand pass
- Rep counted as soon as the overhead hold is confirmed (configurable hold time, default 0.5 s)
- Automatic calibration to your range of motion (or load saved calibration)
- Tunable thresholds for different movements (snatch, jerk, long cycle, press)
- **Elbow-lockout gate** — optionally require locked elbow at top before rep counts
- **Knee-lockout gate** — optionally require locked knees at bottom before rep starts
- **Ankle-grounded check** — warns on screen if heel rises during a rep (push press)
- **Movement name label** — display the movement name as an on-screen tag
- **Skeleton toggle** — show or hide the pose skeleton overlay
- Semi-transparent HUD panel with large rep count, state indicator, progress bar, and status dots
- Per-frame CSV logging for debugging

---

## Requirements

```
Python 3.9+
mediapipe
opencv-python
numpy
```

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install mediapipe opencv-python numpy
```

---

## Quick Start

```bash
# Single kettlebell, right hand (default)
python kb_counter.py video.mp4

# Double kettlebell (both hands, independent counts)
python kb_counter.py video.mp4 --mode double

# Hand-to-hand switch set (combined count, active hand indicator)
python kb_counter.py video.mp4 --mode switch

# Single arm jerk with movement label and form checks
python kb_counter.py video.mp4 --mode switch \
  --rise-threshold -0.3 --drop-threshold 0.05 \
  --knee-lock --movement "Single Arm Jerk"

# Push press with ankle-grounding check and no skeleton
python kb_counter.py video.mp4 --mode switch \
  --rise-threshold -0.3 --drop-threshold 0.05 \
  --ankle-grounded --no-skeleton --movement "Single Arm Push Press"
```

Press **q** to quit. Final rep count is printed to the terminal.

---

## All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `single` | `single` · `double` · `switch` |
| `--min-top-time SECS` | `0.5` | Seconds wrist must hold overhead before the rep is counted |
| `--rise-threshold FLOAT` | `0.4` | norm_y below which wrist is "high". Lower = stricter overhead requirement |
| `--drop-threshold FLOAT` | `0.85` | norm_y above which wrist is "low". Lower = stricter bottom requirement |
| `--elbow-lock` | off | Require elbow lockout (≥ 160°) at top for the hold timer to run |
| `--knee-lock` | off | Require knee lockout (≥ 160°) at bottom before each rep can start |
| `--ankle-grounded` | off | Warn on screen if heel rises during a rep (for push press) |
| `--movement TEXT` | — | Movement name shown as an on-screen label, e.g. `"Single Arm Jerk"` |
| `--no-skeleton` | off | Hide the MediaPipe pose skeleton overlay |
| `--mirror` | off | Flip video horizontally (for selfie/front-facing camera footage) |
| `--log FILE` | — | Write per-frame CSV debug log to FILE |
| `--calibrate` | — | Dedicated calibration pass: move KB through full range for ~5 s, saves `calibration.json` |
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
BOTTOM ──(norm_y < rise_threshold, knees locked if --knee-lock)──► RISING
RISING ──(norm_y < rise_threshold)──────────────────────────────► TOP
TOP    ──(hold for min_top_frames, elbow locked if --elbow-lock)─► TOP  ← rep counted here
TOP    ──(norm_y > rise_threshold after confirmed hold)──────────► FALLING
FALLING──(norm_y > drop_threshold)──────────────────────────────► BOTTOM
```

Rep is **counted at the top** the moment the overhead hold is confirmed (wrist stays overhead for `fps × --min-top-time` frames). A thin progress bar in the HUD shows how far through the hold the current rep is.

Anti-double-count guards:
- **min_top_frames** — wrist must stay overhead for `fps × --min-top-time` frames before counting. Prevents swing-throughs.
- **MIN_REP_LOCKOUT (15 frames)** — after the wrist returns to the bottom, the BOTTOM→RISING transition is blocked for ~0.5 s so the backswing can't start a new cycle immediately.

### Form Gates

| Flag | What it checks | Where it gates |
|------|---------------|----------------|
| `--elbow-lock` | Elbow angle ≥ 160° | Hold timer only runs while elbow is locked; a bent elbow pauses the countdown |
| `--knee-lock` | Knee angle ≥ 160° | BOTTOM→RISING is blocked until knees are straight (rep start requires locked legs) |
| `--ankle-grounded` | Heel Y stays within 3% of frame height above its BOTTOM baseline | Displays a red **HEEL RAISED!** warning in the HUD; rep still counted |

Elbow angle (shoulder→elbow→wrist) and knee angle (hip→knee→ankle) are always computed and shown in the HUD as green/red status dots regardless of whether the gates are active.

### Switch Mode (Hand-to-Hand)

Only the **active** wrist runs the full state machine. The **inactive** wrist is tracked for display and switch detection only.

A switch fires when:
1. Active wrist is at **BOTTOM** (between reps, at rack/hip position)
2. Active wrist is physically **low** (`norm_y > drop_threshold`)
3. Inactive wrist is **overhead** (`norm_y < rise_threshold`) — has the KB

On switch, the incoming tracker is placed directly into **TOP** state with the hold timer pre-filled. The rep is counted on the very next overhead frame — no warm-up delay after the handoff.

A **switch_lockout** of 30 frames (~1 s) prevents ping-pong re-switching.

### HUD Layout

All three modes use a consistent semi-transparent dark panel in the top-left corner:

```
┌──────────────────────────┐     [Movement Name]  ← top-right tag (if --movement)
│ REPS                     │
│   12                     │  ← large rep count
│ ████████░░               │  ← overhead hold progress bar (always reserved)
│ ● TOP                    │  ← state dot (colour-coded)
│ ny  -0.14                │  ← normalised wrist Y
│ ──────────────────────   │
│ ● Elbow  170°            │  ← green = locked, red = bent
│ ● Knees  163°            │  ← if --knee-lock
│ ● Ankle  OK              │  ← if --ankle-grounded
└──────────────────────────┘
```

The progress bar slot is always the same height so the rest of the panel never shifts position.

---

## Tuning for Different Movements

### Snatch / Long Cycle

The KB swings from below the hip to overhead. Defaults (`rise=0.4`, `drop=0.85`) work well.

```bash
python kb_counter.py video.mp4 --mode switch
```

### Jerk / Press

The KB stays between rack (~shoulder) and overhead. It never reaches hip level, so `drop=0.85` is never triggered.

```bash
python kb_counter.py video.mp4 --mode switch \
  --rise-threshold -0.3 \
  --drop-threshold 0.05 \
  --min-top-time 0.5 \
  --knee-lock \
  --movement "Single Arm Jerk"
```

| Threshold | Value | Why |
|-----------|-------|-----|
| `--rise-threshold -0.3` | wrist must be well above shoulder | rack position (≈ 0.07) is above −0.3, so it never falsely triggers RISING |
| `--drop-threshold 0.05` | rack position triggers BOTTOM | rack is norm_y ≈ 0.04–0.12, safely above 0.05 |
| `--min-top-time 0.5` | 0.5 s overhead hold before counting | confirms intentional fixation |

### Push Press

Same thresholds as jerk, with ankle grounding and no skeleton for a cleaner view:

```bash
python kb_counter.py video.mp4 --mode switch \
  --rise-threshold -0.3 \
  --drop-threshold 0.05 \
  --ankle-grounded \
  --no-skeleton \
  --movement "Single Arm Push Press"
```

### Calibration

Run a dedicated calibration pass to derive thresholds from your actual range of motion:

```bash
python kb_counter.py video.mp4 --calibrate
```

Move the KB through the full range for ~5 seconds. Thresholds are saved to `calibration.json` and loaded automatically on subsequent runs. Override any saved value with `--rise-threshold` / `--drop-threshold`.

---

## Debugging with CSV Logs

```bash
python kb_counter.py video.mp4 --mode switch --log debug.csv \
  --rise-threshold -0.3 --drop-threshold 0.05 --min-top-time 0.5
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
pytest test_kb_counter.py -v
```

78 tests covering: smoothing, normalisation, state-machine transitions, lockout guards, calibration, switch-mode detection, and multi-switch N×M rep patterns. No camera or MediaPipe required to run tests.

---

## Project Structure

```
kb_counter.py        # All logic: state machine, calibration, HUD, main loop
test_kb_counter.py   # 78 unit tests (pure Python, no CV/MediaPipe)
calibration.json     # Auto-generated after --calibrate (gitignored)
```

---

## Limitations

- Works best with a stationary camera and good lighting
- Requires the full body (shoulder, hip, wrist, elbow, knee, ankle) to be visible for all form checks; at minimum shoulder, hip, wrist, and elbow must be in frame for basic rep counting
- Accuracy depends on MediaPipe pose detection quality; occlusion or fast motion may cause missed frames
- Currently processes video files only (no live webcam mode)
