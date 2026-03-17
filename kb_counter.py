"""
Kettlebell Rep Counter — entry point.

Usage:
  python kb_counter.py <video_file>                   # single KB (right hand)
  python kb_counter.py <video_file> --mode double     # double KB (both hands)
  python kb_counter.py <video_file> --mode switch     # hand-to-hand KB set
  python kb_counter.py <video_file> --elbow-lock      # require elbow lockout at top
  python kb_counter.py <video_file> --knee-lock       # require knee lockout at bottom
  python kb_counter.py <video_file> --ankle-grounded  # warn if heel rises during rep
  python kb_counter.py <video_file> --movement "Single Arm Jerk"
  python kb_counter.py <video_file> --calibrate       # dedicated calibration pass
  python kb_counter.py <video_file> --no-auto-calib   # skip auto-calibration
"""

import argparse
import sys

import cv2
import mediapipe as mp

from constants import (
    BOTTOM, TOP,
    RISE_THRESHOLD, DROP_THRESHOLD,
)
from calibration import Calibrator, derive_thresholds_from_samples, save_calibration, load_calibration
from tracker import SideTracker
from hud import (
    draw_calibration_hud,
    draw_overlay_single,
    draw_overlay_double,
    draw_overlay_switch,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Kettlebell rep counter")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--mode", choices=["single", "double", "switch"], default="single",
        help="'single' tracks right wrist only (default); "
             "'double' tracks both wrists independently; "
             "'switch' tracks both wrists, combined count (for hand-to-hand sets)",
    )
    parser.add_argument(
        "--min-top-time", type=float, default=0.5, metavar="SECS",
        help="Seconds wrist must hold overhead before the rep is counted (default: 0.5)",
    )
    parser.add_argument(
        "--rise-threshold", type=float, default=None, metavar="FLOAT",
        help=f"norm_y below which wrist is 'high' (default: {RISE_THRESHOLD}). "
             "For jerk/press where wrist doesn't swing low, try 0.6",
    )
    parser.add_argument(
        "--drop-threshold", type=float, default=None, metavar="FLOAT",
        help=f"norm_y above which wrist is 'low' (default: {DROP_THRESHOLD}). "
             "For jerk/press where wrist stays at rack, try 0.5",
    )
    parser.add_argument(
        "--elbow-lock", action="store_true", default=False,
        help="Require elbow lockout at the top of each rep (default: off)",
    )
    parser.add_argument(
        "--knee-lock", action="store_true", default=False,
        help="Require knee lockout at the bottom before each rep starts (default: off)",
    )
    parser.add_argument(
        "--ankle-grounded", action="store_true", default=False,
        help="Warn on screen if heel rises during a rep (for push press; default: off)",
    )
    parser.add_argument(
        "--movement", default="", metavar="TEXT",
        help="Movement name to display on screen, e.g. 'Single Arm Jerk' (default: none)",
    )
    parser.add_argument(
        "--no-skeleton", action="store_true", default=False,
        help="Hide the MediaPipe pose skeleton overlay (default: skeleton shown)",
    )
    parser.add_argument(
        "--mirror", action="store_true", default=False,
        help="Flip video horizontally before processing (use for selfie/front-facing camera)",
    )
    parser.add_argument(
        "--log", metavar="FILE",
        help="Write per-frame CSV log to FILE for debugging (e.g. --log debug.csv)",
    )
    calib_group = parser.add_mutually_exclusive_group()
    calib_group.add_argument(
        "--calibrate", action="store_true",
        help="Dedicated calibration pass: swing KB, save thresholds to calibration.json",
    )
    calib_group.add_argument(
        "--no-auto-calib", action="store_true",
        help="Skip auto-calibration window; load calibration.json if present",
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_partial_calibration(calibrator: Calibrator, label: str) -> None:
    """Save thresholds from whatever samples were collected; skip if too few."""
    samples = calibrator.samples
    if len(samples) >= 10:
        rise, drop = derive_thresholds_from_samples(samples)
        save_calibration(rise, drop)
        print(f"Calibration saved ({label}, {len(samples)} frames): "
              f"rise={rise:.3f}, drop={drop:.3f}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    double_mode          = args.mode == "double"
    switch_mode          = args.mode == "switch"
    dedicated_calib      = args.calibrate
    require_elbow_lock   = args.elbow_lock
    require_knee_lock    = args.knee_lock
    check_ankle_grounded = args.ankle_grounded
    movement_name        = args.movement.strip()
    show_skeleton        = not args.no_skeleton

    # ── Optional per-frame CSV log ─────────────────────────────────────────────
    _log_file = None
    if args.log:
        import csv as _csv
        _log_file = open(args.log, "w", newline="", encoding="utf-8")
        _log_writer = _csv.writer(_log_file)
        _log_writer.writerow([
            "frame", "pose_detected",
            "active_side", "active_state", "active_norm_y",
            "active_needs_init", "active_lockout", "active_top_frames", "active_rep_count",
            "inactive_norm_y",
            "switch_lockout",
        ])

    video_capture = cv2.VideoCapture(args.video)
    if not video_capture.isOpened():
        sys.exit(f"Error: cannot open '{args.video}'")

    frame_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = video_capture.get(cv2.CAP_PROP_FPS) or 30.0

    # ── Auto-correct video rotation from metadata ──────────────────────────────
    _ORIENT_TO_ROTATE = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    try:
        _raw_orient = int(video_capture.get(cv2.CAP_PROP_ORIENTATION_META))
    except AttributeError:
        _raw_orient = 0
    _rotation_fix = _ORIENT_TO_ROTATE.get(_raw_orient)
    if _raw_orient in (90, 270):
        frame_w, frame_h = frame_h, frame_w

    min_top_frames = max(1, round(fps * args.min_top_time))

    mp_pose      = mp.solutions.pose
    PoseLandmark = mp_pose.PoseLandmark
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    right_tracker = SideTracker(min_top_frames=min_top_frames)
    left_tracker  = (SideTracker(min_top_frames=min_top_frames)
                     if (double_mode or switch_mode) else None)

    # ── Calibration setup ──────────────────────────────────────────────────────
    existing_calib = load_calibration()
    if dedicated_calib:
        calibrator = Calibrator()
    elif args.no_auto_calib or existing_calib is not None:
        calibrator = None
        if existing_calib is not None:
            rise, drop = existing_calib
            right_tracker.rise_threshold = rise
            right_tracker.drop_threshold = drop
            if (double_mode or switch_mode) and left_tracker:
                left_tracker.rise_threshold = rise
                left_tracker.drop_threshold = drop
            print(f"Loaded calibration: rise={rise:.3f}, drop={drop:.3f}")
    else:
        calibrator = None

    # ── CLI threshold overrides ────────────────────────────────────────────────
    for tracker in filter(None, [right_tracker, left_tracker]):
        if args.rise_threshold is not None:
            tracker.rise_threshold = args.rise_threshold
        if args.drop_threshold is not None:
            tracker.drop_threshold = args.drop_threshold
    if args.rise_threshold is not None or args.drop_threshold is not None:
        print(f"Threshold overrides: rise={right_tracker.rise_threshold:.3f}  "
              f"drop={right_tracker.drop_threshold:.3f}")

    calib_done_and_saved = False
    switch_side    = 'right'
    switch_lockout = 0
    frame_num      = 0

    while True:
        frame_read_ok, frame = video_capture.read()
        frame_num += 1
        if not frame_read_ok:
            if dedicated_calib and calibrator is not None and not calibrator.done:
                _save_partial_calibration(calibrator, "partial")
            break

        if _rotation_fix is not None:
            frame = cv2.rotate(frame, _rotation_fix)
        if args.mirror:
            frame = cv2.flip(frame, 1)

        rgb_frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb_frame)

        if pose_result.pose_landmarks:
            landmark_list = pose_result.pose_landmarks.landmark

            if calibrator is not None and not calibrator.done:
                # ── Calibration phase ──────────────────────────────────────────
                from geometry import get_landmark_y, normalise_wrist, smooth
                wrist_y    = get_landmark_y(landmark_list, PoseLandmark.RIGHT_WRIST,    frame_h)
                shoulder_y = get_landmark_y(landmark_list, PoseLandmark.RIGHT_SHOULDER, frame_h)
                hip_y      = get_landmark_y(landmark_list, PoseLandmark.RIGHT_HIP,      frame_h)
                raw_norm_y = normalise_wrist(wrist_y, shoulder_y, hip_y)
                smoothed   = smooth(right_tracker.smooth_buf, raw_norm_y)
                calibrator.update(smoothed)

                if calibrator.done:
                    rise, drop = calibrator.derive_thresholds()
                    right_tracker.rise_threshold = rise
                    right_tracker.drop_threshold = drop
                    if (double_mode or switch_mode) and left_tracker:
                        left_tracker.rise_threshold = rise
                        left_tracker.drop_threshold = drop
                    if dedicated_calib:
                        save_calibration(rise, drop)
                        print(f"Calibration saved: rise={rise:.3f}, drop={drop:.3f}")
                        calib_done_and_saved = True
                    else:
                        calibrator = None

            elif calibrator is None:
                # ── Normal counting phase ──────────────────────────────────────
                if switch_mode:
                    PL = PoseLandmark
                    R_lm = (PL.RIGHT_WRIST, PL.RIGHT_SHOULDER, PL.RIGHT_HIP, PL.RIGHT_ELBOW)
                    L_lm = (PL.LEFT_WRIST,  PL.LEFT_SHOULDER,  PL.LEFT_HIP,  PL.LEFT_ELBOW)
                    R_ka = (PL.RIGHT_KNEE, PL.RIGHT_ANKLE)
                    L_ka = (PL.LEFT_KNEE,  PL.LEFT_ANKLE)

                    active_t    = right_tracker if switch_side == 'right' else left_tracker
                    inactive_t  = left_tracker  if switch_side == 'right' else right_tracker
                    active_lm   = R_lm          if switch_side == 'right' else L_lm
                    inactive_lm = L_lm          if switch_side == 'right' else R_lm
                    active_ka   = R_ka          if switch_side == 'right' else L_ka

                    inactive_t.update_display_only(landmark_list,
                                                   *inactive_lm, frame_w, frame_h)

                    # Switch detection
                    if switch_lockout > 0:
                        switch_lockout -= 1
                    elif (active_t.state == BOTTOM
                            and active_t.norm_y > active_t.drop_threshold
                            and inactive_t.norm_y < active_t.rise_threshold):
                        inactive_t.needs_init          = False
                        inactive_t.state               = TOP
                        inactive_t.top_frames          = inactive_t.min_top_frames
                        inactive_t.lockout             = 0
                        inactive_t._counted_this_cycle = False
                        inactive_t.smooth_buf.clear()
                        active_t.state      = BOTTOM
                        active_t.top_frames = 0
                        switch_side    = 'left' if switch_side == 'right' else 'right'
                        switch_lockout = 30
                        active_t, active_lm = inactive_t, inactive_lm

                    active_t.update(landmark_list,
                                    *active_lm, frame_w, frame_h,
                                    require_elbow_lock=require_elbow_lock,
                                    knee_idx=active_ka[0], ankle_idx=active_ka[1],
                                    require_knee_lock=require_knee_lock,
                                    check_ankle_grounded=check_ankle_grounded)

                    draw_overlay_switch(frame, pose_result.pose_landmarks,
                                        frame_w, frame_h,
                                        right_tracker, left_tracker, mp_pose,
                                        switch_side=switch_side,
                                        switch_lockout=switch_lockout,
                                        show_knee=require_knee_lock,
                                        show_ankle=check_ankle_grounded,
                                        movement_name=movement_name,
                                        show_skeleton=show_skeleton)
                else:
                    PL = PoseLandmark
                    right_tracker.update(landmark_list,
                                         PL.RIGHT_WRIST, PL.RIGHT_SHOULDER,
                                         PL.RIGHT_HIP,   PL.RIGHT_ELBOW,
                                         frame_w, frame_h,
                                         require_elbow_lock=require_elbow_lock,
                                         knee_idx=PL.RIGHT_KNEE,
                                         ankle_idx=PL.RIGHT_ANKLE,
                                         require_knee_lock=require_knee_lock,
                                         check_ankle_grounded=check_ankle_grounded)
                    if double_mode:
                        left_tracker.update(landmark_list,
                                            PL.LEFT_WRIST, PL.LEFT_SHOULDER,
                                            PL.LEFT_HIP,   PL.LEFT_ELBOW,
                                            frame_w, frame_h,
                                            require_elbow_lock=require_elbow_lock,
                                            knee_idx=PL.LEFT_KNEE,
                                            ankle_idx=PL.LEFT_ANKLE,
                                            require_knee_lock=require_knee_lock,
                                            check_ankle_grounded=check_ankle_grounded)
                        draw_overlay_double(frame, pose_result.pose_landmarks,
                                            frame_w, frame_h,
                                            right_tracker, left_tracker, mp_pose,
                                            show_knee=require_knee_lock,
                                            show_ankle=check_ankle_grounded,
                                            movement_name=movement_name,
                                            show_skeleton=show_skeleton)
                    else:
                        draw_overlay_single(frame, pose_result.pose_landmarks,
                                            frame_w, frame_h, right_tracker, mp_pose,
                                            show_knee=require_knee_lock,
                                            show_ankle=check_ankle_grounded,
                                            movement_name=movement_name,
                                            show_skeleton=show_skeleton)

        if calibrator is not None and not calib_done_and_saved:
            draw_calibration_hud(frame, calibrator, dedicated_calib)

        # ── Per-frame CSV log ──────────────────────────────────────────────────
        if _log_file is not None and switch_mode:
            pose_ok    = pose_result.pose_landmarks is not None
            active_t   = right_tracker if switch_side == 'right' else left_tracker
            inactive_t = left_tracker  if switch_side == 'right' else right_tracker
            _log_writer.writerow([
                frame_num, int(pose_ok),
                switch_side, active_t.state, f"{active_t.norm_y:.3f}",
                int(active_t.needs_init), active_t.lockout,
                active_t.top_frames, active_t.rep_count,
                f"{inactive_t.norm_y:.3f}",
                switch_lockout,
            ])

        cv2.imshow("Kettlebell Rep Counter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if dedicated_calib and calibrator is not None and not calibrator.done:
                _save_partial_calibration(calibrator, "early exit")
            break
        if calib_done_and_saved:
            break

    video_capture.release()
    pose.close()
    cv2.destroyAllWindows()
    if _log_file is not None:
        _log_file.close()
        print(f"Log written to {args.log}")

    if dedicated_calib:
        return

    if double_mode:
        combined = right_tracker.rep_count + left_tracker.rep_count
        print(f"\nFinal rep count — R: {right_tracker.rep_count}  "
              f"L: {left_tracker.rep_count}  Total: {combined}")
    elif switch_mode:
        combined = right_tracker.rep_count + left_tracker.rep_count
        print(f"\nFinal rep count (hand-to-hand) — R: {right_tracker.rep_count}  "
              f"L: {left_tracker.rep_count}  Total: {combined}")
    else:
        print(f"\nFinal rep count: {right_tracker.rep_count}")


if __name__ == "__main__":
    main()
