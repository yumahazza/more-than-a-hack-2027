"""v1_enhanced_baseline.py - Enhanced possession baseline for More than a Hack 2027.

Key Improvements over baseline.py:
1. Graded Boundaries & Pre-Kickoff 2nd Half:
   - Ungraded blocks (pre-match, halftime, post-match) are set to 'X'.
   - 2nd Half Pre-Kickoff (frames 110808..113741) is strictly set to 'D' (100% Dead Ball).
2. Missing Ball Retention (Persistence):
   - In live play, ball is missing on 40.7% frames due to occlusion/aerial balls.
   - Instead of immediately guessing 'D', retain previous team possession for up to HOLD_FRAMES (~2.5s).
   - If ball is missing longer than HOLD_FRAMES, transition to 'D'.
3. Hysteresis on Contested Balls / Duels:
   - When players from both teams are near the ball, require a distance margin (HYSTERESIS_MARGIN = 0.8m)
     and control radius (< 4.5m) to switch possession from H to A or A to H.
4. Short-Flicker Suppression:
   - Clean up short spurious switches (< 15 frames) that break long continuous spells.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths
import starter.baseline as base_mod
import evaluation.possession_metrics as pm

paths = asset_paths(source="tactical")
TOTAL_FRAMES = GAMES[GAME]["n_video_frames"]
PITCH_X, PITCH_Y = GAMES[GAME]["pitch"]

# Match structure constants
KICKOFF_1ST = 11185
END_1ST = 90651
START_2ND = 110808
KICKOFF_2ND = 113742
END_2ND = 190197

# Tuning parameters
HOLD_FRAMES = 60          # ~2.4 seconds to retain possession when ball is occluded/missing
HYSTERESIS_MARGIN = 0.8   # meter margin required for opponent to take possession
MAX_CONTROL_DIST = 4.5    # meters: maximum distance to be considered possessing ball
MIN_SPELL_FRAMES = 15     # frames (< 0.6s) to filter out single-touch flickers


def stream_player_and_ball_distances():
    """Extract ball visibility and min distance to Home and Away players per frame."""
    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]
    
    team_of_slot = base_mod.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)
    
    ball_visible = np.zeros(TOTAL_FRAMES, dtype=bool)
    dist_home = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    dist_away = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    
    print(f"Reading ATD from {raw_path}...")
    with open(raw_path, "r") as f:
        for idx, line in enumerate(f):
            if idx >= TOTAL_FRAMES:
                break
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            
            ball_part = parts[2]
            if not ball_part.startswith("N"):
                b_tokens = ball_part.split(",")
                bx = float(b_tokens[0]) * PITCH_X
                by = float(b_tokens[1]) * PITCH_Y
                ball_visible[idx] = True
                
                players = parts[1].split(";")
                min_h = 999.0
                min_a = 999.0
                for slot_idx, p_str in enumerate(players):
                    if slot_idx >= n_slots or p_str.startswith("N"):
                        continue
                    coords = p_str.split(",")
                    px = float(coords[0]) * PITCH_X
                    py = float(coords[1]) * PITCH_Y
                    d = np.hypot(px - bx, py - by)
                    if team_of_slot[slot_idx] == base_mod.HOME:
                        if d < min_h:
                            min_h = d
                    else:
                        if d < min_a:
                            min_a = d
                dist_home[idx] = min_h
                dist_away[idx] = min_a
                
    return ball_visible, dist_home, dist_away


def generate_possession_labels(ball_visible, dist_home, dist_away, hold_frames=HOLD_FRAMES, smooth_window=50):
    """Generate labels H, A, D, X using boundary rules, hysteresis, and persistence."""
    labels = np.full(TOTAL_FRAMES, "X", dtype="<U1")
    
    # Process both halves independently
    halves = [
        (1, KICKOFF_1ST, END_1ST),
        (2, KICKOFF_2ND, END_2ND)
    ]
    
    # Pre-kickoff 2nd half is explicitly DEAD BALL
    labels[START_2ND:KICKOFF_2ND] = "D"
    
    for half_num, start_f, end_f in halves:
        curr_state = "D"
        missing_count = 0
        half_labels = []
        
        for f in range(start_f, end_f + 1):
            is_vis = ball_visible[f]
            dh = dist_home[f]
            da = dist_away[f]
            
            if is_vis:
                missing_count = 0
                # Both teams have player distance
                if dh < 900.0 and da < 900.0:
                    if curr_state == "H":
                        # Away must have clear advantage to take over
                        if da < dh - HYSTERESIS_MARGIN and da < MAX_CONTROL_DIST:
                            curr_state = "A"
                    elif curr_state == "A":
                        # Home must have clear advantage to take over
                        if dh < da - HYSTERESIS_MARGIN and dh < MAX_CONTROL_DIST:
                            curr_state = "H"
                    else:  # curr_state == "D"
                        if min(dh, da) < MAX_CONTROL_DIST:
                            curr_state = "H" if dh < da else "A"
                elif dh < MAX_CONTROL_DIST:
                    curr_state = "H"
                elif da < MAX_CONTROL_DIST:
                    curr_state = "A"
                # else: ball is far from everyone (loose ball), keep curr_state
            else:
                # Ball is missing (occlusion, aerial ball, or out of play)
                missing_count += 1
                if missing_count > hold_frames:
                    curr_state = "D"
                # Else: retain curr_state (persistence)
            
            half_labels.append(curr_state)
            
        half_arr = np.array(half_labels, dtype="<U1")
        
        # Suppress short transient flickers (< MIN_SPELL_FRAMES)
        # e.g., H H H A H H H -> H H H H H H H
        i = 0
        n = len(half_arr)
        while i < n:
            val = half_arr[i]
            j = i
            while j < n and half_arr[j] == val:
                j += 1
            run_len = j - i
            if run_len < MIN_SPELL_FRAMES:
                left_val = half_arr[i - 1] if i > 0 else None
                right_val = half_arr[j] if j < n else None
                if left_val and right_val and left_val == right_val:
                    half_arr[i:j] = left_val
            i = j
            
        # Centred majority vote smoothing over active window
        if smooth_window > 0:
            smoothed = np.empty_like(half_arr)
            pad = smooth_window
            # Numeric encoding for fast vote: H=0, A=1, D=2
            code_map = {"H": 0, "A": 1, "D": 2}
            inv_map = np.array(["H", "A", "D"])
            codes = np.array([code_map[x] for x in half_arr], dtype=np.int8)
            padded = np.pad(codes, pad, mode="edge")
            
            for k in range(n):
                window = padded[k : k + 2 * pad + 1]
                counts = np.bincount(window, minlength=3)
                smoothed[k] = inv_map[np.argmax(counts)]
            half_arr = smoothed
            
        labels[start_f : end_f + 1] = half_arr
        
    return labels


def save_submission(labels, name="v1_enhanced_baseline"):
    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.csv")
    
    df = pd.DataFrame({
        "frame": np.arange(TOTAL_FRAMES, dtype=np.int64),
        "label": labels
    })
    df.to_csv(out_path, index=False)
    print(f"Saved submission: {out_path} ({len(df):,} rows)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="v1_enhanced", help="Submission file stem name")
    parser.add_argument("--hold", type=int, default=60, help="Hold frames for missing ball")
    parser.add_argument("--window", type=int, default=50, help="Smoothing half-window in frames")
    args = parser.parse_args()
    
    ball_vis, dist_h, dist_a = stream_player_and_ball_distances()
    
    # Try a couple of window variants to inspect
    for w in [25, 50, 75]:
        variant_name = f"v1_enhanced_w{w}"
        labels = generate_possession_labels(ball_vis, dist_h, dist_a, hold_frames=args.hold, smooth_window=w)
        csv_path = save_submission(labels, name=variant_name)
        
        # Validate format
        print(f"Validating {variant_name}...")
        os.system(f'python evaluation/validate_submission.py "{csv_path}"')

    print("\nRunning evaluation on scoreboard...")
    os.system("python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
