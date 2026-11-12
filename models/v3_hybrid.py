"""v3_hybrid_smart_tagging.py - Hybrid Model combining Physical Hysteresis,
Missing Ball Persistence, and Smart Tagging Set-Piece Dead Ball Detection.

Architecture:
1. Hard Boundaries:
   - Frame 0..11184: X (Pre-match)
   - Frame 90652..110807: X (Halftime)
   - Frame 110808..113741: D (2nd Half Pre-Kickoff 100% Dead Ball)
   - Frame 190198..199318: X (Post-match)
2. Smart Tagging Set Piece Stoppage Mask:
   - During SET PIECES (Throw-in, Goal Kick, Corner Kick, Free Kick, Kickoff),
     the period before ball is struck into play is strictly marked as Dead Ball ('D').
3. Physical Hysteresis & State Persistence:
   - Contested margin (0.8m): opponent cannot take possession without clear lead.
   - Ball missing retention (60 frames / 2.4s): keeps attacking possession through aerial balls/occlusions.
4. Temporal Smoothing (Window Sweep: 60, 75, 90, 110 frames):
   - Preserves continuous long possessions (targeting GT mean duration ~14.2s).
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
import starter.load_smart_tagging as lst
import evaluation.possession_metrics as pm

paths = asset_paths(source="tactical")
TOTAL_FRAMES = GAMES[GAME]["n_video_frames"]
PITCH_X, PITCH_Y = GAMES[GAME]["pitch"]

# Match boundaries
KICKOFF_1ST = 11185
END_1ST = 90651
START_2ND = 110808
KICKOFF_2ND = 113742
END_2ND = 190197

# Hyperparameters
HOLD_FRAMES = 60          # 2.4 seconds persistence during missing ball
HYSTERESIS_MARGIN = 0.8   # meter advantage needed for opponent takeover
MAX_CONTROL_DIST = 4.5    # meters for controlled ball proximity
MIN_SPELL_FRAMES = 15     # filter out transient single-touch flickers


def stream_tracking():
    """Extract ball visibility, positions, speeds, and min distance to each team."""
    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]
    
    team_of_slot = base_mod.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)
    
    ball_vis = np.zeros(TOTAL_FRAMES, dtype=bool)
    dist_h = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    dist_a = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    ball_speed = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    
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
                if len(b_tokens) > 2 and not b_tokens[2].startswith("N"):
                    ball_speed[idx] = float(b_tokens[2])
                ball_vis[idx] = True
                
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
                dist_h[idx] = min_h
                dist_a[idx] = min_a
                
    return ball_vis, dist_h, dist_a, ball_speed


def build_set_piece_stoppage_mask(ball_vis, ball_speed):
    """Identify dead-ball stoppage intervals from Smart Tagging SET PIECES."""
    instances = lst.load()
    is_dead_stoppage = np.zeros(TOTAL_FRAMES, dtype=bool)
    
    set_pieces = [inst for inst in instances if inst["code"].lower() == "set pieces"]
    print(f"Processing {len(set_pieces)} Smart Tagging Set Pieces...")
    
    for sp in set_pieces:
        sf = max(0, sp["start_frame"])
        ef = min(TOTAL_FRAMES, sp["end_frame"])
        
        # During a set piece window, find when the restart actually happens:
        # Before restart, ball is either missing or stationary (speed < 2.5 m/s)
        # We search from the back of the window for where play resumes
        # Most of the first 60-70% of the set piece window is definitely Dead Ball preparation
        stoppage_end = ef
        for f in range(ef - 1, sf + 25, -1):
            if ball_vis[f] and ball_speed[f] > 4.0:
                stoppage_end = f
                break
                
        # Mark the stoppage period before restart as dead ball
        is_dead_stoppage[sf : max(sf, stoppage_end - 10)] = True
        
    return is_dead_stoppage


def generate_hybrid_possession(ball_vis, dist_h, dist_a, is_dead_stoppage, smooth_window=75):
    """Synthesize predictions with hybrid physical rules, Smart Tagging, and temporal smoothing."""
    labels = np.full(TOTAL_FRAMES, "X", dtype="<U1")
    
    # 2nd Half Pre-Kickoff is explicitly DEAD BALL
    labels[START_2ND:KICKOFF_2ND] = "D"
    
    halves = [
        (1, KICKOFF_1ST, END_1ST),
        (2, KICKOFF_2ND, END_2ND)
    ]
    
    for half_num, start_f, end_f in halves:
        curr_state = "D"
        missing_count = 0
        half_labels = []
        
        for f in range(start_f, end_f + 1):
            # Check Smart Tagging dead stoppage mask first
            if is_dead_stoppage[f]:
                curr_state = "D"
                missing_count = 0
                half_labels.append("D")
                continue
                
            is_vis = ball_vis[f]
            dh = dist_h[f]
            da = dist_a[f]
            
            if is_vis:
                missing_count = 0
                # If both teams have proximity to ball
                if dh < 900.0 and da < 900.0:
                    if curr_state == "H":
                        if da < dh - HYSTERESIS_MARGIN and da < MAX_CONTROL_DIST:
                            curr_state = "A"
                    elif curr_state == "A":
                        if dh < da - HYSTERESIS_MARGIN and dh < MAX_CONTROL_DIST:
                            curr_state = "H"
                    else:  # curr_state == "D"
                        if min(dh, da) < MAX_CONTROL_DIST:
                            curr_state = "H" if dh < da else "A"
                elif dh < MAX_CONTROL_DIST:
                    curr_state = "H"
                elif da < MAX_CONTROL_DIST:
                    curr_state = "A"
            else:
                # Ball is occluded or in air
                missing_count += 1
                if missing_count > HOLD_FRAMES:
                    curr_state = "D"
                # Else retain previous state
                
            half_labels.append(curr_state)
            
        half_arr = np.array(half_labels, dtype="<U1")
        
        # Suppress single-touch micro flickers (< MIN_SPELL_FRAMES)
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


def save_and_validate(labels, name):
    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.csv")
    
    df = pd.DataFrame({
        "frame": np.arange(TOTAL_FRAMES, dtype=np.int64),
        "label": labels
    })
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df):,} rows)")
    
    # Run official validator
    os.system(f'python evaluation/validate_submission.py "{out_path}"')
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="v3_hybrid", help="Stem name")
    args = parser.parse_args()
    
    ball_vis, dist_h, dist_a, ball_speed = stream_tracking()
    is_dead_stoppage = build_set_piece_stoppage_mask(ball_vis, ball_speed)
    
    # Sweep smoothing windows around optimal regime (60, 75, 90, 110)
    for w in [60, 75, 90, 110]:
        v_name = f"{args.name}_w{w}"
        print(f"\nGenerating {v_name} (smoothing window = {w})...")
        labels = generate_hybrid_possession(ball_vis, dist_h, dist_a, is_dead_stoppage, smooth_window=w)
        save_and_validate(labels, v_name)
        
    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL SCOREBOARD EVALUATION")
    print("=" * 70)
    os.system("python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()

