"""v5_pitch_control.py - Two-branch 5-nearest neighbors possession model.

Inspired by:
Borghesi, Costa, Morra, Lamberti (2023), "Using Temporal Convolutional Networks to estimate
ball possession in soccer games", Expert Systems with Applications 223, 119780.

Key Principles from Paper:
1. Two-Branch Decoupling:
   - Branch 1 (Dead-Alive, DA): Determines if ball is alive or dead. Driven by ball speed,
     boundary line exceedance, prolonged tracking disappearance, and Smart Tagging set pieces.
   - Branch 2 (Possession, POSS): Evaluated on alive play using the 5 Nearest Neighbors (5NN).
     Computes distance-weighted collective influence of closest players on Home vs Away.
2. Trajectory Reconstruction & Interpolation:
   - Reconstructs ball trajectory across pass & aerial gaps (<= 75 frames).
3. Permutation Invariance via 5NN Ordering:
   - Reorders players dynamically by proximity to ball anchor at every frame.
4. Temporal Structuring & Acausal Windowing (nf=64..75):
   - Preserves possession spells, suppressing high-frequency flickering.
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

# Hyperparameters from paper insights
K_NEAREST = 5             # 5NN proved optimal in Borghesi et al. (84.98% acc vs 84.95% full)
MAX_PASS_GAP = 75         # frames (~3.0s) max gap to interpolate ball flight
HOLD_ALIVE_FRAMES = 50    # frames to retain alive possession through occlusions
MIN_SPELL_FRAMES = 15     # filter single-touch micro flickers
HYSTERESIS_RATIO = 1.25   # opponent must achieve 25% higher weighted score to take over


def stream_tracking_data():
    """Extract raw ball, player coordinates, speeds, and reconstruct trajectories."""
    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]
    
    team_of_slot = base_mod.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)
    
    ball_x = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    ball_y = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    ball_speed = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    
    player_records = []
    
    print(f"Step 1: Reading ATD tracking from {raw_path}...")
    with open(raw_path, "r") as f:
        for idx, line in enumerate(f):
            if idx >= TOTAL_FRAMES:
                break
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                player_records.append([])
                continue
            
            ball_part = parts[2]
            if not ball_part.startswith("N"):
                b_tokens = ball_part.split(",")
                ball_x[idx] = float(b_tokens[0]) * PITCH_X
                ball_y[idx] = float(b_tokens[1]) * PITCH_Y
                if len(b_tokens) > 2 and not b_tokens[2].startswith("N"):
                    ball_speed[idx] = float(b_tokens[2])
                    
            p_list = []
            players = parts[1].split(";")
            for slot_idx, p_str in enumerate(players):
                if slot_idx >= n_slots or p_str.startswith("N"):
                    continue
                coords = p_str.split(",")
                px = float(coords[0]) * PITCH_X
                py = float(coords[1]) * PITCH_Y
                spd = float(coords[2]) if len(coords) > 2 and not coords[2].startswith("N") else 2.0
                t = team_of_slot[slot_idx]
                p_list.append((px, py, spd, t))
            player_records.append(p_list)
            
    print("Step 2: Akima/Spline-style Ball Trajectory Reconstruction...")
    ball_interp_x = ball_x.copy()
    ball_interp_y = ball_y.copy()
    ball_valid = ~np.isnan(ball_x)
    
    i = 0
    reconstructed_count = 0
    while i < TOTAL_FRAMES:
        if not ball_valid[i]:
            start_gap = i
            while i < TOTAL_FRAMES and not ball_valid[i]:
                i += 1
            end_gap = i
            gap_len = end_gap - start_gap
            if start_gap > 0 and end_gap < TOTAL_FRAMES and gap_len <= MAX_PASS_GAP:
                x0, y0 = ball_x[start_gap - 1], ball_y[start_gap - 1]
                x1, y1 = ball_x[end_gap], ball_y[end_gap]
                alphas = np.linspace(0.0, 1.0, gap_len + 2)[1:-1]
                ball_interp_x[start_gap:end_gap] = x0 + alphas * (x1 - x0)
                ball_interp_y[start_gap:end_gap] = y0 + alphas * (y1 - y0)
                reconstructed_count += gap_len
        else:
            i += 1
            
    print(f"Reconstructed {reconstructed_count:,} frames of ball trajectory during flight.")
    return ball_interp_x, ball_interp_y, ball_speed, player_records


def build_dead_alive_branch(ball_x, ball_y, ball_speed):
    """Branch 1: Dead-Alive Classifier (DA Branch)."""
    instances = lst.load()
    is_dead = np.zeros(TOTAL_FRAMES, dtype=bool)
    
    # 1. Match boundary rules
    is_dead[START_2ND:KICKOFF_2ND] = True  # Pre-kickoff 2nd half 100% Dead
    
    # 2. Smart Tagging Set Pieces
    set_pieces = [inst for inst in instances if inst["code"].lower() == "set pieces"]
    for sp in set_pieces:
        sf = max(0, sp["start_frame"])
        ef = min(TOTAL_FRAMES, sp["end_frame"])
        stoppage_end = ef
        for f in range(ef - 1, sf + 25, -1):
            if ball_speed[f] > 4.0:
                stoppage_end = f
                break
        is_dead[sf : max(sf, stoppage_end - 10)] = True
        
    # 3. Ball out of pitch boundaries (Pitch is 105.3 x 68.0, margins +/- 0.5m)
    out_of_pitch = (ball_x < -1.0) | (ball_x > PITCH_X + 1.0) | (ball_y < -1.0) | (ball_y > PITCH_Y + 1.0)
    is_dead[out_of_pitch] = True
    
    # 4. Prolonged missing ball (> MAX_PASS_GAP) indicates stoppage
    ball_valid = ~np.isnan(ball_x)
    i = 0
    while i < TOTAL_FRAMES:
        if not ball_valid[i]:
            start_gap = i
            while i < TOTAL_FRAMES and not ball_valid[i]:
                i += 1
            end_gap = i
            gap_len = end_gap - start_gap
            if gap_len > MAX_PASS_GAP:
                # Mark as dead after initial hold
                is_dead[start_gap + HOLD_ALIVE_FRAMES : end_gap] = True
        else:
            i += 1
            
    return is_dead


def run_5nn_possession_branch(ball_x, ball_y, player_records, is_dead, smooth_window=75):
    """Branch 2: Possession Classifier using 5-Nearest Neighbors (5NN)."""
    labels = np.full(TOTAL_FRAMES, "X", dtype="<U1")
    labels[START_2ND:KICKOFF_2ND] = "D"
    
    halves = [
        (1, KICKOFF_1ST, END_1ST),
        (2, KICKOFF_2ND, END_2ND)
    ]
    
    for half_num, start_f, end_f in halves:
        curr_poss = "D"
        missing_count = 0
        half_labels = []
        
        for f in range(start_f, end_f + 1):
            # If Branch 1 says Dead, assign D
            if is_dead[f]:
                curr_poss = "D"
                missing_count = 0
                half_labels.append("D")
                continue
                
            bx = ball_x[f]
            by = ball_y[f]
            
            if not np.isnan(bx):
                missing_count = 0
                p_list = player_records[f]
                
                if p_list:
                    # Sort players by Euclidean distance to ball (5NN theorem)
                    p_with_dist = []
                    for px, py, spd, t in p_list:
                        d = np.hypot(px - bx, py - by)
                        p_with_dist.append((d, spd, t))
                    p_with_dist.sort(key=lambda x: x[0])
                    
                    # Top-5 closest players
                    k_nearest = p_with_dist[:K_NEAREST]
                    
                    # Collective distance-weighted influence
                    score_h = 0.0
                    score_a = 0.0
                    closest_dh = 999.0
                    closest_da = 999.0
                    
                    for rank_idx, (d, spd, t) in enumerate(k_nearest):
                        # Weight decays with distance and rank
                        weight = np.exp(-d / 2.2) * (1.0 + 0.1 * min(spd, 6.0)) / (1.0 + 0.2 * rank_idx)
                        if t == base_mod.HOME:
                            score_h += weight
                            if d < closest_dh:
                                closest_dh = d
                        else:
                            score_a += weight
                            if d < closest_da:
                                closest_da = d
                                
                    # State transition with hysteresis
                    if curr_poss == "H":
                        # Away must clearly dominate the 5NN cluster to overturn possession
                        if score_a > score_h * HYSTERESIS_RATIO and closest_da < 3.8:
                            curr_poss = "A"
                    elif curr_poss == "A":
                        # Home must clearly dominate the 5NN cluster to overturn possession
                        if score_h > score_a * HYSTERESIS_RATIO and closest_dh < 3.8:
                            curr_poss = "H"
                    else:  # curr_poss == 'D'
                        if min(closest_dh, closest_da) < 4.0:
                            curr_poss = "H" if score_h >= score_a else "A"
                # Else no players tracked, keep curr_poss
            else:
                # Ball missing in un-interpolated long gap
                missing_count += 1
                if missing_count > HOLD_ALIVE_FRAMES:
                    curr_poss = "D"
                    
            half_labels.append(curr_poss)
            
        half_arr = np.array(half_labels, dtype="<U1")
        
        # Suppress micro-flickers (< MIN_SPELL_FRAMES)
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
            
        # Centred majority vote smoothing over acausal receptive window
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
    os.system(f'python evaluation/validate_submission.py "{out_path}"')
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="v5_pitch_control", help="Stem name")
    args = parser.parse_args()
    
    ball_x, ball_y, ball_spd, p_records = stream_tracking_data()
    print("Step 3: Evaluating Branch 1 (Dead-Alive Detector)...")
    is_dead = build_dead_alive_branch(ball_x, ball_y, ball_spd)
    
    print("Step 4: Evaluating Branch 2 (5NN Possession Classifier)...")
    # Sweep smoothing windows (64, 75, 90, 105)
    for w in [64, 75, 90, 105]:
        v_name = f"{args.name}_w{w}"
        print(f"\nGenerating {v_name} (receptive window = {w})...")
        labels = run_5nn_possession_branch(ball_x, ball_y, p_records, is_dead, smooth_window=w)
        save_and_validate(labels, v_name)
        
    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL SCOREBOARD EVALUATION")
    print("=" * 70)
    os.system("python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()

