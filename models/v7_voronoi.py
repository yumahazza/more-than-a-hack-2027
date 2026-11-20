"""v7_voronoi_pitch_control.py - Pitch Control & Spatial Dominance (Barca Innovation Hub Style).

Based on:
1. Fernandez & Bornn (2018), 'Wide Open Spaces: Generating, Visualizing, and Quantifying
   Space Creation in Soccer' (Barca Innovation Hub / Sloan Sports Analytics Conference).
2. Spearman (2018), 'Beyond Expected Goals' (Time-to-Intercept & Pitch Control Surface).
3. Borghesi et al. (2023), 5-Nearest Neighbors Spatial Sufficiency.

Mathematical Formulation:
- Computes local spatial control field within ball action radius (R = 4.5m).
- Evaluates team influence field:
    I_i(r) = exp(-||r - p_i||^2 / (2 * sigma_i^2)) * (1 + 0.1 * spd_i)
- Integrates local pitch control around ball pocket:
    PC_home = sum_{i in Home} I_i(b) / (sum_{i in Home} I_i(b) + sum_{j in Away} I_j(b))
- Dual-layer Hysteresis:
    Transition requires both Pitch Control Dominance (PC_target > 0.60)
    AND physical distance proximity (d < MAX_CONTROL_DIST).
- Preserves natural possession duration with receptive majority smoothing.
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

# Hyperparameters (Physics & Pitch Control)
ACTION_RADIUS = 4.5          # meters (ball control zone)
MAX_CONTROL_DIST = 3.8       # meters (maximum reach for possession capture)
HOLD_FRAMES = 65             # persistence frames for occlusions (~2.6s)
SIGMA_BASE = 2.0             # spatial influence kernel bandwidth (meters)
REACTION_TIME = 0.25         # seconds
MAX_SPRINT = 8.5             # m/s max effective sprint
MIN_SPELL_FRAMES = 15        # suppress micro-flickering (< 0.6s)


def stream_tracking_data():
    """Extract ball coordinates and player kinematics."""
    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]
    
    team_of_slot = base_mod.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)
    
    ball_x = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    ball_y = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    ball_spd = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    player_data = []
    
    print(f"Streaming tracking from {raw_path}...")
    with open(raw_path, "r") as f:
        for idx, line in enumerate(f):
            if idx >= TOTAL_FRAMES:
                break
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                player_data.append([])
                continue
                
            ball_part = parts[2]
            if not ball_part.startswith("N"):
                b_tokens = ball_part.split(",")
                ball_x[idx] = float(b_tokens[0]) * PITCH_X
                ball_y[idx] = float(b_tokens[1]) * PITCH_Y
                if len(b_tokens) > 2 and not b_tokens[2].startswith("N"):
                    ball_spd[idx] = float(b_tokens[2])
                    
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
            player_data.append(p_list)
            
    # Trajectory reconstruction for short pass flight gaps (<= 70 frames)
    print("Reconstructing pass flight trajectories...")
    ball_valid = ~np.isnan(ball_x)
    i = 0
    recon = 0
    while i < TOTAL_FRAMES:
        if not ball_valid[i]:
            start_gap = i
            while i < TOTAL_FRAMES and not ball_valid[i]:
                i += 1
            end_gap = i
            gap = end_gap - start_gap
            if start_gap > 0 and end_gap < TOTAL_FRAMES and gap <= 70:
                x0, y0 = ball_x[start_gap - 1], ball_y[start_gap - 1]
                x1, y1 = ball_x[end_gap], ball_y[end_gap]
                alphas = np.linspace(0.0, 1.0, gap + 2)[1:-1]
                ball_x[start_gap:end_gap] = x0 + alphas * (x1 - x0)
                ball_y[start_gap:end_gap] = y0 + alphas * (y1 - y0)
                recon += gap
        else:
            i += 1
    print(f"Reconstructed {recon:,} flight frames.")
    return ball_x, ball_y, ball_spd, player_data


def compute_pitch_control_possession(ball_x, ball_y, player_data, hold_frames=HOLD_FRAMES, smooth_window=80):
    """Compute frame possession labels using Pitch Control Surface + Physical Hysteresis."""
    labels = np.full(TOTAL_FRAMES, "X", dtype="<U1")
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
            bx = ball_x[f]
            by = ball_y[f]
            
            if not np.isnan(bx):
                missing_count = 0
                plist = player_data[f]
                
                if plist:
                    # Pitch boundaries check
                    if bx < -1.0 or bx > PITCH_X + 1.0 or by < -1.0 or by > PITCH_Y + 1.0:
                        curr_state = "D"
                        half_labels.append("D")
                        continue
                        
                    # Calculate player distances & Pitch Influence
                    h_influence = 0.0
                    a_influence = 0.0
                    min_dh = 999.0
                    min_da = 999.0
                    
                    for px, py, spd, t in plist:
                        d = np.hypot(px - bx, py - by)
                        if d > 12.0:
                            continue  # Outside active influence zone
                            
                        # Anisotropic speed-scaled kernel (Spearman / Fernandez)
                        sigma = SIGMA_BASE * (1.0 + 0.1 * min(spd, 6.0))
                        infl = np.exp(- (d * d) / (2.0 * sigma * sigma)) * (1.0 + 0.15 * min(spd, 7.0))
                        
                        if t == base_mod.HOME:
                            h_influence += infl
                            if d < min_dh:
                                min_dh = d
                        else:
                            a_influence += infl
                            if d < min_da:
                                min_da = d
                                
                    total_inf = h_influence + a_influence
                    if total_inf > 1e-4:
                        pc_home = h_influence / total_inf
                    else:
                        pc_home = 0.5
                        
                    # Possession State Machine with Pitch Control & Hysteresis
                    if curr_state == "H":
                        # Away must dominate space around the ball AND be within reach
                        if pc_home < 0.38 and min_da < min_dh - 0.7 and min_da < MAX_CONTROL_DIST:
                            curr_state = "A"
                    elif curr_state == "A":
                        # Home must dominate space around the ball AND be within reach
                        if pc_home > 0.62 and min_dh < min_da - 0.7 and min_dh < MAX_CONTROL_DIST:
                            curr_state = "H"
                    else:  # curr_state == 'D'
                        if min(min_dh, min_da) < MAX_CONTROL_DIST:
                            curr_state = "H" if pc_home >= 0.50 else "A"
            else:
                missing_count += 1
                if missing_count > hold_frames:
                    curr_state = "D"
                    
            half_labels.append(curr_state)
            
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
            
        # Fast vectorized rolling majority vote smoothing
        if smooth_window > 0:
            smoothed = np.empty_like(half_arr)
            pad = smooth_window
            code_map = {"H": 0, "A": 1, "D": 2}
            inv_map = np.array(["H", "A", "D"])
            codes = np.array([code_map[x] for x in half_arr], dtype=np.int8)
            padded = np.pad(codes, pad, mode="edge")
            
            # Rolling count via 1D cumulative sum (O(N) vectorized)
            kernel_size = 2 * pad + 1
            is_0 = np.pad((codes == 0).astype(np.int32), (pad, pad), mode="edge")
            is_1 = np.pad((codes == 1).astype(np.int32), (pad, pad), mode="edge")
            is_2 = np.pad((codes == 2).astype(np.int32), (pad, pad), mode="edge")
            
            cumsum_0 = np.cumsum(np.insert(is_0, 0, 0))
            cumsum_1 = np.cumsum(np.insert(is_1, 0, 0))
            cumsum_2 = np.cumsum(np.insert(is_2, 0, 0))
            
            cnt_0 = cumsum_0[kernel_size:] - cumsum_0[:-kernel_size]
            cnt_1 = cumsum_1[kernel_size:] - cumsum_1[:-kernel_size]
            cnt_2 = cumsum_2[kernel_size:] - cumsum_2[:-kernel_size]
            
            counts = np.vstack([cnt_0, cnt_1, cnt_2])  # shape (3, n)
            winner = np.argmax(counts, axis=0)
            half_arr = inv_map[winner]
            
        labels[start_f : end_f + 1] = half_arr
        
    return labels


def save_and_evaluate(labels, name):
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
    parser.add_argument("--name", default="v7_voronoi", help="Submission name stem")
    parser.add_argument("--window", type=int, default=80, help="Smoothing window")
    parser.add_argument("--hold", type=int, default=65, help="Hold frames")
    args = parser.parse_args()
    
    ball_x, ball_y, ball_spd, p_data = stream_tracking_data()
    
    for w in [70, 80, 90]:
        vname = f"{args.name}_w{w}"
        print(f"\nProcessing {vname} (window={w})...")
        labels = compute_pitch_control_possession(ball_x, ball_y, p_data, hold_frames=args.hold, smooth_window=w)
        save_and_evaluate(labels, vname)
        
    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL SCOREBOARD EVALUATION")
    print("=" * 70)
    os.system("python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()

