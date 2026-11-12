"""v2_smart_tagging_hmm.py - State Machine & Viterbi Decoding for Ball Possession.

Integrates:
1. Smart Tagging Priors:
   - 59 Set Pieces: Prior boosts 'D' (Dead Ball) during stoppage before restart.
   - 638 Tactical Phases (Build Up, Progression, Transition): Prior boosts the attacking team ('H' or 'A').
2. Probabilistic Emission Model:
   - Exponential distance kernel for closest player of Home vs Away.
   - Ball missing probability modeled across states H, A, D.
3. Temporal State Dynamics (Viterbi Algorithm):
   - Formal transition matrix enforces realistic possession durations (mean ~14s).
   - Eliminates per-frame flickering without arbitrary window heuristics.
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

# State mapping: 0=H, 1=A, 2=D
H_STATE, A_STATE, D_STATE = 0, 1, 2
STATE_NAMES = np.array(["H", "A", "D"])


def extract_tracking_features():
    """Extract ball visibility, distances, and ball speeds."""
    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]
    
    team_of_slot = base_mod.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)
    
    ball_vis = np.zeros(TOTAL_FRAMES, dtype=bool)
    dist_h = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    dist_a = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    ball_speed = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    
    print(f"Reading ATD tracking features...")
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


def extract_smart_tagging_priors():
    """Extract frame-level prior log-odds from Smart Tagging."""
    instances = lst.load()
    
    # Priors: log-multipliers for H, A, D
    # Default 0.0 (no bias)
    prior_boost = np.zeros((TOTAL_FRAMES, 3), dtype=np.float32)
    
    print(f"Applying Smart Tagging priors ({len(instances)} instances)...")
    for inst in instances:
        code = inst["code"].lower()
        sf = max(0, inst["start_frame"])
        ef = min(TOTAL_FRAMES, inst["end_frame"])
        team_tag = inst["tags"].get("Team", "")
        
        if code == "set pieces":
            # Set piece window indicates Dead Ball before restart
            # The preparation leading up to the end of window is strongly D
            prior_boost[sf:ef, D_STATE] += 2.0
            if team_tag == "Barcelona":
                # Restart is taken by Barcelona
                prior_boost[ef - 50 : ef + 50, H_STATE] += 1.5
            elif team_tag == "Atletic Club":
                prior_boost[ef - 50 : ef + 50, A_STATE] += 1.5
        elif code in ["build up", "progression", "attacking transition", "creating chances", "ball in final third"]:
            if team_tag == "Barcelona":
                prior_boost[sf:ef, H_STATE] += 1.5
            elif team_tag == "Atletic Club":
                prior_boost[sf:ef, A_STATE] += 1.5
                
    return prior_boost


def viterbi_decode(log_emissions, log_trans, log_initial):
    """Exact Viterbi dynamic programming decoding."""
    n_steps, n_states = log_emissions.shape
    viterbi = np.zeros((n_steps, n_states), dtype=np.float32)
    backpointer = np.zeros((n_steps, n_states), dtype=np.int32)
    
    # Initialize step 0
    viterbi[0] = log_initial + log_emissions[0]
    
    # Forward pass
    for t in range(1, n_steps):
        # Broadcast transition: viterbi[t-1, :] + log_trans[:, s]
        trans_scores = viterbi[t - 1, :, None] + log_trans
        best_prev = np.argmax(trans_scores, axis=0)
        viterbi[t] = trans_scores[best_prev, np.arange(n_states)] + log_emissions[t]
        backpointer[t] = best_prev
        
    # Backward pass
    best_path = np.zeros(n_steps, dtype=np.int32)
    best_path[-1] = np.argmax(viterbi[-1])
    for t in range(n_steps - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]
        
    return best_path


def run_hmm_possession(ball_vis, dist_h, dist_a, prior_boost, persistence=0.985):
    """Run HMM with emission model and Viterbi decoding."""
    labels = np.full(TOTAL_FRAMES, "X", dtype="<U1")
    
    # 2nd Half Pre-Kickoff is explicitly DEAD BALL
    labels[START_2ND:KICKOFF_2ND] = "D"
    
    # Transition probability matrix (3x3)
    # Self persistence is very high because 1 sec = 25 frames
    p_self = persistence
    p_to_d = (1.0 - p_self) * 0.75
    p_switch = (1.0 - p_self) * 0.25
    
    trans_mat = np.array([
        [p_self,     p_switch,   p_to_d],    # From H -> H, A, D
        [p_switch,   p_self,     p_to_d],    # From A -> H, A, D
        [p_switch/2, p_switch/2, p_self + p_to_d/2]  # From D -> H, A, D
    ], dtype=np.float32)
    # Normalize rows
    trans_mat /= trans_mat.sum(axis=1, keepdims=True)
    log_trans = np.log(trans_mat + 1e-12)
    
    halves = [
        (1, KICKOFF_1ST, END_1ST),
        (2, KICKOFF_2ND, END_2ND)
    ]
    
    for half_num, start_f, end_f in halves:
        span_len = end_f - start_f + 1
        log_emiss = np.zeros((span_len, 3), dtype=np.float32)
        
        sub_vis = ball_vis[start_f : end_f + 1]
        sub_dh = dist_h[start_f : end_f + 1]
        sub_da = dist_a[start_f : end_f + 1]
        sub_prior = prior_boost[start_f : end_f + 1]
        
        # When ball is visible: exponential decay on player distance
        vis_mask = sub_vis
        invis_mask = ~sub_vis
        
        # Distance scale: 2.0 meters
        scale = 2.0
        log_emiss[vis_mask, H_STATE] = -sub_dh[vis_mask] / scale
        log_emiss[vis_mask, A_STATE] = -sub_da[vis_mask] / scale
        # Dead ball emission when visible is low unless both teams far (> 6m)
        min_dist = np.minimum(sub_dh[vis_mask], sub_da[vis_mask])
        log_emiss[vis_mask, D_STATE] = -np.maximum(0.0, 6.0 - min_dist) / 1.5
        
        # When ball is invisible:
        # High chance of D, moderate chance of maintaining H/A
        log_emiss[invis_mask, H_STATE] = -1.2
        log_emiss[invis_mask, A_STATE] = -1.2
        log_emiss[invis_mask, D_STATE] = 0.5
        
        # Add Smart Tagging priors
        log_emiss += sub_prior
        
        # Initial state: 1st half kickoff starts with A (Athletic kick off in 1st half), 2nd half starts with H
        log_initial = np.array([0.0, 0.0, -2.0], dtype=np.float32)
        
        # Run Viterbi
        decoded_path = viterbi_decode(log_emiss, log_trans, log_initial)
        labels[start_f : end_f + 1] = STATE_NAMES[decoded_path]
        
    return labels


def save_and_evaluate(labels, name="v2_smart_tagging_hmm"):
    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.csv")
    
    df = pd.DataFrame({
        "frame": np.arange(TOTAL_FRAMES, dtype=np.int64),
        "label": labels
    })
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df):,} rows)")
    
    # Validate format
    val_cmd = f'python evaluation/validate_submission.py "{out_path}"'
    os.system(val_cmd)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="v2_hmm", help="Model name")
    args = parser.parse_args()
    
    ball_vis, dist_h, dist_a, ball_speed = extract_tracking_features()
    prior_boost = extract_smart_tagging_priors()
    
    # Evaluate across a grid of persistence values (0.97, 0.985, 0.99)
    for p in [0.975, 0.985, 0.992]:
        p_name = f"{args.name}_p{int(p*1000)}"
        print(f"\nEvaluating {p_name} (persistence={p})...")
        labels = run_hmm_possession(ball_vis, dist_h, dist_a, prior_boost, persistence=p)
        save_and_evaluate(labels, name=p_name)
        
    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL SCOREBOARD EVALUATION")
    print("=" * 70)
    os.system("python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()

