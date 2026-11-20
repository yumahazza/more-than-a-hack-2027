"""v9_flight_filter.py - Ball possession estimation with aerial pass speed thresholding.

Model pipeline:
1. Nearest player distance tracking with 0.8m hysteresis margin and 4.0m control radius.
2. Deflection bridging: momentary touches (< 1.6s / 40 frames) without control do not switch possession.
3. Aerial flight filter: when ball speed > 14.0 m/s (~50 km/h), require player distance < 1.2m to intercept.
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

paths = asset_paths(source="tactical")
TOTAL_FRAMES = GAMES[GAME]["n_video_frames"]
PITCH_X, PITCH_Y = GAMES[GAME]["pitch"]

KICKOFF_1ST = 11185
END_1ST = 90651
START_2ND = 110808
KICKOFF_2ND = 113742
END_2ND = 190197

# Physics Hyperparameters
HYSTERESIS_MARGIN = 0.8       # meters required lead for advantage
MAX_CONTROL_DIST = 4.0        # meters active control radius
FLIGHT_TOUCH_DIST = 1.2       # meters required to intercept a high-speed ball
FLIGHT_SPEED_THRESH = 14.0    # m/s (~50 km/h) aerial ball flight threshold
HOLD_FRAMES = 70              # missing ball persistence (~2.8s)
CONFIRM_FRAMES = 40           # deflection bridging duration (~1.6s)
SMOOTH_WINDOW = 80            # receptive smoothing window (~3.2s)


def stream_data():
    cache_vis = os.path.join(ROOT, "scratch", "ball_vis.npy")
    cache_dh = os.path.join(ROOT, "scratch", "dist_h.npy")
    cache_da = os.path.join(ROOT, "scratch", "dist_a.npy")
    cache_spd = os.path.join(ROOT, "scratch", "ball_speed_mps.npy")
    
    if os.path.exists(cache_vis) and os.path.exists(cache_dh) and os.path.exists(cache_da) and os.path.exists(cache_spd):
        print("Loading cached player-ball distances and velocity profile...")
        return np.load(cache_vis), np.load(cache_dh), np.load(cache_da), np.load(cache_spd)
        
    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]
    team_of_slot = base_mod.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)
    
    ball_vis = np.zeros(TOTAL_FRAMES, dtype=bool)
    ball_x = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    ball_y = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    dist_h = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    dist_a = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    
    print(f"Streaming tracking from {raw_path}...")
    with open(raw_path, "r") as f:
        for idx, line in enumerate(f):
            if idx >= TOTAL_FRAMES:
                break
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            ball_part = parts[2]
            if ball_part.startswith("N"):
                continue
            b_tok = ball_part.split(",")
            bx, by = float(b_tok[0]) * PITCH_X, float(b_tok[1]) * PITCH_Y
            ball_vis[idx] = True
            ball_x[idx] = bx
            ball_y[idx] = by
            
            min_h = 999.0
            min_a = 999.0
            for slot, p_str in enumerate(parts[1].split(";")):
                if slot >= n_slots or p_str.startswith("N"):
                    continue
                c = p_str.split(",")
                px, py = float(c[0]) * PITCH_X, float(c[1]) * PITCH_Y
                d = np.hypot(px - bx, py - by)
                if team_of_slot[slot] == base_mod.HOME:
                    if d < min_h: min_h = d
                else:
                    if d < min_a: min_a = d
            dist_h[idx] = min_h
            dist_a[idx] = min_a
            
    # Numerical velocity (25 fps)
    ball_speed = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    for i in range(1, TOTAL_FRAMES):
        if ball_vis[i] and ball_vis[i-1]:
            ball_speed[i] = np.hypot(ball_x[i] - ball_x[i-1], ball_y[i] - ball_y[i-1]) * 25.0
            
    np.save(cache_vis, ball_vis)
    np.save(cache_dh, dist_h)
    np.save(cache_da, dist_a)
    np.save(cache_spd, ball_speed)
    return ball_vis, dist_h, dist_a, ball_speed


def run_ball_flight_model(vis, dh, da, bspd, flight_thresh=FLIGHT_SPEED_THRESH, confirm_frames=CONFIRM_FRAMES, hold_frames=HOLD_FRAMES, smooth_window=SMOOTH_WINDOW):
    labels = np.full(TOTAL_FRAMES, "X", dtype="<U1")
    labels[START_2ND:KICKOFF_2ND] = "D"
    
    halves = [(1, KICKOFF_1ST, END_1ST), (2, KICKOFF_2ND, END_2ND)]
    
    for half_num, start_f, end_f in halves:
        curr_state = "D"
        missing_count = 0
        raw_seq = []
        
        for f in range(start_f, end_f + 1):
            is_vis = vis[f]
            h = dh[f]
            a = da[f]
            spd = bspd[f]
            
            if is_vis:
                missing_count = 0
                is_flight = (spd > flight_thresh)
                allowed_opp_reach = FLIGHT_TOUCH_DIST if is_flight else MAX_CONTROL_DIST
                
                if h < 900.0 and a < 900.0:
                    if curr_state == "H":
                        if a < h - HYSTERESIS_MARGIN and a < allowed_opp_reach:
                            curr_state = "A"
                    elif curr_state == "A":
                        if h < a - HYSTERESIS_MARGIN and h < allowed_opp_reach:
                            curr_state = "H"
                    else:  # D
                        if min(h, a) < MAX_CONTROL_DIST:
                            curr_state = "H" if h < a else "A"
                elif h < MAX_CONTROL_DIST:
                    curr_state = "H"
                elif a < MAX_CONTROL_DIST:
                    curr_state = "A"
            else:
                missing_count += 1
                if missing_count > hold_frames:
                    curr_state = "D"
            raw_seq.append(curr_state)
            
        arr = np.array(raw_seq, dtype="<U1")
        
        # Turnover Confirmation / Deflection Bridging:
        # Avoid breaking long possessions due to momentary (< confirm_frames) opponent contact
        i = 0
        n = len(arr)
        while i < n:
            val = arr[i]
            j = i
            while j < n and arr[j] == val:
                j += 1
            span = j - i
            if span < confirm_frames:
                left = arr[i - 1] if i > 0 else None
                right = arr[j] if j < n else None
                if left and right and left == right and left in ("H", "A"):
                    arr[i:j] = left
            i = j
            
        # Fast vectorized rolling majority vote smoothing
        if smooth_window > 0:
            pad = smooth_window
            inv_map = np.array(["H", "A", "D"])
            codes = np.array([{"H": 0, "A": 1, "D": 2}[x] for x in arr], dtype=np.int8)
            kernel_size = 2 * pad + 1
            
            is_0 = np.pad((codes == 0).astype(np.int32), (pad, pad), mode="edge")
            is_1 = np.pad((codes == 1).astype(np.int32), (pad, pad), mode="edge")
            is_2 = np.pad((codes == 2).astype(np.int32), (pad, pad), mode="edge")
            
            c0 = np.cumsum(np.insert(is_0, 0, 0))
            c1 = np.cumsum(np.insert(is_1, 0, 0))
            c2 = np.cumsum(np.insert(is_2, 0, 0))
            
            cnt0 = c0[kernel_size:] - c0[:-kernel_size]
            cnt1 = c1[kernel_size:] - c1[:-kernel_size]
            cnt2 = c2[kernel_size:] - c2[:-kernel_size]
            
            counts = np.vstack([cnt0, cnt1, cnt2])
            winner = np.argmax(counts, axis=0)
            arr = inv_map[winner]
            
        labels[start_f : end_f + 1] = arr
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
    parser.add_argument("--name", default="v9_flight_filter", help="Stem name")
    parser.add_argument("--v_thresh", type=float, default=14.0, help="Flight speed threshold m/s")
    parser.add_argument("--confirm", type=int, default=40, help="Turnover confirmation frames")
    parser.add_argument("--hold", type=int, default=70, help="Hold frames")
    args = parser.parse_args()
    
    vis, dh, da, bspd = stream_data()
    
    for w in [75, 80, 85]:
        vname = f"{args.name}_w{w}"
        print(f"\nGenerating {vname} (window={w}, v_thresh={args.v_thresh} m/s)...")
        labels = run_ball_flight_model(vis, dh, da, bspd, flight_thresh=args.v_thresh, confirm_frames=args.confirm, hold_frames=args.hold, smooth_window=w)
        save_and_evaluate(labels, vname)
        
    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL SCOREBOARD EVALUATION")
    print("=" * 70)
    os.system("python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()

