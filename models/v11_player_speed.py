"""v11_player_speed.py - Ball possession estimation with collective player speed filtering.

Model pipeline:
1. Nearest player distance tracking with 0.8m hysteresis margin and 4.0m control radius.
2. Deflection bridging: momentary touches (< 1.6s / 40 frames) without control do not switch possession.
3. Aerial flight filter: when ball speed > 14.0 m/s (~50 km/h), require player distance < 1.2m to intercept.
4. Set piece restart physics: maintain dead ball ('D') during set pieces until restart kick (> 3.5 m/s).
5. Low player speed dead ball filter: when mean player speed < 1.2 m/s sustained for >= 2.0s.
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
RESTART_KICK_SPEED = 3.5      # m/s velocity impulse marking set piece restart
FLOCKING_MAX_SPEED = 1.2      # m/s collective player walk speed marking dead ball
FLOCKING_MIN_FRAMES = 50      # frames (2.0s) sustained lull to trigger dead ball
BALL_DEAD_MAX_SPEED = 8.0     # m/s ball speed ceiling during player lull
HOLD_FRAMES = 70              # missing ball persistence (~2.8s)
CONFIRM_FRAMES = 40           # deflection bridging duration (~1.6s)
SMOOTH_WINDOW = 80            # receptive smoothing window (~3.2s)


def stream_data():
    cache_vis = os.path.join(ROOT, "scratch", "ball_vis.npy")
    cache_dh = os.path.join(ROOT, "scratch", "dist_h.npy")
    cache_da = os.path.join(ROOT, "scratch", "dist_a.npy")
    cache_spd = os.path.join(ROOT, "scratch", "ball_speed_mps.npy")
    cache_flock = os.path.join(ROOT, "scratch", "mean_player_speed.npy")
    
    if (os.path.exists(cache_vis) and os.path.exists(cache_dh) and 
        os.path.exists(cache_da) and os.path.exists(cache_spd) and os.path.exists(cache_flock)):
        print("Loading cached player-ball distances, velocity profile, and flocking speeds...")
        return (np.load(cache_vis), np.load(cache_dh), np.load(cache_da), 
                np.load(cache_spd), np.load(cache_flock))
        
    raise FileNotFoundError("Precomputed caches missing. Run extract_all_player_speeds.py first.")


def run_v11_model(
    vis, dh, da, bspd, flock_spd,
    flight_thresh=FLIGHT_SPEED_THRESH, 
    confirm_frames=CONFIRM_FRAMES, 
    hold_frames=HOLD_FRAMES, 
    smooth_window=SMOOTH_WINDOW,
    flock_thresh=FLOCKING_MAX_SPEED,
    flock_min_len=FLOCKING_MIN_FRAMES
):
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
            inv_map = np.array(["H", "A", "D"])
            arr = inv_map[winner]
            
        labels[start_f:end_f + 1] = arr
        
    # Multimodal Set-Piece Dead-Ball Physics:
    print("Applying multimodal set-piece restart physics from Smart Tagging...")
    events = lst.load()
    set_pieces = [e for e in events if e['code'].lower() == 'set pieces']
    print(f"Incorporating {len(set_pieces)} set piece intervals...")
    
    for sp in set_pieces:
        st = sp['start_frame']
        en = sp['end_frame']
        kicked = False
        for f in range(st, min(en, len(labels))):
            if not kicked:
                if vis[f] and bspd[f] > RESTART_KICK_SPEED and min(dh[f], da[f]) < 3.0:
                    kicked = True
                else:
                    labels[f] = "D"
                    
    # Collective Flocking Speed Dead-Ball Detection:
    print("Applying collective flocking speed dead-ball filter...")
    for half_num, start_f, end_f in halves:
        h_flock = flock_spd[start_f:end_f + 1]
        h_bspd = bspd[start_f:end_f + 1]
        is_slow = (h_flock < flock_thresh) & (h_bspd < BALL_DEAD_MAX_SPEED)
        
        i = 0
        n = len(is_slow)
        while i < n:
            if is_slow[i]:
                st = i
                while i < n and is_slow[i]:
                    i += 1
                en = i
                if (en - st) >= flock_min_len:
                    labels[start_f + st : start_f + en] = "D"
            else:
                i += 1
                
    return labels


def main():
    parser = argparse.ArgumentParser(description="Run v11 Collective Flocking model.")
    parser.add_argument("--window", type=int, default=80, help="Smoothing window (e.g. 75, 80)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    vis, dh, da, bspd, flock_spd = stream_data()
    print(f"Executing v11 Collective Flocking (window={args.window})...")
    
    labels = run_v11_model(
        vis, dh, da, bspd, flock_spd,
        flight_thresh=FLIGHT_SPEED_THRESH,
        confirm_frames=CONFIRM_FRAMES, 
        smooth_window=args.window,
        flock_thresh=FLOCKING_MAX_SPEED,
        flock_min_len=FLOCKING_MIN_FRAMES
    )
    
    out_name = args.output or f"v11_player_speed_w{args.window}.csv"
    out_path = os.path.join(ROOT, "evaluation", "results", out_name)
    
    df = pd.DataFrame({"frame": np.arange(TOTAL_FRAMES), "label": labels})
    df.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path} ({len(df)} rows).")


if __name__ == "__main__":
    main()

