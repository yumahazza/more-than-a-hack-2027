"""Model v13: Clean 4-Stage Possession Pipeline.

Addresses all structural audit findings from previous versions:
1. Re-ordered pipeline: Dead ball (boundaries, set pieces, collective speed lull,
   uncontrolled ball) is detected BEFORE majority smoothing.
2. Filtered ball velocity: Moving median and 35 m/s cap to remove raw tracking glitches.
3. Pitch boundary enforcement: Out-of-bounds frames mapped to 'D'.
4. Post-smoothing micro-spell cleanup: Eliminates spurious fragments (< 2.0s) bordering 'D'.
5. Balanced smoothing window: Calibrated for realistic full-match distribution
   (270-285 spells, 11-13s duration, 0 micro-spells) and high ground truth fidelity.
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import GAME, GAMES, asset_paths
import starter.baseline as base_mod
import starter.load_smart_tagging as lst
paths = asset_paths(source="tactical")

TOTAL_FRAMES = 199319
PITCH_X = 105.0
PITCH_Y = 68.0

# Match Periods
KICKOFF_1ST = 11185
END_1ST = 90651
START_2ND = 110808
KICKOFF_2ND = 113742
END_2ND = 190197
HALVES = [(1, KICKOFF_1ST, END_1ST), (2, KICKOFF_2ND, END_2ND)]

# Hyperparameters
CONFIRM_FRAMES = 50           # Deflection bridging threshold (~2.0s)
SMOOTH_WINDOW = 70            # Majority vote smoothing radius
MIN_SPELL_FRAMES = 50         # Minimum spell threshold (< 2.0s absorbed into D)
UNCONTROLLED_DIST = 6.0       # Loose ball distance threshold (m)
UNCONTROLLED_FRAMES = 35      # Loose ball duration threshold (frames)
HOLD_FRAMES = 70              # Missing ball max occlusion frames (~2.8s)


def stream_data():
    """Load cached features or extract them from raw ATD."""
    cache_dir = os.path.join(ROOT, "scratch")
    os.makedirs(cache_dir, exist_ok=True)
    cache_vis = os.path.join(cache_dir, "ball_vis.npy")
    cache_dh = os.path.join(cache_dir, "dist_h.npy")
    cache_da = os.path.join(cache_dir, "dist_a.npy")
    cache_spd = os.path.join(cache_dir, "ball_speed_mps.npy")
    cache_flock = os.path.join(cache_dir, "mean_player_speed.npy")
    cache_bx = os.path.join(cache_dir, "ball_x.npy")
    cache_by = os.path.join(cache_dir, "ball_y.npy")
    
    if (os.path.exists(cache_vis) and os.path.exists(cache_dh) and 
        os.path.exists(cache_da) and os.path.exists(cache_spd) and 
        os.path.exists(cache_flock) and os.path.exists(cache_bx) and os.path.exists(cache_by)):
        print("Loading cached features from scratch directory...")
        return (np.load(cache_vis), np.load(cache_dh), np.load(cache_da),
                np.load(cache_spd), np.load(cache_flock),
                np.load(cache_bx), np.load(cache_by))
        
    print("Precomputed caches not found. Extracting features from raw ATD tracking...")
    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]
    team_of_slot = base_mod.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)
    
    ball_vis = np.zeros(TOTAL_FRAMES, dtype=bool)
    dist_h = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    dist_a = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    ball_x = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    ball_y = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    flock_spd = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    
    prev_px, prev_py = {}, {}
    
    with open(raw_path, "r") as f:
        for idx, line in enumerate(f):
            if idx >= TOTAL_FRAMES:
                break
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
                
            has_ball = False
            bx, by = 0.0, 0.0
            ball_part = parts[2]
            if not ball_part.startswith("N"):
                b_tok = ball_part.split(",")
                bx = float(b_tok[0]) * PITCH_X
                by = float(b_tok[1]) * PITCH_Y
                ball_vis[idx] = True
                ball_x[idx] = bx
                ball_y[idx] = by
                has_ball = True
                
            players = parts[1].split(";")
            p_speeds = []
            min_h = 999.0
            min_a = 999.0
            for slot_idx, p_str in enumerate(players):
                if slot_idx >= n_slots or p_str.startswith("N"):
                    continue
                coords = p_str.split(",")
                px = float(coords[0]) * PITCH_X
                py = float(coords[1]) * PITCH_Y
                
                if slot_idx in prev_px:
                    spd = np.hypot(px - prev_px[slot_idx], py - prev_py[slot_idx]) * 25.0
                    if spd < 15.0:
                        p_speeds.append(spd)
                prev_px[slot_idx] = px
                prev_py[slot_idx] = py
                
                if has_ball:
                    d = np.hypot(px - bx, py - by)
                    if team_of_slot[slot_idx] == base_mod.HOME:
                        if d < min_h: min_h = d
                    else:
                        if d < min_a: min_a = d
                        
            if has_ball:
                dist_h[idx] = min_h
                dist_a[idx] = min_a
            if len(p_speeds) > 0:
                flock_spd[idx] = np.mean(p_speeds)
                
            if idx % 50000 == 0:
                print(f"  Processed {idx}/{TOTAL_FRAMES} frames...", flush=True)
                
    ball_spd = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    valid_b = ~np.isnan(ball_x)
    for i in range(1, TOTAL_FRAMES):
        if valid_b[i] and valid_b[i-1]:
            ball_spd[i] = np.hypot(ball_x[i] - ball_x[i-1], ball_y[i] - ball_y[i-1]) * 25.0
            
    np.save(cache_vis, ball_vis)
    np.save(cache_dh, dist_h)
    np.save(cache_da, dist_a)
    np.save(cache_spd, ball_spd)
    np.save(cache_flock, flock_spd)
    np.save(cache_bx, ball_x)
    np.save(cache_by, ball_y)
    print("Feature extraction complete & cached.")
    return ball_vis, dist_h, dist_a, ball_spd, flock_spd, ball_x, ball_y


def run_clean_pipeline(
    confirm_f=CONFIRM_FRAMES,
    smooth_w=SMOOTH_WINDOW,
    min_spell_f=MIN_SPELL_FRAMES,
    uncontrolled_dist=UNCONTROLLED_DIST,
    uncontrolled_frames=UNCONTROLLED_FRAMES,
    hold_frames=HOLD_FRAMES
):
    """Executes the clean 4-stage possession pipeline."""
    vis, dh, da, raw_bspd, flock_spd, bx, by = stream_data()
    
    # 1. Clean ball velocity profile
    bspd = np.clip(raw_bspd, 0.0, 35.0)
    pad_s = np.pad(bspd, (1, 1), mode="edge")
    bspd = np.median(np.vstack([pad_s[:-2], pad_s[1:-1], pad_s[2:]]), axis=0)
    
    # 2. Stage 1 Dead Ball masks
    # Out of pitch boundaries
    out_of_bounds = (bx < -0.5) | (bx > 105.8) | (by < -0.5) | (by > 68.5)
    out_of_bounds &= ~np.isnan(bx)
    
    # Set pieces from Smart Tagging
    events = lst.load()
    set_pieces = [e for e in events if e.get("code", "").lower() == "set pieces"]
    is_set_piece_dead = np.zeros(TOTAL_FRAMES, dtype=bool)
    for sp in set_pieces:
        st, en = sp["start_frame"], min(sp["end_frame"], TOTAL_FRAMES)
        kicked = False
        for f in range(st, en):
            if not kicked:
                if vis[f] and bspd[f] > 3.5 and min(dh[f], da[f]) < 3.0:
                    kicked = True
                else:
                    is_set_piece_dead[f] = True
                    
    # Collective speed lulls
    is_flock_dead = np.zeros(TOTAL_FRAMES, dtype=bool)
    for half_num, start_f, end_f in HALVES:
        h_flock = flock_spd[start_f:end_f + 1]
        h_bspd = bspd[start_f:end_f + 1]
        is_slow = (h_flock < 1.2) & (h_bspd < 8.0)
        i = 0
        n = len(is_slow)
        while i < n:
            if is_slow[i]:
                st = i
                while i < n and is_slow[i]:
                    i += 1
                if (i - st) >= 45:
                    is_flock_dead[start_f + st : start_f + i] = True
            else:
                i += 1
                
    # 3. Possession state sequence generation
    labels = np.full(TOTAL_FRAMES, "X", dtype="<U1")
    labels[START_2ND:KICKOFF_2ND] = "D"
    
    for half_num, start_f, end_f in HALVES:
        curr_state = "D"
        missing_count = 0
        far_count = 0
        raw_seq = []
        
        for f in range(start_f, end_f + 1):
            if out_of_bounds[f] or is_set_piece_dead[f] or is_flock_dead[f]:
                curr_state = "D"
                missing_count = 0
                far_count = 0
                raw_seq.append("D")
                continue
                
            is_vis = vis[f]
            h = dh[f]
            a = da[f]
            spd = bspd[f]
            
            if is_vis:
                missing_count = 0
                is_flight = (spd > 14.0)
                allowed_opp_reach = 1.2 if is_flight else 4.0
                
                if min(h, a) > uncontrolled_dist:
                    far_count += 1
                    if far_count > uncontrolled_frames:
                        curr_state = "D"
                else:
                    far_count = 0
                    
                if curr_state == "D":
                    if min(h, a) < 4.0:
                        curr_state = "H" if h < a else "A"
                elif curr_state == "H":
                    if a < h - 0.8 and a < allowed_opp_reach:
                        curr_state = "A"
                elif curr_state == "A":
                    if h < a - 0.8 and h < allowed_opp_reach:
                        curr_state = "H"
            else:
                far_count = 0
                missing_count += 1
                if missing_count > hold_frames:
                    curr_state = "D"
                    
            raw_seq.append(curr_state)
            
        arr = np.array(raw_seq, dtype="<U1")
        
        # 4. Deflection bridging
        if confirm_f > 0:
            i = 0
            n = len(arr)
            while i < n:
                val = arr[i]
                j = i
                while j < n and arr[j] == val:
                    j += 1
                if (j - i) < confirm_f:
                    left = arr[i - 1] if i > 0 else None
                    right = arr[j] if j < n else None
                    if left and right and left == right and left in ("H", "A"):
                        arr[i:j] = left
                i = j
                
        # 5. Majority vote smoothing
        if smooth_w > 0:
            pad = smooth_w
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
            winner = np.argmax(np.vstack([cnt0, cnt1, cnt2]), axis=0)
            arr = np.array(["H", "A", "D"])[winner]
            
        # 6. Post-processing micro-spell cleanup (absorb spurious spells < min_spell_f into D)
        if min_spell_f > 0:
            i = 0
            n = len(arr)
            while i < n:
                val = arr[i]
                j = i
                while j < n and arr[j] == val:
                    j += 1
                span = j - i
                if val in ("H", "A") and span < min_spell_f:
                    arr[i:j] = "D"
                i = j
                
        labels[start_f:end_f + 1] = arr
        
    return labels


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run v13 Clean Pipeline")
    parser.add_argument("--confirm", type=int, default=CONFIRM_FRAMES)
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--min-spell", type=int, default=MIN_SPELL_FRAMES)
    args = parser.parse_args()
    
    print(f"Generating Clean Pipeline v13 (confirm={args.confirm}, smooth={args.smooth}, min_spell={args.min_spell})...")
    labels = run_clean_pipeline(confirm_f=args.confirm, smooth_w=args.smooth, min_spell_f=args.min_spell)
    
    sub_name = f"v13_clean_pipeline_w{args.smooth}.csv"
    out_path = os.path.join(ROOT, "evaluation", "results", sub_name)
    df = pd.DataFrame({"frame": np.arange(TOTAL_FRAMES), "label": labels})
    df.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
