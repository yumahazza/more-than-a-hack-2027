"""extract_all_player_speeds.py - Precompute ATD tracking caches into scratch/.

Generates the five .npy arrays expected by v8-v12 possession models:
- scratch/ball_vis.npy
- scratch/dist_h.npy
- scratch/dist_a.npy
- scratch/ball_speed_mps.npy
- scratch/mean_player_speed.npy
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths
import starter.baseline as base_mod


def main():
    paths = asset_paths(source="tactical")
    total_frames = GAMES[GAME]["n_video_frames"]
    pitch_x, pitch_y = GAMES[GAME]["pitch"]

    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]
    team_of_slot = base_mod.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)

    scratch_dir = os.path.join(ROOT, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    ball_vis = np.zeros(total_frames, dtype=bool)
    ball_x = np.full(total_frames, np.nan, dtype=np.float32)
    ball_y = np.full(total_frames, np.nan, dtype=np.float32)
    dist_h = np.full(total_frames, 999.0, dtype=np.float32)
    dist_a = np.full(total_frames, 999.0, dtype=np.float32)
    mean_player_speed = np.zeros(total_frames, dtype=np.float32)

    print(f"Streaming tracking from {raw_path}...")
    with open(raw_path, "r") as f:
        for idx, line in enumerate(f):
            if idx >= total_frames:
                break
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue

            ball_part = parts[2]
            has_ball = not ball_part.startswith("N")
            if has_ball:
                b_tok = ball_part.split(",")
                bx = float(b_tok[0]) * pitch_x
                by = float(b_tok[1]) * pitch_y
                ball_vis[idx] = True
                ball_x[idx] = bx
                ball_y[idx] = by

            min_h = 999.0
            min_a = 999.0
            spd_sum = 0.0
            spd_cnt = 0

            for slot, p_str in enumerate(parts[1].split(";")):
                if slot >= n_slots or p_str.startswith("N"):
                    continue
                c = p_str.split(",")
                if len(c) > 2 and not c[2].startswith("N"):
                    spd_sum += float(c[2])
                    spd_cnt += 1

                if has_ball:
                    px = float(c[0]) * pitch_x
                    py = float(c[1]) * pitch_y
                    d = float(np.hypot(px - bx, py - by))
                    if team_of_slot[slot] == base_mod.HOME:
                        if d < min_h:
                            min_h = d
                    else:
                        if d < min_a:
                            min_a = d

            if has_ball:
                dist_h[idx] = min_h
                dist_a[idx] = min_a
            if spd_cnt > 0:
                mean_player_speed[idx] = spd_sum / spd_cnt

    ball_speed = np.zeros(total_frames, dtype=np.float32)
    for i in range(1, total_frames):
        if ball_vis[i] and ball_vis[i - 1]:
            ball_speed[i] = np.hypot(ball_x[i] - ball_x[i - 1], ball_y[i] - ball_y[i - 1]) * 25.0

    np.save(os.path.join(scratch_dir, "ball_vis.npy"), ball_vis)
    np.save(os.path.join(scratch_dir, "dist_h.npy"), dist_h)
    np.save(os.path.join(scratch_dir, "dist_a.npy"), dist_a)
    np.save(os.path.join(scratch_dir, "ball_speed_mps.npy"), ball_speed)
    np.save(os.path.join(scratch_dir, "mean_player_speed.npy"), mean_player_speed)
    print(f"Saved precomputed caches to {scratch_dir} ({total_frames} frames).")


if __name__ == "__main__":
    main()
