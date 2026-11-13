"""v16_unified_adaptive_viterbi.py - Two-Stage Unified Adaptive Viterbi Possession Model.

Fixes the Stage-1 vs Stage-2 structural bottleneck identified in v13-v15:
1. Preserves v15's strong Stage-1 Alive/Dead segmentation (Savitzky-Golay ball kinematics,
   vector Spearman T_int, collective team shape, self-supervised Ridge occlusion imputer,
   and Smart Tagging priors), while recalibrating short-stoppage exit barriers so quick
   set-piece restarts across the full match are not over-merged.
2. Replaces Stage-2's sluggish causal streak counter (CONFIRM_TAKEOVER_FRAMES=32) and
   wide post-hoc boxcar smoothing (smooth_1d) with an Exact 2-State Adaptive Viterbi (H/A)
   inside each contiguous Alive span:
   - Integrates evidence bi-directionally so genuine 1.5s-3.5s possessions survive cleanly
     with frame-exact boundaries (higher IoU at 0.6/0.7) while sub-second deflections are
     naturally suppressed by the Viterbi transition barrier.
   - Relaxes H <-> A transition barriers locally at physical ball-touch / vector-kink events
     (Link & Hoernig 2017) so turnovers snap directly onto physical touches.
3. Scopes flight protection and contextual priors cleanly:
   - Flight suppression only gates mid-flight frames away from players without touch impulse,
     preventing rapid interceptions followed by immediate passes from freezing turnovers.
   - Collective team-shape and phase-tag team priors guide missing-ball and contested frames
     without overriding clear close-range visible ball control.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths
import starter.baseline as base_mod
import starter.load_smart_tagging as lst

TOTAL_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
PITCH_X, PITCH_Y = GAMES[GAME]["pitch"]

# Match half boundaries (DATA.md / games.py)
KICKOFF_1ST = 11185
END_1ST = 90651
START_2ND = 110808
KICKOFF_2ND = 113742
END_2ND = 190197

# Physics & Sports Science Constants (derived from kinematics & sports science, never tuned on public GT)
REACTION_TIME_S = 0.25          # Spearman (2018) human reaction latency
MAX_SPRINT_MPS = 8.5            # Effective sprint speed cap
K_NEAREST = 5                   # Borghesi et al. (2023) 5NN neighborhood
HSR_SPEED_MPS = 3.5             # High-speed running threshold (Jerome et al. 2023)
PASS_FLIGHT_SPEED_MPS = 8.5     # Driven pass speed threshold (Link & Hoernig 2017)
AERIAL_FLIGHT_SPEED_MPS = 13.0  # Fast aerial pass threshold
FLIGHT_TOUCH_DIST_M = 1.30      # Direct interception proximity required during aerial flight
RESTART_KICK_SPEED_MPS = 4.2    # Smoothed ball speed confirming set-piece restart kick
OOB_MARGIN_M = 0.5              # Pitch line tolerance (105.3m x 68.0m)
SIDELINE_BAND_M = 2.5           # Sideline throw-in setup band
UNCONTROLLED_DIST_M = 6.5       # Loose ball radius
WALK_SPEED_MPS = 1.35           # Collective walking threshold during stoppages
SHORT_OCCLUSION_FRAMES = 15     # Micro-dropout interpolation window (0.6s)
MAX_TRACK_JUMP_M = 2.4          # Max physical ball step per frame (60 m/s)
SG_WINDOW = 13                  # Savitzky-Golay derivative window (0.52s)

# Stage-1 (Alive vs Dead) & Stage-2 (2-State H/A Viterbi) parameters
DEFAULT_SWITCH_PENALTY = -82.0  # Stage-1 H <-> A log-barrier
DEFAULT_DEAD_PENALTY = -66.0    # Stage-1 Alive <-> Dead log-barrier (proven v15 calibration)
STAGE2_SWITCH_PENALTY = -78.0   # Stage-2 exact 2-state H <-> A log-barrier
STAGE2_TOUCH_RELAX = 18.0       # Directional relaxation when receiving team touches ball
MIN_ALIVE_SPELL_FRAMES = 25     # Minimum surviving spell duration (~1.00s)
MIN_DEAD_RUN_FRAMES = 38        # Minimum surviving stoppage duration (~1.52s)
BRIDGE_CHALLENGE_FRAMES = 34    # Bridge momentary (<1.36s) deflections per CHALLENGE.md
DEAD_ISLAND_MAX_FRAMES = 120    # Max D -> Alive -> D low-tempo retrieval island (~4.8s)


def savgol_coeffs_deriv1(window_length, polyorder=2, dt=1.0 / FPS):
    """Compute Savitzky-Golay first-derivative convolution kernel in pure numpy."""
    half = window_length // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    vander = np.vstack([x ** p for p in range(polyorder + 1)]).T
    pinv = np.linalg.pinv(vander)
    return (pinv[1] / dt)[::-1]


def smooth_1d(arr, half_win):
    """Centered moving average with edge padding in O(N)."""
    if half_win <= 0:
        return arr.copy()
    k = 2 * half_win + 1
    padded = np.pad(arr.astype(np.float64), (half_win, half_win), mode="edge")
    csum = np.cumsum(np.insert(padded, 0, 0.0))
    return (csum[k:] - csum[:-k]) / float(k)


def _compute_team_shape(px_list, py_list, vx_list, vy_list):
    """Compute permutation-invariant collective team metrics (Frencken et al. 2011) quickly."""
    n = len(px_list)
    if n < 3:
        return 52.65, 34.0, 12.0, 400.0, 40.0, 65.0, 0.0, 0.0
    px = np.asarray(px_list, dtype=np.float64)
    py = np.asarray(py_list, dtype=np.float64)
    cx = float(px.mean())
    cy = float(py.mean())
    dx = px - cx
    dy = py - cy
    stretch = float(np.sqrt(dx * dx + dy * dy).mean())
    area = float(np.pi * px.std() * py.std())
    s_px = np.sort(px)
    i20 = max(0, min(n - 1, int(0.2 * (n - 1))))
    i80 = max(0, min(n - 1, int(0.8 * (n - 1))))
    mvx = float(sum(vx_list) / n)
    mvy = float(sum(vy_list) / n)
    return cx, cy, stretch, area, float(s_px[i20]), float(s_px[i80]), mvx, mvy


def extract_kinematic_features():
    """Stream ATD once and compute vector kinematics, collective team shape, and touch kinks."""
    paths = asset_paths(source="tactical")
    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]

    team_of_slot = base_mod.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)

    vel_lag = 5
    hist_x = np.full((n_slots, vel_lag), np.nan, dtype=np.float64)
    hist_y = np.full((n_slots, vel_lag), np.nan, dtype=np.float64)
    hist_f = np.full((n_slots, vel_lag), -100, dtype=np.int32)

    ball_vis = np.zeros(TOTAL_FRAMES, dtype=bool)
    ball_x = np.full(TOTAL_FRAMES, np.nan, dtype=np.float64)
    ball_y = np.full(TOTAL_FRAMES, np.nan, dtype=np.float64)

    dist_h = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    dist_a = np.full(TOTAL_FRAMES, 999.0, dtype=np.float32)
    tint_h = np.full(TOTAL_FRAMES, 99.0, dtype=np.float32)
    tint_a = np.full(TOTAL_FRAMES, 99.0, dtype=np.float32)
    score_h = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    score_a = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    mean_player_spd = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    hsr_players = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    opp_gap_arr = np.full(TOTAL_FRAMES, 99.0, dtype=np.float32)
    team_shape = np.zeros((TOTAL_FRAMES, 16), dtype=np.float32)

    hypot = np.hypot
    HOME = base_mod.HOME

    print(f"Streaming ATD vector + collective kinematics from {raw_path}...")
    with open(raw_path, "r") as handle:
        for idx, line in enumerate(handle):
            if idx >= TOTAL_FRAMES:
                break
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue

            ball_part = parts[2]
            has_ball = not ball_part.startswith("N")
            bx = by = np.nan
            if has_ball:
                btok = ball_part.split(",")
                bx = float(btok[0]) * PITCH_X
                by = float(btok[1]) * PITCH_Y
                ball_vis[idx] = True
                ball_x[idx] = bx
                ball_y[idx] = by

            spd_sum = 0.0
            spd_cnt = 0
            hsr_cnt = 0
            candidates = []
            hx, hy, hvx, hvy = [], [], [], []
            ax, ay, avx, avy = [], [], [], []

            buf_col = idx % vel_lag
            ref_col = (idx + 1) % vel_lag

            for slot, p_str in enumerate(parts[1].split(";")):
                if slot >= n_slots or p_str.startswith("N"):
                    continue
                c = p_str.split(",")
                px = float(c[0]) * PITCH_X
                py = float(c[1]) * PITCH_Y

                pspd = 1.5
                if len(c) > 2 and not c[2].startswith("N"):
                    pspd = float(c[2])
                    spd_sum += pspd
                    spd_cnt += 1
                    if pspd >= HSR_SPEED_MPS:
                        hsr_cnt += 1

                vx = vy = 0.0
                dt_f = idx - int(hist_f[slot, ref_col])
                if 1 <= dt_f <= 8 and not np.isnan(hist_x[slot, ref_col]):
                    inv_dt = FPS / float(dt_f)
                    raw_vx = (px - hist_x[slot, ref_col]) * inv_dt
                    raw_vy = (py - hist_y[slot, ref_col]) * inv_dt
                    if (raw_vx * raw_vx + raw_vy * raw_vy) <= 132.25:  # 11.5^2 m/s
                        vx, vy = raw_vx, raw_vy

                hist_x[slot, buf_col] = px
                hist_y[slot, buf_col] = py
                hist_f[slot, buf_col] = idx

                team = team_of_slot[slot]
                if team == HOME:
                    hx.append(px)
                    hy.append(py)
                    hvx.append(vx)
                    hvy.append(vy)
                else:
                    ax.append(px)
                    ay.append(py)
                    avx.append(vx)
                    avy.append(vy)

                if has_ball:
                    dx = bx - px
                    dy = by - py
                    d = float(hypot(dx, dy))
                    v_toward = (vx * dx + vy * dy) / d if d > 0.25 else max(pspd, 0.0)
                    turn_penalty = 0.30 if v_toward < -1.0 else 0.0
                    eff_spd = min(MAX_SPRINT_MPS, max(2.2, v_toward + 2.8))
                    t_int = REACTION_TIME_S + turn_penalty + d / eff_spd
                    candidates.append((d, t_int, pspd, v_toward, team))

            if spd_cnt > 0:
                mean_player_spd[idx] = spd_sum / spd_cnt
            hsr_players[idx] = float(hsr_cnt)

            team_shape[idx, :8] = _compute_team_shape(hx, hy, hvx, hvy)
            team_shape[idx, 8:] = _compute_team_shape(ax, ay, avx, avy)

            if has_ball and candidates:
                candidates.sort(key=lambda item: item[0])
                min_dh = 999.0
                min_da = 999.0
                min_th = 99.0
                min_ta = 99.0
                sh = 0.0
                sa = 0.0

                for rank, (d, t_int, pspd, v_toward, team) in enumerate(candidates):
                    if team == HOME:
                        if d < min_dh:
                            min_dh = d
                        if t_int < min_th:
                            min_th = t_int
                    else:
                        if d < min_da:
                            min_da = d
                        if t_int < min_ta:
                            min_ta = t_int

                    if rank < K_NEAREST:
                        vt_clip = -2.0 if v_toward < -2.0 else (5.0 if v_toward > 5.0 else v_toward)
                        approach_boost = 1.0 + 0.08 * vt_clip
                        w = float(np.exp(-d / 2.2)) * approach_boost / (1.0 + 0.15 * rank)
                        if team == HOME:
                            sh += w
                        else:
                            sa += w

                dist_h[idx] = min_dh
                dist_a[idx] = min_da
                tint_h[idx] = min_th
                tint_a[idx] = min_ta
                score_h[idx] = sh
                score_a[idx] = sa
                opp_gap_arr[idx] = min_da if min_dh <= min_da else min_dh

    # Reject multi-frame ball tracking glitches/teleports before differentiation
    clean_vis = ball_vis.copy()
    last_good = -1
    for i in range(TOTAL_FRAMES):
        if clean_vis[i]:
            if last_good >= 0:
                dt_frames = i - last_good
                jump = float(hypot(ball_x[i] - ball_x[last_good], ball_y[i] - ball_y[last_good]))
                if jump > 1.8 * dt_frames + 1.0:
                    clean_vis[i] = False
                    ball_x[i] = np.nan
                    ball_y[i] = np.nan
                    continue
            last_good = i

    # Interpolate micro-dropouts (<= SHORT_OCCLUSION_FRAMES)
    interp_x = ball_x.copy()
    interp_y = ball_y.copy()
    i = 0
    while i < TOTAL_FRAMES:
        if not clean_vis[i]:
            st = i
            while i < TOTAL_FRAMES and not clean_vis[i]:
                i += 1
            en = i
            gap = en - st
            if st > 0 and en < TOTAL_FRAMES and gap <= SHORT_OCCLUSION_FRAMES:
                jump = float(hypot(ball_x[en] - ball_x[st - 1], ball_y[en] - ball_y[st - 1]))
                if jump <= MAX_TRACK_JUMP_M * (gap + 1):
                    alphas = np.linspace(0.0, 1.0, gap + 2)[1:-1]
                    interp_x[st:en] = ball_x[st - 1] + alphas * (ball_x[en] - ball_x[st - 1])
                    interp_y[st:en] = ball_y[st - 1] + alphas * (ball_y[en] - ball_y[st - 1])
        else:
            i += 1

    # Savitzky-Golay smoothed ball velocity vector (vx, vy)
    ball_vx = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    ball_vy = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    ball_speed_sg = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    sg_kernel = savgol_coeffs_deriv1(SG_WINDOW, polyorder=2)
    half_sg = SG_WINDOW // 2
    valid_interp = ~np.isnan(interp_x)

    i = 0
    while i < TOTAL_FRAMES:
        if valid_interp[i]:
            st = i
            while i < TOTAL_FRAMES and valid_interp[i]:
                i += 1
            en = i
            seg_len = en - st
            if seg_len >= SG_WINDOW:
                px_seg = np.pad(interp_x[st:en], (half_sg, half_sg), mode="edge")
                py_seg = np.pad(interp_y[st:en], (half_sg, half_sg), mode="edge")
                vx = np.convolve(px_seg, sg_kernel, mode="valid")
                vy = np.convolve(py_seg, sg_kernel, mode="valid")
                ball_vx[st:en] = vx
                ball_vy[st:en] = vy
                ball_speed_sg[st:en] = np.hypot(vx, vy)
            elif seg_len >= 2:
                vx = np.gradient(interp_x[st:en]) * FPS
                vy = np.gradient(interp_y[st:en]) * FPS
                ball_vx[st:en] = vx
                ball_vy[st:en] = vy
                ball_speed_sg[st:en] = np.hypot(vx, vy)
        else:
            i += 1

    # Link & Hoernig (2017) ball touch impulse: vector |dv| + direction-change kink
    ball_impulse = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    lag = 4
    dvx = ball_vx[lag:] - ball_vx[:-lag]
    dvy = ball_vy[lag:] - ball_vy[:-lag]
    both_vis = clean_vis[lag:] & clean_vis[:-lag]
    ball_impulse[lag:] = np.where(both_vis, np.hypot(dvx, dvy), 0.0)

    vector_kink = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    k_lag = 3
    if TOTAL_FRAMES > 2 * k_lag:
        vx_b = ball_vx[:-2 * k_lag].astype(np.float64)
        vy_b = ball_vy[:-2 * k_lag].astype(np.float64)
        vx_a = ball_vx[2 * k_lag:].astype(np.float64)
        vy_a = ball_vy[2 * k_lag:].astype(np.float64)
        sp_b = np.hypot(vx_b, vy_b)
        sp_a = np.hypot(vx_a, vy_a)
        valid_kink = (
            clean_vis[:-2 * k_lag]
            & clean_vis[2 * k_lag:]
            & (sp_b >= 2.0)
            & (sp_a >= 2.0)
        )
        cos_sim = np.where(
            valid_kink,
            (vx_b * vx_a + vy_b * vy_a) / np.maximum(sp_b * sp_a, 1e-6),
            1.0,
        )
        vector_kink[k_lag:-k_lag] = np.where(valid_kink, 1.0 - np.clip(cos_sim, -1.0, 1.0), 0.0)

    touch_score = ball_impulse + 3.0 * vector_kink

    clean_vis = clean_vis & (~np.isnan(ball_x))
    is_oob = np.zeros(TOTAL_FRAMES, dtype=bool)
    is_oob[clean_vis] = (
        (ball_x[clean_vis] < -OOB_MARGIN_M)
        | (ball_x[clean_vis] > PITCH_X + OOB_MARGIN_M)
        | (ball_y[clean_vis] < -OOB_MARGIN_M)
        | (ball_y[clean_vis] > PITCH_Y + OOB_MARGIN_M)
    )
    near_sideline = np.zeros(TOTAL_FRAMES, dtype=bool)
    near_sideline[clean_vis] = (
        (ball_y[clean_vis] < SIDELINE_BAND_M)
        | (ball_y[clean_vis] > PITCH_Y - SIDELINE_BAND_M)
    )

    flock_smooth = smooth_1d(mean_player_spd, half_win=18).astype(np.float32)
    hsr_smooth = smooth_1d(hsr_players, half_win=12).astype(np.float32)

    for col in range(16):
        team_shape[:, col] = smooth_1d(team_shape[:, col], half_win=6).astype(np.float32)

    return {
        "vis": clean_vis,
        "oob": is_oob,
        "sideline": near_sideline,
        "dh": dist_h,
        "da": dist_a,
        "th": tint_h,
        "ta": tint_a,
        "sh": score_h,
        "sa": score_a,
        "bspd": ball_speed_sg,
        "touch": touch_score,
        "flock": flock_smooth,
        "hsr": hsr_smooth,
        "opp_gap": opp_gap_arr,
        "shape": team_shape,
    }


def fit_self_supervised_shape_prior(feat):
    """Train a pure-NumPy Ridge team-control estimator per half strictly on automatic ATD
    visible-ball frames (zero human labels, per RULES.md) from permutation-invariant
    collective team shape features (Frencken et al. 2011; Kim et al. 2023 principle).
    """
    shape = feat["shape"].astype(np.float64)
    vis = feat["vis"]
    oob = feat["oob"]
    dh = feat["dh"].astype(np.float64)
    da = feat["da"].astype(np.float64)

    vcx_h = np.zeros(TOTAL_FRAMES, dtype=np.float64)
    vcy_h = np.zeros(TOTAL_FRAMES, dtype=np.float64)
    vcx_a = np.zeros(TOTAL_FRAMES, dtype=np.float64)
    vcy_a = np.zeros(TOTAL_FRAMES, dtype=np.float64)
    lag = 12
    vcx_h[lag:] = (shape[lag:, 0] - shape[:-lag, 0]) * (FPS / lag)
    vcy_h[lag:] = (shape[lag:, 1] - shape[:-lag, 1]) * (FPS / lag)
    vcx_a[lag:] = (shape[lag:, 8] - shape[:-lag, 8]) * (FPS / lag)
    vcy_a[lag:] = (shape[lag:, 9] - shape[:-lag, 9]) * (FPS / lag)

    stretch_diff = (shape[:, 2] - shape[:, 10])[:, None]
    area_ratio = np.log((shape[:, 3] + 10.0) / (shape[:, 11] + 10.0))[:, None]
    centroid_xgap = (shape[:, 0] - shape[:, 8])[:, None]

    X_full = np.hstack([
        shape,
        vcx_h[:, None],
        vcy_h[:, None],
        vcx_a[:, None],
        vcy_a[:, None],
        stretch_diff,
        area_ratio,
        centroid_xgap,
    ])

    shape_adv = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    halves = [(KICKOFF_1ST, END_1ST), (KICKOFF_2ND, END_2ND)]

    for sf, ef in halves:
        sl = slice(sf, ef + 1)
        Xh = X_full[sl]
        min_d = np.minimum(dh[sl], da[sl])
        train_mask = vis[sl] & (~oob[sl]) & (min_d <= 3.5) & (np.abs(dh[sl] - da[sl]) >= 0.8)
        if np.sum(train_mask) < 200:
            continue

        X_tr = Xh[train_mask]
        y_tr = np.where(dh[sl][train_mask] < da[sl][train_mask], 1.0, -1.0)

        mu = np.mean(X_tr, axis=0)
        sd = np.maximum(np.std(X_tr, axis=0), 1e-4)
        Xn_tr = (X_tr - mu) / sd
        Xn_all = (Xh - mu) / sd

        X_aug = np.hstack([Xn_tr, np.ones((len(Xn_tr), 1))])
        reg = 150.0 * np.eye(X_aug.shape[1])
        reg[-1, -1] = 0.0
        w = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_tr)

        raw_pred = Xn_all @ w[:-1] + w[-1]
        sm_pred = smooth_1d(raw_pred, half_win=12)
        shape_adv[sl] = np.clip(sm_pred * 1.6, -1.5, 1.5).astype(np.float32)

    return shape_adv


def _find_restart_kick(feat, sf, ef):
    """Detect set-piece restart kick via sustained ball speed or touch impulse near a player."""
    vis = feat["vis"]
    oob = feat["oob"]
    dh = feat["dh"]
    da = feat["da"]
    bspd = feat["bspd"]
    touch = feat["touch"]

    consec = 0
    for f in range(sf, ef):
        near_player = vis[f] and min(dh[f], da[f]) < 3.5 and not oob[f]
        if near_player and (bspd[f] >= RESTART_KICK_SPEED_MPS or (touch[f] >= 3.2 and bspd[f] >= 2.8)):
            consec += 1
            if consec >= 3:
                return f - 2
        else:
            consec = 0
    return None


def _resolve_team_at_restart(feat, kicked_at, fallback_end):
    """Infer executing team (0=H, 1=A, or None) for N/A-labeled Set Pieces from player proximity."""
    vis = feat["vis"]
    dh = feat["dh"]
    da = feat["da"]
    st = max(0, kicked_at - 5)
    en = min(TOTAL_FRAMES, min(fallback_end, kicked_at + 12))
    h_votes = 0
    a_votes = 0
    for f in range(st, en):
        if vis[f]:
            if dh[f] < da[f] - 0.5 and dh[f] < 4.5:
                h_votes += 1
            elif da[f] < dh[f] - 0.5 and da[f] < 4.5:
                a_votes += 1
    if h_votes > a_votes:
        return 0
    if a_votes > h_votes:
        return 1
    return None


def build_smart_tagging_priors(feat):
    """Convert Smart Tagging phases and set pieces into Stage-1 priors and restart hints."""
    events = lst.load()
    priors = np.zeros((TOTAL_FRAMES, 3), dtype=np.float32)         # Stage-1: H, A, D
    restart_priors = np.zeros((TOTAL_FRAMES, 2), dtype=np.float32) # Direct set-piece restart team cue
    active_play_mask = np.zeros(TOTAL_FRAMES, dtype=bool)
    long_ball_mask = np.zeros(TOTAL_FRAMES, dtype=bool)
    stoppage_mask = np.zeros(TOTAL_FRAMES, dtype=bool)

    vis = feat["vis"]
    oob = feat["oob"]
    sideline = feat["sideline"]
    dh = feat["dh"]
    da = feat["da"]
    bspd = feat["bspd"]
    flock = feat["flock"]
    hsr = feat["hsr"]

    attack_codes = {
        "build up",
        "progression",
        "attacking transition",
        "creating chances",
        "ball in final third",
        "ball in the box",
    }
    defend_codes = {
        "defending in defensive third",
        "defending in middle third",
        "defending in attacking third",
        "defensive transition",
    }

    set_pieces = []
    goals = []
    pos_h_votes = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    pos_a_votes = np.zeros(TOTAL_FRAMES, dtype=np.float32)

    for ev in events:
        code = (ev["code"] or "").strip().lower()
        sf = max(0, ev["start_frame"])
        ef = min(TOTAL_FRAMES, ev["end_frame"])
        team_tag = (ev["tags"].get("Team") or "").strip()

        if code == "set pieces":
            set_pieces.append((sf, ef, team_tag))
        elif code == "goals":
            goals.append((sf, ef, team_tag))
        elif code in attack_codes:
            active_play_mask[sf:ef] = True
            if team_tag == "Barcelona":
                pos_h_votes[sf:ef] += 1.0
            elif team_tag == "Atletic Club":
                pos_a_votes[sf:ef] += 1.0
        elif code == "long ball":
            active_play_mask[sf:ef] = True
            long_ball_mask[sf:ef] = True
            if team_tag == "Barcelona":
                pos_h_votes[sf:ef] += 0.8
            elif team_tag == "Atletic Club":
                pos_a_votes[sf:ef] += 0.8
        elif code in defend_codes:
            active_play_mask[sf:ef] = True
            weight = 0.85 if code == "defensive transition" else 0.55
            if team_tag == "Barcelona":
                pos_a_votes[sf:ef] += weight
            elif team_tag == "Atletic Club":
                pos_h_votes[sf:ef] += weight

    for sf, ef, team_tag in goals:
        stoppage_mask[sf:ef] = True
        priors[sf:ef, 2] += 5.0
        post_end = min(TOTAL_FRAMES, ef + 60)
        if team_tag == "Barcelona":
            priors[ef:post_end, 1] += 1.2
            restart_priors[ef:post_end, 1] += 1.2
        elif team_tag == "Atletic Club":
            priors[ef:post_end, 0] += 1.2
            restart_priors[ef:post_end, 0] += 1.2

    for sf, ef, team_tag in set_pieces:
        ext_sf = sf
        lookback_limit = max(0, sf - 225)
        f = sf - 1
        while f >= lookback_limit:
            is_dead_evidence = (
                oob[f]
                or (not vis[f] and flock[f] < 1.35 and hsr[f] < 0.8)
                or (sideline[f] and min(dh[f], da[f]) > 3.2 and bspd[f] < 2.0 and flock[f] < 1.45)
                or (flock[f] < WALK_SPEED_MPS and hsr[f] < 1.0)
                or (bspd[f] < 2.2 and flock[f] < 1.55 and hsr[f] < 1.0)
                or (min(dh[f], da[f]) > UNCONTROLLED_DIST_M and bspd[f] < 3.0)
            )
            if is_dead_evidence:
                ext_sf = f
                f -= 1
            else:
                break

        kicked_at = _find_restart_kick(feat, sf, ef)
        dead_end = kicked_at if kicked_at is not None else ef
        if dead_end > ext_sf:
            stoppage_mask[ext_sf:dead_end] = True
            priors[ext_sf:dead_end, 2] += 4.5

        if kicked_at is not None:
            post_end = min(TOTAL_FRAMES, kicked_at + 50)
            if team_tag == "Barcelona":
                priors[kicked_at:post_end, 0] += 1.5
                restart_priors[kicked_at:post_end, 0] += 1.5
            elif team_tag == "Atletic Club":
                priors[kicked_at:post_end, 1] += 1.5
                restart_priors[kicked_at:post_end, 1] += 1.5
            else:
                resolved = _resolve_team_at_restart(feat, kicked_at, ef)
                if resolved is not None:
                    priors[kicked_at:post_end, resolved] += 1.2
                    restart_priors[kicked_at:post_end, resolved] += 1.2

    active_play_mask[stoppage_mask] = False
    pos_h_votes[stoppage_mask] = 0.0
    pos_a_votes[stoppage_mask] = 0.0

    priors[active_play_mask, 2] -= 1.40
    net_vote = pos_h_votes - pos_a_votes
    priors[net_vote > 0.3, 0] += np.minimum(1.1, 0.65 * net_vote[net_vote > 0.3])
    priors[net_vote < -0.3, 1] += np.minimum(1.1, 0.65 * (-net_vote[net_vote < -0.3]))

    return priors, restart_priors, active_play_mask, long_ball_mask


def adaptive_viterbi_decode(log_emissions, switch_pen_arr, dead_pen_arr, log_init):
    """Exact 3-state Viterbi with time-varying transition penalties (H=0, A=1, D=2)."""
    n_steps, n_states = log_emissions.shape
    dp = np.zeros((n_steps, n_states), dtype=np.float64)
    bp = np.zeros((n_steps, n_states), dtype=np.int8)

    dp[0] = log_init + log_emissions[0]
    for t in range(1, n_steps):
        sp = switch_pen_arr[t]
        dp_in = dead_pen_arr[t]
        dp_out = dp_in * 0.90 - (14.0 if dp_in < DEFAULT_DEAD_PENALTY else 0.0)
        log_trans = np.array([
            [0.0, sp, dp_in],
            [sp, 0.0, dp_in],
            [dp_out, dp_out, 0.0],
        ], dtype=np.float64)
        scores = dp[t - 1, :, None] + log_trans
        best_prev = np.argmax(scores, axis=0)
        dp[t] = scores[best_prev, np.arange(n_states)] + log_emissions[t]
        bp[t] = best_prev

    path = np.zeros(n_steps, dtype=np.int8)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_steps - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def viterbi_2state_decode(emit_ha, sp_01_arr, sp_10_arr, log_init):
    """Exact 2-state Viterbi decoder (0=H, 1=A) with directional touch-adaptive switch penalties."""
    n = len(emit_ha)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    dp0 = float(log_init[0] + emit_ha[0, 0])
    dp1 = float(log_init[1] + emit_ha[0, 1])
    bp = np.zeros((n, 2), dtype=np.int8)

    for t in range(1, n):
        sp_10 = float(sp_10_arr[t])  # cost to switch A -> H
        sp_01 = float(sp_01_arr[t])  # cost to switch H -> A

        # Transition to state 0 (break exact ties in favor of later switch per CHALLENGE.md)
        s00 = dp0
        s10 = dp1 + sp_10
        if s00 > s10:
            ndp0 = s00 + float(emit_ha[t, 0])
            bp[t, 0] = 0
        else:
            ndp0 = s10 + float(emit_ha[t, 0])
            bp[t, 0] = 1

        # Transition to state 1 (break exact ties in favor of later switch per CHALLENGE.md)
        s01 = dp0 + sp_01
        s11 = dp1
        if s11 > s01:
            ndp1 = s11 + float(emit_ha[t, 1])
            bp[t, 1] = 1
        else:
            ndp1 = s01 + float(emit_ha[t, 1])
            bp[t, 1] = 0

        dp0, dp1 = ndp0, ndp1

    path = np.empty(n, dtype=np.int8)
    path[-1] = 0 if dp0 >= dp1 else 1
    for t in range(n - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def cleanup_segments(
    codes,
    h_flock,
    h_hsr,
    h_active,
    h_bspd,
    min_alive=MIN_ALIVE_SPELL_FRAMES,
    min_dead=MIN_DEAD_RUN_FRAMES,
):
    """Remove sub-second artifacts and low-tempo D -> Alive -> D dead-ball retrieval islands."""
    out = codes.copy()
    n = len(out)

    # Pass 1: absorb short D runs flanked by identical alive team
    i = 0
    while i < n:
        val = out[i]
        j = i
        while j < n and out[j] == val:
            j += 1
        span = j - i
        if val == 2 and span < min_dead:
            left = out[i - 1] if i > 0 else None
            right = out[j] if j < n else None
            if left is not None and right is not None and left == right and left in (0, 1):
                out[i:j] = left
            elif left is not None and left in (0, 1) and right is None:
                out[i:j] = left
        i = j

    # Pass 2: absorb short low-tempo Alive islands flanked by D (D -> Alive -> D)
    i = 0
    while i < n:
        if out[i] != 2:
            j = i
            while j < n and out[j] != 2:
                j += 1
            span = j - i
            if span <= DEAD_ISLAND_MAX_FRAMES and i > 0 and j < n and out[i - 1] == 2 and out[j] == 2:
                mean_fl = float(np.mean(h_flock[i:j]))
                mean_hs = float(np.mean(h_hsr[i:j]))
                has_act = bool(np.any(h_active[i:j]))
                max_bs = float(np.max(h_bspd[i:j]))
                if (mean_fl < 1.35 and mean_hs < 0.70) and (not has_act or max_bs < 4.2):
                    out[i:j] = 2
            i = j
        else:
            i += 1

    # Pass 3: bridge sub-1s opponent challenges (H -> A -> H or A -> H -> A < BRIDGE_CHALLENGE_FRAMES)
    i = 0
    while i < n:
        val = out[i]
        j = i
        while j < n and out[j] == val:
            j += 1
        span = j - i
        if val in (0, 1) and span < BRIDGE_CHALLENGE_FRAMES:
            left = out[i - 1] if i > 0 else None
            right = out[j] if j < n else None
            if left is not None and right is not None and left == right and left in (0, 1):
                out[i:j] = left
        i = j

    # Pass 4: absorb any remaining sub-second H/A fragments (< min_alive)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            val = out[i]
            j = i
            while j < n and out[j] == val:
                j += 1
            span = j - i
            if val in (0, 1) and span < min_alive:
                left = out[i - 1] if i > 0 else None
                right = out[j] if j < n else None
                if left is not None and right is not None:
                    out[i:j] = left if left == right else right
                    changed = True
                elif left is not None:
                    out[i:j] = left
                    changed = True
                elif right is not None:
                    out[i:j] = right
                    changed = True
            i = j

    return out


def decode_alive_span_viterbi(
    h_vis,
    h_oob,
    h_dh,
    h_da,
    h_th,
    h_ta,
    h_sh,
    h_sa,
    h_bspd,
    h_touch,
    h_long,
    h_shape_adv,
    h_phase_prior,
    h_restart_prior,
    init_team=0,
):
    """Exact 2-state touch-adaptive Viterbi H/A decoder inside one contiguous Alive span.

    Replaces causal streak counters + boxcar smoothing with bi-directional MAP decoding:
    - Clear close-range visible ball control drives emissions directly.
    - Mid-flight passes (far from players, no touch impulse) attenuate emissions softly.
    - Contested duels (|dh - da| < 0.65m) attenuate team flips so current holder persists.
    - Shape & phase priors support occluded/contested intervals without overriding clear touches.
    - Transition barrier relaxes at physical ball-touch / vector-kink peaks for sharp boundaries.
    """
    n = len(h_vis)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    in_play_vis = h_vis & (~h_oob)
    min_d = np.minimum(h_dh, h_da)
    d_diff = np.clip(h_da - h_dh, -4.0, 4.0)
    abs_ddiff = np.abs(h_da - h_dh)

    # Flight protection: high-speed aerial balls or un-touched driven passes carry zero/low flip evidence
    in_aerial = h_bspd > AERIAL_FLIGHT_SPEED_MPS
    in_driven = (h_bspd > PASS_FLIGHT_SPEED_MPS) & (h_touch < 2.2) & (min_d > 1.8)
    in_long = h_long & (h_bspd > 6.5) & (h_touch < 2.2) & (min_d > 1.8)

    flight_gate = np.where(in_aerial, 0.0, np.where(in_driven | in_long, 0.15, 1.0))
    # Smooth control reach: full within 3.2m, tapering cleanly to 0 beyond 4.8m
    control_gate = np.where(min_d <= 3.2, 1.0, np.where(min_d <= 4.8, np.exp(-(min_d - 3.2) / 1.0), 0.0))
    gate = control_gate * flight_gate

    t_diff = np.clip(h_ta - h_th, -1.5, 1.5)
    nn_ratio = np.clip(np.log((h_sh + 1e-4) / (h_sa + 1e-4)), -2.0, 2.0)
    nn_weight = np.where(min_d < 1.5, 0.08, 0.20)

    team_adv = gate * (0.85 * t_diff + 0.60 * d_diff + nn_weight * nn_ratio)

    # Contested 50-50 duels attenuate emission so Viterbi persistence holds current possessor
    contested = in_play_vis & (min_d < 3.0) & (abs_ddiff < 0.70)
    team_adv = np.where(contested, team_adv * 0.15, team_adv)

    emit_ha = np.zeros((n, 2), dtype=np.float64)
    emit_ha[in_play_vis, 0] += team_adv[in_play_vis]
    emit_ha[in_play_vis, 1] -= team_adv[in_play_vis]

    # Set-piece restart priors apply directly at restart onset
    emit_ha += h_restart_prior

    # Shape & phase priors act ONLY during missing-ball occlusions (never during visible flight!)
    occluded = ~in_play_vis
    phase_adv = h_phase_prior[:, 0] - h_phase_prior[:, 1]
    ctx_adv = 0.30 * h_shape_adv + 0.25 * phase_adv
    emit_ha[occluded, 0] += ctx_adv[occluded]
    emit_ha[occluded, 1] -= ctx_adv[occluded]

    # Directional touch-snap relaxation: only discount switching INTO team X when team X
    # actually touches/controls the ball (prevents backward jumps onto the previous team's pass!)
    touch_h = in_play_vis & (h_dh <= 2.4) & (h_dh < h_da - 0.5) & (h_touch >= 2.2)
    touch_a = in_play_vis & (h_da <= 2.4) & (h_da < h_dh - 0.5) & (h_touch >= 2.2)
    sp_10_arr = np.where(touch_h, STAGE2_SWITCH_PENALTY + STAGE2_TOUCH_RELAX, STAGE2_SWITCH_PENALTY)
    sp_01_arr = np.where(touch_a, STAGE2_SWITCH_PENALTY + STAGE2_TOUCH_RELAX, STAGE2_SWITCH_PENALTY)

    # Prior preference at span start from Stage-1 / restart hint
    log_init = np.array([-0.15, -0.15], dtype=np.float64)
    if init_team in (0, 1):
        log_init[init_team] = 0.0

    return viterbi_2state_decode(emit_ha, sp_01_arr, sp_10_arr, log_init)


def run_v16_model(
    feat,
    shape_adv,
    priors,
    restart_priors,
    active_play_mask,
    long_ball_mask,
    switch_penalty=DEFAULT_SWITCH_PENALTY,
    dead_penalty=DEFAULT_DEAD_PENALTY,
):
    """Two-stage unified adaptive Viterbi decoder."""
    labels = np.full(TOTAL_FRAMES, "X", dtype="<U1")
    labels[START_2ND:KICKOFF_2ND] = "D"

    state_names = np.array(["H", "A", "D"])
    halves = [(1, KICKOFF_1ST, END_1ST), (2, KICKOFF_2ND, END_2ND)]

    vis = feat["vis"]
    oob = feat["oob"]
    sideline = feat["sideline"]
    dh = feat["dh"]
    da = feat["da"]
    th = feat["th"]
    ta = feat["ta"]
    sh = feat["sh"]
    sa = feat["sa"]
    bspd = feat["bspd"]
    touch = feat["touch"]
    flock = feat["flock"]
    hsr = feat["hsr"]
    opp_gap = feat["opp_gap"]

    for _, start_f, end_f in halves:
        sl = slice(start_f, end_f + 1)
        m = end_f - start_f + 1
        emit = np.zeros((m, 3), dtype=np.float64)

        h_vis = vis[sl]
        h_oob = oob[sl]
        h_sideline = sideline[sl]
        h_dh = dh[sl].astype(np.float64)
        h_da = da[sl].astype(np.float64)
        h_th = th[sl].astype(np.float64)
        h_ta = ta[sl].astype(np.float64)
        h_sh = sh[sl].astype(np.float64)
        h_sa = sa[sl].astype(np.float64)
        h_bspd = bspd[sl].astype(np.float64)
        h_touch = touch[sl].astype(np.float64)
        h_flock = flock[sl].astype(np.float64)
        h_hsr = hsr[sl].astype(np.float64)
        h_gap = opp_gap[sl].astype(np.float64)
        h_shape_adv = shape_adv[sl].astype(np.float64)
        h_prior = priors[sl].astype(np.float64)
        h_restart = restart_priors[sl].astype(np.float64)
        h_active = active_play_mask[sl]
        h_long = long_ball_mask[sl]

        # Stage 1: Alive vs Dead emissions (proven v15 context-aware formulation)
        in_play_vis = h_vis & (~h_oob)
        min_d = np.minimum(h_dh, h_da)
        d_diff = np.clip(h_da - h_dh, -4.0, 4.0)

        is_flight = (h_bspd > AERIAL_FLIGHT_SPEED_MPS) & (min_d > FLIGHT_TOUCH_DIST_M)
        control_gate = np.where(min_d <= 4.0, 1.0, np.exp(-(min_d - 4.0) / 1.8))
        flight_gate = np.where(is_flight, 0.10, 1.0)
        gate = control_gate * flight_gate

        t_diff = np.clip(h_ta - h_th, -1.5, 1.5)
        nn_ratio = np.clip(np.log((h_sh + 1e-4) / (h_sa + 1e-4)), -2.0, 2.0)
        nn_weight = np.where(min_d < 1.5, 0.08, 0.25)

        team_adv = gate * (1.05 * t_diff + 0.50 * d_diff + nn_weight * nn_ratio)
        team_adv = np.where(np.abs(d_diff) < 0.5, team_adv * 0.25, team_adv)

        emit[in_play_vis, 0] += team_adv[in_play_vis]
        emit[in_play_vis, 1] -= team_adv[in_play_vis]

        miss_play = ~in_play_vis
        emit[miss_play, 0] += 0.55 * h_shape_adv[miss_play]
        emit[miss_play, 1] -= 0.55 * h_shape_adv[miss_play]

        loose_vis = in_play_vis & (min_d > UNCONTROLLED_DIST_M)
        no_close_control = min_d > 3.2
        low_activity = (h_flock < WALK_SPEED_MPS) | ((h_flock < 1.55) & (h_hsr < 0.8))

        emit[loose_vis & low_activity, 2] += 3.6
        emit[loose_vis & (~low_activity), 2] += 0.7
        emit[in_play_vis & no_close_control & low_activity & (h_bspd < 5.0), 2] += 2.9

        throwin_setup = (
            in_play_vis
            & (~h_active)
            & h_sideline
            & (min_d > 2.2)
            & (h_bspd < 1.8)
            & (h_flock < 1.35)
            & (h_hsr < 0.6)
        )
        emit[throwin_setup, 2] += 2.6
        emit[throwin_setup, 0] *= 0.35
        emit[throwin_setup, 1] *= 0.35

        ball_placement = (
            in_play_vis
            & (~h_active)
            & (h_flock < 1.15)
            & (h_hsr < 0.8)
            & (h_bspd < 2.5)
            & ((h_gap > 4.2) | (h_flock < 0.95))
        )
        emit[ball_placement, 2] += 2.2
        emit[ball_placement, 0] *= 0.25
        emit[ball_placement, 1] *= 0.25

        emit[h_oob, 2] += 5.5
        emit[h_oob, 0] -= 2.5
        emit[h_oob, 1] -= 2.5

        i = 0
        while i < m:
            if not h_vis[i]:
                st = i
                while i < m and not h_vis[i]:
                    i += 1
                en = i
                gap = en - st
                mean_fl = float(np.mean(h_flock[st:en]))
                mean_hsr = float(np.mean(h_hsr[st:en]))

                if gap <= 55:
                    emit[st:en, 2] -= 0.4
                else:
                    for k in range(st, en):
                        elapsed = k - st
                        if h_active[k]:
                            if h_hsr[k] >= 1.2 or h_flock[k] >= 1.55:
                                emit[k, 2] -= 1.0
                            elif h_flock[k] < 1.10 and h_hsr[k] < 0.5 and elapsed > 100:
                                emit[k, 2] += min(0.8, (elapsed - 100) / 80.0)
                            else:
                                emit[k, 2] -= 0.4
                        elif h_hsr[k] >= 2.2 and elapsed < 140:
                            emit[k, 2] -= 0.3
                        else:
                            ramp = min(3.8, max(0.0, (elapsed - 55) / 30.0))
                            if mean_fl < WALK_SPEED_MPS or mean_hsr < 0.7:
                                ramp += 1.5
                            elif h_hsr[k] < 0.6 and h_flock[k] < 1.80:
                                ramp += 0.7
                            emit[k, 2] += ramp
            else:
                i += 1

        touch_event = in_play_vis & (min_d <= 2.2) & (h_touch >= 2.5)
        switch_pen_arr = np.where(touch_event, switch_penalty + 22.0, switch_penalty)
        low_tempo_stoppage = (~h_active) & (h_flock < 1.30) & (h_hsr < 0.6) & (h_bspd < 2.8)
        dead_pen_arr = np.where(low_tempo_stoppage, dead_penalty - 14.0, dead_penalty)

        emit += h_prior
        log_init = np.array([-0.5, -0.5, -1.0], dtype=np.float64)
        vit_path = adaptive_viterbi_decode(emit, switch_pen_arr, dead_pen_arr, log_init)
        vit_clean = cleanup_segments(
            vit_path,
            h_flock,
            h_hsr,
            h_active,
            h_bspd,
            min_alive=MIN_ALIVE_SPELL_FRAMES,
            min_dead=MIN_DEAD_RUN_FRAMES,
        )

        # Stage 2: Exact 2-State Touch-Adaptive Viterbi inside each Alive span (no boxcar smoothing)
        final_codes = vit_clean.copy()
        idx = 0
        while idx < m:
            if final_codes[idx] != 2:
                st = idx
                while idx < m and final_codes[idx] != 2:
                    idx += 1
                en = idx
                final_codes[st:en] = decode_alive_span_viterbi(
                    h_vis[st:en],
                    h_oob[st:en],
                    h_dh[st:en],
                    h_da[st:en],
                    h_th[st:en],
                    h_ta[st:en],
                    h_sh[st:en],
                    h_sa[st:en],
                    h_bspd[st:en],
                    h_touch[st:en],
                    h_long[st:en],
                    h_shape_adv[st:en],
                    h_prior[st:en],
                    h_restart[st:en],
                    init_team=int(final_codes[st]),
                )
            else:
                idx += 1

        final_codes = cleanup_segments(
            final_codes,
            h_flock,
            h_hsr,
            h_active,
            h_bspd,
            min_alive=MIN_ALIVE_SPELL_FRAMES,
            min_dead=MIN_DEAD_RUN_FRAMES,
        )
        labels[sl] = state_names[final_codes]

    return labels


def main():
    parser = argparse.ArgumentParser(description="Run v16 Unified Adaptive Viterbi possession model.")
    parser.add_argument("--name", default="v16_unified_adaptive_viterbi", help="Output submission stem")
    parser.add_argument("--switch-pen", type=float, default=DEFAULT_SWITCH_PENALTY, help="Stage-1 H<->A penalty")
    parser.add_argument("--dead-pen", type=float, default=DEFAULT_DEAD_PENALTY, help="Stage-1 <->D penalty")
    args = parser.parse_args()

    feat = extract_kinematic_features()
    shape_adv = fit_self_supervised_shape_prior(feat)
    priors, restart_priors, active_mask, long_ball_mask = build_smart_tagging_priors(feat)
    print(f"Decoding full match with v16 unified adaptive Viterbi (switch={args.switch_pen}, dead={args.dead_pen})...")
    labels = run_v16_model(
        feat,
        shape_adv,
        priors,
        restart_priors,
        active_mask,
        long_ball_mask,
        switch_penalty=args.switch_pen,
        dead_penalty=args.dead_pen,
    )

    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.name}.csv")
    pd.DataFrame({"frame": np.arange(TOTAL_FRAMES), "label": labels}).to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(labels)} frames).")


if __name__ == "__main__":
    main()
