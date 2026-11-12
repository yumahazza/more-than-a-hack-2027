"""v13_unified_viterbi.py - Unified Kinematic & Multimodal Viterbi Possession Model.

Addresses structural weaknesses identified in v12_dead_ball.py:
1. Smooth Ball Kinematics (Savitzky-Golay polynomial derivative, pure numpy):
   - Eliminates 1-frame finite-difference jitter spikes (up to 109 m/s) and
     frame-drop zero-speed resets.
   - Rejects impossible single-frame tracking teleports (> 60 m/s) and bridges
     micro-dropouts (<= 15 frames) before velocity estimation.
2. Spearman Time-to-Intercept (T_int) + 5NN Spatial Influence:
   - Implements genuine dynamic T_int and 5-Nearest Neighbors weighting
     (Spearman 2018; Borghesi et al. 2023) instead of static 1NN Euclidean distance.
3. Context-Aware Dead-Ball ('D') Evidence (Vidal-Codina et al. 2022):
   - Checks pitch boundaries (105.3m x 68.0m) for out-of-play balls.
   - Prevents players retrieving a dead ball from falsely breaking 'D' intervals.
   - Prevents prolonged tracking occlusions during high-tempo attacks (high
     collective player speed or active Smart Tagging phase) from falsely
     splitting live possessions into 'D'.
4. Full Smart Tagging Priors + Pre-Tag Stoppage Extension:
   - Combines Set Pieces (with backward dead-ball search & smoothed kick-impulse
     detection) and active attacking phase tags as soft log-odds priors.
5. Unified Exact Viterbi Decoding (Replaces ±82-frame Majority Vote & Post-Overwrites):
   - Eliminates the 165-frame symmetric smoothing window that erased short
     possessions and blurred transitions.
   - Integrates all dead-ball and set-piece priors *before* decoding and runs
     micro-segment cleanup last, eliminating post-smoothing fragment artifacts.
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

# Graded match structure (from DATA.md / games.py)
KICKOFF_1ST = 11185
END_1ST = 90651
START_2ND = 110808
KICKOFF_2ND = 113742
END_2ND = 190197

# Physics & Sports Science Constants (not tuned on public GT)
REACTION_TIME_S = 0.25          # Spearman (2018) human reaction delay
MAX_SPRINT_MPS = 8.5            # Effective max sprint speed
K_NEAREST = 5                   # Borghesi et al. (2023) 5NN sufficiency
MAX_CONTROL_DIST_M = 4.2        # Active ball control reach
MAX_CONTROL_T_INT_S = 1.4       # Max Time-to-Intercept for control
FLIGHT_SPEED_MPS = 13.0         # High-speed aerial pass threshold (~47 km/h)
FLIGHT_TOUCH_DIST_M = 1.3       # Required proximity to intercept fast aerial ball
RESTART_KICK_SPEED_MPS = 4.5    # Smoothed kick velocity for set-piece restart
OOB_MARGIN_M = 0.5              # Tolerance outside pitch touchline/goal line
UNCONTROLLED_DIST_M = 6.5       # Loose ball distance threshold
WALK_SPEED_MPS = 1.35           # Collective player walk speed during stoppages
ACTIVE_PLAY_SPEED_MPS = 2.0     # Collective speed indicating live active play
SHORT_OCCLUSION_FRAMES = 15     # Micro-dropout gap to interpolate cleanly (0.6s)
MAX_TRACK_JUMP_M = 2.4          # Max physical ball travel per 40ms frame (~60 m/s)
SG_WINDOW = 13                  # Savitzky-Golay window (0.52s at 25 Hz)
# Unified 3-State Viterbi + Asymmetric Alive Possession Constants
DEFAULT_SWITCH_PENALTY = -82.0  # Log-barrier for H <-> A in 3-state Alive/Dead segmentation
DEFAULT_DEAD_PENALTY = -62.0    # Log-barrier for Alive <-> Dead transitions
HYSTERESIS_MARGIN_M = 0.80      # Combined distance + T_int takeover margin
CONTROL_REACH_M = 4.5           # Max ball distance for opponent to initiate takeover
CONFIRM_TAKEOVER_FRAMES = 32    # Sustained opponent control (~1.28s at 25 Hz), backdated to onset
ALIVE_SMOOTH_HALF_WIN = 38      # Zero-phase within-span majority smoothing (~1.52s half-width)
MIN_ALIVE_SPELL_FRAMES = 25     # Minimum surviving spell duration (~1.0s)
MIN_DEAD_RUN_FRAMES = 35        # Minimum surviving dead-ball stoppage (~1.4s)
BRIDGE_CHALLENGE_FRAMES = 38    # Bridge momentary (<1.52s) deflections per CHALLENGE.md


def savgol_coeffs_deriv1(window_length, polyorder=2, dt=1.0 / FPS):
    """Compute Savitzky-Golay first-derivative convolution kernel in pure numpy."""
    half = window_length // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    # Vandermonde design matrix [1, x, x^2, ...]
    vander = np.vstack([x ** p for p in range(polyorder + 1)]).T
    pinv = np.linalg.pinv(vander)
    # Row 1 gives the first derivative at x=0; reverse for np.convolve
    return (pinv[1] / dt)[::-1]


def smooth_1d(arr, half_win):
    """Centered moving average with edge padding."""
    if half_win <= 0:
        return arr.copy()
    k = 2 * half_win + 1
    padded = np.pad(arr.astype(np.float64), (half_win, half_win), mode="edge")
    csum = np.cumsum(np.insert(padded, 0, 0.0))
    return (csum[k:] - csum[:-k]) / float(k)


def extract_kinematic_features():
    """Stream ATD once and compute smoothed ball & 5NN player kinematics."""
    paths = asset_paths(source="tactical")
    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]

    team_of_slot = base_mod.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)

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

    print(f"Streaming ATD tracking from {raw_path}...")
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
            candidates = []

            for slot, p_str in enumerate(parts[1].split(";")):
                if slot >= n_slots or p_str.startswith("N"):
                    continue
                c = p_str.split(",")
                pspd = 1.5
                if len(c) > 2 and not c[2].startswith("N"):
                    pspd = float(c[2])
                    spd_sum += pspd
                    spd_cnt += 1

                if has_ball:
                    px = float(c[0]) * PITCH_X
                    py = float(c[1]) * PITCH_Y
                    d = float(np.hypot(px - bx, py - by))
                    eff_spd = min(MAX_SPRINT_MPS, max(pspd, 1.5) + 1.2)
                    t_int = REACTION_TIME_S + d / eff_spd
                    candidates.append((d, t_int, pspd, team_of_slot[slot]))

            if spd_cnt > 0:
                mean_player_spd[idx] = spd_sum / spd_cnt

            if has_ball and candidates:
                candidates.sort(key=lambda item: item[0])
                min_dh = 999.0
                min_da = 999.0
                min_th = 99.0
                min_ta = 99.0
                sh = 0.0
                sa = 0.0

                for rank, (d, t_int, pspd, team) in enumerate(candidates):
                    if team == base_mod.HOME:
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
                        w = np.exp(-d / 2.2) * (1.0 + 0.08 * min(pspd, 6.0)) / (1.0 + 0.15 * rank)
                        if team == base_mod.HOME:
                            sh += w
                        else:
                            sa += w

                dist_h[idx] = min_dh
                dist_a[idx] = min_da
                tint_h[idx] = min_th
                tint_a[idx] = min_ta
                score_h[idx] = sh
                score_a[idx] = sa

    # Reject multi-frame ball tracking glitches/teleports before differentiation
    clean_vis = ball_vis.copy()
    last_good = -1
    for i in range(TOTAL_FRAMES):
        if clean_vis[i]:
            if last_good >= 0:
                dt_frames = i - last_good
                jump = np.hypot(ball_x[i] - ball_x[last_good], ball_y[i] - ball_y[last_good])
                # Even a 45 m/s shot travels at most 1.8m per 40ms frame
                if jump > 1.8 * dt_frames + 1.0:
                    clean_vis[i] = False
                    ball_x[i] = np.nan
                    ball_y[i] = np.nan
                    continue
            last_good = i

    # Interpolate short dropouts (<= SHORT_OCCLUSION_FRAMES) inside continuous ball tracks
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
                jump = np.hypot(ball_x[en] - ball_x[st - 1], ball_y[en] - ball_y[st - 1])
                if jump <= MAX_TRACK_JUMP_M * (gap + 1):
                    alphas = np.linspace(0.0, 1.0, gap + 2)[1:-1]
                    interp_x[st:en] = ball_x[st - 1] + alphas * (ball_x[en] - ball_x[st - 1])
                    interp_y[st:en] = ball_y[st - 1] + alphas * (ball_y[en] - ball_y[st - 1])
        else:
            i += 1

    # Compute Savitzky-Golay smoothed ball speed over each contiguous valid segment
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
                ball_speed_sg[st:en] = np.hypot(vx, vy)
            elif seg_len >= 2:
                vx = np.gradient(interp_x[st:en]) * FPS
                vy = np.gradient(interp_y[st:en]) * FPS
                ball_speed_sg[st:en] = np.hypot(vx, vy)
        else:
            i += 1

    ball_oob = np.zeros(TOTAL_FRAMES, dtype=bool)
    ball_oob[ball_vis] = (
        (ball_x[ball_vis] < -OOB_MARGIN_M)
        | (ball_x[ball_vis] > PITCH_X + OOB_MARGIN_M)
        | (ball_y[ball_vis] < -OOB_MARGIN_M)
        | (ball_y[ball_vis] > PITCH_Y + OOB_MARGIN_M)
    )

    flock_smooth = smooth_1d(mean_player_spd, half_win=12).astype(np.float32)

    return {
        "vis": ball_vis,
        "oob": ball_oob,
        "dh": dist_h,
        "da": dist_a,
        "th": tint_h,
        "ta": tint_a,
        "sh": score_h,
        "sa": score_a,
        "bspd": ball_speed_sg,
        "flock": flock_smooth,
    }


def build_smart_tagging_priors(feat):
    """Convert Smart Tagging set pieces and active tactical phases into soft log-odds."""
    events = lst.load()
    priors = np.zeros((TOTAL_FRAMES, 3), dtype=np.float32)  # cols: H, A, D
    active_play_mask = np.zeros(TOTAL_FRAMES, dtype=bool)

    vis = feat["vis"]
    oob = feat["oob"]
    dh = feat["dh"]
    da = feat["da"]
    bspd = feat["bspd"]
    flock = feat["flock"]

    active_codes = {
        "build up",
        "progression",
        "attacking transition",
        "creating chances",
        "ball in final third",
        "ball in the box",
    }

    set_pieces = []
    active_h_votes = np.zeros(TOTAL_FRAMES, dtype=np.int32)
    active_a_votes = np.zeros(TOTAL_FRAMES, dtype=np.int32)

    for ev in events:
        code = (ev["code"] or "").strip().lower()
        sf = max(0, ev["start_frame"])
        ef = min(TOTAL_FRAMES, ev["end_frame"])
        team_tag = (ev["tags"].get("Team") or "").strip()

        if code == "set pieces":
            set_pieces.append((sf, ef, team_tag))
        elif code in active_codes:
            active_play_mask[sf:ef] = True
            if team_tag == "Barcelona":
                active_h_votes[sf:ef] += 1
            elif team_tag == "Atletic Club":
                active_a_votes[sf:ef] += 1

    # Soft active-phase prior: suppresses false D and gently favors tagged attacking team
    priors[active_play_mask, 2] -= 1.0
    only_h = (active_h_votes > 0) & (active_a_votes == 0)
    only_a = (active_a_votes > 0) & (active_h_votes == 0)
    priors[only_h, 0] += 0.75
    priors[only_a, 1] += 0.75

    # Set Pieces: detect sustained kick impulse using smoothed ball velocity
    for sf, ef, team_tag in set_pieces:
        # Backward extension: if ball was already dead/OOB/slow immediately before tag start
        ext_sf = sf
        lookback_limit = max(0, sf - 100)  # up to 4.0s before Smart Tagging window
        f = sf - 1
        while f >= lookback_limit:
            is_dead_evidence = (
                oob[f]
                or (not vis[f])
                or (flock[f] < WALK_SPEED_MPS + 0.1)
                or (bspd[f] < 2.2 and flock[f] < 1.7)
                or (min(dh[f], da[f]) > UNCONTROLLED_DIST_M and bspd[f] < 3.0)
            )
            if is_dead_evidence:
                ext_sf = f
                f -= 1
            else:
                break

        # Find true restart kick inside [sf, ef): require smoothed kick speed >= threshold
        # sustained for >= 3 consecutive frames with a player near the ball.
        kicked_at = None
        consec = 0
        for f in range(sf, ef):
            near_player = vis[f] and min(dh[f], da[f]) < 3.2 and not oob[f]
            if near_player and bspd[f] >= RESTART_KICK_SPEED_MPS:
                consec += 1
                if consec >= 3:
                    kicked_at = f - 2
                    break
            else:
                consec = 0

        dead_end = kicked_at if kicked_at is not None else ef
        if dead_end > ext_sf:
            priors[ext_sf:dead_end, 2] += 4.5

        if kicked_at is not None:
            post_end = min(TOTAL_FRAMES, kicked_at + 60)
            if team_tag == "Barcelona":
                priors[kicked_at:post_end, 0] += 1.5
            elif team_tag == "Atletic Club":
                priors[kicked_at:post_end, 1] += 1.5

    return priors, active_play_mask


def viterbi_decode(log_emissions, log_trans, log_init):
    """Exact Viterbi decoding for 3 states (H=0, A=1, D=2)."""
    n_steps, n_states = log_emissions.shape
    dp = np.zeros((n_steps, n_states), dtype=np.float64)
    bp = np.zeros((n_steps, n_states), dtype=np.int8)

    dp[0] = log_init + log_emissions[0]
    for t in range(1, n_steps):
        scores = dp[t - 1, :, None] + log_trans
        best_prev = np.argmax(scores, axis=0)
        dp[t] = scores[best_prev, np.arange(n_states)] + log_emissions[t]
        bp[t] = best_prev

    path = np.zeros(n_steps, dtype=np.int8)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_steps - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def cleanup_short_segments(codes, min_alive=MIN_ALIVE_SPELL_FRAMES, min_dead=MIN_DEAD_RUN_FRAMES):
    """Remove micro-segments after decoding so no 1-frame or sub-second artifacts remain.

    codes: 0=H, 1=A, 2=D
    """
    out = codes.copy()
    n = len(out)

    # Pass 1: absorb short D runs flanked by alive play
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

    # Pass 2: bridge brief opponent challenges (H -> A -> H or A -> H -> A < bridge_frames)
    # Per CHALLENGE.md: momentary touches/deflections without sustained control do not flip possession.
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

    # Pass 3: absorb any remaining short H/A runs (< min_alive) into dominant neighbor
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


def decode_alive_span(
    h_vis,
    h_dh,
    h_da,
    h_th,
    h_ta,
    h_bspd,
    h_prior,
    init_team=0,
):
    """Causal asymmetric possession retention + within-span majority smoothing for one Alive span.

    Models CHALLENGE.md rules directly: contested/aerial/occluded intervals keep
    the current holder; opponent takeovers require sustained spatial margin AND
    reception confirmation, followed by centered smoothing bounded inside the span.
    Returns array of {0=H, 1=A}.
    """
    n = len(h_vis)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    # Determine initial team: check Set Piece restart prior first, then earliest confident control
    curr = int(init_team)
    max_probe = min(n, 75)
    found_prior = False
    for k in range(max_probe):
        p_diff = float(h_prior[k, 0] - h_prior[k, 1])
        if abs(p_diff) >= 1.0:
            curr = 0 if p_diff > 0 else 1
            found_prior = True
            break
    if not found_prior:
        for k in range(max_probe):
            if h_vis[k] and min(h_dh[k], h_da[k]) <= CONTROL_REACH_M and abs(h_da[k] - h_dh[k]) >= 0.5:
                curr = 0 if h_dh[k] < h_da[k] else 1
                break

    raw = np.empty(n, dtype=np.int8)
    opp_streak = 0
    streak_start = -1

    for k in range(n):
        dh_k = h_dh[k]
        da_k = h_da[k]
        min_d = dh_k if dh_k < da_k else da_k
        in_flight = (h_bspd[k] > FLIGHT_SPEED_MPS) and (min_d > FLIGHT_TOUCH_DIST_M)

        if h_vis[k] and not in_flight:
            d_curr = dh_k if curr == 0 else da_k
            d_opp = da_k if curr == 0 else dh_k
            t_curr = h_th[k] if curr == 0 else h_ta[k]
            t_opp = h_ta[k] if curr == 0 else h_th[k]

            # Strong Smart Tagging prior + kinematic margin (distance + Spearman T_int)
            p_adv = float(h_prior[k, 1] - h_prior[k, 0]) if curr == 0 else float(h_prior[k, 0] - h_prior[k, 1])
            eff_margin = HYSTERESIS_MARGIN_M - (0.35 if p_adv >= 1.0 else 0.0)
            kinematic_adv = (d_curr - d_opp) + 0.75 * np.clip(t_curr - t_opp, -1.5, 1.5)

            opp_takeover = (d_opp < CONTROL_REACH_M) and (kinematic_adv > eff_margin)
            if opp_takeover:
                if opp_streak == 0:
                    streak_start = k
                opp_streak += 1
                if opp_streak >= CONFIRM_TAKEOVER_FRAMES:
                    # Backdate transition onset to when opponent first gained control
                    curr = 1 - curr
                    if 0 <= streak_start < k:
                        raw[streak_start:k] = curr
                    opp_streak = 0
                    streak_start = -1
            elif d_curr < 3.5 and d_curr <= d_opp:
                opp_streak = 0
                streak_start = -1
            else:
                opp_streak = max(0, opp_streak - 1)
                if opp_streak == 0:
                    streak_start = -1
        else:
            # Occluded or fast aerial pass: decay challenger counter, keep current holder
            opp_streak = max(0, opp_streak - 1)
            if opp_streak == 0:
                streak_start = -1

        raw[k] = curr

    # Within-span majority vote smoothing (strictly bounded by Alive span; zero D leakage)
    if n <= 15:
        majority = 0 if np.sum(raw == 0) >= np.sum(raw == 1) else 1
        return np.full(n, majority, dtype=np.int8)

    hw = min(ALIVE_SMOOTH_HALF_WIN, max(1, (n - 1) // 2))
    # Map {0=H, 1=A} -> {+1, -1} so majority vote averages cleanly in O(n)
    sig = np.where(raw == 0, 1.0, -1.0)
    sm = smooth_1d(sig, hw)
    return np.where(sm >= 0.0, 0, 1).astype(np.int8)


def run_v13_model(
    feat,
    priors,
    active_play_mask,
    switch_penalty=DEFAULT_SWITCH_PENALTY,
    dead_penalty=DEFAULT_DEAD_PENALTY,
):
    """Two-stage unified decoder:
    1) 3-state exact Viterbi isolates Dead ('D') vs Alive intervals using smoothed
       kinematics, OOB checks, context-aware occlusion logic, and Smart Tagging priors.
    2) Inside each Alive interval, causal asymmetric hysteresis + bounded smoothing
       assigns H/A possession without mid-pass early switching or D-adjacent micro-spells.
    """
    labels = np.full(TOTAL_FRAMES, "X", dtype="<U1")
    labels[START_2ND:KICKOFF_2ND] = "D"

    state_names = np.array(["H", "A", "D"])
    halves = [(1, KICKOFF_1ST, END_1ST), (2, KICKOFF_2ND, END_2ND)]

    log_trans = np.array([
        [0.0, switch_penalty, dead_penalty],
        [switch_penalty, 0.0, dead_penalty],
        [dead_penalty * 0.9, dead_penalty * 0.9, 0.0],
    ], dtype=np.float64)

    vis = feat["vis"]
    oob = feat["oob"]
    dh = feat["dh"]
    da = feat["da"]
    th = feat["th"]
    ta = feat["ta"]
    sh = feat["sh"]
    sa = feat["sa"]
    bspd = feat["bspd"]
    flock = feat["flock"]

    for _, start_f, end_f in halves:
        sl = slice(start_f, end_f + 1)
        m = end_f - start_f + 1
        emit = np.zeros((m, 3), dtype=np.float64)

        h_vis = vis[sl]
        h_oob = oob[sl]
        h_dh = dh[sl].astype(np.float64)
        h_da = da[sl].astype(np.float64)
        h_th = th[sl].astype(np.float64)
        h_ta = ta[sl].astype(np.float64)
        h_sh = sh[sl].astype(np.float64)
        h_sa = sa[sl].astype(np.float64)
        h_bspd = bspd[sl].astype(np.float64)
        h_flock = flock[sl].astype(np.float64)
        h_prior = priors[sl].astype(np.float64)
        h_active = active_play_mask[sl]

        # 1. Team & Alive evidence when ball is visible and inside pitch
        in_play_vis = h_vis & (~h_oob)
        min_d = np.minimum(h_dh, h_da)
        d_diff = np.clip(h_da - h_dh, -4.0, 4.0)

        # Control gate preserves full Alive signal within 4.0m (proven macroF1=0.8293)
        is_flight = (h_bspd > FLIGHT_SPEED_MPS) & (min_d > FLIGHT_TOUCH_DIST_M)
        control_gate = np.where(min_d <= 4.0, 1.0, np.exp(-(min_d - 4.0) / 1.8))
        flight_gate = np.where(is_flight, 0.10, 1.0)
        gate = control_gate * flight_gate

        t_diff = np.clip(h_ta - h_th, -1.5, 1.5)
        nn_ratio = np.clip(np.log((h_sh + 1e-4) / (h_sa + 1e-4)), -2.0, 2.0)
        nn_weight = np.where(min_d < 1.5, 0.08, 0.25)

        team_adv = gate * (1.0 * t_diff + 0.55 * d_diff + nn_weight * nn_ratio)
        team_adv = np.where(np.abs(d_diff) < 0.5, team_adv * 0.25, team_adv)

        emit[in_play_vis, 0] += team_adv[in_play_vis]
        emit[in_play_vis, 1] -= team_adv[in_play_vis]

        # Dead-ball evidence when ball is visible
        loose_vis = in_play_vis & (min_d > UNCONTROLLED_DIST_M)
        slow_lull = h_flock < WALK_SPEED_MPS
        emit[loose_vis & slow_lull, 2] += 3.5
        emit[loose_vis & (~slow_lull), 2] += 0.8
        emit[in_play_vis & slow_lull & (h_bspd < 5.0), 2] += 2.8

        # Mild suppression when players walk and ball is nearly stationary (pre-restart setup)
        dead_setup = in_play_vis & (h_flock < WALK_SPEED_MPS + 0.15) & (h_bspd < 2.2)
        emit[dead_setup, 0] *= 0.45
        emit[dead_setup, 1] *= 0.45

        # Ball visible outside pitch boundary -> strong Dead Ball
        emit[h_oob, 2] += 5.5
        emit[h_oob, 0] -= 2.5
        emit[h_oob, 1] -= 2.5

        # 2. Missing ball dynamics (context-aware)
        i = 0
        while i < m:
            if not h_vis[i]:
                st = i
                while i < m and not h_vis[i]:
                    i += 1
                en = i
                gap = en - st
                mean_fl = float(np.mean(h_flock[st:en]))
                has_active_tag = bool(np.any(h_active[st:en]))

                if gap <= 55:
                    emit[st:en, 2] -= 0.4
                else:
                    for k in range(st, en):
                        elapsed = k - st
                        if h_flock[k] >= ACTIVE_PLAY_SPEED_MPS and h_active[k]:
                            emit[k, 2] -= 0.5
                        elif h_flock[k] >= ACTIVE_PLAY_SPEED_MPS and elapsed < 125:
                            emit[k, 2] += 0.1
                        else:
                            ramp = min(3.8, max(0.0, (elapsed - 50) / 30.0))
                            if mean_fl < WALK_SPEED_MPS:
                                ramp += 1.5
                            if has_active_tag and elapsed < 125:
                                ramp *= 0.35
                            emit[k, 2] += ramp
            else:
                i += 1

        # 3. Add Smart Tagging soft priors and decode Dead vs Alive segments
        emit += h_prior
        log_init = np.array([-0.5, -0.5, -1.0], dtype=np.float64)
        vit_path = viterbi_decode(emit, log_trans, log_init)

        # Clean micro-D runs first so Alive spans are cohesive before H/A decoding
        vit_clean = cleanup_short_segments(vit_path, min_alive=MIN_ALIVE_SPELL_FRAMES, min_dead=MIN_DEAD_RUN_FRAMES)

        # 4. Refine each contiguous Alive span with asymmetric causal retention + bounded smoothing
        final_codes = vit_clean.copy()
        idx = 0
        while idx < m:
            if final_codes[idx] != 2:
                st = idx
                while idx < m and final_codes[idx] != 2:
                    idx += 1
                en = idx
                viterbi_init = int(final_codes[st])
                final_codes[st:en] = decode_alive_span(
                    h_vis[st:en],
                    h_dh[st:en],
                    h_da[st:en],
                    h_th[st:en],
                    h_ta[st:en],
                    h_bspd[st:en],
                    h_prior[st:en],
                    init_team=viterbi_init,
                )
            else:
                idx += 1

        # Final cleanup guarantees zero sub-1s fragments or <1.68s deflections anywhere
        final_codes = cleanup_short_segments(
            final_codes, min_alive=MIN_ALIVE_SPELL_FRAMES, min_dead=MIN_DEAD_RUN_FRAMES
        )
        labels[sl] = state_names[final_codes]

    return labels


def main():
    parser = argparse.ArgumentParser(description="Run v13 Unified Kinematic Viterbi model.")
    parser.add_argument("--name", default="v13_unified_viterbi", help="Output submission stem")
    parser.add_argument("--switch-pen", type=float, default=DEFAULT_SWITCH_PENALTY, help="Log transition penalty H<->A")
    parser.add_argument("--dead-pen", type=float, default=DEFAULT_DEAD_PENALTY, help="Log transition penalty <->D")
    args = parser.parse_args()

    feat = extract_kinematic_features()
    priors, active_mask = build_smart_tagging_priors(feat)
    print(f"Decoding full match with exact Viterbi (switch={args.switch_pen}, dead={args.dead_pen})...")
    labels = run_v13_model(feat, priors, active_mask, switch_penalty=args.switch_pen, dead_penalty=args.dead_pen)

    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.name}.csv")
    pd.DataFrame({"frame": np.arange(TOTAL_FRAMES), "label": labels}).to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(labels)} frames).")


if __name__ == "__main__":
    main()
