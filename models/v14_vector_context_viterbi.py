"""v14_vector_context_viterbi.py - Vector Kinematics & Full-Tagging Multimodal Viterbi Model.

Builds on v13_unified_viterbi.py to address the remaining per-frame and boundary errors:
1. Vector Time-to-Intercept (Spearman 2018) & Pass Trajectory Alignment (Link & Hoernig 2017):
   - Computes short-horizon player velocity vectors (vx, vy) from ATD positions to measure
     true radial closing velocity toward the ball rather than scalar speed alone.
   - Tracks smoothed ball velocity vector (bvx, bvy), trajectory curvature/impulse events,
     and protects driven ground passes (>= 7.5 m/s) in addition to aerial passes.
2. BIP vs BOP Activity Features (Vidal-Codina et al. 2022; Jerome et al. 2023):
   - Computes high-speed running (HSR) player count and nearest-opponent spacing to holder.
   - Prevents dead-ball setup/walking ball-carrying (low flock + low HSR + wide opponent gap)
     from spuriously overriding 'D' with team control.
   - Distinguishes missing-ball stoppages from occluded counter-attacks via HSR & phase tags.
3. Full Smart Tagging Multimodal Priors (Otting & Karlis):
   - Integrates DEFENDING IN ... THIRD and DEFENSIVE TRANSITION tags as live-play + opponent
     possession evidence, LONG BALL as pass-flight protection, GOALS as dead-ball reset blocks,
     and resolves 'Team = N/A' Set Pieces using player proximity at the detected kick impulse.
4. Touch-Snapped Asymmetric Hysteresis & Light Within-Span Smoothing:
   - Snaps confirmed transitions to local physical touch/impulse frames and uses narrower
     within-span smoothing (+/-0.96s) so ~1.5-3s opponent possessions and spell edges remain sharp.
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

# Physics & Sports Science Constants (grounded in literature, never tuned on public GT)
REACTION_TIME_S = 0.25          # Spearman (2018) reaction latency
MAX_SPRINT_MPS = 8.5            # Effective sprint speed cap
K_NEAREST = 5                   # Borghesi et al. (2023) 5NN neighborhood
HSR_SPEED_MPS = 3.5             # High-speed running threshold separating BIP vs BOP (Jerome et al. 2023)
PASS_FLIGHT_SPEED_MPS = 7.5     # Driven pass flight protection threshold (Link & Hoernig 2017)
AERIAL_FLIGHT_SPEED_MPS = 13.0  # Fast aerial pass threshold
FLIGHT_TOUCH_DIST_M = 1.25      # Direct interception proximity required during ball flight
RESTART_KICK_SPEED_MPS = 4.2    # Smoothed ball speed confirming set-piece restart kick
OOB_MARGIN_M = 0.5              # Pitch line tolerance (105.3m x 68.0m)
UNCONTROLLED_DIST_M = 6.5       # Loose ball radius
WALK_SPEED_MPS = 1.35           # Collective walking threshold during stoppages
ACTIVE_PLAY_SPEED_MPS = 2.0     # Collective active-play speed
SHORT_OCCLUSION_FRAMES = 15     # Micro-dropout interpolation window (0.6s)
MAX_TRACK_JUMP_M = 2.4          # Max physical ball step per frame (60 m/s)
SG_WINDOW = 13                  # Savitzky-Golay derivative window (0.52s)

# Two-stage decoding parameters
DEFAULT_SWITCH_PENALTY = -82.0  # Viterbi H <-> A penalty for Stage-1 Alive/Dead segmentation
DEFAULT_DEAD_PENALTY = -66.0    # Viterbi Alive <-> Dead transition barrier (prevents occlusion splits)
HYSTERESIS_MARGIN_M = 0.80      # Combined distance + vector T_int margin
CONTROL_REACH_M = 4.5           # Max distance for active opponent takeover
CONFIRM_TAKEOVER_FRAMES = 32    # Sustained opponent control (~1.28s at 25 Hz)
ALIVE_SMOOTH_HALF_WIN = 34      # Bounded zero-phase within-span smoothing (~1.36s half-width)
MIN_ALIVE_SPELL_FRAMES = 28     # Minimum surviving spell duration (~1.12s)
MIN_DEAD_RUN_FRAMES = 38        # Minimum surviving stoppage duration (~1.52s)
BRIDGE_CHALLENGE_FRAMES = 42    # Bridge momentary (<1.68s) deflections per CHALLENGE.md


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


def extract_kinematic_features():
    """Stream ATD once and compute vector player/ball kinematics and BIP/BOP activity."""
    paths = asset_paths(source="tactical")
    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]

    team_of_slot = base_mod.team_of_each_track(meta_path)
    n_slots = len(team_of_slot)

    # Rolling 5-frame position ring buffer for smooth causal player velocity vectors
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

    print(f"Streaming ATD vector kinematics from {raw_path}...")
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
            buf_col = idx % vel_lag
            ref_col = (idx + 1) % vel_lag  # vel_lag-1 frames ago (~0.16s window)

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

                # Compute causal velocity vector from ring buffer before overwriting
                vx = vy = 0.0
                dt_f = idx - int(hist_f[slot, ref_col])
                if 1 <= dt_f <= 8 and not np.isnan(hist_x[slot, ref_col]):
                    inv_dt = FPS / float(dt_f)
                    raw_vx = (px - hist_x[slot, ref_col]) * inv_dt
                    raw_vy = (py - hist_y[slot, ref_col]) * inv_dt
                    vmag = float(np.hypot(raw_vx, raw_vy))
                    if vmag <= 11.5:  # Guard against rare ID-swap teleports
                        vx, vy = raw_vx, raw_vy

                hist_x[slot, buf_col] = px
                hist_y[slot, buf_col] = py
                hist_f[slot, buf_col] = idx

                if has_ball:
                    dx = bx - px
                    dy = by - py
                    d = float(np.hypot(dx, dy))
                    if d > 0.25:
                        v_toward = (vx * dx + vy * dy) / d
                    else:
                        v_toward = max(pspd, 0.0)

                    # Spearman vector T_int: closing players reach faster; receding players incur turn delay
                    turn_penalty = 0.30 if v_toward < -1.0 else 0.0
                    eff_spd = min(MAX_SPRINT_MPS, max(2.2, v_toward + 2.8))
                    t_int = REACTION_TIME_S + turn_penalty + d / eff_spd
                    candidates.append((d, t_int, pspd, v_toward, team_of_slot[slot]))

            if spd_cnt > 0:
                mean_player_spd[idx] = spd_sum / spd_cnt
            hsr_players[idx] = float(hsr_cnt)

            if has_ball and candidates:
                candidates.sort(key=lambda item: item[0])
                min_dh = 999.0
                min_da = 999.0
                min_th = 99.0
                min_ta = 99.0
                sh = 0.0
                sa = 0.0

                for rank, (d, t_int, pspd, v_toward, team) in enumerate(candidates):
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
                        # Combine proximity, closing intent, and rank decay
                        approach_boost = 1.0 + 0.08 * np.clip(v_toward, -2.0, 5.0)
                        w = np.exp(-d / 2.2) * approach_boost / (1.0 + 0.15 * rank)
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
                # Distance to nearest opponent relative to holder team
                opp_gap_arr[idx] = min_da if min_dh <= min_da else min_dh

    # Reject multi-frame ball tracking glitches/teleports before differentiation
    clean_vis = ball_vis.copy()
    last_good = -1
    for i in range(TOTAL_FRAMES):
        if clean_vis[i]:
            if last_good >= 0:
                dt_frames = i - last_good
                jump = float(np.hypot(ball_x[i] - ball_x[last_good], ball_y[i] - ball_y[last_good]))
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
                jump = float(np.hypot(ball_x[en] - ball_x[st - 1], ball_y[en] - ball_y[st - 1]))
                if jump <= MAX_TRACK_JUMP_M * (gap + 1):
                    alphas = np.linspace(0.0, 1.0, gap + 2)[1:-1]
                    interp_x[st:en] = ball_x[st - 1] + alphas * (ball_x[en] - ball_x[st - 1])
                    interp_y[st:en] = ball_y[st - 1] + alphas * (ball_y[en] - ball_y[st - 1])
        else:
            i += 1

    # Savitzky-Golay smoothed ball velocity vector (vx, vy), speed, and touch impulse
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

    # Link & Hoernig (2017) physical ball touch / deflection impulse: vector change |dv_b| over 4 frames
    ball_impulse = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    lag = 4
    dvx = ball_vx[lag:] - ball_vx[:-lag]
    dvy = ball_vy[lag:] - ball_vy[:-lag]
    both_vis = clean_vis[lag:] & clean_vis[:-lag]
    ball_impulse[lag:] = np.where(both_vis, np.hypot(dvx, dvy), 0.0)

    is_oob = np.zeros(TOTAL_FRAMES, dtype=bool)
    is_oob[clean_vis] = (
        (ball_x[clean_vis] < -OOB_MARGIN_M)
        | (ball_x[clean_vis] > PITCH_X + OOB_MARGIN_M)
        | (ball_y[clean_vis] < -OOB_MARGIN_M)
        | (ball_y[clean_vis] > PITCH_Y + OOB_MARGIN_M)
    )

    flock_smooth = smooth_1d(mean_player_spd, half_win=18).astype(np.float32)
    hsr_smooth = smooth_1d(hsr_players, half_win=12).astype(np.float32)
    clean_vis = clean_vis & (~np.isnan(ball_x))

    return {
        "vis": clean_vis,
        "oob": is_oob,
        "dh": dist_h,
        "da": dist_a,
        "th": tint_h,
        "ta": tint_a,
        "sh": score_h,
        "sa": score_a,
        "bspd": ball_speed_sg,
        "bimp": ball_impulse,
        "flock": flock_smooth,
        "hsr": hsr_smooth,
        "opp_gap": opp_gap_arr,
    }


def _find_restart_kick(feat, sf, ef):
    """Detect set-piece restart kick via sustained ball speed or acceleration impulse near a player."""
    vis = feat["vis"]
    oob = feat["oob"]
    dh = feat["dh"]
    da = feat["da"]
    bspd = feat["bspd"]
    bimp = feat["bimp"]

    consec = 0
    for f in range(sf, ef):
        near_player = vis[f] and min(dh[f], da[f]) < 3.5 and not oob[f]
        if near_player and (bspd[f] >= RESTART_KICK_SPEED_MPS or (bimp[f] >= 3.2 and bspd[f] >= 3.0)):
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
    """Convert ALL Smart Tagging phases (attack, defense, transitions, long ball, goals, set pieces)
    into soft log-odds priors and flight protection masks.
    """
    events = lst.load()
    priors = np.zeros((TOTAL_FRAMES, 3), dtype=np.float32)  # cols: H, A, D
    active_play_mask = np.zeros(TOTAL_FRAMES, dtype=bool)
    long_ball_mask = np.zeros(TOTAL_FRAMES, dtype=bool)

    vis = feat["vis"]
    oob = feat["oob"]
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
    stoppage_mask = np.zeros(TOTAL_FRAMES, dtype=bool)
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
            # Defending / defensive transition strongly implies live play & opponent possession
            active_play_mask[sf:ef] = True
            weight = 0.85 if code == "defensive transition" else 0.55
            if team_tag == "Barcelona":
                pos_a_votes[sf:ef] += weight
            elif team_tag == "Atletic Club":
                pos_h_votes[sf:ef] += weight

    # Goals: post-goal celebration + reset until next kickoff is strongly Dead Ball
    for sf, ef, team_tag in goals:
        stoppage_mask[sf:ef] = True
        priors[sf:ef, 2] += 5.0
        post_end = min(TOTAL_FRAMES, ef + 60)
        # Conceding team kicks off after a goal
        if team_tag == "Barcelona":
            priors[ef:post_end, 1] += 1.2
        elif team_tag == "Atletic Club":
            priors[ef:post_end, 0] += 1.2

    # Set Pieces: pre-tag dead lookback + kick detection + N/A team resolution
    for sf, ef, team_tag in set_pieces:
        ext_sf = sf
        lookback_limit = max(0, sf - 125)  # up to 5.0s before Smart Tagging window
        f = sf - 1
        while f >= lookback_limit:
            is_dead_evidence = (
                oob[f]
                or (not vis[f])
                or (flock[f] < WALK_SPEED_MPS + 0.1 and hsr[f] < 1.5)
                or (bspd[f] < 2.2 and flock[f] < 1.7 and hsr[f] < 1.5)
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
            post_end = min(TOTAL_FRAMES, kicked_at + 60)
            if team_tag == "Barcelona":
                priors[kicked_at:post_end, 0] += 1.5
            elif team_tag == "Atletic Club":
                priors[kicked_at:post_end, 1] += 1.5
            else:
                resolved = _resolve_team_at_restart(feat, kicked_at, ef)
                if resolved is not None:
                    priors[kicked_at:post_end, resolved] += 1.2

    # Stoppages (pre-kick set pieces & goals) take precedence over overlapping live-phase tags
    active_play_mask[stoppage_mask] = False
    pos_h_votes[stoppage_mask] = 0.0
    pos_a_votes[stoppage_mask] = 0.0

    # Soft live-play & possession-side priors
    priors[active_play_mask, 2] -= 1.40
    net_vote = pos_h_votes - pos_a_votes
    priors[net_vote > 0.3, 0] += np.minimum(1.1, 0.65 * net_vote[net_vote > 0.3])
    priors[net_vote < -0.3, 1] += np.minimum(1.1, 0.65 * (-net_vote[net_vote < -0.3]))

    return priors, active_play_mask, long_ball_mask


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
    """Remove micro-segments after decoding so no sub-second artifacts remain."""
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

    # Pass 2: bridge brief opponent challenges (H -> A -> H or A -> H -> A < BRIDGE_CHALLENGE_FRAMES)
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

    # Pass 3: absorb any remaining short H/A runs (< min_alive)
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
    h_bimp,
    h_long,
    h_prior,
    init_team=0,
):
    """Asymmetric possession retention + physical touch-snapped onset + light bounded smoothing.

    - Protects aerial passes (>= 13 m/s) and clean driven passes (>= 7.5 m/s without touch impulse)
      from premature turnovers (Link & Hoernig 2017).
    - Uses vector Spearman T_int + Euclidean margin for opponent takeovers.
    - Snaps backdated transitions onto local touch impulse / reception peaks for sharp IoU boundaries.
    """
    n = len(h_vis)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    # Check Set Piece restart prior first, else earliest confident ball control
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

        # Protect aerial balls & clean fast passes (unless physical touch/turn impulse occurs nearby)
        in_aerial = (h_bspd[k] > AERIAL_FLIGHT_SPEED_MPS) and (min_d > FLIGHT_TOUCH_DIST_M)
        in_driven_pass = (h_bspd[k] > PASS_FLIGHT_SPEED_MPS) and (h_bimp[k] < 2.5) and (min_d > 1.8)
        in_long_ball = bool(h_long[k]) and (h_bspd[k] > 5.0) and (min_d > 1.6)
        in_flight = in_aerial or in_driven_pass or in_long_ball

        if h_vis[k] and not in_flight:
            d_curr = dh_k if curr == 0 else da_k
            d_opp = da_k if curr == 0 else dh_k
            t_curr = h_th[k] if curr == 0 else h_ta[k]
            t_opp = h_ta[k] if curr == 0 else h_th[k]

            p_adv = float(h_prior[k, 1] - h_prior[k, 0]) if curr == 0 else float(h_prior[k, 0] - h_prior[k, 1])
            eff_margin = HYSTERESIS_MARGIN_M - (0.35 if p_adv >= 0.7 else 0.0)
            kinematic_adv = (d_curr - d_opp) + 0.85 * np.clip(t_curr - t_opp, -1.5, 1.5)

            opp_takeover = (d_opp < CONTROL_REACH_M) and (kinematic_adv > eff_margin)
            if opp_takeover:
                if opp_streak == 0:
                    streak_start = k
                opp_streak += 1
                if opp_streak >= CONFIRM_TAKEOVER_FRAMES:
                    new_team = 1 - curr
                    # Snap onset to nearest physical touch/deflection impulse around streak_start
                    win_lo = max(0, streak_start - 10)
                    win_hi = min(k + 1, streak_start + 16)
                    onset = streak_start
                    if win_hi > win_lo:
                        imp_slice = h_bimp[win_lo:win_hi]
                        max_imp_rel = int(np.argmax(imp_slice))
                        if imp_slice[max_imp_rel] >= 2.2:
                            onset = win_lo + max_imp_rel
                    curr = new_team
                    if 0 <= onset < k:
                        raw[onset:k] = curr
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
            opp_streak = max(0, opp_streak - 1)
            if opp_streak == 0:
                streak_start = -1

        raw[k] = curr

    # Light bounded majority smoothing (+/- 0.96s) inside Alive span only
    if n <= 15:
        majority = 0 if np.sum(raw == 0) >= np.sum(raw == 1) else 1
        return np.full(n, majority, dtype=np.int8)

    hw = min(ALIVE_SMOOTH_HALF_WIN, max(1, (n - 1) // 2))
    sig = np.where(raw == 0, 1.0, -1.0)
    sm = smooth_1d(sig, hw)
    return np.where(sm >= 0.0, 0, 1).astype(np.int8)


def run_v14_model(
    feat,
    priors,
    active_play_mask,
    long_ball_mask,
    switch_penalty=DEFAULT_SWITCH_PENALTY,
    dead_penalty=DEFAULT_DEAD_PENALTY,
):
    """Two-stage vector-kinematic & multimodal Viterbi decoder."""
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
    bimp = feat["bimp"]
    flock = feat["flock"]
    hsr = feat["hsr"]
    opp_gap = feat["opp_gap"]

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
        h_bimp = bimp[sl].astype(np.float64)
        h_flock = flock[sl].astype(np.float64)
        h_hsr = hsr[sl].astype(np.float64)
        h_gap = opp_gap[sl].astype(np.float64)
        h_prior = priors[sl].astype(np.float64)
        h_active = active_play_mask[sl]
        h_long = long_ball_mask[sl]

        # 1. Alive team emissions when ball is visible inside pitch
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

        # 2. Context-aware visible Dead-Ball evidence (BIP vs BOP: Vidal-Codina 2022; Jerome 2023)
        loose_vis = in_play_vis & (min_d > UNCONTROLLED_DIST_M)
        no_close_control = min_d > 3.2
        low_activity = (h_flock < WALK_SPEED_MPS) | ((h_flock < 1.55) & (h_hsr < 0.8))

        emit[loose_vis & low_activity, 2] += 3.6
        emit[loose_vis & (~low_activity), 2] += 0.7
        # Guard against falsely marking slow in-play build-up as D when a player holds the ball
        emit[in_play_vis & no_close_control & low_activity & (h_bspd < 5.0), 2] += 2.9

        # Walking ball-placement/retrieval guard: only outside tagged live play when everyone walks
        # and the ball is nearly stationary/slowly rolled into set-piece position.
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

        # Ball visible outside touchlines/goal lines -> strong Dead Ball
        emit[h_oob, 2] += 5.5
        emit[h_oob, 0] -= 2.5
        emit[h_oob, 1] -= 2.5

        # 3. Missing-ball dynamics driven by HSR player count + Smart Tagging context
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
                            # Active non-stoppage Smart Tagging phase: keep play alive even during
                            # slower build-up or >5s box occlusions unless players come to a near-standstill.
                            if h_hsr[k] >= 1.2 or h_flock[k] >= 1.55:
                                emit[k, 2] -= 1.0
                            elif h_flock[k] < 1.10 and h_hsr[k] < 0.5 and elapsed > 100:
                                emit[k, 2] += min(0.8, (elapsed - 100) / 80.0)
                            else:
                                emit[k, 2] -= 0.4
                        elif h_hsr[k] >= 2.2 and elapsed < 140:
                            # Multiple players sprinting despite missing ball -> live play
                            emit[k, 2] -= 0.3
                        else:
                            # Stoppage ramp outside active play tags
                            ramp = min(3.8, max(0.0, (elapsed - 55) / 30.0))
                            if mean_fl < WALK_SPEED_MPS or mean_hsr < 0.7:
                                ramp += 1.5
                            elif h_hsr[k] < 0.6 and h_flock[k] < 1.80:
                                ramp += 0.7
                            emit[k, 2] += ramp
            else:
                i += 1

        # 4. Multimodal Smart Tagging priors + Stage-1 Viterbi (Alive vs Dead)
        emit += h_prior
        log_init = np.array([-0.5, -0.5, -1.0], dtype=np.float64)
        vit_path = viterbi_decode(emit, log_trans, log_init)
        vit_clean = cleanup_short_segments(
            vit_path, min_alive=MIN_ALIVE_SPELL_FRAMES, min_dead=MIN_DEAD_RUN_FRAMES
        )

        # 5. Stage-2 touch-snapped asymmetric H/A decoding strictly within Alive intervals
        final_codes = vit_clean.copy()
        idx = 0
        while idx < m:
            if final_codes[idx] != 2:
                st = idx
                while idx < m and final_codes[idx] != 2:
                    idx += 1
                en = idx
                final_codes[st:en] = decode_alive_span(
                    h_vis[st:en],
                    h_dh[st:en],
                    h_da[st:en],
                    h_th[st:en],
                    h_ta[st:en],
                    h_bspd[st:en],
                    h_bimp[st:en],
                    h_long[st:en],
                    h_prior[st:en],
                    init_team=int(final_codes[st]),
                )
            else:
                idx += 1

        final_codes = cleanup_short_segments(
            final_codes, min_alive=MIN_ALIVE_SPELL_FRAMES, min_dead=MIN_DEAD_RUN_FRAMES
        )
        labels[sl] = state_names[final_codes]

    return labels


def main():
    parser = argparse.ArgumentParser(description="Run v14 Vector Context Viterbi possession model.")
    parser.add_argument("--name", default="v14_vector_context_viterbi", help="Output submission stem")
    parser.add_argument("--switch-pen", type=float, default=DEFAULT_SWITCH_PENALTY, help="Log transition penalty H<->A")
    parser.add_argument("--dead-pen", type=float, default=DEFAULT_DEAD_PENALTY, help="Log transition penalty <->D")
    args = parser.parse_args()

    feat = extract_kinematic_features()
    priors, active_mask, long_ball_mask = build_smart_tagging_priors(feat)
    print(f"Decoding full match with v14 multimodal Viterbi (switch={args.switch_pen}, dead={args.dead_pen})...")
    labels = run_v14_model(
        feat,
        priors,
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
