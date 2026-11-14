"""Rule-based + 3-state Viterbi possession model (v4: 5-Pillar Football Physics Architecture).

Key physical and tactical upgrades over v3 (`v3_physics_viterbi`):
1. Pillar 1 — Exact Set-Piece & Stoppage Kick-Release Detector:
   - Stationary ball waiting for a free-kick/corner/throw-in/goal-kick remains Dead ('D')
     until the physical kick/throw impulse occurs (`verified_restart_kick`).
   - Releases live play immediately upon a verified kick impulse in-bounds where collective
     player speeds (`p80`) accelerate into live play over the next 40 frames (`walk_fut40 <= 0.72`),
     preventing 22-player `walk_fraction` inertia from delaying restarts while ignoring casual
     ball tosses during stoppages.
   - Early wing restart dribble release from `post_oob` once a receiver controls and dribbles
     the ball inside the pitch with rising collective player speed.
2. Pillar 2 — Goalkeeper Goal-Side Identity Correction (`upgrade_atd_features_v4`):
   - Automatically identifies goalkeeper track slots (`median(|x - 0.5|) > 0.35`) and corrects
     goal-end slot swaps (`x < 0.25` vs `x > 0.75`) relative to each team's defending half,
     eliminating false opponent turnovers when a goalkeeper holds or distributes the ball inside
     their own penalty/6-yard box.
3. Pillar 3 — Sustained Unchased Boundary Stoppage Extrapolation (`unchased_boundary_stop`):
   - Detects balls rolling into the boundary corridor (`x <= 0.10 | x >= 0.90 | y <= 0.12 | y >= 0.88`)
     with no player within 6.0 m, receding nearest player (`nd_fut15 - nd >= 1.0 m`), and
     collapsing collective player speed (`p80 - p80_fut45 >= 0.85 m/s`), sustaining the dead-ball
     state smoothly until collective stoppage walking takes over.
4. Pillar 4 — Co-Moving Sprint Dribble Vector & Pass-Flight Lock:
   - Computes player-ball relative velocity (`rel_spd_home`, `rel_spd_away`) to preserve control
     anchors during high-speed sprint dribbles (`bspd <= 9.8 m/s` with `rel_spd <= 3.2 m/s`)
     while rejecting high-speed pass fly-bys (`min(bspd) > 7.2 m/s` and `min(rel_spd) > 3.2 m/s`).
   - Trims leading pre-reception approach frames on incoming passes (`bspd >= 4.4 m/s`, `accel < 9.5 m/s^2`,
     `nd > 1.05 m`, `rel_spd > 3.0 m/s`) so turnovers align with actual foot contact.
   - Blocks same-team pass bridging across gaps where the opposing team executed a strong
     deflection/interception impulse (`accel >= 18.0 m/s^2`, `abs(margin) >= 2.2 m`).
5. Pillar 5 — Close-Combat Tackle Detector, Interception Kick & In-Play Occlusion Discipline:
   - Detects close-combat boot tackles (`nd <= 0.68 m`, `abs(margin) >= 0.30 m`) when the tackling
     team emerges with sustained spatial margin (`>= 0.85 m`) over the subsequent 25 frames, as well
     as Newtonian unseen-foot kick impulses when an optical player track is briefly occluded.
   - Preserves short active interception kicks (`len >= 6`, `accel >= 15.0 m/s^2`, `nd <= 1.0 m`,
     `abs(margin) >= 1.1 m`) without false hold-until-control bleed.
   - Requires collective walking (`walk >= 0.68`) before treating a missing-ball occlusion gap
     (`gap_len >= 75`) as a real stoppage trigger in `heal_short_ungrounded_dead_spells_v4`.
"""

import numpy as np

from src.atd_features import nan_box_smooth_blocked
from src.atd_loader import AWAY, HOME
from src.possession_model import LABEL_CHARS, STATE_A, STATE_D, STATE_H
from src.possession_model_v2 import _compute_post_oob_recovery, live_half_blocks
from src.possession_model_v3 import viterbi_decode_half_v3

NOBODY = -1
STATE_X = 3


def upgrade_atd_features_v4(atd, atd_feats, live_blocks):
    """Apply Pillar 2 (GK goal-side identity correction) and Pillar 4 (player-ball relative speed)."""
    px = atd["player_x"]
    py = atd["player_y"]
    bx = atd["ball_x"]
    by = atd["ball_y"]
    team_slot = atd["team_of_slot"]
    length, width = atd["pitch"]
    fps = atd["fps"]
    n = len(bx)

    with np.errstate(all="ignore"):
        med_abs_x = np.nanmedian(np.abs(px - 0.5), axis=0)
    gk_slots = np.flatnonzero(med_abs_x > 0.35)

    ball_vis = atd_feats["ball_visible"]
    vis_rows = np.flatnonzero(ball_vis)
    dx = (px[vis_rows] - bx[vis_rows, None]) * length
    dy = (py[vis_rows] - by[vis_rows, None]) * width
    dist = np.hypot(dx, dy).astype(np.float64)
    dist_safe = np.where(np.isnan(dist), np.inf, dist)

    dyn_team = np.broadcast_to(team_slot[None, :], (len(vis_rows), len(team_slot))).copy()
    for s, e in live_blocks:
        with np.errstate(all="ignore"):
            h_mean_x = np.nanmean(px[s:e, team_slot == HOME])
            a_mean_x = np.nanmean(px[s:e, team_slot == AWAY])
        left_team = HOME if h_mean_x < a_mean_x else AWAY
        right_team = AWAY if left_team == HOME else HOME
        in_half = (vis_rows >= s) & (vis_rows < e)
        for g_slot in gk_slots:
            gx = px[vis_rows[in_half], g_slot]
            t_col = dyn_team[in_half, g_slot].copy()
            t_col[gx < 0.25] = left_team
            t_col[gx > 0.75] = right_team
            dyn_team[in_half, g_slot] = t_col

    dist_h_mat = np.where(dyn_team == HOME, dist_safe, np.inf)
    dist_a_mat = np.where(dyn_team == AWAY, dist_safe, np.inf)
    slot_h = np.argmin(dist_h_mat, axis=1)
    slot_a = np.argmin(dist_a_mat, axis=1)
    d_home_v4 = dist_h_mat[np.arange(len(vis_rows)), slot_h]
    d_away_v4 = dist_a_mat[np.arange(len(vis_rows)), slot_a]
    d_home_v4[np.isinf(d_home_v4)] = np.nan
    d_away_v4[np.isinf(d_away_v4)] = np.nan

    prev_rows = np.maximum(0, vis_rows - 1)
    vx_h = (px[vis_rows, slot_h] - px[prev_rows, slot_h]).astype(np.float64) * length * fps
    vy_h = (py[vis_rows, slot_h] - py[prev_rows, slot_h]).astype(np.float64) * width * fps
    vx_a = (px[vis_rows, slot_a] - px[prev_rows, slot_a]).astype(np.float64) * length * fps
    vy_a = (py[vis_rows, slot_a] - py[prev_rows, slot_a]).astype(np.float64) * width * fps
    bvx_vis = (bx[vis_rows] - bx[prev_rows]).astype(np.float64) * length * fps
    bvy_vis = (by[vis_rows] - by[prev_rows]).astype(np.float64) * width * fps

    rel_spd_h = np.hypot(bvx_vis - vx_h, bvy_vis - vy_h)
    rel_spd_a = np.hypot(bvx_vis - vx_a, bvy_vis - vy_a)

    out = dict(atd_feats)
    out["min_dist_home"] = np.full(n, np.nan, dtype=np.float64)
    out["min_dist_away"] = np.full(n, np.nan, dtype=np.float64)
    out["rel_spd_home"] = np.full(n, np.nan, dtype=np.float64)
    out["rel_spd_away"] = np.full(n, np.nan, dtype=np.float64)
    out["min_dist_home"][vis_rows] = d_home_v4
    out["min_dist_away"][vis_rows] = d_away_v4
    out["rel_spd_home"][vis_rows] = rel_spd_h
    out["rel_spd_away"][vis_rows] = rel_spd_a
    out["nearest_dist"] = np.fmin(out["min_dist_home"], out["min_dist_away"])
    out["dist_margin"] = out["min_dist_away"] - out["min_dist_home"]
    return out


def compute_dead_ball_score_v4(atd_feats, st_feats, live_blocks):
    """Compute per-frame dead-ball log-odds (> 0 = Dead, < 0 = Live) for v4."""
    n_f = len(atd_feats["ball_visible"])
    score = np.zeros(n_f, dtype=np.float64)

    vis = atd_feats["ball_visible"]
    oob = atd_feats["ball_oob"]
    vis_in = vis & (~oob)
    bx = atd_feats["ball_x"]
    by = atd_feats["ball_y"]
    b_spd = atd_feats["ball_speed"]
    b_spd_safe = np.where(np.isnan(b_spd), 2.0, b_spd)
    b_acc = np.where(np.isnan(atd_feats["ball_accel"]), 0.0, atd_feats["ball_accel"])
    nd = atd_feats["nearest_dist"]
    nd_safe = np.where(np.isnan(nd), 99.0, nd)
    margin = np.where(np.isnan(atd_feats["dist_margin"]), 0.0, atd_feats["dist_margin"])
    in_sp = st_feats["in_set_piece"]

    m_spd = np.where(np.isnan(atd_feats["mean_player_speed_smooth"]), 1.5, atd_feats["mean_player_speed_smooth"])
    p80 = np.where(np.isnan(atd_feats["p80_player_speed_smooth"]), 2.0, atd_feats["p80_player_speed_smooth"])
    p90 = np.where(np.isnan(atd_feats["p90_player_speed_smooth"]), 2.6, atd_feats["p90_player_speed_smooth"])
    walk = np.where(np.isnan(atd_feats["walk_fraction_smooth"]), 0.55, atd_feats["walk_fraction_smooth"])

    post_oob = _compute_post_oob_recovery(atd_feats, live_blocks, max_recovery_frames=185)

    # 1. Collective player engagement & tail sprint speed
    speed_live = np.clip((m_spd - 1.45) * 2.2, -2.0, 2.0)
    p80_live = np.clip((p80 - 1.85) * 1.3, -1.6, 1.8)
    p90_live = np.clip((p90 - 2.20) * 0.9, -1.0, 1.2)
    walk_dead = np.clip((walk - 0.65) * 3.5, -1.2, 1.5)

    p80_prev25 = np.roll(p80, 25)
    p80_fut40 = np.roll(p80, -40)
    walk_fut40 = np.roll(walk, -40)

    # Pillar 1A: Wing restart dribble release from post_oob
    wing_restart_dribble = (
        vis_in
        & (~np.isnan(nd))
        & (nd <= 1.5)
        & (b_spd_safe >= 1.8)
        & (p80 >= 1.85)
        & ((p80 - p80_prev25) >= 0.25)
        & (walk < 0.62)
    )
    post_oob = post_oob & (~wing_restart_dribble)

    live_kin = speed_live + p80_live + p90_live
    live_kin = np.where(post_oob, np.minimum(0.0, live_kin), live_kin)

    score -= live_kin
    score += walk_dead

    # 2. Visible in-bounds ball kinematics (guarded by walk < 0.73 against stoppage ball-tossing)
    players_engaged = (p80 >= 1.80) | (p90 >= 2.25) | (walk < 0.65)
    active_moving_in = vis_in & (b_spd_safe >= 1.8) & players_engaged
    fast_kick_in = vis_in & (~post_oob) & (~in_sp) & (b_spd_safe >= 6.0) & (~players_engaged) & (walk < 0.73)
    still_wait = vis_in & (b_spd_safe < 0.6) & (p80 < 1.80)

    score[active_moving_in] -= 1.8
    score[fast_kick_in] -= 2.4
    score[still_wait] += 2.0

    # 3. Missing-ball gap context & Stoppage-Gap Inertia Compensation
    miss = ~vis
    gap_oob = atd_feats["gap_oob"]
    gap_edge = atd_feats["gap_near_edge"]
    gap_len = atd_feats["gap_len"]

    score[miss & gap_oob] += 4.2
    score[miss & gap_edge & (gap_len >= 30) & (p80 < 2.35)] += 2.0

    flat_stop = (p80 < 1.70) & (p90 < 2.05) & (walk >= 0.68)
    score[miss & (gap_len >= 100) & flat_stop] += 1.6

    for s, e in live_blocks:
        missing = miss[s:e].astype(np.int8)
        edges = np.diff(np.concatenate(([0], missing, [0])))
        starts = np.flatnonzero(edges == 1) + s
        stops = np.flatnonzero(edges == -1) + s
        for gs, ge in zip(starts, stops):
            glen = ge - gs
            if glen < 100:
                continue
            if np.nanpercentile(p80[gs:ge], 30) < 1.60 and np.nanmedian(walk[gs:ge]) >= 0.72:
                decel = p80[gs:ge] < 4.9
                score[gs:ge][decel] = np.maximum(score[gs:ge][decel], 2.5)
                post_end = min(e, ge + 45)
                burst = (
                    vis_in[ge:post_end]
                    & (~in_sp[ge:post_end])
                    & (~post_oob[ge:post_end])
                    & (b_spd_safe[ge:post_end] >= 3.2)
                )
                score[ge:post_end][burst] -= 3.2

    internal_live_occ = miss & (~gap_oob) & (~gap_edge) & (~in_sp) & (~flat_stop) & (score < 1.5)
    score[internal_live_occ] -= 1.2

    # 4. Pillar 3: Sustained Unchased Boundary Stoppage Extrapolation
    nd_fut15 = np.roll(nd_safe, -15)
    p80_fut45 = np.roll(p80, -45)
    walk_fut45 = np.roll(walk, -45)
    in_corridor = (by >= 0.88) | (by <= 0.12) | (bx <= 0.10) | (bx >= 0.90)
    unchased_trigger = (
        vis_in
        & in_corridor
        & (nd_safe >= 6.0)
        & ((nd_fut15 - nd_safe) >= 1.0)
        & ((p80 - p80_fut45) >= 0.85)
        & (p80_fut45 < 1.95)
        & (walk_fut45 >= 0.58)
    )
    unchased_boundary_stop = np.zeros(n_f, dtype=bool)
    for s, e in live_blocks:
        active = False
        for t in range(s, e):
            if unchased_trigger[t]:
                active = True
            elif active:
                if (not vis_in[t]) or (not in_corridor[t]) or (nd_safe[t] < 5.5) or (walk[t] >= 0.68):
                    active = False
            if active:
                unchased_boundary_stop[t] = True

    score[unchased_boundary_stop] = np.maximum(score[unchased_boundary_stop], 3.4)

    # 5. Pillar 1B: Set-piece waiting vs Verified Restart Kick-Release Detector
    score[in_sp & (~oob) & (p80 < 2.10)] += 0.8
    active_phase = np.abs(st_feats["st_bias"]) >= 0.8
    sp_restart_kick = vis_in & in_sp & active_phase & (b_spd_safe >= 5.0) & (nd_safe >= 1.5)
    score[sp_restart_kick] -= 2.2

    verified_restart_kick = (
        vis_in
        & (~post_oob)
        & (b_spd_safe >= 4.2)
        & ((b_acc >= 10.0) | (b_spd_safe >= 7.5))
        & (np.abs(margin) >= 2.5)
        & ((p80_fut40 - p80) >= 0.28)
        & (walk_fut40 <= 0.72)
    )
    restart_release = np.zeros(n_f, dtype=bool)
    for k in np.flatnonzero(verified_restart_kick):
        for t in range(k, min(n_f, k + 50)):
            if not vis_in[t] or post_oob[t]:
                break
            if (b_spd_safe[t] >= 1.8) or (nd_safe[t] <= 1.5):
                restart_release[t] = True
            else:
                break
    score[restart_release] = np.minimum(score[restart_release], -1.8)

    score = nan_box_smooth_blocked(score, half_win=3, blocks=live_blocks)
    score = np.where(np.isnan(score), 0.0, score)

    # 6. Sharp physical boundary overrides AFTER smoothing
    score[oob] = np.maximum(score[oob] + 6.0, 4.5)
    score[post_oob] = np.maximum(score[post_oob] + 2.6, 2.8)
    score[unchased_boundary_stop] = np.maximum(score[unchased_boundary_stop], 2.8)

    for s, e in live_blocks:
        ko_end = min(e, s + 15)
        not_oob = ~oob[s:ko_end]
        score[s:ko_end][not_oob] -= 2.5

    score = np.clip(score, -5.5, 6.5)
    score[st_feats["pre_kickoff_dead_mask"]] = 10.0
    return score, post_oob


def _find_control_anchors_v4(atd_feats, dead_score, radius_m=2.1, margin_m=0.75, min_run=5):
    """Identify control anchors with Pillar 4 (co-moving sprint dribble, pass-flight trim)
    and Pillar 5 (close-combat boot tackle & unseen-foot kick impulse).
    """
    vis_in = atd_feats["ball_visible"] & (~atd_feats["ball_oob"]) & (dead_score < 1.0)
    nd = atd_feats["nearest_dist"]
    margin = atd_feats["dist_margin"]
    bspd = np.where(np.isnan(atd_feats["ball_speed"]), 0.0, atd_feats["ball_speed"])
    accel = np.where(np.isnan(atd_feats["ball_accel"]), 0.0, atd_feats["ball_accel"])
    rel_h = np.where(np.isnan(atd_feats["rel_spd_home"]), 99.0, atd_feats["rel_spd_home"])
    rel_a = np.where(np.isnan(atd_feats["rel_spd_away"]), 99.0, atd_feats["rel_spd_away"])

    # Pillar 4A: Co-moving sprint dribble allows controllable up to 9.8 m/s when rel_spd <= 3.2 m/s
    controllable_h = (bspd <= 7.8) | ((bspd <= 9.8) & (rel_h <= 3.2)) | ((accel >= 12.0) & (nd <= 1.9))
    controllable_a = (bspd <= 7.8) | ((bspd <= 9.8) & (rel_a <= 3.2)) | ((accel >= 12.0) & (nd <= 1.9))

    raw = np.full(len(vis_in), NOBODY, dtype=np.int8)
    raw[vis_in & controllable_h & (nd <= radius_m) & (margin >= margin_m)] = HOME
    raw[vis_in & controllable_a & (nd <= radius_m) & (margin <= -margin_m)] = AWAY

    # Pillar 5A: Close-combat boot tackle (nd <= 0.68m, abs(margin) >= 0.30m, sustained team margin over next 25f)
    close_h = vis_in & (bspd <= 6.5) & (nd <= 0.68) & (margin >= 0.30) & (margin < margin_m)
    close_a = vis_in & (bspd <= 6.5) & (nd <= 0.68) & (margin <= -0.30) & (margin > -margin_m)
    for idx_t in np.flatnonzero(close_h):
        fut_m = margin[idx_t + 5 : min(len(vis_in), idx_t + 30)]
        valid_m = fut_m[~np.isnan(fut_m)]
        if len(valid_m) >= 5 and np.mean(valid_m) >= 0.85:
            raw[idx_t : min(len(vis_in), idx_t + 6)] = HOME
    for idx_t in np.flatnonzero(close_a):
        fut_m = margin[idx_t + 5 : min(len(vis_in), idx_t + 30)]
        valid_m = fut_m[~np.isnan(fut_m)]
        if len(valid_m) >= 5 and np.mean(valid_m) <= -0.85:
            raw[idx_t : min(len(vis_in), idx_t + 6)] = AWAY

    # Pillar 5B: Newtonian Unseen-Foot Kick Impulse (ball stopped < 1.2 m/s in live play, then kicked >= 10 m/s^2
    # while kicker's optical track is momentarily occluded nd > 2.5m, with margin shifting strongly to kicking team)
    for t in range(20, len(vis_in) - 25):
        if not vis_in[t] or raw[t] != NOBODY:
            continue
        if nd[t] > 2.5 and accel[t] >= 10.0 and bspd[t] >= 1.5 and bspd[t] <= 6.5:
            if np.nanmin(bspd[t - 18 : t]) <= 1.2 and np.nanmax(bspd[t : t + 18]) >= 5.5:
                m_now = margin[t]
                fut_slice = margin[t + 5 : t + 22]
                valid_f = fut_slice[~np.isnan(fut_slice)]
                if len(valid_f) >= 5:
                    m_fut = float(np.mean(valid_f))
                    if m_now >= 0.40 and m_fut >= 1.60 and (m_fut - m_now) >= 0.60:
                        raw[t - 2 : t + 6] = HOME
                    elif m_now <= -0.40 and m_fut <= -1.60 and (m_fut - m_now) <= -0.60:
                        raw[t - 2 : t + 6] = AWAY

    anchors = np.full(len(vis_in), NOBODY, dtype=np.int8)
    for team in (HOME, AWAY):
        mask = (raw == team).astype(np.int8)
        edges = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        rel_team = rel_h if team == HOME else rel_a
        for s, e in zip(starts, stops):
            truly_controlled = (np.nanmin(bspd[s:e]) <= 7.2) or (np.nanmin(rel_team[s:e]) <= 3.2)
            if not truly_controlled:
                continue
            while (e - s) > min_run and bspd[s] >= 4.4 and accel[s] < 9.5 and nd[s] > 1.05 and rel_team[s] > 3.0:
                s += 1
            length_r = e - s
            if length_r >= min_run or (
                length_r >= 3
                and np.nanmax(np.abs(margin[s:e])) >= 1.6
                and np.nanmin(bspd[s:e]) <= 6.5
            ):
                anchors[s:e] = team
    return anchors


def compute_team_evidence_v4(
    atd_feats, st_feats, dead_score, live_blocks, max_bridge_frames=110, hold_tau_frames=85.0
):
    """Compute per-frame Home-vs-Away log-odds (> 0 = Home, < 0 = Away) for v4."""
    n_f = len(atd_feats["ball_visible"])
    score = np.zeros(n_f, dtype=np.float64)

    vis_in = atd_feats["ball_visible"] & (~atd_feats["ball_oob"])
    nd = atd_feats["nearest_dist"]
    margin = atd_feats["dist_margin"]
    bspd = np.where(np.isnan(atd_feats["ball_speed"]), 0.0, atd_feats["ball_speed"])
    accel = np.where(np.isnan(atd_feats["ball_accel"]), 0.0, atd_feats["ball_accel"])

    flight_damp = np.where(
        (bspd > 7.5) & (accel < 12.0),
        np.clip(1.0 - (bspd - 7.5) / 5.5, 0.12, 1.0),
        1.0,
    )
    prox_weight = np.where(vis_in & (~np.isnan(nd)), np.exp(-nd / 2.8) * flight_damp, 0.0)
    margin_clipped = np.where(np.isnan(margin), 0.0, np.clip(margin, -4.0, 4.0))
    score += prox_weight * (margin_clipped * 0.45)

    anchors = _find_control_anchors_v4(atd_feats, dead_score)

    idx = np.flatnonzero(anchors >= 0)
    runs = []
    if len(idx) >= 1:
        start = idx[0]
        prev = idx[0]
        prev_team = int(anchors[idx[0]])
        for k in idx[1:]:
            t = int(anchors[k])
            if k == prev + 1 and t == prev_team:
                prev = k
            else:
                runs.append((start, prev, prev_team, prev - start + 1))
                start = k
                prev = k
                prev_team = t
        runs.append((start, prev, prev_team, prev - start + 1))

    filtered_anchors = anchors.copy()
    interception_mask = np.zeros(n_f, dtype=bool)
    if len(runs) >= 3:
        for i in range(1, len(runs) - 1):
            s_prev, e_prev, t_prev, _ = runs[i - 1]
            s_cur, e_cur, t_cur, len_cur = runs[i]
            s_next, e_next, t_next, _ = runs[i + 1]
            if (
                t_prev == t_next
                and t_cur != t_prev
                and len_cur <= 18
                and (s_next - e_prev) <= 130
                and np.max(dead_score[e_prev:s_next]) < 1.2
            ):
                is_active_interception = (
                    len_cur >= 6
                    and np.nanmax(accel[s_cur : e_cur + 1]) >= 15.0
                    and np.nanmin(nd[s_cur : e_cur + 1]) <= 1.0
                    and np.nanmax(np.abs(margin[s_cur : e_cur + 1])) >= 1.1
                )
                if is_active_interception:
                    w_s = max(e_prev + 1, s_cur - 12)
                    w_e = min(s_next, e_cur + 2)
                    interception_mask[w_s:w_e] = True
                    sign_int = 1.0 if t_cur == HOME else -1.0
                    score[w_s:w_e] += sign_int * 2.15
                filtered_anchors[s_cur : e_cur + 1] = NOBODY

    idx = np.flatnonzero(filtered_anchors >= 0)
    runs2 = []
    if len(idx) >= 1:
        start = idx[0]
        prev = idx[0]
        prev_team = int(filtered_anchors[idx[0]])
        for k in idx[1:]:
            t = int(filtered_anchors[k])
            if k == prev + 1 and t == prev_team:
                prev = k
            else:
                runs2.append((start, prev, prev_team))
                start = k
                prev = k
                prev_team = t
        runs2.append((start, prev, prev_team))

    # 1. Symmetric same-team anchor bridge (blocked if opponent intercepted in between)
    for (_, e1, t1), (s2, _, t2) in zip(runs2[:-1], runs2[1:]):
        if t1 != t2:
            continue
        gap = s2 - e1
        if gap <= 1 or gap > max_bridge_frames:
            continue
        if np.max(dead_score[e1:s2]) >= 1.2:
            continue
        opp_interception = (
            (margin[e1 + 1 : s2] <= -2.2 if t1 == HOME else margin[e1 + 1 : s2] >= 2.2)
            & (accel[e1 + 1 : s2] >= 18.0)
            & (bspd[e1 + 1 : s2] <= 6.5)
        )
        if np.sum(opp_interception) >= 4:
            continue
        sign = 1.0 if t1 == HOME else -1.0
        score[e1 + 1 : s2] += sign * 0.95

    # 2. Causal hold-until-control inertia (with Pass-Flight Lock)
    real_opp_control = (bspd <= 7.5) | ((accel >= 14.0) & (nd <= 1.75))
    for s, e in live_blocks:
        last_team = NOBODY
        elapsed = 0
        for t in range(s, e):
            if dead_score[t] >= 1.2:
                last_team = NOBODY
                elapsed = 0
                continue
            if filtered_anchors[t] >= 0:
                last_team = int(filtered_anchors[t])
                elapsed = 0
                continue
            if last_team == NOBODY:
                continue
            elapsed += 1
            if elapsed > 200:
                last_team = NOBODY
                continue
            if vis_in[t] and (not np.isnan(nd[t])) and nd[t] <= 2.2 and real_opp_control[t]:
                opp_closer = (
                    (last_team == HOME and margin[t] <= -0.6)
                    or (last_team == AWAY and margin[t] >= 0.6)
                )
                if opp_closer:
                    last_team = NOBODY
                    continue
            if (not vis_in[t]) or np.isnan(nd[t]) or nd[t] > 2.2 or (not real_opp_control[t]):
                sign = 1.0 if last_team == HOME else -1.0
                score[t] += sign * 0.85 * np.exp(-elapsed / hold_tau_frames)

    score[filtered_anchors == HOME] += 0.65
    score[filtered_anchors == AWAY] -= 0.65

    st_tanh = np.tanh(st_feats["st_bias"] / 1.5)
    st_weight = np.where(~vis_in, 0.35, np.where(np.isnan(nd) | (nd > 2.5), 0.08, 0.06))
    score += st_weight * st_tanh

    sp_team = st_feats["set_piece_team"]
    live_in_sp = (dead_score < 1.0) & (sp_team >= 0)
    score[live_in_sp & (sp_team == HOME)] += 0.75
    score[live_in_sp & (sp_team == AWAY)] -= 0.75

    return np.clip(score, -2.4, 2.4), filtered_anchors, interception_mask


def heal_short_ungrounded_dead_spells_v4(
    codes, atd_feats, st_feats, post_oob, live_blocks, max_heal_len=70
):
    """Heal brief Dead ('D') blips in open play that lack a physical stoppage trigger."""
    out = codes.copy()
    oob = atd_feats["ball_oob"]
    gap_oob = atd_feats["gap_oob"]
    gap_len = atd_feats["gap_len"]
    in_sp = st_feats["in_set_piece"]
    walk = np.where(np.isnan(atd_feats["walk_fraction_smooth"]), 0.55, atd_feats["walk_fraction_smooth"])

    for s, e in live_blocks:
        seg = out[s:e]
        is_d = (seg == STATE_D).astype(np.int8)
        edges = np.diff(np.concatenate(([0], is_d, [0])))
        starts = np.flatnonzero(edges == 1) + s
        stops = np.flatnonzero(edges == -1) + s
        for ds_f, de_f in zip(starts, stops):
            d_len = de_f - ds_f
            if d_len > max_heal_len:
                continue
            has_trigger = (
                bool(np.any(oob[ds_f:de_f]))
                or bool(np.any(gap_oob[ds_f:de_f]))
                or bool(np.any(post_oob[ds_f:de_f]))
                or bool(np.any(in_sp[ds_f:de_f]))
                or bool(np.any((gap_len[ds_f:de_f] >= 75) & (walk[ds_f:de_f] >= 0.68)))
            )
            if has_trigger:
                continue
            prev_state = int(out[ds_f - 1]) if ds_f > s else STATE_D
            next_state = int(out[de_f]) if de_f < e else STATE_D
            if prev_state in (STATE_H, STATE_A):
                out[ds_f:de_f] = prev_state
            elif next_state in (STATE_H, STATE_A):
                out[ds_f:de_f] = next_state
    return out


def predict_possession_v4(atd_feats, st_feats, atd_raw=None):
    """Run full v4 possession inference across all frames."""
    n = len(atd_feats["ball_visible"])
    blocks = live_half_blocks(st_feats, n)

    if atd_raw is not None:
        atd_feats = upgrade_atd_features_v4(atd_raw, atd_feats, blocks)

    dead_s, post_oob = compute_dead_ball_score_v4(atd_feats, st_feats, blocks)
    team_s, fa4, int_mask4 = compute_team_evidence_v4(atd_feats, st_feats, dead_s, blocks)

    abs_team = np.abs(team_s)
    emit = np.empty((n, 3), dtype=np.float64)
    emit[:, STATE_H] = -0.5 * dead_s + team_s - abs_team
    emit[:, STATE_A] = -0.5 * dead_s - team_s - abs_team
    emit[:, STATE_D] = 0.5 * dead_s

    nd = atd_feats["nearest_dist"]
    bspd = np.where(np.isnan(atd_feats["ball_speed"]), 0.0, atd_feats["ball_speed"])
    accel = np.where(np.isnan(atd_feats["ball_accel"]), 0.0, atd_feats["ball_accel"])
    touch_gate = (
        (
            atd_feats["ball_visible"]
            & (~atd_feats["ball_oob"])
            & (~np.isnan(nd))
            & (nd <= 2.1)
            & ((bspd <= 6.5) | (accel >= 11.0))
        )
        | (fa4 >= 0)
        | int_mask4
    )
    touch_gate_s = nan_box_smooth_blocked(touch_gate.astype(np.float64), half_win=2, blocks=blocks) > 0.0

    codes = np.full(n, STATE_X, dtype=np.int8)
    for s, e in blocks:
        codes[s:e] = viterbi_decode_half_v3(emit[s:e], touch_gate_s[s:e])

    codes = heal_short_ungrounded_dead_spells_v4(codes, atd_feats, st_feats, post_oob, blocks, max_heal_len=70)
    codes[st_feats["pre_kickoff_dead_mask"]] = STATE_D

    labels = LABEL_CHARS[codes]
    return labels, codes, dead_s, team_s
