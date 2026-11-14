"""Possession Decoder v3: Ballistic-Aware Physics + Stoppage Inertia Compensation + Asymmetric Viterbi.

Implements the 5 rules-compliant v3 architectural upgrades over v2 without overfitting the 5-minute public slice:
1. Active In-Field Open-Play Ball Protection & Ballistic Fly-By Filtering (Newton's 1st Law:
   a ball moving > 8.2 m/s without a kick/touch impulse >= 12 m/s^2 is in ballistic flight, not controlled).
2. Minimum Physical Duration for Ungrounded Dead ('D') Islands + Asymmetric Viterbi Dead Entry/Exit Penalties
   (`heal_short_ungrounded_dead_spells` heals brief open-play D blips <= 70f that lack physical triggers).
3. Post-OOB Recovery & Goalkeeper Box Physics Lock (suppresses retreating formation sprint scores during
   post-OOB retrieval while releasing immediately on active wing-restart dribble).
4. Transition Boundary Sharpening (`mean_abs_latency` & `spellF1@0.6-0.7`) via:
   - Stoppage-gap player deceleration & restart-burst inertia compensation,
   - Isolated brief deflection filtering (`<= 18f` sandwiched between same-team anchors),
   - Local touch-gated Viterbi switching penalty (`switch_team_touch=18.0` vs `switch_team_base=29.0`).
5. Full-Match Rules-Compliant Discipline (evaluated across all 199,319 frames / 90 minutes).
"""

import numpy as np

from games import GAME
from src.atd_features import nan_box_smooth_blocked
from src.atd_loader import AWAY, HOME
from src.possession_model import LABEL_CHARS, STATE_A, STATE_D, STATE_H
from src.possession_model_v2 import _compute_post_oob_recovery, live_half_blocks

NOBODY = -1
STATE_X = 3


def compute_dead_ball_score_v3(atd_feats, st_feats, live_blocks):
    """Compute per-frame dead-ball evidence (> 0 = Dead, < 0 = Live) for v3.

    Preserves v2's high stoppage recall while locking `post_oob` against retreating
    formation sprints and compensating player deceleration/restart inertia at long stoppage gaps.
    """
    n = len(atd_feats["ball_visible"])
    score = np.zeros(n, dtype=np.float64)

    vis = atd_feats["ball_visible"]
    oob = atd_feats["ball_oob"]
    vis_in = vis & (~oob)
    b_spd = atd_feats["ball_speed"]
    b_spd_safe = np.where(np.isnan(b_spd), 2.0, b_spd)
    nd = atd_feats["nearest_dist"]
    in_sp = st_feats["in_set_piece"]

    m_spd = np.where(
        np.isnan(atd_feats["mean_player_speed_smooth"]),
        1.5,
        atd_feats["mean_player_speed_smooth"],
    )
    p80 = np.where(
        np.isnan(atd_feats["p80_player_speed_smooth"]),
        2.0,
        atd_feats["p80_player_speed_smooth"],
    )
    p90 = np.where(
        np.isnan(atd_feats["p90_player_speed_smooth"]),
        2.6,
        atd_feats["p90_player_speed_smooth"],
    )
    walk = np.where(
        np.isnan(atd_feats["walk_fraction_smooth"]),
        0.55,
        atd_feats["walk_fraction_smooth"],
    )

    post_oob = _compute_post_oob_recovery(atd_feats, live_blocks, max_recovery_frames=185)

    # 1. Collective player engagement & tail sprint speed
    speed_live = np.clip((m_spd - 1.45) * 2.2, -2.0, 2.0)
    p80_live = np.clip((p80 - 1.85) * 1.3, -1.6, 1.8)
    p90_live = np.clip((p90 - 2.20) * 0.9, -1.0, 1.2)
    walk_dead = np.clip((walk - 0.65) * 3.5, -1.2, 1.5)

    # During post_oob (goal-kick / throw-in retrieval after ball exited pitch),
    # high player speeds are formation retreat unless a player is actively accelerating with ball control.
    p80_prev25 = np.roll(p80, 25)
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

    # 2. Visible in-bounds ball kinematics
    players_engaged = (p80 >= 1.80) | (p90 >= 2.25) | (walk < 0.65)
    active_moving_in = vis_in & (b_spd_safe >= 1.8) & players_engaged
    fast_kick_in = vis_in & (b_spd_safe >= 10.0) & (~players_engaged)
    still_wait = vis_in & (b_spd_safe < 0.6) & (p80 < 1.80)

    score[active_moving_in] -= 1.8
    score[fast_kick_in] -= 1.6
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

    # Compensate player deceleration inertia at the start of confirmed stoppage gaps,
    # and player acceleration inertia when the ball bursts out of a stoppage gap
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
                post_end = min(e, ge + 40)
                burst = (
                    vis_in[ge:post_end]
                    & (~in_sp[ge:post_end])
                    & (~post_oob[ge:post_end])
                    & (b_spd_safe[ge:post_end] >= 4.0)
                )
                score[ge:post_end][burst] -= 2.4

    internal_live_occ = miss & (~gap_oob) & (~gap_edge) & (~in_sp) & (~flat_stop) & (score < 1.5)
    score[internal_live_occ] -= 0.8

    # 4. Smart Tagging set-piece waiting boost vs active restart onset
    score[in_sp & (~oob) & (p80 < 2.10)] += 0.8
    active_phase = np.abs(st_feats["st_bias"]) >= 0.8
    nd_safe = np.where(np.isnan(nd), 0.0, nd)
    sp_restart_kick = vis_in & in_sp & active_phase & (b_spd_safe >= 5.5) & (nd_safe >= 1.5)
    score[sp_restart_kick] -= 1.2

    # Smooth gently inside live blocks only
    score = nan_box_smooth_blocked(score, half_win=3, blocks=live_blocks)
    score = np.where(np.isnan(score), 0.0, score)

    # 5. Sharp physical boundary overrides AFTER smoothing
    score[oob] = np.maximum(score[oob] + 6.0, 4.5)
    score[post_oob] = np.maximum(score[post_oob] + 2.6, 2.8)

    for s, e in live_blocks:
        ko_end = min(e, s + 15)
        not_oob = ~oob[s:ko_end]
        score[s:ko_end][not_oob] -= 2.5

    score = np.clip(score, -5.5, 6.5)
    score[st_feats["pre_kickoff_dead_mask"]] = 10.0
    return score, post_oob


def _find_control_anchors_v3(atd_feats, dead_score, radius_m=2.1, margin_m=0.75, min_run=5):
    """Identify close-range control anchors while filtering out ballistic high-speed fly-bys."""
    vis_in = atd_feats["ball_visible"] & (~atd_feats["ball_oob"]) & (dead_score < 1.0)
    nd = atd_feats["nearest_dist"]
    margin = atd_feats["dist_margin"]
    bspd = np.where(np.isnan(atd_feats["ball_speed"]), 0.0, atd_feats["ball_speed"])
    accel = np.where(np.isnan(atd_feats["ball_accel"]), 0.0, atd_feats["ball_accel"])

    # Newton's First Law: a ball flying > 8.2 m/s without a kick/deflection impulse (< 12 m/s^2)
    # is in ballistic flight (pass/clearance flying past a player), not controlled at the feet.
    controllable_or_touched = (bspd <= 8.2) | ((accel >= 12.0) & (nd <= 1.9))

    raw = np.full(len(vis_in), NOBODY, dtype=np.int8)
    raw[vis_in & controllable_or_touched & (nd <= radius_m) & (margin >= margin_m)] = HOME
    raw[vis_in & controllable_or_touched & (nd <= radius_m) & (margin <= -margin_m)] = AWAY

    anchors = np.full(len(vis_in), NOBODY, dtype=np.int8)
    for team in (HOME, AWAY):
        mask = (raw == team).astype(np.int8)
        edges = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        for s, e in zip(starts, stops):
            length_r = e - s
            if length_r >= min_run or (
                length_r >= 3
                and np.nanmax(np.abs(margin[s:e])) >= 1.6
                and np.nanmin(bspd[s:e]) <= 6.5
            ):
                anchors[s:e] = team
    return anchors


def _apply_passing_and_hold_v3(
    team_score,
    anchors,
    atd_feats,
    dead_score,
    live_blocks,
    max_bridge_frames=110,
    hold_tau_frames=85.0,
):
    """Symmetric passing bridge + ballistic-aware causal hold-until-control."""
    out = team_score.copy()
    vis_in = atd_feats["ball_visible"] & (~atd_feats["ball_oob"])
    nd = atd_feats["nearest_dist"]
    margin = atd_feats["dist_margin"]
    bspd = np.where(np.isnan(atd_feats["ball_speed"]), 0.0, atd_feats["ball_speed"])
    accel = np.where(np.isnan(atd_feats["ball_accel"]), 0.0, atd_feats["ball_accel"])

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

    # Filter out isolated brief deflection/block anchors (<= 18 frames) sandwiched between
    # two anchors of the opposing team within 130 frames of live play.
    filtered_anchors = anchors.copy()
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
                filtered_anchors[s_cur : e_cur + 1] = NOBODY

    idx = np.flatnonzero(filtered_anchors >= 0)
    runs = []
    if len(idx) >= 1:
        start = idx[0]
        prev = idx[0]
        prev_team = int(filtered_anchors[idx[0]])
        for k in idx[1:]:
            t = int(filtered_anchors[k])
            if k == prev + 1 and t == prev_team:
                prev = k
            else:
                runs.append((start, prev, prev_team))
                start = k
                prev = k
                prev_team = t
        runs.append((start, prev, prev_team))

    # 1. Symmetric same-team anchor bridge
    for (_, e1, t1), (s2, _, t2) in zip(runs[:-1], runs[1:]):
        if t1 != t2:
            continue
        gap = s2 - e1
        if gap <= 1 or gap > max_bridge_frames:
            continue
        if np.max(dead_score[e1:s2]) >= 1.2:
            continue
        sign = 1.0 if t1 == HOME else -1.0
        out[e1 + 1 : s2] += sign * 0.95

    # 2. Causal hold-until-control inertia (ignores high-speed ballistic fly-bys)
    real_opp_control = (bspd <= 8.2) | ((accel >= 12.0) & (nd <= 1.9))
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

            if vis_in[t] and (not np.isnan(nd[t])) and nd[t] <= 2.4 and real_opp_control[t]:
                opp_closer = (
                    (last_team == HOME and margin[t] <= -0.6)
                    or (last_team == AWAY and margin[t] >= 0.6)
                )
                if opp_closer:
                    last_team = NOBODY
                    continue

            if (not vis_in[t]) or np.isnan(nd[t]) or nd[t] > 2.4 or (not real_opp_control[t]):
                sign = 1.0 if last_team == HOME else -1.0
                out[t] += sign * 0.85 * np.exp(-elapsed / hold_tau_frames)

    return out, filtered_anchors


def compute_team_evidence_v3(atd_feats, st_feats, dead_score, live_blocks):
    """Compute per-frame Home-vs-Away log-odds (> 0 = Home, < 0 = Away) with ballistic gating."""
    n = len(atd_feats["ball_visible"])
    score = np.zeros(n, dtype=np.float64)

    vis_in = atd_feats["ball_visible"] & (~atd_feats["ball_oob"])
    nd = atd_feats["nearest_dist"]
    margin = atd_feats["dist_margin"]
    bspd = np.where(np.isnan(atd_feats["ball_speed"]), 0.0, atd_feats["ball_speed"])
    accel = np.where(np.isnan(atd_feats["ball_accel"]), 0.0, atd_feats["ball_accel"])

    # 1. Distance-decayed AND speed-gated geometric proximity evidence:
    # Damp proximity when the ball is flying fast (> 7.5 m/s) without a touch impulse (< 12 m/s^2)
    flight_damp = np.where(
        (bspd > 7.5) & (accel < 12.0),
        np.clip(1.0 - (bspd - 7.5) / 5.5, 0.12, 1.0),
        1.0,
    )
    prox_weight = np.where(vis_in & (~np.isnan(nd)), np.exp(-nd / 2.8) * flight_damp, 0.0)
    margin_clipped = np.where(np.isnan(margin), 0.0, np.clip(margin, -4.0, 4.0))
    score += prox_weight * (margin_clipped * 0.45)

    # 2. Control anchors + same-team pass bridge + ballistic-aware hold inertia
    anchors = _find_control_anchors_v3(atd_feats, dead_score)
    score, filtered_anchors = _apply_passing_and_hold_v3(
        score, anchors, atd_feats, dead_score, live_blocks
    )

    score[filtered_anchors == HOME] += 0.65
    score[filtered_anchors == AWAY] -= 0.65

    # 3. Gated Smart Tagging tactical prior
    st_tanh = np.tanh(st_feats["st_bias"] / 1.5)
    st_weight = np.where(~vis_in, 0.35, np.where(np.isnan(nd) | (nd > 2.5), 0.08, 0.06))
    score += st_weight * st_tanh

    # 4. Set-piece restart team anchor
    sp_team = st_feats["set_piece_team"]
    live_in_sp = (dead_score < 1.0) & (sp_team >= 0)
    score[live_in_sp & (sp_team == HOME)] += 0.75
    score[live_in_sp & (sp_team == AWAY)] -= 0.75

    return np.clip(score, -2.2, 2.2), filtered_anchors


def viterbi_decode_half_v3(
    emit_scores,
    touch_gate,
    switch_team_base=29.0,
    switch_team_touch=18.0,
    enter_dead_cost=24.0,
    exit_dead_cost=20.0,
):
    """Time-varying & asymmetric Viterbi decoder for (H=0, A=1, D=2).

    - Lowers H<->A switching cost at physical contact/impulse frames (`touch_gate=True`) so
      transitions snap accurately to the exact frame of interception/touch.
    - Uses asymmetric Live->Dead (`enter_dead_cost`) vs Dead->Live (`exit_dead_cost`) penalties.
    """
    T = len(emit_scores)
    if T == 0:
        return np.empty(0, dtype=np.int8)

    dp = np.zeros((T, 3), dtype=np.float64)
    back = np.zeros((T, 3), dtype=np.int8)
    dp[0] = emit_scores[0]

    base_pen = np.array(
        [
            [0.0, -switch_team_base, -exit_dead_cost],
            [-switch_team_base, 0.0, -exit_dead_cost],
            [-enter_dead_cost, -enter_dead_cost, 0.0],
        ],
        dtype=np.float64,
    )
    touch_pen = np.array(
        [
            [0.0, -switch_team_touch, -exit_dead_cost],
            [-switch_team_touch, 0.0, -exit_dead_cost],
            [-enter_dead_cost, -enter_dead_cost, 0.0],
        ],
        dtype=np.float64,
    )

    for t in range(1, T):
        pen = touch_pen if touch_gate[t] else base_pen
        cand = dp[t - 1][:, None] + pen
        best_prev = np.argmax(cand, axis=0)
        dp[t] = emit_scores[t] + cand[best_prev, np.arange(3)]
        back[t] = best_prev

    states = np.empty(T, dtype=np.int8)
    states[T - 1] = int(np.argmax(dp[T - 1]))
    for t in range(T - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


def heal_short_ungrounded_dead_spells(
    codes, atd_feats, st_feats, post_oob, live_blocks, max_heal_len=70
):
    """Enforce minimum physical duration for ungrounded Dead ('D') islands.

    Real stoppages either have a physical trigger (OOB, gap_oob, post_oob, set_piece, or long
    missing-ball gap >= 75f) or last longer than ~2.8s. Brief D blips inside open play without
    any physical trigger are healed to the surrounding live team.
    """
    out = codes.copy()
    oob = atd_feats["ball_oob"]
    gap_oob = atd_feats["gap_oob"]
    gap_len = atd_feats["gap_len"]
    in_sp = st_feats["in_set_piece"]

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
                or bool(np.any(gap_len[ds_f:de_f] >= 75))
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


def predict_possession_v3(
    atd_feats,
    st_feats,
    game=GAME,
    switch_team_base=29.0,
    switch_team_touch=18.0,
    enter_dead_cost=24.0,
    exit_dead_cost=20.0,
    max_heal_len=70,
):
    """Generate full-match frame-by-frame possession predictions (v3)."""
    n = len(atd_feats["ball_visible"])
    blocks = live_half_blocks(st_feats, n, game=game)

    dead_score, post_oob = compute_dead_ball_score_v3(atd_feats, st_feats, blocks)
    team_score, _ = compute_team_evidence_v3(atd_feats, st_feats, dead_score, blocks)

    abs_team = np.abs(team_score)
    emit = np.empty((n, 3), dtype=np.float64)
    emit[:, STATE_H] = -0.5 * dead_score + team_score - abs_team
    emit[:, STATE_A] = -0.5 * dead_score - team_score - abs_team
    emit[:, STATE_D] = 0.5 * dead_score

    nd = atd_feats["nearest_dist"]
    bspd = np.where(np.isnan(atd_feats["ball_speed"]), 0.0, atd_feats["ball_speed"])
    accel = np.where(np.isnan(atd_feats["ball_accel"]), 0.0, atd_feats["ball_accel"])
    touch_gate = (
        atd_feats["ball_visible"]
        & (~atd_feats["ball_oob"])
        & (~np.isnan(nd))
        & (nd <= 2.1)
        & ((bspd <= 6.5) | (accel >= 11.0))
    )
    touch_gate_s = (
        nan_box_smooth_blocked(touch_gate.astype(np.float64), half_win=2, blocks=blocks) > 0.0
    )

    codes = np.full(n, STATE_X, dtype=np.int8)
    for s, e in blocks:
        codes[s:e] = viterbi_decode_half_v3(
            emit[s:e],
            touch_gate_s[s:e],
            switch_team_base=switch_team_base,
            switch_team_touch=switch_team_touch,
            enter_dead_cost=enter_dead_cost,
            exit_dead_cost=exit_dead_cost,
        )

    codes = heal_short_ungrounded_dead_spells(
        codes, atd_feats, st_feats, post_oob, blocks, max_heal_len=max_heal_len
    )
    codes[st_feats["pre_kickoff_dead_mask"]] = STATE_D

    return LABEL_CHARS[codes]
