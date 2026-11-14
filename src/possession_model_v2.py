"""Possession inference v2: decoupled hierarchical emissions, physical dead-ball kinetics,
hold-until-control momentum, and gated Smart Tagging priors.

Addresses the failure modes identified in v1 error analysis:
1. Decouples Live-vs-Dead from team proximity so set-piece takers near a dead ball do
   not bias emissions away from 'D'.
2. Prevents ball movement caused by carrying/retrieving a dead ball from cancelling
   collective walking stoppages, makes out-of-bounds boundaries physically dominant,
   and models post-OOB retrieval setups cleanly.
3. Holds possession on the last controlling team while a ball is in fast flight in open
   space until an opponent closes within control range or a stoppage occurs.
4. Gates long-horizon Smart Tagging tactical priors so they fill occlusions/restarts
   without overriding clear ATD player proximity.
5. Respects active half-block / kickoff boundaries during smoothing and decoding.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from src.atd_features import nan_box_smooth_blocked  # noqa: E402
from src.atd_loader import AWAY, HOME  # noqa: E402
from src.possession_model import (  # noqa: E402
    LABEL_CHARS,
    STATE_A,
    STATE_D,
    STATE_H,
    viterbi_decode_half,
)

NOBODY = -1


def live_half_blocks(st_feats, n_frames, game=GAME):
    """Return list of [start, end) video-frame blocks starting at each half's kickoff."""
    pre_ko = st_feats["pre_kickoff_dead_mask"]
    blocks = []
    for _, first_f, last_f in half_video_blocks(game):
        s = max(0, int(first_f))
        e = min(n_frames, int(last_f) + 1)
        if e <= s:
            continue
        # Advance start past any graded pre-kickoff dead frames (e.g. 2nd half warmup)
        live_mask = ~pre_ko[s:e]
        if not np.any(live_mask):
            continue
        first_live = s + int(np.argmax(live_mask))
        blocks.append((first_live, e))
    return blocks


def _compute_post_oob_recovery(atd_feats, live_blocks, max_recovery_frames=175):
    """Detect ball retrieval / set-piece setup immediately following an out-of-bounds exit.

    After the ball crosses touchline/endline (ball_oob), a player often walks it back inside
    the pitch near the edge or goal area while teammates jog back into formation. Without
    explicit memory of the preceding OOB exit, clearing ball_oob immediately flips predictions
    to Live before play actually restarts.
    """
    n = len(atd_feats["ball_visible"])
    recovery = np.zeros(n, dtype=bool)

    vis = atd_feats["ball_visible"]
    oob = atd_feats["ball_oob"]
    gap_oob = atd_feats["gap_oob"]
    bx = atd_feats["ball_x"]
    by = atd_feats["ball_y"]
    bspd = atd_feats["ball_speed"]

    # Boundary band: within 8% of endlines (covers 6-yard goal area) or 6% of touchlines
    in_interior = vis & (~oob) & (bx > 0.09) & (bx < 0.91) & (by > 0.07) & (by < 0.93)
    fast_restart = vis & (~oob) & (~np.isnan(bspd)) & (bspd >= 9.0)

    for s, e in live_blocks:
        frames_since_oob = max_recovery_frames + 1
        active = False
        fast_run = 0
        for t in range(s, e):
            if oob[t] or ((~vis[t]) and gap_oob[t]):
                active = True
                frames_since_oob = 0
                fast_run = 0
                continue

            if not active:
                continue

            frames_since_oob += 1
            if frames_since_oob > max_recovery_frames:
                active = False
                continue

            if in_interior[t]:
                active = False
                continue

            if fast_restart[t]:
                fast_run += 1
                if fast_run >= 3:
                    active = False
                    continue
            else:
                fast_run = 0

            # Active post-OOB retrieval setup inside near-boundary zone or ball-missing gap
            if (not vis[t]) or (~in_interior[t]):
                recovery[t] = True

    return recovery


def compute_dead_ball_score_v2(atd_feats, st_feats, live_blocks):
    """Compute per-frame dead-ball evidence (> 0 = Dead, < 0 = Live) from first principles."""
    n = len(atd_feats["ball_visible"])
    score = np.zeros(n, dtype=np.float64)

    vis = atd_feats["ball_visible"]
    oob = atd_feats["ball_oob"]
    vis_in = vis & (~oob)
    b_spd = atd_feats["ball_speed"]
    nd = atd_feats["nearest_dist"]

    m_spd = np.where(np.isnan(atd_feats["mean_player_speed_smooth"]), 1.5, atd_feats["mean_player_speed_smooth"])
    p80 = np.where(np.isnan(atd_feats["p80_player_speed_smooth"]), 2.0, atd_feats["p80_player_speed_smooth"])
    p90 = np.where(np.isnan(atd_feats["p90_player_speed_smooth"]), 2.6, atd_feats["p90_player_speed_smooth"])
    walk = np.where(np.isnan(atd_feats["walk_fraction_smooth"]), 0.55, atd_feats["walk_fraction_smooth"])

    # 1. Collective player engagement & tail sprint speed
    # Calibrated around full-match bimodal separation between open play and stoppages.
    speed_live = np.clip((m_spd - 1.45) * 2.2, -2.0, 2.0)
    p80_live = np.clip((p80 - 1.85) * 1.3, -1.6, 1.8)
    p90_live = np.clip((p90 - 2.20) * 0.9, -1.0, 1.2)
    walk_dead = np.clip((walk - 0.65) * 3.5, -1.2, 1.5)
    score -= (speed_live + p80_live + p90_live)
    score += walk_dead

    # 2. Visible in-bounds ball kinematics conditioned on collective player engagement
    # Players often roll or carry a dead ball at 2-7 m/s while everyone walks; only count
    # moderate ball motion as Live when players are engaged or the kick itself is fast.
    b_spd_safe = np.where(np.isnan(b_spd), 2.0, b_spd)
    players_engaged = (p80 >= 1.80) | (p90 >= 2.25) | (walk < 0.65)

    active_moving_in = vis_in & (b_spd_safe >= 1.8) & players_engaged
    fast_kick_in = vis_in & (b_spd_safe >= 10.0) & (~players_engaged)
    still_wait = vis_in & (b_spd_safe < 0.6) & (p80 < 1.80)

    score[active_moving_in] -= 1.8
    score[fast_kick_in] -= 1.6
    score[still_wait] += 2.0

    # 3. Missing-ball gap context
    miss = ~vis
    gap_oob = atd_feats["gap_oob"]
    gap_edge = atd_feats["gap_near_edge"]
    gap_len = atd_feats["gap_len"]
    in_sp = st_feats["in_set_piece"]

    score[miss & gap_oob] += 4.2
    score[miss & gap_edge & (gap_len >= 30) & (p80 < 2.35)] += 2.0

    # Prolonged missing-ball gap inside pitch indicates stoppage ONLY if collective speeds stay flat
    flat_stop = (p80 < 1.70) & (p90 < 2.05) & (walk >= 0.68)
    score[miss & (gap_len >= 100) & flat_stop] += 1.6

    # Brief/moderate internal tracking occlusions without OOB/edge/set-piece are typically live
    internal_live_occ = miss & (~gap_oob) & (~gap_edge) & (~in_sp) & (~flat_stop)
    score[internal_live_occ] -= 0.8

    # 4. Smart Tagging set-piece waiting boost vs active restart onset
    score[in_sp & (~oob) & (p80 < 2.10)] += 0.8
    active_phase = np.abs(st_feats["st_bias"]) >= 0.8
    nd_safe = np.where(np.isnan(nd), 0.0, nd)
    sp_restart_kick = vis_in & in_sp & active_phase & (b_spd_safe >= 5.5) & (nd_safe >= 1.5)
    score[sp_restart_kick] -= 1.2

    # Smooth gently inside live blocks only (zero bleed across kickoff / halftime)
    score = nan_box_smooth_blocked(score, half_win=3, blocks=live_blocks)
    score = np.where(np.isnan(score), 0.0, score)

    # 5. Sharp, non-blurred physical boundary overrides applied AFTER smoothing
    # Out-of-bounds ball is physically dead regardless of how fast players sprint/jog.
    post_oob = _compute_post_oob_recovery(atd_feats, live_blocks)
    score[oob] = np.maximum(score[oob] + 6.0, 4.5)
    score[post_oob] += 2.4

    # Kickoff frame begins live play immediately (do not linger in pre-kickoff D)
    for s, e in live_blocks:
        ko_end = min(e, s + 15)
        not_oob = ~oob[s:ko_end]
        score[s:ko_end][not_oob] -= 2.5

    score = np.clip(score, -5.5, 6.5)
    score[st_feats["pre_kickoff_dead_mask"]] = 10.0
    return score


def _find_control_anchors_v2(atd_feats, dead_score, radius_m=2.1, margin_m=0.75, min_run=5):
    """Identify sustained close-range control anchors (-1=none, 0=Home, 1=Away)."""
    vis_in = atd_feats["ball_visible"] & (~atd_feats["ball_oob"]) & (dead_score < 1.0)
    nd = atd_feats["nearest_dist"]
    margin = atd_feats["dist_margin"]

    raw = np.full(len(vis_in), NOBODY, dtype=np.int8)
    raw[vis_in & (nd <= radius_m) & (margin >= margin_m)] = HOME
    raw[vis_in & (nd <= radius_m) & (margin <= -margin_m)] = AWAY

    anchors = np.full(len(vis_in), NOBODY, dtype=np.int8)
    for team in (HOME, AWAY):
        mask = (raw == team).astype(np.int8)
        edges = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        for s, e in zip(starts, stops):
            length = e - s
            if length >= min_run or (length >= 2 and np.nanmax(np.abs(margin[s:e])) >= 1.6):
                anchors[s:e] = team
    return anchors


def _apply_passing_and_hold(team_score, anchors, atd_feats, dead_score, live_blocks,
                            max_bridge_frames=100, hold_tau_frames=80.0):
    """Combine symmetric passing bridges with causal hold-until-control momentum.

    1) Symmetric bridge: reinforces Team T across flight/occlusions between two same-team
       anchors within max_bridge_frames and no dead-ball pause in between.
    2) Causal hold-until-control: after Team T controls the ball, maintains possession on T
       while the ball is in flight in open space (nearest_dist > 2.6 m) or occluded until
       an opponent physically closes on the ball (nearest_dist <= 2.6 m with >= 0.5 m
       margin for opponent) or a stoppage occurs. Eliminates premature mid-flight flips.
    """
    out = team_score.copy()

    # 1. Symmetric same-team anchor bridge
    idx = np.flatnonzero(anchors >= 0)
    if len(idx) >= 2:
        runs = []
        start = idx[0]
        prev = idx[0]
        prev_team = anchors[idx[0]]
        for k in idx[1:]:
            t = anchors[k]
            if k == prev + 1 and t == prev_team:
                prev = k
            else:
                runs.append((start, prev, prev_team))
                start = k
                prev = k
                prev_team = t
        runs.append((start, prev, prev_team))

        for (_, e1, t1), (s2, _, t2) in zip(runs[:-1], runs[1:]):
            if t1 != t2:
                continue
            gap = s2 - e1
            if gap <= 1 or gap > max_bridge_frames:
                continue
            if np.max(dead_score[e1:s2]) >= 1.0:
                continue
            sign = 1.0 if t1 == HOME else -1.0
            out[e1 + 1:s2] += sign * 0.90

    # 2. Forward hold-until-control inertia during in-flight / occlusion segments
    vis_in = atd_feats["ball_visible"] & (~atd_feats["ball_oob"])
    nd = atd_feats["nearest_dist"]
    margin = atd_feats["dist_margin"]

    for s, e in live_blocks:
        last_team = NOBODY
        elapsed = 0
        for t in range(s, e):
            if dead_score[t] >= 1.2:
                last_team = NOBODY
                elapsed = 0
                continue

            if anchors[t] >= 0:
                last_team = int(anchors[t])
                elapsed = 0
                continue

            if last_team == NOBODY:
                continue

            elapsed += 1
            if elapsed > 200:
                last_team = NOBODY
                continue

            # Check if the opposing team has visibly closed onto the ball
            if vis_in[t] and (not np.isnan(nd[t])) and nd[t] <= 2.6:
                opp_closer = (
                    (last_team == HOME and margin[t] <= -0.5)
                    or (last_team == AWAY and margin[t] >= 0.5)
                )
                if opp_closer:
                    last_team = NOBODY
                    continue

            # Ball is uncontested in flight (> 2.6 m) or missing -> hold last controller
            if (not vis_in[t]) or np.isnan(nd[t]) or nd[t] > 2.6:
                sign = 1.0 if last_team == HOME else -1.0
                out[t] += sign * 0.85 * np.exp(-elapsed / hold_tau_frames)

    return out


def compute_team_evidence_v2(atd_feats, st_feats, dead_score, live_blocks):
    """Compute per-frame Home-vs-Away log-odds (> 0 = Home, < 0 = Away)."""
    n = len(atd_feats["ball_visible"])
    score = np.zeros(n, dtype=np.float64)

    vis_in = atd_feats["ball_visible"] & (~atd_feats["ball_oob"])
    nd = atd_feats["nearest_dist"]
    margin = atd_feats["dist_margin"]

    # 1. Distance-decayed geometric proximity evidence
    prox_weight = np.where(vis_in & (~np.isnan(nd)), np.exp(-nd / 2.8), 0.0)
    margin_clipped = np.where(np.isnan(margin), 0.0, np.clip(margin, -4.0, 4.0))
    score += prox_weight * (margin_clipped * 0.45)

    clear_home = vis_in & (nd <= 2.1) & (margin >= 0.75)
    clear_away = vis_in & (nd <= 2.1) & (margin <= -0.75)
    score[clear_home] += 0.65
    score[clear_away] -= 0.65

    # 2. Passing bridge + causal hold-until-control inertia
    anchors = _find_control_anchors_v2(atd_feats, dead_score)
    score = _apply_passing_and_hold(score, anchors, atd_feats, dead_score, live_blocks)

    # 3. Gated Smart Tagging tactical prior:
    # Strong during ball occlusions (~vis_in), gentle when ball is in clear view near players
    # or mid-flight so long phase tags never override real ATD touches or cause flight flips.
    st_tanh = np.tanh(st_feats["st_bias"] / 1.5)
    st_weight = np.where(~vis_in, 0.35, np.where(np.isnan(nd) | (nd > 2.5), 0.08, 0.06))
    score += st_weight * st_tanh

    # 4. Set-piece restart team anchor right around set-piece release
    sp_team = st_feats["set_piece_team"]
    live_in_sp = (dead_score < 1.0) & (sp_team >= 0)
    score[live_in_sp & (sp_team == HOME)] += 0.75
    score[live_in_sp & (sp_team == AWAY)] -= 0.75

    return np.clip(score, -2.2, 2.2)


def predict_possession_v2(atd_feats, st_feats, game=GAME, switch_team_cost=28.0, switch_dead_cost=22.0):
    """Run v2 possession pipeline and return length-N array of labels ('H','A','D','X')."""
    n_frames = len(atd_feats["ball_visible"])
    blocks = live_half_blocks(st_feats, n_frames, game=game)

    dead_score = compute_dead_ball_score_v2(atd_feats, st_feats, blocks)
    team_score = compute_team_evidence_v2(atd_feats, st_feats, dead_score, blocks)

    # Exact decoupled hierarchical 3-state emissions:
    #   max(emit[H], emit[A]) - emit[D] == -dead_score identically for any |team_score|,
    #   emit[H] - emit[A] == 2 * team_score.
    abs_team = np.abs(team_score)
    emit = np.empty((n_frames, 3), dtype=np.float64)
    emit[:, STATE_H] = -0.5 * dead_score + team_score - abs_team
    emit[:, STATE_A] = -0.5 * dead_score - team_score - abs_team
    emit[:, STATE_D] = 0.5 * dead_score

    # Default everything outside graded halves to X (3)
    codes = np.full(n_frames, 3, dtype=np.int8)
    for s, e in blocks:
        codes[s:e] = viterbi_decode_half(
            emit[s:e],
            switch_team_cost=switch_team_cost,
            switch_dead_cost=switch_dead_cost,
        )

    # Enforce pre-kickoff graded frames strictly as Dead ('D')
    codes[st_feats["pre_kickoff_dead_mask"]] = STATE_D
    return LABEL_CHARS[codes]
