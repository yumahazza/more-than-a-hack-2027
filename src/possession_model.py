"""Possession inference model combining ATD kinematics, passing bridges, Smart Tagging, and Viterbi decoding.

Translates frame-level evidence into 3-state (H, A, D) sequence decoding per half
to avoid per-frame flickering that destroys spellF1, while setting ungraded
frames (before match, halftime, after full-time) to 'X'.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks  # noqa: E402
from src.atd_features import nan_box_smooth  # noqa: E402
from src.atd_loader import AWAY, HOME  # noqa: E402

STATE_H, STATE_A, STATE_D = 0, 1, 2
LABEL_CHARS = np.array(["H", "A", "D", "X"], dtype="<U1")


def compute_dead_ball_score(atd_feats, st_feats):
    """Compute per-frame log-odds-like dead-ball evidence (> 0 = Dead, < 0 = Live).

    Replaces the baseline's 'ball is NaN -> D' rule with football kinematics:
    distinguishes active occlusions (high player speeds inside pitch) from real
    stoppages (out-of-bounds ball, stationary ball, or slow-moving players).
    """
    n = len(atd_feats["ball_visible"])
    score = np.zeros(n, dtype=np.float64)

    vis = atd_feats["ball_visible"]
    oob = atd_feats["ball_oob"]
    b_spd = atd_feats["ball_speed"]
    m_spd = atd_feats["mean_player_speed_smooth"]
    p80_spd = atd_feats["p80_player_speed_smooth"]

    # Fill missing collective speeds neutrally
    m_spd_safe = np.where(np.isnan(m_spd), 1.5, m_spd)
    p80_safe = np.where(np.isnan(p80_spd), 2.0, p80_spd)

    # 1. Collective player speed signal (available on 100% of match frames)
    # Slow walking (< 1.35 m/s mean and < 1.85 m/s p80) indicates stoppage;
    # fast running (> 1.65 m/s mean or > 2.35 m/s p80) indicates live play.
    speed_live = np.clip((m_spd_safe - 1.45) * 2.5, -2.0, 2.0)
    sprint_live = np.clip((p80_safe - 2.05) * 1.2, -1.2, 1.5)
    score -= (speed_live + sprint_live)

    # 2. Visible ball signals
    vis_in = vis & (~oob)
    b_spd_safe = np.where(np.isnan(b_spd), 2.0, b_spd)

    # Ball moving inside pitch -> strong live signal
    moving_in = vis_in & (b_spd_safe >= 1.5)
    score[moving_in] -= 1.8

    # Ball sitting nearly still while players also move slowly -> set-piece wait (Dead)
    still_wait = vis_in & (b_spd_safe < 0.5) & (m_spd_safe < 1.45)
    score[still_wait] += 2.2

    # Ball visibly outside pitch bounds -> strong Dead signal
    score[oob] += 3.5

    # 3. Missing-ball gap context
    miss = ~vis
    gap_oob = atd_feats["gap_oob"]
    gap_edge = atd_feats["gap_near_edge"]
    gap_len = atd_feats["gap_len"]

    score[miss & gap_oob] += 2.8
    score[miss & gap_edge & (gap_len >= 40)] += 1.4
    # Very long missing-ball gap (> 6 s) without high player speed is usually a stoppage
    score[miss & (gap_len >= 150) & (m_spd_safe < 1.55)] += 1.2

    # 4. Smart Tagging set-piece modulation when players slow down
    in_sp = st_feats["in_set_piece"]
    score[in_sp & (m_spd_safe < 1.45)] += 0.8

    # Smooth slightly to remove 1-frame noise without blunting restarts
    score = nan_box_smooth(score, half_win=4)
    score = np.clip(score, -4.0, 4.5)

    # Hard lock pre-kickoff graded frames (e.g., before 2nd half kickoff) to Dead
    score[st_feats["pre_kickoff_dead_mask"]] = 10.0
    return score


def _find_control_anchors(atd_feats, dead_score, radius_m=2.0, margin_m=0.8, min_run=3):
    """Identify frames where one team clearly controls the ball at close range."""
    vis = atd_feats["ball_visible"] & (~atd_feats["ball_oob"]) & (dead_score < 1.0)
    nd = atd_feats["nearest_dist"]
    margin = atd_feats["dist_margin"]

    raw = np.full(len(vis), -1, dtype=np.int8)
    home_ctrl = vis & (nd <= radius_m) & (margin >= margin_m)
    away_ctrl = vis & (nd <= radius_m) & (margin <= -margin_m)
    raw[home_ctrl] = HOME
    raw[away_ctrl] = AWAY

    # Keep only control runs of at least `min_run` frames (or isolated clear touches > 1.2 m margin)
    anchors = np.full(len(vis), -1, dtype=np.int8)
    for team in (HOME, AWAY):
        mask = (raw == team).astype(np.int8)
        edges = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        for s, e in zip(starts, stops):
            if (e - s) >= min_run or np.nanmax(np.abs(margin[s:e])) >= 1.5:
                anchors[s:e] = team
    return anchors


def _apply_passing_bridge(team_score, anchors, dead_score, max_bridge_frames=85):
    """Bridge live intervals between same-team control anchors (Idea 2: pass continuity).

    When Team T clearly controls the ball at t1 and the next clear control anchor
    (within max_bridge_frames and without a Dead-Ball stoppage in between) also
    belongs to Team T, reinforce Team T across the intervening flight/occlusion frames.
    """
    bridged = team_score.copy()
    idx = np.flatnonzero(anchors >= 0)
    if len(idx) < 2:
        return bridged

    # Group contiguous anchor runs
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

    for (s1, e1, t1), (s2, e2, t2) in zip(runs[:-1], runs[1:]):
        if t1 != t2:
            continue
        gap = s2 - e1
        if gap <= 1 or gap > max_bridge_frames:
            continue
        # Do not bridge across a genuine dead-ball stoppage
        if np.max(dead_score[e1:s2]) >= 1.2:
            continue
        sign = 1.0 if t1 == HOME else -1.0
        bridged[e1 + 1:s2] += sign * 0.9

    return bridged


def compute_team_evidence(atd_feats, st_feats, dead_score):
    """Compute per-frame Home-vs-Away evidence (> 0 = Home, < 0 = Away)."""
    n = len(atd_feats["ball_visible"])
    score = np.zeros(n, dtype=np.float64)

    vis_in = atd_feats["ball_visible"] & (~atd_feats["ball_oob"])
    nd = atd_feats["nearest_dist"]
    margin = atd_feats["dist_margin"]

    # 1. Distance-decayed proximity evidence (Idea 3 upgraded)
    # Decays smoothly when the ball is in the air far from any player.
    prox_weight = np.where(vis_in & (~np.isnan(nd)), np.exp(-nd / 3.0), 0.0)
    margin_clipped = np.where(np.isnan(margin), 0.0, np.clip(margin, -4.0, 4.0))
    score += prox_weight * (margin_clipped * 0.45)

    # Extra confidence when ball is within control radius (< 2 m) with clear margin
    clear_home = vis_in & (nd <= 2.0) & (margin >= 0.8)
    clear_away = vis_in & (nd <= 2.0) & (margin <= -0.8)
    score[clear_home] += 0.6
    score[clear_away] -= 0.6

    # 2. Passing-bridge continuity across same-team control anchors (Idea 2)
    anchors = _find_control_anchors(atd_feats, dead_score)
    score = _apply_passing_bridge(score, anchors, dead_score)

    # 3. Soft Smart Tagging tactical prior (helpful during long ball occlusions)
    score += 0.30 * np.tanh(st_feats["st_bias"] / 1.5)

    # 4. Set-piece restart team prior
    sp_team = st_feats["set_piece_team"]
    live_in_sp = (dead_score < 0.5) & (sp_team >= 0)
    score[live_in_sp & (sp_team == HOME)] += 0.7
    score[live_in_sp & (sp_team == AWAY)] -= 0.7

    return np.clip(score, -2.0, 2.0)


def viterbi_decode_half(emit_scores, switch_team_cost=18.0, switch_dead_cost=20.0):
    """Exact 3-state (H, A, D) Viterbi sequence decoder over one contiguous half.

    Maximises sum_t emit_scores[t, state_t] - transition_cost(state_{t-1}, state_t).
    """
    n = len(emit_scores)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    trans = np.array([
        [0.0, switch_team_cost, switch_dead_cost],
        [switch_team_cost, 0.0, switch_dead_cost],
        [switch_dead_cost, switch_dead_cost, 0.0],
    ], dtype=np.float64)

    backptr = np.zeros((n, 3), dtype=np.int8)
    dp = emit_scores[0].copy()

    for t in range(1, n):
        # scores[prev, curr] = dp[prev] - trans[prev, curr]
        cand = dp[:, None] - trans
        best_prev = np.argmax(cand, axis=0)
        backptr[t] = best_prev
        dp = cand[best_prev, [0, 1, 2]] + emit_scores[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(dp))
    for t in range(n - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def predict_possession(atd_feats, st_feats, game=GAME):
    """Run full possession pipeline and return length-N array of labels ('H','A','D','X')."""
    n_frames = len(atd_feats["ball_visible"])
    dead_score = compute_dead_ball_score(atd_feats, st_feats)
    team_score = compute_team_evidence(atd_feats, st_feats, dead_score)

    # Symmetric 3-state emission scores:
    # (emit[H] + emit[A])/2 - emit[D] = -dead_score, and emit[H] - emit[A] = 2 * team_score.
    emit = np.empty((n_frames, 3), dtype=np.float64)
    emit[:, STATE_H] = -0.5 * dead_score + team_score
    emit[:, STATE_A] = -0.5 * dead_score - team_score
    emit[:, STATE_D] = 0.5 * dead_score

    # Default everything outside graded halves to X (3)
    codes = np.full(n_frames, 3, dtype=np.int8)
    for _, first_f, last_f in half_video_blocks(game):
        s = max(0, first_f)
        e = min(n_frames, last_f + 1)
        codes[s:e] = viterbi_decode_half(emit[s:e])

    # Enforce pre-kickoff graded frames strictly as Dead ('D')
    codes[st_feats["pre_kickoff_dead_mask"]] = STATE_D
    return LABEL_CHARS[codes]
