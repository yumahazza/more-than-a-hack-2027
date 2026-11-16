"""Model v14: Touch-Gated 3-State Physics Viterbi & Post-Decode Cleanup.

Combines the 5-pillar physical emission + touch-gated 3-state Viterbi architecture with
audit-hardened fixes and post-Viterbi micro-fragment cleanup:
1. Block-aware temporal shifting (`shift_blocked`) preventing look-ahead/look-back bleed
   across half boundaries into halftime/pre-match frames.
2. Goalkeeper goal-side dynamic identity correction (`upgrade_atd_features`) resolving
   optical tracker GK slot swaps across goal ends.
3. Co-moving relative velocity (`rel_spd`), pre-reception approach trimming, close-combat
   boot tackles, and post-OOB ball-retrieval memory (`_compute_post_oob_recovery`).
4. Touch-gated asymmetric 3-state Viterbi decoding (`viterbi_decode_half`) calibrated on the
   robust interior plateau `(19.5, 14.5, 26.0, 22.0)`.
5. Post-Viterbi deflection bridging & unanchored tail absorption (`post_clean_v14`).
"""

import argparse
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import DATA_DIR, DEFAULT_SOURCE, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES, load as load_xml  # noqa: E402

HOME, AWAY = 0, 1
NOBODY = -1
STATE_H, STATE_A, STATE_D, STATE_X = 0, 1, 2, 3
LABEL_CHARS = np.array(["H", "A", "D", "X"])

ATTACK_CODE_WEIGHTS = {
    "BUILD UP": 1.0,
    "PROGRESSION": 1.0,
    "ATTACKING TRANSITION": 1.0,
    "CREATING CHANCES": 0.8,
    "BALL IN FINAL THIRD": 0.6,
    "BALL IN THE BOX": 0.6,
    "PLAYERS IN THE BOX": 0.5,
    "LONG BALL": 0.6,
}

DEFEND_CODE_WEIGHTS = {
    "DEFENDING IN ATTACKING THIRD": 0.8,
    "DEFENDING IN MIDDLE THIRD": 0.8,
    "DEFENDING IN DEFENSIVE THIRD": 0.8,
    "DEFENSIVE TRANSITION": 0.8,
}

VITERBI_PARAMS = dict(
    switch_team_base=19.5,
    switch_team_touch=14.5,
    enter_dead_cost=26.0,
    exit_dead_cost=22.0,
)


# ---------------------------------------------------------------------------
# 1. Fast Streaming ATD Loader (with local npz cache)
# ---------------------------------------------------------------------------

def load_team_mapping(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


def _parse_triplet(token):
    if token[0] == "N":
        return np.nan, np.nan, np.nan
    c1 = token.index(",")
    c2 = token.index(",", c1 + 1)
    x = float(token[:c1])
    y = float(token[c1 + 1 : c2])
    spd_str = token[c2 + 1 :]
    spd = np.nan if spd_str[0] == "N" else float(spd_str)
    return x, y, spd


def parse_atd_raw(raw_path, n_frames, n_slots):
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            vframe = int(parts[0]) - 1
            if vframe < 0 or vframe >= n_frames:
                continue
            bx_val, by_val, _ = _parse_triplet(parts[2])
            bx[vframe] = bx_val
            by[vframe] = by_val
            for slot, token in enumerate(parts[1].split(";")):
                if slot >= n_slots or token[0] == "N":
                    continue
                x, y, spd = _parse_triplet(token)
                px[vframe, slot] = x
                py[vframe, slot] = y
                ps[vframe, slot] = spd
    return px, py, ps, bx, by


def load_atd_arrays(game=GAME, data_dir=None, source=DEFAULT_SOURCE, use_cache=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    n_frames = source_frames(source, game)
    team_of_slot = load_team_mapping(meta_path)
    n_slots = len(team_of_slot)

    cache_root = data_dir or DATA_DIR
    cache_dir = os.path.join(cache_root, "_cache")
    cache_path = os.path.join(cache_dir, f"atd_{game}_{source}.npz")

    if use_cache and os.path.isfile(cache_path):
        cached = np.load(cache_path)
        return {
            "player_x": cached["player_x"],
            "player_y": cached["player_y"],
            "player_s": cached["player_s"],
            "ball_x": cached["ball_x"],
            "ball_y": cached["ball_y"],
            "team_of_slot": cached["team_of_slot"],
            "pitch": tuple(GAMES[game]["pitch"]),
            "fps": GAMES[game]["fps"],
            "n_frames": n_frames,
        }

    px, py, ps, bx, by = parse_atd_raw(raw_path, n_frames, n_slots)
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(
            cache_path,
            player_x=px,
            player_y=py,
            player_s=ps,
            ball_x=bx,
            ball_y=by,
            team_of_slot=team_of_slot,
        )
    return {
        "player_x": px,
        "player_y": py,
        "player_s": ps,
        "ball_x": bx,
        "ball_y": by,
        "team_of_slot": team_of_slot,
        "pitch": tuple(GAMES[game]["pitch"]),
        "fps": GAMES[game]["fps"],
        "n_frames": n_frames,
    }


# ---------------------------------------------------------------------------
# 2. Smart Tagging & ATD Kinematic Feature Extraction
# ---------------------------------------------------------------------------

def parse_team_name(team_text):
    if not team_text:
        return NOBODY
    norm = team_text.strip().lower()
    if norm in ("n/a", "none", ""):
        return NOBODY
    if "barca" in norm or "barcel" in norm:
        return HOME
    if "atlet" in norm or "athlet" in norm:
        return AWAY
    return NOBODY


def extract_smart_tagging_features(n_frames, game=GAME, data_dir=None):
    fps = GAMES[game]["fps"]
    events = load_xml(game=game, data_dir=data_dir, fps=fps)

    home_prior = np.zeros(n_frames, dtype=np.float64)
    away_prior = np.zeros(n_frames, dtype=np.float64)
    in_set_piece = np.zeros(n_frames, dtype=bool)
    set_piece_team = np.full(n_frames, NOBODY, dtype=np.int8)

    set_pieces, goals = [], []
    for ev in events:
        code = (ev["code"] or "").strip().upper()
        s = max(0, int(ev["start_frame"]))
        e = min(n_frames, int(ev["end_frame"]))
        if e <= s:
            continue
        team_code = parse_team_name(ev["tags"].get("Team"))
        if code in ATTACK_CODE_WEIGHTS and team_code != NOBODY:
            w = ATTACK_CODE_WEIGHTS[code]
            if team_code == HOME:
                home_prior[s:e] += w
            else:
                away_prior[s:e] += w
        elif code in DEFEND_CODE_WEIGHTS and team_code != NOBODY:
            w = DEFEND_CODE_WEIGHTS[code]
            if team_code == HOME:
                away_prior[s:e] += w
            else:
                home_prior[s:e] += w
        elif code == "GOALS":
            goals.append({"start_frame": s, "end_frame": e, "team_code": team_code})
        elif code == "SET PIECES":
            set_pieces.append({
                "start_frame": s,
                "end_frame": e,
                "type": (ev["tags"].get("Type") or "").strip(),
                "team_code": team_code,
                "half": ev["tags"].get("Half"),
            })

    for sp in set_pieces:
        if sp["type"].lower() == "kick off (after goal)" and sp["team_code"] == NOBODY:
            preceding = [g for g in goals if g["start_frame"] <= sp["start_frame"] <= g["end_frame"] + 50]
            if preceding and preceding[-1]["team_code"] != NOBODY:
                scorer = preceding[-1]["team_code"]
                sp["team_code"] = AWAY if scorer == HOME else HOME
        s, e = sp["start_frame"], sp["end_frame"]
        in_set_piece[s:e] = True
        if sp["team_code"] != NOBODY:
            set_piece_team[s:e] = sp["team_code"]

    active_half_mask = np.zeros(n_frames, dtype=bool)
    pre_kickoff_dead_mask = np.zeros(n_frames, dtype=bool)
    half_names = {1: "1st Half", 2: "2nd Half"}
    for half_num, first_f, last_f in half_video_blocks(game):
        first_f = max(0, first_f)
        last_f = min(n_frames - 1, last_f)
        active_half_mask[first_f : last_f + 1] = True
        ko_frame = KICKOFF_VIDEO_FRAMES.get(half_names.get(half_num))
        if ko_frame is not None and ko_frame > first_f:
            pre_kickoff_dead_mask[first_f : min(ko_frame, last_f + 1)] = True

    return {
        "home_prior": home_prior,
        "away_prior": away_prior,
        "st_bias": home_prior - away_prior,
        "in_set_piece": in_set_piece,
        "set_piece_team": set_piece_team,
        "set_pieces": set_pieces,
        "active_half_mask": active_half_mask,
        "pre_kickoff_dead_mask": pre_kickoff_dead_mask,
    }


def live_half_blocks(st_feats, n_frames, game=GAME):
    pre_ko = st_feats["pre_kickoff_dead_mask"]
    blocks = []
    for _, first_f, last_f in half_video_blocks(game):
        s = max(0, int(first_f))
        e = min(n_frames, int(last_f) + 1)
        if e <= s:
            continue
        live_mask = ~pre_ko[s:e]
        if not np.any(live_mask):
            continue
        first_live = s + int(np.argmax(live_mask))
        blocks.append((first_live, e))
    return blocks


def nan_box_smooth(values, half_win):
    if half_win <= 0:
        return values.astype(np.float64, copy=True)
    valid = ~np.isnan(values)
    filled = np.where(valid, values.astype(np.float64), 0.0)
    c_sum = np.concatenate(([0.0], np.cumsum(filled)))
    c_cnt = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    n = len(values)
    idx = np.arange(n)
    lo = np.maximum(0, idx - half_win)
    hi = np.minimum(n, idx + half_win + 1)
    cnt = c_cnt[hi] - c_cnt[lo]
    out = np.full(n, np.nan, dtype=np.float64)
    has = cnt > 0
    out[has] = (c_sum[hi[has]] - c_sum[lo[has]]) / cnt[has]
    return out


def nan_box_smooth_blocked(values, half_win, blocks=None):
    if blocks is None:
        return nan_box_smooth(values, half_win)
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    for s, e in blocks:
        s_c, e_c = max(0, int(s)), min(n, int(e))
        if e_c > s_c:
            out[s_c:e_c] = nan_box_smooth(values[s_c:e_c], half_win)
    return out


def shift_blocked(values, k, blocks):
    """Block-aware shift preventing wrap-around across half boundaries."""
    out = values.astype(np.float64, copy=True)
    for s, e in blocks:
        seg = values[s:e].astype(np.float64)
        m = len(seg)
        if m == 0:
            continue
        kk = max(-m, min(m, int(k)))
        if kk > 0:
            out[s:e] = np.concatenate((np.full(kk, seg[0]), seg[: m - kk]))
        elif kk < 0:
            out[s:e] = np.concatenate((seg[-kk:], np.full(-kk, seg[-1])))
    return out


def extract_atd_features(atd, active_blocks=None):
    bx, by = atd["ball_x"], atd["ball_y"]
    px, py, ps = atd["player_x"], atd["player_y"], atd["player_s"]
    team_of_slot = atd["team_of_slot"]
    length, width = atd["pitch"]
    fps = atd["fps"]
    n = len(bx)

    ball_vis = (~np.isnan(bx)) & (~np.isnan(by))
    ball_oob = np.zeros(n, dtype=bool)
    ball_near_edge = np.zeros(n, dtype=bool)
    ball_oob[ball_vis] = (
        (bx[ball_vis] < -0.005) | (bx[ball_vis] > 1.005)
        | (by[ball_vis] < -0.005) | (by[ball_vis] > 1.005)
    )
    ball_near_edge[ball_vis] = (
        (bx[ball_vis] < 0.02) | (bx[ball_vis] > 0.98)
        | (by[ball_vis] < 0.02) | (by[ball_vis] > 0.98)
    )

    ball_speed = np.full(n, np.nan, dtype=np.float64)
    vis_idx = np.flatnonzero(ball_vis)
    if len(vis_idx) >= 2:
        dt = np.diff(vis_idx)
        ok = (dt >= 1) & (dt <= 5)
        dx_m = (bx[vis_idx[1:]].astype(np.float64) - bx[vis_idx[:-1]].astype(np.float64)) * length
        dy_m = (by[vis_idx[1:]].astype(np.float64) - by[vis_idx[:-1]].astype(np.float64)) * width
        spd = np.clip(np.hypot(dx_m, dy_m) * (fps / dt.astype(np.float64)), 0.0, 45.0)
        ball_speed[vis_idx[1:][ok]] = spd[ok]

    ball_speed_smooth = nan_box_smooth_blocked(ball_speed, half_win=2, blocks=active_blocks)
    ball_speed_smooth[~ball_vis] = np.nan

    vx = np.full(n, np.nan, dtype=np.float64)
    vy = np.full(n, np.nan, dtype=np.float64)
    if len(vis_idx) >= 2:
        ok_a = (dt >= 1) & (dt <= 3)
        bx_m = bx.astype(np.float64) * length
        by_m = by.astype(np.float64) * width
        vx[vis_idx[1:][ok_a]] = (bx_m[vis_idx[1:][ok_a]] - bx_m[vis_idx[:-1][ok_a]]) * (fps / dt[ok_a])
        vy[vis_idx[1:][ok_a]] = (by_m[vis_idx[1:][ok_a]] - by_m[vis_idx[:-1][ok_a]]) * (fps / dt[ok_a])
    ax = np.diff(vx, prepend=np.nan) * fps
    ay = np.diff(vy, prepend=np.nan) * fps
    ball_accel = nan_box_smooth_blocked(np.hypot(ax, ay), half_win=1, blocks=active_blocks)
    ball_accel[~ball_vis] = np.nan

    gap_len = np.zeros(n, dtype=np.int32)
    gap_near_edge = np.zeros(n, dtype=bool)
    gap_oob = np.zeros(n, dtype=bool)
    gap_pre_speed = np.full(n, np.nan, dtype=np.float64)

    scan_blocks = active_blocks if active_blocks is not None else [(0, n)]
    for b_s, b_e in scan_blocks:
        b_s, b_e = max(0, int(b_s)), min(n, int(b_e))
        if b_e <= b_s:
            continue
        missing = (~ball_vis[b_s:b_e]).astype(np.int8)
        edges = np.diff(np.concatenate(([0], missing, [0])))
        starts = np.flatnonzero(edges == 1) + b_s
        stops = np.flatnonzero(edges == -1) + b_s
        for s, e in zip(starts, stops):
            gap_len[s:e] = e - s
            gap_near_edge[s:e] = (bool(ball_near_edge[s - 1]) if s > b_s else False) or (
                bool(ball_near_edge[e]) if e < b_e else False
            )
            gap_oob[s:e] = (bool(ball_oob[s - 1]) if s > b_s else False) or (
                bool(ball_oob[e]) if e < b_e else False
            )
            if s > b_s:
                gap_pre_speed[s:e] = ball_speed_smooth[s - 1]

    valid_ps = ~np.isnan(ps)
    n_spd = np.sum(valid_ps, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        n_players = np.sum(~np.isnan(px), axis=1).astype(np.int16)
        mean_speed = np.nanmean(ps, axis=1).astype(np.float64)
        p80_speed = np.nanpercentile(ps, 80, axis=1).astype(np.float64)
        p90_speed = np.nanpercentile(ps, 90, axis=1).astype(np.float64)

    walk_fraction = np.sum((ps < 1.5) & valid_ps, axis=1) / np.maximum(1, n_spd)
    too_few = n_players < 6
    too_few_spd = too_few | (n_spd < 6)
    mean_speed[too_few] = np.nan
    p80_speed[too_few] = np.nan
    p90_speed[too_few_spd] = np.nan
    walk_fraction[too_few_spd] = np.nan

    home_mask = team_of_slot == HOME
    away_mask = team_of_slot == AWAY
    min_dist_home = np.full(n, np.nan, dtype=np.float64)
    min_dist_away = np.full(n, np.nan, dtype=np.float64)
    nearest_player_speed = np.full(n, np.nan, dtype=np.float64)

    if np.any(ball_vis):
        dx = (px[ball_vis] - bx[ball_vis, None]) * length
        dy = (py[ball_vis] - by[ball_vis, None]) * width
        dist = np.hypot(dx, dy).astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            d_home = np.nanmin(dist[:, home_mask], axis=1)
            d_away = np.nanmin(dist[:, away_mask], axis=1)
            dist_safe = np.where(np.isnan(dist), np.inf, dist)
            nearest_slot = np.argmin(dist_safe, axis=1)
            vis_rows = np.flatnonzero(ball_vis)
            n_spd_near = ps[vis_rows, nearest_slot].astype(np.float64)
        min_dist_home[ball_vis] = d_home
        min_dist_away[ball_vis] = d_away
        nearest_player_speed[ball_vis] = n_spd_near

    nearest_dist = np.fmin(min_dist_home, min_dist_away)
    dist_margin = min_dist_away - min_dist_home

    return {
        "ball_x": bx,
        "ball_y": by,
        "ball_visible": ball_vis,
        "ball_oob": ball_oob,
        "ball_near_edge": ball_near_edge,
        "ball_speed": ball_speed_smooth,
        "ball_accel": ball_accel,
        "gap_len": gap_len,
        "gap_near_edge": gap_near_edge,
        "gap_oob": gap_oob,
        "gap_pre_speed": gap_pre_speed,
        "n_players": n_players,
        "mean_player_speed": mean_speed,
        "p80_player_speed": p80_speed,
        "p90_player_speed": p90_speed,
        "walk_fraction": walk_fraction,
        "mean_player_speed_smooth": nan_box_smooth_blocked(mean_speed, half_win=12, blocks=active_blocks),
        "p80_player_speed_smooth": nan_box_smooth_blocked(p80_speed, half_win=12, blocks=active_blocks),
        "p90_player_speed_smooth": nan_box_smooth_blocked(p90_speed, half_win=12, blocks=active_blocks),
        "walk_fraction_smooth": nan_box_smooth_blocked(walk_fraction, half_win=12, blocks=active_blocks),
        "min_dist_home": min_dist_home,
        "min_dist_away": min_dist_away,
        "nearest_dist": nearest_dist,
        "dist_margin": dist_margin,
        "nearest_player_speed": nearest_player_speed,
    }


# ---------------------------------------------------------------------------
# 3. Physics Emission Scoring & Touch-Gated Viterbi Decoder
# ---------------------------------------------------------------------------

def _compute_post_oob_recovery(atd_feats, live_blocks, max_recovery_frames=185):
    n = len(atd_feats["ball_visible"])
    recovery = np.zeros(n, dtype=bool)
    vis = atd_feats["ball_visible"]
    oob = atd_feats["ball_oob"]
    gap_oob = atd_feats["gap_oob"]
    bx, by = atd_feats["ball_x"], atd_feats["ball_y"]
    bspd = atd_feats["ball_speed"]

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
            if frames_since_oob > max_recovery_frames or in_interior[t]:
                active = False
                continue
            if fast_restart[t]:
                fast_run += 1
                if fast_run >= 3:
                    active = False
                    continue
            else:
                fast_run = 0
            if (not vis[t]) or (~in_interior[t]):
                recovery[t] = True
    return recovery


def upgrade_atd_features(atd, atd_feats, live_blocks):
    px, py = atd["player_x"], atd["player_y"]
    bx, by = atd["ball_x"], atd["ball_y"]
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
    d_home = dist_h_mat[np.arange(len(vis_rows)), slot_h]
    d_away = dist_a_mat[np.arange(len(vis_rows)), slot_a]
    d_home[np.isinf(d_home)] = np.nan
    d_away[np.isinf(d_away)] = np.nan

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
    out["min_dist_home"][vis_rows] = d_home
    out["min_dist_away"][vis_rows] = d_away
    out["rel_spd_home"][vis_rows] = rel_spd_h
    out["rel_spd_away"][vis_rows] = rel_spd_a
    out["nearest_dist"] = np.fmin(out["min_dist_home"], out["min_dist_away"])
    out["dist_margin"] = out["min_dist_away"] - out["min_dist_home"]
    return out


def compute_dead_ball_score(atd_feats, st_feats, live_blocks):
    n_f = len(atd_feats["ball_visible"])
    score = np.zeros(n_f, dtype=np.float64)

    vis = atd_feats["ball_visible"]
    oob = atd_feats["ball_oob"]
    vis_in = vis & (~oob)
    bx, by = atd_feats["ball_x"], atd_feats["ball_y"]
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

    speed_live = np.clip((m_spd - 1.45) * 2.2, -2.0, 2.0)
    p80_live = np.clip((p80 - 1.85) * 1.3, -1.6, 1.8)
    p90_live = np.clip((p90 - 2.20) * 0.9, -1.0, 1.2)
    walk_dead = np.clip((walk - 0.65) * 3.5, -1.2, 1.5)

    p80_prev25 = shift_blocked(p80, 25, live_blocks)
    p80_fut40 = shift_blocked(p80, -40, live_blocks)
    walk_fut40 = shift_blocked(walk, -40, live_blocks)

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

    players_engaged = (p80 >= 1.80) | (p90 >= 2.25) | (walk < 0.65)
    active_moving_in = vis_in & (b_spd_safe >= 1.8) & players_engaged
    fast_kick_in = vis_in & (~post_oob) & (~in_sp) & (b_spd_safe >= 6.0) & (~players_engaged) & (walk < 0.73)
    still_wait = vis_in & (b_spd_safe < 0.6) & (p80 < 1.80)

    score[active_moving_in] -= 1.8
    score[fast_kick_in] -= 2.4
    score[still_wait] += 2.0

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
            if (ge - gs) < 100:
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

    nd_fut15 = shift_blocked(nd_safe, -15, live_blocks)
    p80_fut45 = shift_blocked(p80, -45, live_blocks)
    walk_fut45 = shift_blocked(walk, -45, live_blocks)
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


def _find_control_anchors(atd_feats, dead_score, radius_m=2.1, margin_m=0.75, min_run=5):
    vis_in = atd_feats["ball_visible"] & (~atd_feats["ball_oob"]) & (dead_score < 1.0)
    nd = atd_feats["nearest_dist"]
    margin = atd_feats["dist_margin"]
    bspd = np.where(np.isnan(atd_feats["ball_speed"]), 0.0, atd_feats["ball_speed"])
    accel = np.where(np.isnan(atd_feats["ball_accel"]), 0.0, atd_feats["ball_accel"])
    rel_h = np.where(np.isnan(atd_feats["rel_spd_home"]), 99.0, atd_feats["rel_spd_home"])
    rel_a = np.where(np.isnan(atd_feats["rel_spd_away"]), 99.0, atd_feats["rel_spd_away"])

    controllable_h = (bspd <= 7.8) | ((bspd <= 9.8) & (rel_h <= 3.2)) | ((accel >= 12.0) & (nd <= 1.9))
    controllable_a = (bspd <= 7.8) | ((bspd <= 9.8) & (rel_a <= 3.2)) | ((accel >= 12.0) & (nd <= 1.9))

    raw = np.full(len(vis_in), NOBODY, dtype=np.int8)
    raw[vis_in & controllable_h & (nd <= radius_m) & (margin >= margin_m)] = HOME
    raw[vis_in & controllable_a & (nd <= radius_m) & (margin <= -margin_m)] = AWAY

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


def compute_team_evidence(
    atd_feats, st_feats, dead_score, live_blocks, max_bridge_frames=110, hold_tau_frames=85.0
):
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

    anchors = _find_control_anchors(atd_feats, dead_score)

    idx = np.flatnonzero(anchors >= 0)
    runs = []
    if len(idx) >= 1:
        start = prev = idx[0]
        prev_team = int(anchors[idx[0]])
        for k in idx[1:]:
            t = int(anchors[k])
            if k == prev + 1 and t == prev_team:
                prev = k
            else:
                runs.append((start, prev, prev_team, prev - start + 1))
                start = prev = k
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
        start = prev = idx[0]
        prev_team = int(filtered_anchors[idx[0]])
        for k in idx[1:]:
            t = int(filtered_anchors[k])
            if k == prev + 1 and t == prev_team:
                prev = k
            else:
                runs2.append((start, prev, prev_team))
                start = prev = k
                prev_team = t
        runs2.append((start, prev, prev_team))

    for (_, e1, t1), (s2, _, t2) in zip(runs2[:-1], runs2[1:]):
        if t1 != t2:
            continue
        gap = s2 - e1
        if gap <= 1 or gap > max_bridge_frames or np.max(dead_score[e1:s2]) >= 1.2:
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


def viterbi_decode_half(
    emit_scores,
    touch_gate,
    switch_team_base=19.5,
    switch_team_touch=14.5,
    enter_dead_cost=26.0,
    exit_dead_cost=22.0,
):
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
            if (de_f - ds_f) > max_heal_len:
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


def post_clean_v14(
    codes, live_blocks, fa=None, bridge_same=12, absorb_dead_island=30, absorb_tail_to_dead=45
):
    """Post-Viterbi micro-fragment & unanchored-tail cleanup."""
    out = codes.copy()
    for s, e in live_blocks:
        arr = out[s:e]
        if bridge_same > 0:
            i, m = 0, len(arr)
            while i < m:
                j = i
                while j < m and arr[j] == arr[i]:
                    j += 1
                if arr[i] in (STATE_H, STATE_A) and (j - i) < bridge_same:
                    left = arr[i - 1] if i > 0 else -1
                    right = arr[j] if j < m else -1
                    if left == right and left in (STATE_H, STATE_A):
                        arr[i:j] = left
                i = j
        if absorb_dead_island > 0:
            i, m = 0, len(arr)
            while i < m:
                j = i
                while j < m and arr[j] == arr[i]:
                    j += 1
                if arr[i] in (STATE_H, STATE_A) and (j - i) < absorb_dead_island:
                    left = arr[i - 1] if i > 0 else STATE_D
                    right = arr[j] if j < m else STATE_D
                    if left == STATE_D and right == STATE_D:
                        arr[i:j] = STATE_D
                i = j
        if absorb_tail_to_dead > 0 and fa is not None:
            i, m = 0, len(arr)
            while i < m:
                j = i
                while j < m and arr[j] == arr[i]:
                    j += 1
                if arr[i] in (STATE_H, STATE_A) and (j - i) < absorb_tail_to_dead:
                    left = arr[i - 1] if i > 0 else -1
                    right = arr[j] if j < m else -1
                    anch_cnt = int(np.sum(fa[s + i : s + j] == arr[i]))
                    if right == STATE_D and left in (STATE_H, STATE_A) and left != arr[i] and anch_cnt <= 3:
                        arr[i:j] = left
                i = j
        out[s:e] = arr
    return out


def predict_possession_v14(atd_feats, st_feats, atd_raw=None, viterbi_params=None):
    vp = dict(VITERBI_PARAMS)
    if viterbi_params:
        vp.update(viterbi_params)

    n = len(atd_feats["ball_visible"])
    blocks = live_half_blocks(st_feats, n)

    if atd_raw is not None:
        atd_feats = upgrade_atd_features(atd_raw, atd_feats, blocks)

    dead_s, post_oob = compute_dead_ball_score(atd_feats, st_feats, blocks)
    team_s, fa, int_mask = compute_team_evidence(atd_feats, st_feats, dead_s, blocks)

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
        | (fa >= 0)
        | int_mask
    )
    touch_gate_s = nan_box_smooth_blocked(touch_gate.astype(np.float64), half_win=2, blocks=blocks) > 0.0

    codes = np.full(n, STATE_X, dtype=np.int8)
    for s, e in blocks:
        codes[s:e] = viterbi_decode_half(emit[s:e], touch_gate_s[s:e], **vp)

    codes = heal_short_ungrounded_dead_spells(codes, atd_feats, st_feats, post_oob, blocks, max_heal_len=70)
    codes = post_clean_v14(codes, blocks, fa=fa)
    codes[st_feats["pre_kickoff_dead_mask"]] = STATE_D

    labels = LABEL_CHARS[codes]
    return labels, codes, dead_s, team_s


def main():
    parser = argparse.ArgumentParser(description="Run v14 Touch-Gated Physics Viterbi model.")
    parser.add_argument("--game", default=GAME, help="Game identifier (default: BAR-ATH)")
    parser.add_argument("--data-dir", default=None, help="Optional data directory override")
    parser.add_argument("--source", default="tactical", help="Optional data source override")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "evaluation", "results", "v14_physics_viterbi.csv"),
        help="Output submission CSV path",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"[1/4] Loading ATD tracking data for {args.game} ...")
    atd = load_atd_arrays(game=args.game, data_dir=args.data_dir, source=args.source)
    n_frames = len(atd["ball_x"])
    print(f"      Loaded {n_frames:,} frames in {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"[2/4] Extracting Smart Tagging features for {args.game} ...")
    st_feats = extract_smart_tagging_features(n_frames=n_frames, game=args.game, data_dir=args.data_dir)
    active_blocks = live_half_blocks(st_feats, n_frames, game=args.game)
    print(f"      Done in {time.time() - t1:.1f}s")

    t2 = time.time()
    print("[3/4] Extracting physics & geometry features from ATD ...")
    atd_feats = extract_atd_features(atd, active_blocks=active_blocks)
    print(f"      Done in {time.time() - t2:.1f}s")

    t3 = time.time()
    print("[4/4] Running v14 Touch-Gated 3-State Viterbi + Post-Decode Cleanup ...")
    labels, _, _, _ = predict_possession_v14(atd_feats, st_feats, atd_raw=atd)
    print(f"      Decoded in {time.time() - t3:.1f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df = pd.DataFrame({"frame": np.arange(n_frames, dtype=np.int64), "label": labels})
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(out_df):,} predictions -> {args.output} (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

