"""Physics and geometry feature extraction from Automatic Tracking Data (ATD).

All distances are converted from normalised [0, 1] pitch coordinates into
metres using the true pitch dimensions before comparing distances across axes.
"""

import warnings

import numpy as np

from src.atd_loader import AWAY, HOME


def nan_box_smooth(values, half_win):
    """O(N) centred moving average ignoring NaNs; returns NaN if window has no valid values."""
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


def _compute_ball_kinematics(bx, by, pitch, fps, max_gap_frames=5):
    """Compute ball visibility, boundary flags, smoothed ball speed (m/s), and gap context."""
    length, width = pitch
    n = len(bx)
    ball_vis = (~np.isnan(bx)) & (~np.isnan(by))

    # Boundary flags on visible ball frames
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

    # Raw ATD ball speed is NaN on ~91% of visible frames; derive velocity from coordinates
    ball_speed = np.full(n, np.nan, dtype=np.float64)
    vis_idx = np.flatnonzero(ball_vis)
    if len(vis_idx) >= 2:
        dt = np.diff(vis_idx)
        ok = (dt >= 1) & (dt <= max_gap_frames)
        dx_m = (bx[vis_idx[1:]].astype(np.float64) - bx[vis_idx[:-1]].astype(np.float64)) * length
        dy_m = (by[vis_idx[1:]].astype(np.float64) - by[vis_idx[:-1]].astype(np.float64)) * width
        spd = np.hypot(dx_m, dy_m) * (fps / dt.astype(np.float64))
        spd = np.clip(spd, 0.0, 45.0)
        ball_speed[vis_idx[1:][ok]] = spd[ok]

    ball_speed_smooth = nan_box_smooth(ball_speed, half_win=2)
    # Only keep smoothed speed where ball is actually visible (don't bleed into large gaps)
    ball_speed_smooth[~ball_vis] = np.nan

    # Missing-ball gap length & boundary context (last seen before gap / first seen after gap)
    gap_len = np.zeros(n, dtype=np.int32)
    gap_near_edge = np.zeros(n, dtype=bool)
    gap_oob = np.zeros(n, dtype=bool)
    gap_pre_speed = np.full(n, np.nan, dtype=np.float64)

    missing = (~ball_vis).astype(np.int8)
    edges = np.diff(np.concatenate(([0], missing, [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for s, e in zip(starts, stops):
        length_g = e - s
        gap_len[s:e] = length_g
        pre_edge = bool(ball_near_edge[s - 1]) if s > 0 else False
        post_edge = bool(ball_near_edge[e]) if e < n else False
        gap_near_edge[s:e] = pre_edge or post_edge

        pre_oob = bool(ball_oob[s - 1]) if s > 0 else False
        post_oob = bool(ball_oob[e]) if e < n else False
        gap_oob[s:e] = pre_oob or post_oob

        if s > 0:
            gap_pre_speed[s:e] = ball_speed_smooth[s - 1]

    return {
        "ball_visible": ball_vis,
        "ball_oob": ball_oob,
        "ball_near_edge": ball_near_edge,
        "ball_speed": ball_speed_smooth,
        "gap_len": gap_len,
        "gap_near_edge": gap_near_edge,
        "gap_oob": gap_oob,
        "gap_pre_speed": gap_pre_speed,
    }


def _compute_player_kinematics(px, ps):
    """Compute collective player speeds (works on 100% of frames even when ball is NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        n_players = np.sum(~np.isnan(px), axis=1).astype(np.int16)
        mean_speed = np.nanmean(ps, axis=1).astype(np.float64)
        p80_speed = np.nanpercentile(ps, 80, axis=1).astype(np.float64)

    # Require at least 6 tracked players to trust collective speed
    too_few = n_players < 6
    mean_speed[too_few] = np.nan
    p80_speed[too_few] = np.nan

    return {
        "n_players": n_players,
        "mean_player_speed": mean_speed,
        "p80_player_speed": p80_speed,
        "mean_player_speed_smooth": nan_box_smooth(mean_speed, half_win=12),
        "p80_player_speed_smooth": nan_box_smooth(p80_speed, half_win=12),
    }


def _compute_ball_proximity(px, py, ps, bx, by, team_of_slot, pitch):
    """Compute ball-to-nearest-player distances (in metres) for Home and Away teams."""
    length, width = pitch
    n = len(bx)
    ball_vis = (~np.isnan(bx)) & (~np.isnan(by))

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
            # Track speed of overall nearest player
            dist_safe = np.where(np.isnan(dist), np.inf, dist)
            nearest_slot = np.argmin(dist_safe, axis=1)
            vis_rows = np.flatnonzero(ball_vis)
            n_spd = ps[vis_rows, nearest_slot].astype(np.float64)

        min_dist_home[ball_vis] = d_home
        min_dist_away[ball_vis] = d_away
        nearest_player_speed[ball_vis] = n_spd

    nearest_dist = np.fmin(min_dist_home, min_dist_away)
    dist_margin = min_dist_away - min_dist_home  # > 0 means Home is closer

    return {
        "min_dist_home": min_dist_home,
        "min_dist_away": min_dist_away,
        "nearest_dist": nearest_dist,
        "dist_margin": dist_margin,
        "nearest_player_speed": nearest_player_speed,
    }


def extract_atd_features(atd):
    """Build frame-aligned feature dictionary from loaded ATD arrays."""
    feats = {}
    feats.update(
        _compute_ball_kinematics(
            atd["ball_x"], atd["ball_y"], atd["pitch"], atd["fps"]
        )
    )
    feats.update(_compute_player_kinematics(atd["player_x"], atd["player_s"]))
    feats.update(
        _compute_ball_proximity(
            atd["player_x"],
            atd["player_y"],
            atd["player_s"],
            atd["ball_x"],
            atd["ball_y"],
            atd["team_of_slot"],
            atd["pitch"],
        )
    )
    return feats
