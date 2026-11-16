"""Model v15: Physics Viterbi with Savitzky-Golay Kick Calibration & Team-Shape Prior.

Combines the v14 Touch-Gated 3-State Physics Viterbi architecture with the literature-grounded
modules from `feat/v1-possession-pipeline` (`solution/features.py` & `solution/shape.py`):
1. Goalkeeper goal-side dynamic identity correction (`upgrade_atd_features`), co-moving sprint
   dribble relative velocity (`rel_spd`), pre-reception approach trimming, and post-OOB
   ball-retrieval memory (`_compute_post_oob_recovery`).
2. Unsupervised per-provider match calibration (`calibrate_match_provider`, Vidal-Codina et al. 2022)
   and Savitzky-Golay 2nd-order local acceleration peak kick detection (`detect_sg_kicks`,
   Link & Hoernig 2017) outside contested duels.
3. Self-supervised 1-D logistic regression on team Stretch Index difference (`stretch_index`,
   Clemente et al. 2012) fitted on high-confidence pseudo-labels and normalized by its empirical
   autocorrelation time `tau` (`correlation_frames`) to guide missing-ball live intervals.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME  # noqa: E402
import models.v14_physics_viterbi as m14  # noqa: E402

HOME, AWAY = m14.HOME, m14.AWAY
NOBODY = m14.NOBODY
STATE_H, STATE_A, STATE_D, STATE_X = m14.STATE_H, m14.STATE_A, m14.STATE_D, m14.STATE_X
LABEL_CHARS = m14.LABEL_CHARS

SG_WINDOW = 7
CAL_SLOW_BALL = 2.0
CAL_ZONE_PCT = 75
CAL_KICK_PCT = 90
R_POSSESSION_FLOOR = 1.0
KICK_ACCEL_FLOOR = 4.0


# ---------------------------------------------------------------------------
# 1. Savitzky-Golay Ball Acceleration & Unsupervised Provider Calibration
# ---------------------------------------------------------------------------

def savgol_coeffs(window, deriv):
    """Savitzky-Golay polynomial order-2 convolution weights for value (0), 1st (1), or 2nd (2) derivative."""
    half = window // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(t, 3, increasing=True)
    pinv = np.linalg.pinv(design)
    factor = (1.0, 1.0, 2.0)[deriv]
    return factor * pinv[deriv]


def savgol(x, window, deriv):
    """Centred Savitzky-Golay filter; NaN wherever the window touches a missing frame."""
    w = savgol_coeffs(window, deriv)[::-1]
    out = np.convolve(np.nan_to_num(x), w, mode="same")
    bad = np.convolve(np.isnan(x).astype(np.float64), np.ones(window), mode="same") > 0
    out[bad] = np.nan
    half = window // 2
    out[:half] = np.nan
    out[-half:] = np.nan
    return out


def compute_sg_kicks_and_calibration(atd, atd_feats, st_feats):
    """Calibrate possession/duel radius and detect local Savitzky-Golay acceleration peak kicks."""
    length, width = atd["pitch"]
    fps = float(atd["fps"])
    bx = atd["ball_x"].astype(np.float64) * length
    by = atd["ball_y"].astype(np.float64) * width
    px = atd["player_x"].astype(np.float32) * length
    py = atd["player_y"].astype(np.float32) * width

    vis = atd_feats["ball_visible"] & (~atd_feats["ball_oob"])
    graded = st_feats["active_half_mask"]
    nd = atd_feats["nearest_dist"]
    dh = atd_feats["min_dist_home"]
    da = atd_feats["min_dist_away"]
    bspd = np.where(np.isnan(atd_feats["ball_speed"]), 99.0, atd_feats["ball_speed"])
    m_spd = np.where(np.isnan(atd_feats["mean_player_speed_smooth"]), 0.0, atd_feats["mean_player_speed_smooth"])

    # Second-nearest player distance caps the calibrated possession zone
    second_d = np.full(len(bx), np.nan, dtype=np.float64)
    vis_rows = np.flatnonzero(vis)
    if len(vis_rows) > 0:
        dx = px[vis_rows] - bx[vis_rows, None].astype(np.float32)
        dy = py[vis_rows] - by[vis_rows, None].astype(np.float32)
        dist = np.hypot(dx, dy)
        part = np.partition(np.where(np.isfinite(dist), dist, np.inf), 1, axis=1)[:, 1]
        part[~np.isfinite(part)] = np.nan
        second_d[vis_rows] = part

    accel_sg = np.hypot(savgol(bx, SG_WINDOW, 2), savgol(by, SG_WINDOW, 2)) * (fps * fps)

    slow_live = vis & graded & np.isfinite(nd) & (bspd < CAL_SLOW_BALL) & (m_spd >= 1.4)
    if int(slow_live.sum()) >= 100:
        r_raw = float(np.percentile(nd[slow_live], CAL_ZONE_PCT))
        sec = second_d[slow_live]
        cap = float(np.median(sec[np.isfinite(sec)])) if np.isfinite(sec).any() else r_raw
    else:
        r_raw = cap = R_POSSESSION_FLOOR
    r_zone = float(max(min(r_raw, cap), R_POSSESSION_FLOOR))

    a_valid = accel_sg[vis & graded & np.isfinite(accel_sg)]
    kick_thresh = float(max(KICK_ACCEL_FLOOR, np.percentile(a_valid, CAL_KICK_PCT))) if a_valid.size >= 100 else KICK_ACCEL_FLOOR

    duel = (np.nan_to_num(dh, nan=np.inf) <= r_zone) & (np.nan_to_num(da, nan=np.inf) <= r_zone)
    a_clean = np.nan_to_num(accel_sg, nan=0.0)
    peak = np.zeros(len(a_clean), dtype=bool)
    peak[1:-1] = (a_clean[1:-1] >= a_clean[:-2]) & (a_clean[1:-1] > a_clean[2:])
    kick_mask = vis & peak & (a_clean >= kick_thresh) & np.isfinite(nd) & (nd <= r_zone) & (~duel)
    kick_sign = np.where(kick_mask, np.where(dh <= da, 1.0, -1.0), 0.0)

    return kick_sign, kick_mask, {"r_zone": r_zone, "kick_thresh": kick_thresh}


# ---------------------------------------------------------------------------
# 2. Stretch-Index Team Shape & Autocorrelation-Normalized Logistic Model
# ---------------------------------------------------------------------------

def compute_stretch_diff(atd, live_blocks):
    """Mean distance of each team's players to their centroid (Clemente et al. 2012), smoothed +/-1s."""
    length, width = atd["pitch"]
    px = atd["player_x"].astype(np.float64) * length
    py = atd["player_y"].astype(np.float64) * width
    team = atd["team_of_slot"]

    def _team_stretch(side):
        cols = team == side
        x, y = px[:, cols], py[:, cols]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cx = np.nanmean(x, axis=1, keepdims=True)
            cy = np.nanmean(y, axis=1, keepdims=True)
            return np.nanmean(np.hypot(x - cx, y - cy), axis=1)

    diff = _team_stretch(HOME) - _team_stretch(AWAY)
    return m14.nan_box_smooth_blocked(diff, half_win=25, blocks=live_blocks)


def correlation_frames(x, max_lag=250):
    v = x[np.isfinite(x)]
    if v.size < 10 * max_lag:
        return max_lag
    v = v - v.mean()
    var = float(np.dot(v, v))
    if var <= 0:
        return max_lag
    for lag in range(1, max_lag + 1):
        if float(np.dot(v[:-lag], v[lag:])) / var < np.exp(-1.0):
            return lag
    return max_lag


def fit_logistic_1d(x, y, iters=30):
    mu, sd = float(x.mean()), float(x.std())
    sd = sd if sd > 0 else 1.0
    z = (x - mu) / sd
    X = np.column_stack([np.ones_like(z), z])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -20.0, 20.0)))
        grad = X.T @ (y - p)
        hess = X.T @ (X * (p * (1.0 - p))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    b = float(w[1] / sd)
    a = float(w[0] - b * mu)
    return a, b


def compute_shape_missing_evidence(stretch_diff, fa, vis_in, dead_s, max_lag=250):
    """Fit 1-D logistic regression on high-confidence control anchors and normalize by tau."""
    train = vis_in & (dead_s < 0.0) & (fa >= 0) & np.isfinite(stretch_diff)
    if int(train.sum()) < 1000:
        return np.zeros(len(stretch_diff), dtype=np.float64)
    y = (fa[train] == HOME).astype(np.float64)
    a, b = fit_logistic_1d(stretch_diff[train], y)
    prior = float(np.clip(y.mean(), 0.05, 0.95))
    prior_lo = float(np.log(prior / (1.0 - prior)))
    tau = float(max(25, correlation_frames(stretch_diff, max_lag=max_lag)))
    out = np.zeros(len(stretch_diff), dtype=np.float64)
    ok = (~vis_in) & np.isfinite(stretch_diff)
    out[ok] = (a + b * stretch_diff[ok] - prior_lo) / tau
    return np.clip(out, -0.15, 0.15)


# ---------------------------------------------------------------------------
# 3. Unified v15 Decoder
# ---------------------------------------------------------------------------

def predict_possession_v15(
    atd_feats, st_feats, atd_raw=None, kick_sign=None, stretch_diff=None, viterbi_params=None
):
    vp = dict(m14.VITERBI_PARAMS)
    if viterbi_params:
        vp.update(viterbi_params)

    n = len(atd_feats["ball_visible"])
    blocks = m14.live_half_blocks(st_feats, n)

    if atd_raw is not None:
        atd_feats = m14.upgrade_atd_features(atd_raw, atd_feats, blocks)
        if kick_sign is None:
            kick_sign, _, _ = compute_sg_kicks_and_calibration(atd_raw, atd_feats, st_feats)
        if stretch_diff is None:
            stretch_diff = compute_stretch_diff(atd_raw, blocks)

    dead_s, post_oob = m14.compute_dead_ball_score(atd_feats, st_feats, blocks)
    team_s, fa, int_mask = m14.compute_team_evidence(atd_feats, st_feats, dead_s, blocks)

    vis_in = atd_feats["ball_visible"] & (~atd_feats["ball_oob"])
    if kick_sign is not None:
        team_s = team_s + 0.20 * kick_sign
    if stretch_diff is not None:
        team_s = team_s + compute_shape_missing_evidence(stretch_diff, fa, vis_in, dead_s)
    team_s = np.clip(team_s, -2.4, 2.4)

    abs_team = np.abs(team_s)
    emit = np.empty((n, 3), dtype=np.float64)
    emit[:, STATE_H] = -0.5 * dead_s + team_s - abs_team
    emit[:, STATE_A] = -0.5 * dead_s - team_s - abs_team
    emit[:, STATE_D] = 0.5 * dead_s

    nd = atd_feats["nearest_dist"]
    bspd = np.where(np.isnan(atd_feats["ball_speed"]), 0.0, atd_feats["ball_speed"])
    accel = np.where(np.isnan(atd_feats["ball_accel"]), 0.0, atd_feats["ball_accel"])
    touch_gate = (
        (vis_in & (~np.isnan(nd)) & (nd <= 2.1) & ((bspd <= 6.5) | (accel >= 11.0)))
        | (fa >= 0)
        | int_mask
    )
    touch_gate_s = m14.nan_box_smooth_blocked(touch_gate.astype(np.float64), half_win=2, blocks=blocks) > 0.0

    codes = np.full(n, STATE_X, dtype=np.int8)
    for s, e in blocks:
        codes[s:e] = m14.viterbi_decode_half(emit[s:e], touch_gate_s[s:e], **vp)

    codes = m14.heal_short_ungrounded_dead_spells(codes, atd_feats, st_feats, post_oob, blocks, max_heal_len=70)
    codes = m14.post_clean_v14(codes, blocks, fa=fa)
    codes[st_feats["pre_kickoff_dead_mask"]] = STATE_D

    labels = LABEL_CHARS[codes]
    return labels, codes, dead_s, team_s


def main():
    parser = argparse.ArgumentParser(description="Run v15 Adaptive Viterbi with SG Kicks & Team Shape.")
    parser.add_argument("--game", default=GAME, help="Game identifier (default: BAR-ATH)")
    parser.add_argument("--data-dir", default=None, help="Optional data directory override")
    parser.add_argument("--source", default="tactical", help="Optional data source override")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "evaluation", "results", "v15_adaptive_viterbi.csv"),
        help="Output submission CSV path",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"[1/4] Loading ATD tracking data for {args.game} ...")
    atd = m14.load_atd_arrays(game=args.game, data_dir=args.data_dir, source=args.source)
    n_frames = len(atd["ball_x"])
    print(f"      Loaded {n_frames:,} frames in {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"[2/4] Extracting Smart Tagging features for {args.game} ...")
    st_feats = m14.extract_smart_tagging_features(n_frames=n_frames, game=args.game, data_dir=args.data_dir)
    active_blocks = m14.live_half_blocks(st_feats, n_frames, game=args.game)
    print(f"      Done in {time.time() - t1:.1f}s")

    t2 = time.time()
    print("[3/4] Extracting physics, SG kick calibration & stretch-index team shape ...")
    atd_feats = m14.extract_atd_features(atd, active_blocks=active_blocks)
    up_feats = m14.upgrade_atd_features(atd, atd_feats, active_blocks)
    kick_sign, _, calib = compute_sg_kicks_and_calibration(atd, up_feats, st_feats)
    stretch_diff = compute_stretch_diff(atd, active_blocks)
    print(
        f"      Calibrated r_zone={calib['r_zone']:.2f}m, kick_thresh={calib['kick_thresh']:.1f}m/s^2 "
        f"| done in {time.time() - t2:.1f}s"
    )

    t3 = time.time()
    print("[4/4] Running v15 Adaptive Viterbi decoder ...")
    labels, _, _, _ = predict_possession_v15(
        atd_feats, st_feats, atd_raw=atd, kick_sign=kick_sign, stretch_diff=stretch_diff
    )
    print(f"      Decoded in {time.time() - t3:.1f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df = pd.DataFrame({"frame": np.arange(n_frames, dtype=np.int64), "label": labels})
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(out_df):,} predictions -> {args.output} (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
