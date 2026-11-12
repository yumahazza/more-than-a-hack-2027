"""Per-frame features from the tracking arrays (positions already in metres, see atd.py).

The ATD speed columns are NOT used: explore2.py showed the ball's speed column does not agree with
its own positions (correlation 0.07), so every speed here is computed from positions.
"""

import warnings

import numpy as np

import params as P

HOME, AWAY = 0, 1


def shift(x, k):
    """x shifted so that out[t] = x[t - k]; NaN where undefined."""
    out = np.full_like(x, np.nan)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def rolling_nanmean(x, half):
    valid = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    m = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    count = m[hi] - m[lo]
    return np.where(count > 0, (c[hi] - c[lo]) / np.maximum(count, 1), np.nan)


def compute(arrays, length, width):
    ball, players, team = arrays["ball"], arrays["players"], arrays["team"]
    bx = ball[:, 0].astype(np.float64)
    by = ball[:, 1].astype(np.float64)
    vis = np.isfinite(bx) & np.isfinite(by)

    # ball speed (central difference over +/- 2 frames) and velocity change ("impulse")
    speed = np.hypot(shift(bx, -2) - shift(bx, 2), shift(by, -2) - shift(by, 2)) * P.FPS / 4
    k = P.K_VEL
    vbx, vby = (bx - shift(bx, k)) * P.FPS / k, (by - shift(by, k)) * P.FPS / k
    vax, vay = (shift(bx, -k) - bx) * P.FPS / k, (shift(by, -k) - by) * P.FPS / k
    impulse = np.hypot(vax - vbx, vay - vby)

    # distance from the ball to the nearest player of each team
    px = players[:, :, 0].astype(np.float32)
    py = players[:, :, 1].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = np.hypot(px - bx[:, None].astype(np.float32), py - by[:, None].astype(np.float32))
        d_home = np.nanmin(np.where(team[None, :] == HOME, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(team[None, :] == AWAY, dist, np.nan), axis=1)

        # player activity: mean speed over all tracked players, computed from positions
        s = P.ACTIVITY_SPAN
        sx = (np.roll(px, -s, axis=0) - np.roll(px, s, axis=0))
        sy = (np.roll(py, -s, axis=0) - np.roll(py, s, axis=0))
        pspeed = np.hypot(sx, sy) * P.FPS / (2 * s)
        pspeed[:s] = np.nan
        pspeed[-s:] = np.nan
        pspeed[pspeed > P.MAX_PLAYER_SPEED] = np.nan
        activity = np.nanmean(pspeed, axis=1)
    activity_s = rolling_nanmean(activity, P.ACTIVITY_SMOOTH)

    near_d = np.fmin(d_home, d_away)
    near_team = np.where(np.isnan(d_away) | (d_home <= d_away), HOME, AWAY).astype(np.int8)
    margin = np.abs(d_home - d_away)
    margin[np.isnan(margin)] = np.inf          # one side not tracked near: no contest
    outside = -np.minimum.reduce([bx, length - bx, by, width - by])   # >0 = metres outside

    return {
        "n": len(ball), "vis": vis, "bx": bx, "by": by, "speed": speed, "impulse": impulse,
        "d_home": d_home, "d_away": d_away, "near_d": near_d, "near_team": near_team,
        "margin": margin, "outside": outside, "activity": activity, "activity_s": activity_s,
        "length": length, "width": width,
    }
