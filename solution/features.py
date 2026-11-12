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


def savgol_coeffs(window, deriv):
    """Savitzky-Golay convolution weights, polynomial order 2, for the centre point.

    Least-squares fit of a parabola over `window` frames; the weights give the value (deriv 0),
    first derivative (1) or second derivative (2), per frame^deriv. numpy only, no scipy.
    """
    half = window // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(t, 3, increasing=True)          # columns 1, t, t^2
    pinv = np.linalg.pinv(design)                      # rows give the coefficients c0, c1, c2
    factor = (1.0, 1.0, 2.0)[deriv]                    # d^2/dt^2 of c2*t^2 is 2*c2
    return factor * pinv[deriv]


def savgol(x, window, deriv):
    """Centred Savitzky-Golay filter. NaN wherever the window touches a missing frame."""
    w = savgol_coeffs(window, deriv)[::-1]
    out = np.convolve(np.nan_to_num(x), w, mode="same")
    bad = np.convolve(np.isnan(x).astype(np.float64), np.ones(window), mode="same") > 0
    out[bad] = np.nan
    half = window // 2
    out[:half] = np.nan            # the window runs off the array: no estimate at the edges
    out[-half:] = np.nan
    return out


def ball_acceleration(bx, by, window, fps):
    """Magnitude of the Savitzky-Golay ball acceleration, m/s^2."""
    return np.hypot(savgol(bx, window, 2), savgol(by, window, 2)) * fps * fps


def kick_frames(acc, vis, near_d, near_team, duel, threshold, reach):
    """Per frame: the team credited with a kick (0/1), or -1.

    Link & Hoernig (2017): a kick is a local maximum of the smoothed ball acceleration of at least
    `threshold` m/s^2, given to the nearest player if he is within reach. A kick inside a duel is
    contested and credited to nobody.
    """
    a = np.nan_to_num(acc, nan=0.0)
    peak = np.zeros(len(a), dtype=bool)
    peak[1:-1] = (a[1:-1] >= a[:-2]) & (a[1:-1] > a[2:])
    kick = vis & peak & (a >= threshold) & np.isfinite(near_d) & (near_d <= reach) & ~duel
    out = np.full(len(a), -1, dtype=np.int8)
    out[kick] = near_team[kick]
    return out


def calibrate(vis, speed, near_d, activity_s, acc):
    """v6: possession radius and kick threshold from this match's own distributions (no labels)."""
    slow_live = (vis & np.isfinite(near_d) & (np.nan_to_num(speed, nan=99) < P.CAL_SLOW_BALL)
                 & (np.nan_to_num(activity_s, nan=0) >= P.SLOW_PLAYERS))
    if slow_live.sum() >= 100:
        r = float(np.percentile(near_d[slow_live], P.CAL_ZONE_PCT))
    else:
        r = P.R_POSSESSION
    r = float(np.clip(r, P.R_POSSESSION, P.CAL_ZONE_MAX))
    a = acc[vis & np.isfinite(acc)]
    k = float(np.percentile(a, P.CAL_KICK_PCT)) if a.size >= 100 else P.KICK_ACCEL
    k = max(k, P.KICK_ACCEL)
    return {"r_zone": r, "r_duel": r, "kick_accel": k, "n_slow_live": int(slow_live.sum())}


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

    # v5 structure (Vidal-Codina et al. 2022; Link & Hoernig 2017), v6 values calibrated on this match
    accel = ball_acceleration(bx, by, P.SG_WINDOW, P.FPS)
    calib = calibrate(vis, speed, near_d, activity_s, accel)
    r_duel = calib["r_duel"]
    duel = (np.nan_to_num(d_home, nan=np.inf) <= r_duel) & (np.nan_to_num(d_away, nan=np.inf) <= r_duel)
    kick_team = kick_frames(accel, vis, near_d, near_team, duel, calib["kick_accel"], calib["r_zone"])

    return {
        "duel": duel, "kick_team": kick_team, "accel": accel, "calib": calib,
        "r_zone": calib["r_zone"],
        "n": len(ball), "vis": vis, "bx": bx, "by": by, "speed": speed, "impulse": impulse,
        "d_home": d_home, "d_away": d_away, "near_d": near_d, "near_team": near_team,
        "margin": margin, "outside": outside, "activity": activity, "activity_s": activity_s,
        "length": length, "width": width,
    }
