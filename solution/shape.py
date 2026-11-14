"""v8: team shape as possession evidence while the ball is lost.

The ball is missing on 41% of graded frames, and inside live play the model had almost nothing to
go on there (only Smart Tagging, at a small weight). Two lines of work say the players' movement
alone carries the possession:

- Team spread changes with possession: stretch index and surface area differ between the moments
  with and without the ball (Clemente et al. 2012, 2013); teams spread out more when attacking.
- PathCRF (Kim et al. 2026) infers possession from player trajectories without the ball, with
  92.4% frame-level team accuracy (supervised, so not usable as is).

Here, without any human label:

1. stretch index per team = mean distance of its tracked players to the team centroid, smoothed;
   feature x = stretch(home) - stretch(away).
2. Pseudo-labels: frames where the ball IS visible, play is live, and the first-pass model is well
   inside a possession (at least SHAPE_MARGIN from any change). These labels come from our own
   algorithm (allowed by RULES.md: "pseudo-labels produced by an algorithm").
3. A one-feature logistic regression p(home | x) is fitted on them. Its log-odds minus the prior
   log-odds is the likelihood ratio of x, i.e. the HMM evidence of one frame.
4. Consecutive frames of a smooth feature are not independent: the per-frame evidence is divided by
   the feature's correlation time (frames until its autocorrelation drops below 1/e).
"""

import warnings

import numpy as np

import params as P
from deadball import runs

HOME, AWAY = 0, 1


def stretch_index(px, py, team, side):
    """Mean distance of a team's tracked players to their centroid, per frame (m)."""
    cols = team == side
    x, y = px[:, cols], py[:, cols]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cx = np.nanmean(x, axis=1, keepdims=True)
        cy = np.nanmean(y, axis=1, keepdims=True)
        return np.nanmean(np.hypot(x - cx, y - cy), axis=1)


def correlation_frames(x, max_lag):
    """Frames until the autocorrelation of x drops below 1/e (at least 1)."""
    v = x[np.isfinite(x)]
    if v.size < 10 * max_lag:
        return max_lag
    v = v - v.mean()
    var = np.dot(v, v)
    if var <= 0:
        return max_lag
    for lag in range(1, max_lag + 1):
        if np.dot(v[:-lag], v[lag:]) / var < np.exp(-1):
            return lag
    return max_lag


def fit_logistic(x, y, iters=50):
    """1-D logistic regression by Newton's method on standardised x. Returns (a, b) in raw units."""
    mu, sd = x.mean(), x.std()
    sd = sd if sd > 0 else 1.0
    z = (x - mu) / sd
    X = np.column_stack([np.ones_like(z), z])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        grad = X.T @ (y - p)
        hess = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    b = w[1] / sd
    a = w[0] - b * mu
    return a, b


def fit(f, labels, dead):
    """Fit the shape model on first-pass pseudo-labels. Returns a dict, or None if too little data."""
    x = f["shape"]
    live = (labels == HOME) | (labels == AWAY)
    near_change = np.zeros(len(labels), dtype=bool)
    change = np.flatnonzero(np.diff(labels.astype(np.int16)) != 0) + 1
    for c in change:
        near_change[max(c - P.SHAPE_MARGIN, 0):c + P.SHAPE_MARGIN] = True
    train = live & f["vis"] & ~dead & ~near_change & np.isfinite(x)
    if train.sum() < P.SHAPE_MIN_TRAIN:
        return None
    y = (labels[train] == HOME).astype(np.float64)
    a, b = fit_logistic(x[train], y)
    prior = y.mean()
    prior_logodds = np.log(prior / (1 - prior))
    tau = correlation_frames(x, P.SHAPE_TAU_MAX)
    pred = (a + b * x[train]) > prior_logodds
    agree = float(np.mean(pred == (y == 1)))
    return {"a": a, "b": b, "prior_logodds": prior_logodds, "tau": tau,
            "n_train": int(train.sum()), "agree": agree}


def evidence(f, model, where):
    """Per-frame HMM evidence (+ = home) from team shape, on the frames in `where`."""
    out = np.zeros(f["n"])
    if model is None:
        return out
    x = f["shape"]
    ok = where & np.isfinite(x)
    llr = model["a"] + model["b"] * x[ok] - model["prior_logodds"]
    out[ok] = llr / model["tau"]
    return out
