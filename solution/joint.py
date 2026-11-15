"""v10: home / away / dead ball decided together in one three-state HMM.

Why: up to v9 the dead ball was decided first by hard yes/no rules ("quiet for 3 s AND a set-piece
tag", "quiet for 5 s AND few phase tags"), then the team by an HMM. diagnose.py showed the stoppages
we miss (a foul, an injury) are many WEAK signals together: players slowing, the ball lost or
rolling slowly, few passes, no phase-of-play tag. Each fails its threshold alone. In an HMM weak
evidence ADDS UP, the same way possession evidence has since v4. A teammate's model, which treats
dead ball as a Viterbi state, finds far more of the dead ball, which points the same way.

Without human labels:

1. First pass = the v9 pipeline (hard dead-ball rules + two-state H/A Viterbi).
2. Dead-ball evidence model: a logistic regression of "dead" on simple per-frame cues, fitted on
   PSEUDO-LABELS from the first pass: frames well inside a first-pass dead stretch vs frames well
   inside live play (RULES.md allows algorithmic pseudo-labels). Its log-odds minus the prior
   log-odds is the per-frame evidence, divided by its correlation time (frames are not
   independent), exactly as for team shape in v8.
3. The first-pass hard rules stay as strong evidence (they are 99% precise on the public window).
4. Transition costs come from the first pass itself: entering dead ball costs ln(mean live
   stretch), leaving it ln(mean dead stretch) + ln 2 (two live states to go to); a team change
   costs SWITCH_COST as before.
5. One Viterbi over each half decides H / A / D jointly.
"""

import numpy as np

import params as P
from deadball import rolling_max, runs, sustained
from shape import correlation_frames

HOME, AWAY, DEAD, NONE = 0, 1, 2, -1


def cues(f, tags):
    """Per-frame dead-ball cues (columns) and their names."""
    vis, speed = f["vis"], np.nan_to_num(f["speed"], nan=0.0)
    passing = sustained(vis & (speed > P.PASS_SPEED), P.FAST_MIN)
    near_pass = rolling_max(passing, P.QUIET_WINDOW)
    sp_near = np.zeros(f["n"], dtype=bool)
    for sp in tags["setpieces"]:
        sp_near[max(sp["start"] - P.STOP_EVIDENCE, 0):sp["end"]] = True
    loose = vis & (np.nan_to_num(f["near_d"], nan=np.inf) > P.LOOSE_ZONES * f["r_zone"])
    act = np.nan_to_num(f["activity_s"], nan=np.nanmedian(f["activity_s"]))
    cols = [
        ("player activity (m/s)", act),
        ("pass within 1 s", near_pass.astype(float)),
        ("ball lost", (~vis).astype(float)),
        ("phase-of-play tag", tags["phase"].astype(float)),
        ("set-piece tag within 10 s", sp_near.astype(float)),
        ("nobody near the ball", loose.astype(float)),
        ("ball speed (m/s, visible)", np.where(vis, np.minimum(speed, 20.0), 0.0)),
    ]
    return np.column_stack([c for _, c in cols]), [name for name, _ in cols]


def fit_logistic_multi(X, y, iters=50, ridge=1e-3):
    """Multivariate logistic regression (Newton, small ridge) on standardised columns."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (y - p) - ridge * np.r_[0, w[1:]]
        hess = Z.T @ (Z * (p * (1 - p))[:, None]) + ridge * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w, mu, sd


def fit_dead_model(f, tags, labels, graded):
    """Fit the dead-ball evidence model on first-pass pseudo-labels."""
    dead = labels == DEAD
    live = (labels == HOME) | (labels == AWAY)
    # well inside: shrink dead stretches by 1 s at each end, keep live frames 3 s from any dead
    dead_core = dead & ~rolling_max(~dead & graded, P.FPS)
    live_core = live & ~rolling_max(dead, 3 * P.FPS)
    train = graded & (dead_core | live_core)
    X, names = cues(f, tags)
    if dead_core[train].sum() < 500 or live_core[train].sum() < 500:
        return None
    y = dead_core[train].astype(float)
    w, mu, sd = fit_logistic_multi(X[train], y)
    prior = y.mean()
    z = np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ w
    llr = np.clip(z, -30, 30) - np.log(prior / (1 - prior))
    tau = correlation_frames(np.where(graded, llr, np.nan), P.SHAPE_TAU_MAX)
    pred = z[train] > np.log(prior / (1 - prior))
    return {"w": w, "mu": mu, "sd": sd, "names": names, "tau": tau, "llr": llr,
            "agree": float(np.mean(pred == (y == 1))), "n_dead": int(dead_core[train].sum()),
            "n_live": int(live_core[train].sum())}


def run_lengths(labels, graded, code):
    lengths = []
    for s, l in zip(*runs((labels == code) & graded)):
        lengths.append(l)
    return np.array(lengths)


def viterbi3(emit, cost):
    """emit (n,3) log scores; cost (3,3) transition costs (0 on the diagonal). Late changes on ties."""
    n = len(emit)
    score = emit[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] - cost             # cand[i, j]: from i to j
        best = np.empty(3)
        for j in range(3):
            stay = cand[j, j]
            others = [i for i in range(3) if i != j]
            i = max(others, key=lambda k: cand[k, j])
            # ">=": on a tie the change is taken as LATE as possible (the old state keeps going
            # while there is no evidence either way), as in the two-state decoder of v4
            if cand[i, j] >= stay:
                back[t, j], best[j] = i, cand[i, j]
            else:
                back[t, j], best[j] = j, stay
        score = best + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(f, tags, halves, first_pass, hard_dead, llr_ha):
    """Joint H/A/D labels. Returns (labels, info) or (first_pass, None) if the model cannot fit."""
    n = f["n"]
    graded = np.zeros(n, dtype=bool)
    for _, first, last in halves:
        graded[first:min(last + 1, n)] = True
    model = fit_dead_model(f, tags, first_pass, graded)
    if model is None:
        return first_pass, None
    alive_len = np.concatenate([run_lengths(np.where(first_pass == AWAY, HOME, first_pass), graded, HOME)])
    dead_len = run_lengths(first_pass, graded, DEAD)
    mean_alive = float(alive_len.mean()) if alive_len.size else 1000.0
    mean_dead = float(dead_len.mean()) if dead_len.size else 500.0
    to_dead, from_dead = np.log(mean_alive), np.log(mean_dead) + np.log(2.0)
    cost = np.array([[0.0, P.SWITCH_COST, to_dead],
                     [P.SWITCH_COST, 0.0, to_dead],
                     [from_dead, from_dead, 0.0]])

    d_llr = model["llr"] / model["tau"]
    d_llr = d_llr + np.where(hard_dead, P.HARD_DEAD_EVIDENCE, 0.0)
    emit = np.column_stack([llr_ha / 2.0, -llr_ha / 2.0, d_llr])

    labels = np.full(n, NONE, dtype=np.int8)
    for _, first, last in halves:
        hi = min(last + 1, n)
        labels[first:hi] = viterbi3(emit[first:hi], cost)
    info = dict(model)
    info.update({"mean_alive_s": mean_alive / P.FPS, "mean_dead_s": mean_dead / P.FPS,
                 "cost_to_dead": to_dead, "cost_from_dead": from_dead})
    info.pop("llr")
    return labels, info
