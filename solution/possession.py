"""Which team has the ball while it is alive.

v4: inside every stretch of live play, the team is the most likely H/A sequence of a two-state
hidden Markov model, decoded with Viterbi.

  evidence per frame  v5: the ball in one team's 1 m possession zone (no evidence in a duel),
                      a kick credited to a team (worth one change on its own), or a reliable Smart
                      Tagging team while the ball is lost (see docs/REFERENCES.md)
  change cost         SWITCH_COST, the log of the expected frames between possession changes

A change happens only where the evidence for the other team, summed over the following frames,
outweighs the cost. A defender briefly next to the ball does not pay it; an interception followed
by half a second of play does. Nearest-player flips ~29 times a minute; real possession changes
~3 times.

The touch detector of v1-v3 (touch_frames, episodes) is kept: it still names the team at a restart
and diagnose.py reports on it.
"""

import numpy as np

import params as P
from deadball import runs

HOME, AWAY, DEAD, NONE = 0, 1, 2, -1


def touch_frames(f):
    """Per frame: team credited with playing the ball (0/1), or -1."""
    vis, near_d, margin = f["vis"], f["near_d"], f["margin"]
    clear = vis & np.isfinite(near_d) & (margin >= P.MARGIN)
    impulse = clear & (np.nan_to_num(f["impulse"]) > P.IMPULSE) & (near_d < P.R_TOUCH)
    control = (vis & np.isfinite(near_d) & (margin >= P.CONTROL_MARGIN) & (near_d < P.R_CONTROL)
               & (np.nan_to_num(f["speed"], nan=99) < P.CONTROL_SPEED))

    team = f["near_team"]
    out = np.full(f["n"], NONE, dtype=np.int8)
    out[impulse] = team[impulse]
    for side in (HOME, AWAY):
        mask = control & (team == side)
        for s, l in zip(*runs(mask)):
            if l >= P.CONTROL_MIN:
                out[s:s + l] = side
    return out


def episodes(touch):
    """[(start, team)] for each run of frames credited to one team."""
    result = []
    for side in (HOME, AWAY):
        for s, _ in zip(*runs(touch == side)):
            result.append((int(s), side))
    return sorted(result)


def start_team(s, e, f, ep_starts, ep_teams, tags, previous):
    """Team in possession when live play (re)starts at frame s."""
    i = np.searchsorted(ep_starts, s)
    if i < len(ep_starts) and ep_starts[i] < min(s + P.START_LOOKAHEAD, e):
        return int(ep_teams[i])
    for sp in tags["setpieces"]:
        if sp["side"] is not None and sp["start"] - P.SETPIECE_WINDOW <= s <= sp["end"] + P.SETPIECE_WINDOW:
            return sp["side"]
    if tags["tag_team"][s] != NONE:
        return int(tags["tag_team"][s])
    if previous is not None:
        return previous
    seen = np.flatnonzero(f["vis"][s:e])
    return int(f["near_team"][s + seen[0]]) if seen.size else HOME


def gap_switches(s, e, f, tags):
    """[(frame, team)] where, inside a lost-ball stretch of live play, the tags name a team."""
    out = []
    lost = ~f["vis"][s:e]
    tag_team = tags["tag_team"][s:e]
    for gs, gl in zip(*runs(lost)):
        seg = tag_team[gs:gs + gl]
        for side in (HOME, AWAY):
            for ts, tl in zip(*runs(seg == side)):
                if tl >= P.GAP_TAG_MIN:
                    out.append((s + gs + int(ts), side))
    return out


def absorb_short(labels, min_len, lo, hi):
    """Possession runs shorter than min_len take the label of the run before them."""
    seg = labels[lo:hi]
    change = np.flatnonzero(np.diff(seg)) + 1
    bounds = np.concatenate(([0], change, [len(seg)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a > 0 and seg[a] in (HOME, AWAY) and b - a < min_len and seg[a - 1] in (HOME, AWAY):
            seg[a:b] = seg[a - 1]
    labels[lo:hi] = seg


def evidence(f, tags):
    """Per-frame log-likelihood ratio, positive = home has the ball (v5).

    - ball in one team's possession zone (1 m), not a duel: +/- EVIDENCE_CAP
    - a kick credited to a team (not in a duel): +/- KICK_EVIDENCE on that frame
    - ball lost and a reliable Smart Tagging team: +/- TAG_EVIDENCE
    - anything else (ball in flight, duel, no information): 0
    """
    vis, near_d, team = f["vis"], f["near_d"], f["near_team"]
    sign = np.where(team == HOME, 1.0, -1.0)
    zone = vis & np.isfinite(near_d) & (near_d <= f["r_zone"]) & ~f["duel"]
    llr = np.where(zone, sign * P.EVIDENCE_CAP, 0.0)
    kick = f["kick_team"]
    llr[kick == HOME] += P.KICK_EVIDENCE
    llr[kick == AWAY] -= P.KICK_EVIDENCE
    tag = tags["tag_team"]
    lost = ~vis & (tag != NONE)
    llr[lost] = np.where(tag[lost] == HOME, P.TAG_EVIDENCE, -P.TAG_EVIDENCE)
    return llr


def touch_costs(f, dead, halves):
    """v12: per-frame cost of a team change (see params.USE_TOUCH_GATE). Returns (costs, info)."""
    n = f["n"]
    costs = np.full(n, P.SWITCH_COST)
    if not P.USE_TOUCH_GATE:
        return costs, None
    graded = np.zeros(n, dtype=bool)
    for _, first, last in halves:
        graded[first:min(last + 1, n)] = True
    live = graded & ~dead
    vis = f["vis"]
    touch = vis & np.isfinite(f["near_d"]) & (f["near_d"] <= f["r_zone"])
    away = vis & ~touch
    lost = ~vis
    n_live = max(int(live.sum()), 1)
    f_t, f_n, f_l = (touch & live).sum() / n_live, (away & live).sum() / n_live, (lost & live).sum() / n_live
    if min(f_t, f_n) <= 0:
        return costs, None
    T = P.MEAN_POSSESSION_S * P.FPS
    c_t = np.log(T * f_t / (P.TOUCH_SHARE * (1 - f_l)))
    c_n = np.log(T * f_n / ((1 - P.TOUCH_SHARE) * (1 - f_l)))
    costs[touch] = c_t
    costs[away] = c_n
    return costs, {"f_touch": f_t, "f_away": f_n, "f_lost": f_l,
                   "cost_touch": c_t, "cost_away": c_n, "cost_lost": float(np.log(T))}


def viterbi(llr, start, cost=None):
    """Most likely H/A path for one live stretch, starting (almost surely) with `start`.
    `cost` = per-frame team-change cost (v12); None = SWITCH_COST everywhere."""
    n = len(llr)
    if cost is None:
        cost = np.full(n, P.SWITCH_COST)
    half = llr / 2.0
    score = np.array([0.0, 0.0])
    score[1 - start] = -P.START_PRIOR
    back = np.zeros((n, 2), dtype=np.int8)       # back[t, s] = state at t-1 on the best path to s
    for t in range(n):
        emit = (half[t], -half[t])
        if t:
            c = cost[t]
            stay_h, move_h = score[HOME], score[AWAY] - c
            stay_a, move_a = score[AWAY], score[HOME] - c
            # On a tie, take the change as LATE as possible: with no evidence either way the old
            # team keeps the ball, as the scorer itself does when it fills an abstention.
            back[t, HOME] = HOME if stay_h > move_h else AWAY
            back[t, AWAY] = AWAY if stay_a > move_a else HOME
            score = np.array([max(stay_h, move_h), max(stay_a, move_a)])
        score = score + emit
    path = np.empty(n, dtype=np.int8)
    path[-1] = HOME if score[HOME] >= score[AWAY] else AWAY
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def assign(f, dead, tags, halves):
    """Label codes per frame: 0 H, 1 A, 2 D, -1 outside the halves (v4/v5, Viterbi)."""
    n = f["n"]
    labels = np.full(n, NONE, dtype=np.int8)
    # v5: the restart team is named by the first KICK after the restart (the restart kick itself)
    kick_at = np.flatnonzero(f["kick_team"] != NONE)
    ep_starts = kick_at.astype(np.int64)
    ep_teams = f["kick_team"][kick_at].astype(np.int8)
    llr = evidence(f, tags)
    costs, f["touch_gate"] = touch_costs(f, dead, halves)

    def decode(llr):
        out = np.full(n, NONE, dtype=np.int8)
        for _, first, last in halves:
            hi = min(last + 1, n)
            out[first:hi][dead[first:hi]] = DEAD
            previous = None
            for rs, rl in zip(*runs(~dead[first:hi])):
                s, e = first + int(rs), first + int(rs + rl)
                start = start_team(s, e, f, ep_starts, ep_teams, tags, previous)
                out[s:e] = viterbi(llr[s:e], start, costs[s:e])
                previous = int(out[e - 1])
            absorb_short(out, P.MIN_POSSESSION, first, hi)
        return out

    labels = decode(llr)
    f["shape_model"] = None
    llr_ha = llr
    if P.USE_SHAPE:
        # v8: second pass with team-shape evidence on lost-ball live frames (see shape.py)
        import shape
        model = shape.fit(f, labels, dead)
        f["shape_model"] = model
        if model is not None:
            lost_live = ~f["vis"] & ~dead
            labels = decode(llr + shape.evidence(f, model, lost_live))
            llr_ha = llr + shape.evidence(f, model, ~f["vis"])
    f["joint_model"] = None
    if P.USE_JOINT:
        # v10: H / A / D decided together (see joint.py); the labels above are its first pass
        import joint
        joint_labels, info = joint.decode(f, tags, halves, labels, dead, llr_ha)
        f["joint_model"] = info
        if info is not None:
            labels = joint_labels
            for _, first, last in halves:
                hi = min(last + 1, n)
                drop_short_dead(labels, P.MIN_DEAD, first, hi)
                absorb_short(labels, P.MIN_POSSESSION, first, hi)
    return labels


def drop_short_dead(labels, min_len, lo, hi):
    """Dead stretches shorter than min_len take the live label before them (or after, at a start)."""
    seg = labels[lo:hi]
    for s, l in zip(*runs(seg == DEAD)):
        if l < min_len:
            fill = seg[s - 1] if s > 0 else (seg[s + l] if s + l < len(seg) else HOME)
            seg[s:s + l] = fill if fill in (HOME, AWAY) else HOME
    labels[lo:hi] = seg


def assign_events(f, dead, tags, halves):
    """v1-v3 assignment (switch on touch events), kept for comparison."""
    """Label codes per frame: 0 H, 1 A, 2 D, -1 outside the halves."""
    n = f["n"]
    labels = np.full(n, NONE, dtype=np.int8)
    eps = episodes(touch_frames(f))
    ep_starts = np.array([s for s, _ in eps], dtype=np.int64)
    ep_teams = np.array([t for _, t in eps], dtype=np.int8)

    for _, first, last in halves:
        hi = min(last + 1, n)
        labels[first:hi][dead[first:hi]] = DEAD
        previous = None
        for rs, rl in zip(*runs(~dead[first:hi])):
            s, e = first + int(rs), first + int(rs + rl)
            cur = start_team(s, e, f, ep_starts, ep_teams, tags, previous)
            lo_i, hi_i = np.searchsorted(ep_starts, s), np.searchsorted(ep_starts, e)
            events = [(int(ep_starts[j]), int(ep_teams[j])) for j in range(lo_i, hi_i)]
            events += gap_switches(s, e, f, tags)
            events.sort()
            mark = s
            for frame, team in events:
                if team != cur and frame > mark:
                    labels[mark:frame] = cur
                    mark, cur = frame, team
                elif team != cur:
                    cur = team
            labels[mark:e] = cur
            previous = cur
        absorb_short(labels, P.MIN_POSSESSION, first, hi)
    return labels
