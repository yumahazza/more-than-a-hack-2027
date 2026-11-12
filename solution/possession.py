"""Which team has the ball while it is alive.

Lesson from the public ground truth and the video: a single touch (even a header) already counts
as a possession, so the team changes when an opponent PLAYS THE BALL, not when he is merely the
nearest player. Nearest-player flips ~29 times a minute; real possession changes ~3 times.

  touch    = sudden change of ball velocity with a player within R_TOUCH, not contested
  control  = slow ball within R_CONTROL of one team for CONTROL_MIN frames (dribbling)

While the ball is lost inside live play, the team is kept unless the reliable Smart Tagging phases
point to the other team for a sustained stretch.
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
    control = clear & (near_d < P.R_CONTROL) & (np.nan_to_num(f["speed"], nan=99) < P.CONTROL_SPEED)

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


def assign(f, dead, tags, halves):
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
