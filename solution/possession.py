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


def viterbi(llr, start):
    """Most likely H/A path for one live stretch, starting (almost surely) with `start`."""
    n = len(llr)
    half = llr / 2.0
    score = np.array([0.0, 0.0])
    score[1 - start] = -P.START_PRIOR
    back = np.zeros((n, 2), dtype=np.int8)       # back[t, s] = state at t-1 on the best path to s
    for t in range(n):
        emit = (half[t], -half[t])
        if t:
            stay_h, move_h = score[HOME], score[AWAY] - P.SWITCH_COST
            stay_a, move_a = score[AWAY], score[HOME] - P.SWITCH_COST
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

    for _, first, last in halves:
        hi = min(last + 1, n)
        labels[first:hi][dead[first:hi]] = DEAD
        previous = None
        for rs, rl in zip(*runs(~dead[first:hi])):
            s, e = first + int(rs), first + int(rs + rl)
            start = start_team(s, e, f, ep_starts, ep_teams, tags, previous)
            labels[s:e] = viterbi(llr[s:e], start)
            previous = int(labels[e - 1])
        absorb_short(labels, P.MIN_POSSESSION, first, hi)
    return labels


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
