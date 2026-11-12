"""Which frames are dead ball (D).

Three sources, combined:
  1. ball out of play  -- the ball crosses a line; dead until the restart kick/throw
  2. stoppage          -- players slow and the ball still or lost, for long enough, with evidence
                          (a set-piece tag nearby, the ball going out, or a long lost-ball stretch)
  3. before kick-off   -- from the start of a half until the ball is kicked from the centre
"""

import numpy as np

import params as P


def runs(mask):
    """(start, length) arrays for every run of True."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return edges[0::2], edges[1::2] - edges[0::2]


def close_holes(mask, max_hole):
    """Fill runs of False no longer than max_hole that sit between two runs of True."""
    out = mask.copy()
    starts, lengths = runs(~mask)
    for s, l in zip(starts, lengths):
        if 0 < s and s + l < len(mask) and l <= max_hole:
            out[s:s + l] = True
    return out


def drop_short(mask, min_len):
    out = mask.copy()
    for s, l in zip(*runs(mask)):
        if l < min_len:
            out[s:s + l] = False
    return out


def find_restart(f, start, stop):
    """First frame in [start, stop) where the ball is back inside the pitch and moving fast."""
    stop = min(stop, f["n"])
    if start >= stop:
        return None
    seg = slice(start, stop)
    ok = (f["vis"][seg] & (f["outside"][seg] <= -P.RESTART_INSIDE)
          & (f["speed"][seg] > P.RESTART_SPEED))
    hit = np.flatnonzero(ok)
    return start + int(hit[0]) if hit.size else None


def detect(f, tags, halves):
    """Boolean dead-ball mask over all frames. `halves` = [(half, first, last)] video frames."""
    n = f["n"]
    vis, speed = f["vis"], f["speed"]
    dead = np.zeros(n, dtype=bool)
    in_play = np.zeros(n, dtype=bool)
    for _, first, last in halves:
        in_play[first:min(last + 1, n)] = True

    # 1. ball out of play
    oob = vis & (f["outside"] > P.OOB_MARGIN)
    oob = drop_short(oob, P.OOB_MIN)
    oob_starts, _ = runs(oob)
    for o in oob_starts:
        restart = find_restart(f, o + P.RESTART_MIN_DELAY, o + P.RESTART_MAX)
        end = restart if restart is not None else min(o + P.OOB_FALLBACK, n)
        dead[o:end] = True

    # 2. stoppages
    slow = np.nan_to_num(f["activity_s"], nan=np.inf) < P.SLOW_PLAYERS
    quiet_ball = ~vis | (np.nan_to_num(speed, nan=0.0) < P.BALL_STILL)
    quiet = close_holes(slow & quiet_ball, P.CLOSE_HOLES)
    sp = tags["sp_start"]
    for s, l in zip(*runs(quiet)):
        if l < P.STOP_MIN:
            continue
        e = s + l
        lost = (~vis[s:e]).mean()
        evidence = (sp[max(s - P.STOP_EVIDENCE, 0):min(e + P.STOP_EVIDENCE, n)].any()
                    or oob[max(s - 3 * P.FPS, 0):e].any()
                    or (l >= P.STOP_LONG and lost > 0.5))
        if evidence:
            dead[s:e] = True

    # 3. before each half's kick-off
    lx, ly = f["length"] / 2, f["width"] / 2
    centre = np.hypot(f["bx"] - lx, f["by"] - ly) < P.KICKOFF_RADIUS
    kickoff_tags = [s for s in tags["setpieces"] if s["type"].startswith("kick off (start)")]
    for _, first, last in halves:
        # Search only up to the end of this half's kick-off tag: if the tracking missed the kick,
        # the first fast ball through the centre could be minutes into play.
        tag = [s for s in kickoff_tags if first - 10 * P.FPS <= s["start"] <= first + P.KICKOFF_SEARCH]
        if not tag:
            continue
        stop = min(tag[0]["end"] + 5 * P.FPS, last + 1, n)
        ok = vis[first:stop] & centre[first:stop] & (np.nan_to_num(speed[first:stop]) > P.RESTART_SPEED)
        hit = np.flatnonzero(ok)
        if hit.size:
            dead[first:first + int(hit[0])] = True

    dead = close_holes(dead, P.MERGE_DEAD)
    dead = drop_short(dead, P.MIN_DEAD)
    return dead & in_play
