"""Stage 1.3: summary statistics of the match data, to understand it before building a solution.

    .\\.venv\\Scripts\\python.exe solution\\explore.py

Prints SUMMARY NUMBERS only (shares, percentiles, counts) over the graded frames. Nothing is
printed per frame, so the output can be shared with the team (and with an AI assistant) without
leaking the data.

The public ground truth is deliberately NOT used: statistics per ground-truth class would invite
picking thresholds from those 5 minutes, which RULES.md forbids.
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402
import load_smart_tagging  # noqa: E402
from atd import HOME, AWAY, load_arrays  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean/nanmin on empty rows

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]

# Smart Tagging: for these codes Team is the team WITH the ball; for the defensive ones it is the
# team WITHOUT it, so their side is flipped before comparing with the tracking.
ATTACKING_CODES = {"build up", "progression", "creating chances", "attacking transition",
                   "ball in final third", "ball in the box", "players in the box", "long ball"}
DEFENDING_CODES = {"defending in attacking third", "defending in middle third",
                   "defending in defensive third", "defensive transition"}


# ------------------------------------------------------------------------------------------------
def header(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def pct(x):
    return "{:5.1f}%".format(100.0 * x)


def percentiles(values, qs=(5, 25, 50, 75, 95)):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "(no data)"
    return "  ".join("p{}={:.2f}".format(q, v) for q, v in zip(qs, np.percentile(values, qs)))


def runs(mask):
    """(start, length) of every run of True in a boolean array."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts, ends = edges[0::2], edges[1::2]
    return starts, ends - starts


def graded_mask(n):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:min(last + 1, n)] = True
    return mask


def side_of_team_tag(value):
    v = (value or "").strip().lower()
    if "barcelona" in v:
        return HOME
    if "atletic" in v or "athletic" in v:
        return AWAY
    return None


# ------------------------------------------------------------------------------------------------
def main():
    a = load_arrays()
    ball, players, team = a["ball"], a["players"], a["team"]
    n = len(ball)
    graded = graded_mask(n)
    n_graded = int(graded.sum())
    minutes = n_graded / FPS / 60.0
    print("graded frames: {} ({:.1f} minutes)".format(n_graded, minutes))
    print("track slots: {} (home {}, away {})".format(len(team), int((team == HOME).sum()),
                                                       int((team == AWAY).sum())))

    visible = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    vis_g = visible & graded

    # -- 1. missing ball ------------------------------------------------------------------------
    header("1. MISSING BALL (graded frames)")
    print("ball visible: {}   missing: {}".format(pct(vis_g.sum() / n_graded),
                                                 pct(1 - vis_g.sum() / n_graded)))
    starts, lengths = runs(graded & ~visible)
    print("number of gaps (runs of frames without the ball): {}".format(len(lengths)))
    print("gap length (frames): " + percentiles(lengths.astype(float)))
    bins = [(1, 5, "1-5 fr  (<0.2 s)"), (6, 25, "6-25 fr (<=1 s)"),
            (26, 125, "26-125 fr (1-5 s)"), (126, 500, "126-500 fr (5-20 s)"),
            (501, 10 ** 9, ">500 fr (>20 s)")]
    total_missing = max(int(lengths.sum()), 1)
    print("  {:<22} {:>8} {:>22}".format("bucket", "gaps", "share of missing frames"))
    for lo, hi, name in bins:
        sel = (lengths >= lo) & (lengths <= hi)
        print("  {:<22} {:>8} {:>22}".format(name, int(sel.sum()),
                                             pct(lengths[sel].sum() / total_missing)))

    # -- 2. ball position & speed ---------------------------------------------------------------
    header("2. BALL POSITION & SPEED (when visible)")
    bx, by = ball[vis_g, 0], ball[vis_g, 1]
    out = (bx < 0) | (bx > LENGTH) | (by < 0) | (by > WIDTH)
    print("ball outside the pitch lines: {} of visible frames".format(pct(out.mean())))
    edge = np.minimum.reduce([bx, LENGTH - bx, by, WIDTH - by])
    print("ball distance to nearest line (m, negative = outside): " + percentiles(edge))
    oob = (ball[:, 0] < 0) | (ball[:, 0] > LENGTH) | (ball[:, 1] < 0) | (ball[:, 1] > WIDTH)
    _, o_len = runs(graded & visible & oob)
    print("ball-out events (runs): {}  length (frames): {}".format(
        len(o_len), percentiles(o_len.astype(float))))
    bs = ball[vis_g, 2]
    if np.isfinite(bs).any():
        print("ball speed, ATD column (m/s): " + percentiles(bs))
        print("  still/slow ball (<1 m/s): {}   fast ball (>10 m/s): {}".format(
            pct(np.nanmean(bs < 1)), pct(np.nanmean(bs > 10))))
    else:
        print("ball speed is not available in the ATD")

    # -- 3. players -----------------------------------------------------------------------------
    header("3. PLAYERS (graded frames)")
    present = np.isfinite(players[:, :, 0])
    count = present.sum(axis=1)[graded]
    print("tracked objects per frame: " + percentiles(count.astype(float)))
    print("  frames with <18 objects: {}   >22 objects: {}".format(pct(np.mean(count < 18)),
                                                                  pct(np.mean(count > 22))))
    for side, name in ((HOME, "home"), (AWAY, "away")):
        c = present[:, team == side].sum(axis=1)[graded]
        print("  {} objects: {}".format(name, percentiles(c.astype(float))))
    mean_speed = np.nanmean(players[:, :, 2], axis=1)[graded]
    print("mean speed of all players per frame (m/s): " + percentiles(mean_speed))
    for thr in (1.0, 1.5, 2.0):
        print("  frames with mean < {} m/s: {}".format(thr, pct(np.nanmean(mean_speed < thr))))

    # -- 4. nearest player to the ball ----------------------------------------------------------
    header("4. NEAREST PLAYER TO THE BALL (when the ball is visible)")
    dx = players[:, :, 0] - ball[:, None, 0]
    dy = players[:, :, 1] - ball[:, None, 1]
    dist = np.hypot(dx, dy)
    d_home = np.nanmin(np.where(team[None, :] == HOME, dist, np.nan), axis=1)
    d_away = np.nanmin(np.where(team[None, :] == AWAY, dist, np.nan), axis=1)
    nearest = np.fmin(d_home, d_away)
    ok = vis_g & np.isfinite(nearest)
    print("nearest player distance (m): " + percentiles(nearest[ok]))
    for thr in (1, 2, 3, 5, 10):
        print("  nearest < {:>2} m: {}".format(thr, pct(np.mean(nearest[ok] < thr))))
    both = vis_g & np.isfinite(d_home) & np.isfinite(d_away)
    margin = np.abs(d_home - d_away)[both]
    print("gap between nearest of each team (m): " + percentiles(margin))
    print("  gap < 1 m (contested/ambiguous): {}".format(pct(np.mean(margin < 1))))

    # a side with nobody tracked has NaN distance; the other side is then the nearest
    near_team = np.where(np.isnan(d_away) | (d_home <= d_away), HOME, AWAY)
    switches, vis_frames = 0, 0
    v_starts, v_len = runs(ok)
    for s, l in zip(v_starts, v_len):
        seg = near_team[s:s + l]
        switches += int(np.count_nonzero(np.diff(seg)))
        vis_frames += l
    print("'nearest team' changes per minute of visible ball: {:.1f}".format(
        switches / max(vis_frames / FPS / 60.0, 1e-9)))
    print("share of frames where 'nearest team' = home: {}".format(
        pct(np.mean(near_team[ok] == HOME))))

    # -- 5. Smart Tagging -----------------------------------------------------------------------
    header("5. SMART TAGGING")
    events = load_smart_tagging.load()
    in_graded = [e for e in events if graded[min(max(e["start_frame"], 0), n - 1)]]
    print("tags total: {}   starting in graded frames: {}".format(len(events), len(in_graded)))

    att = np.zeros((2, n), dtype=bool)  # [side] -> frame covered by "this side has the ball"
    per_code = {}
    for e in events:
        code = (e["code"] or "").strip().lower()
        side = side_of_team_tag(e["tags"].get("Team"))
        s, t = max(e["start_frame"], 0), min(e["end_frame"], n)
        if side is None or t <= s:
            continue
        if code in DEFENDING_CODES:
            side = 1 - side
        elif code not in ATTACKING_CODES:
            continue
        att[side, s:t] = True
        span = np.zeros(n, dtype=bool)
        span[s:t] = True
        sel = span & ok
        rec = per_code.setdefault(code, [0, 0, 0])
        rec[0] += 1
        rec[1] += int(sel.sum())
        rec[2] += int((near_team[sel] == side).sum())

    cov_any = (att[0] | att[1]) & graded
    conflict = att[0] & att[1] & graded
    print("graded frames covered by a team tag (defensive tags flipped): " +
          pct(cov_any.sum() / n_graded))
    print("  ...covered for both teams at once (conflict): " + pct(conflict.sum() / n_graded))
    print("\nagreement of tag team vs 'nearest team to the ball' (ball-visible frames):")
    print("  {:<32} {:>5} {:>10} {:>10}".format("code", "tags", "frames", "agree"))
    for code in sorted(per_code, key=lambda c: -per_code[c][1]):
        k, f, m = per_code[code]
        print("  {:<32} {:>5} {:>10} {:>10}".format(code, k, f, pct(m / f) if f else "-"))

    team_vals = {}
    for e in events:
        v = e["tags"].get("Team")
        team_vals[v] = team_vals.get(v, 0) + 1
    print("\nTeam values: " + ", ".join("{}={}".format(k, v) for k, v in team_vals.items()))

    print("\nset pieces: is there a dead-ball sign before/inside the tag window?")
    print("  (window = 10 s before the tag starts until the tag ends)")
    rows = {}
    for e in events:
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        kind = (e["tags"].get("Type") or "?").strip()
        s, t = max(e["start_frame"] - 10 * FPS, 0), min(e["end_frame"], n)
        w_missing = ~visible[s:t]
        longest_gap = int(runs(w_missing)[1].max()) if w_missing.any() else 0
        r = rows.setdefault(kind, [0, 0, 0, 0])
        r[0] += 1
        r[1] += int(oob[s:t].any())
        r[2] += int(longest_gap >= FPS)
        r[3] += int(np.nanmean(players[s:t, :, 2]) < 1.5) if t > s else 0
    print("  {:<24} {:>4} {:>10} {:>16} {:>18}".format(
        "type", "n", "ball out", "ball lost >=1s", "players slow<1.5"))
    for kind in sorted(rows, key=lambda k: -rows[k][0]):
        k, o, g, sl = rows[kind]
        print("  {:<24} {:>4} {:>10} {:>16} {:>18}".format(kind, k, o, g, sl))

    print("\ndone. Every number above is a summary; safe to share with the team.")


if __name__ == "__main__":
    main()
