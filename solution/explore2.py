"""Stage 1.3 (continued): two questions raised by explore.py.

    .\\.venv\\Scripts\\python.exe solution\\explore2.py

A. The ATD ball speed is never above 10 m/s. Is the column odd, or are the ball positions
   themselves smoothed / stuck to players? -> compute the speed ourselves from the positions.
B. The ball goes missing in long gaps (median ~11 s). During those gaps, is the ball dead or still
   in play?
C. Where is the ball last seen before a gap, and where does it reappear after it?

Same as explore.py: summary numbers only, no ground truth, nothing printed per frame.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from explore import (ATTACKING_CODES, DEFENDING_CODES, FPS, LENGTH, WIDTH,  # noqa: E402
                     graded_mask, header, load_smart_tagging, pct, percentiles, runs,
                     side_of_team_tag)
from atd import load_arrays  # noqa: E402

LONG_GAP = 5 * FPS          # a "long" gap: 5 s or more
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16


def dist_to_line(x, y):
    """Distance to the nearest touch/goal line in metres (negative = outside the pitch)."""
    return np.minimum.reduce([x, LENGTH - x, y, WIDTH - y])


def in_box(x, y):
    return ((x < BOX_DEPTH) | (x > LENGTH - BOX_DEPTH)) & (np.abs(y - WIDTH / 2) < BOX_HALF_WIDTH)


def near_centre(x, y, radius=3.0):
    return np.hypot(x - LENGTH / 2, y - WIDTH / 2) < radius


def main():
    a = load_arrays()
    ball, players = a["ball"], a["players"]
    n = len(ball)
    graded = graded_mask(n)
    visible = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    bx, by = ball[:, 0], ball[:, 1]

    # -- A. ball speed computed from positions ---------------------------------------------------
    header("A. BALL SPEED COMPUTED FROM DISPLACEMENT")
    for lag, name in ((1, "over 1 frame (0.04 s)"), (5, "over 5 frames (0.2 s)"),
                      (25, "over 25 frames (1 s)")):
        ok = visible[lag:] & visible[:-lag] & graded[lag:] & graded[:-lag]
        step = np.hypot(bx[lag:] - bx[:-lag], by[lag:] - by[:-lag])[ok]
        speed = step * FPS / lag
        print("{}: {}".format(name, percentiles(speed)))
        print("    >10 m/s: {}   >20 m/s: {}   <1 m/s: {}".format(
            pct(np.mean(speed > 10)), pct(np.mean(speed > 20)), pct(np.mean(speed < 1))))

    ok1 = visible[1:] & visible[:-1] & graded[1:]
    step1 = np.hypot(np.diff(bx), np.diff(by))[ok1]
    print("jumps > 2 m between frames (= >50 m/s, physically impossible): {}".format(
        pct(np.mean(step1 > 2))))
    print("ball not moving at all between frames (< 1 cm): {}".format(
        pct(np.mean(step1 < 0.01))))

    col = ball[1:, 2][ok1]
    calc = (step1 * FPS)
    good = np.isfinite(col) & (calc > 0.05)
    if good.any():
        ratio = col[good] / calc[good]
        corr = np.corrcoef(col[good], calc[good])[0, 1]
        print("ATD ball speed column / computed speed: " + percentiles(ratio))
        print("correlation column vs computed: {:.2f}  (1 = perfectly in step)".format(corr))

    dist = np.hypot(players[:, :, 0] - bx[:, None], players[:, :, 1] - by[:, None])
    nearest = np.nanmin(dist, axis=1)
    okn = visible & graded & np.isfinite(nearest)
    print("ball 'stuck' to a player (distance < 0.1 m): {}   (< 0.3 m): {}".format(
        pct(np.mean(nearest[okn] < 0.1)), pct(np.mean(nearest[okn] < 0.3))))
    near_speed = np.take_along_axis(
        players[:, :, 2], np.nanargmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)[:, None],
        axis=1)[:, 0]
    both = okn & np.isfinite(near_speed) & np.isfinite(ball[:, 2])
    if both.any():
        print("correlation ball speed (column) vs nearest player speed: {:.2f}".format(
            np.corrcoef(ball[both, 2], near_speed[both])[0, 1]))

    # -- B. what happens inside long gaps --------------------------------------------------------
    header("B. LONG GAPS (ball missing >= 5 s, graded frames)")
    events = load_smart_tagging.load()
    live_tag = np.zeros(n, dtype=bool)          # frame covered by any team-tagged phase
    sp_start = np.zeros(n, dtype=bool)          # frame where a set-piece tag starts
    for e in events:
        code = (e["code"] or "").strip().lower()
        s, t = max(e["start_frame"], 0), min(e["end_frame"], n)
        if t <= s:
            continue
        if code == "set pieces":
            sp_start[s] = True
        elif code in ATTACKING_CODES | DEFENDING_CODES and side_of_team_tag(e["tags"].get("Team")) is not None:
            live_tag[s:t] = True

    starts, lengths = runs(graded & ~visible)
    long_sel = lengths >= LONG_GAP
    starts, lengths = starts[long_sel], lengths[long_sel]
    mean_speed = np.nanmean(players[:, :, 2], axis=1)
    oob = (bx < 0) | (bx > LENGTH) | (by < 0) | (by > WIDTH)

    rows = []
    for s, l in zip(starts, lengths):
        e = s + l
        before = slice(max(s - FPS, 0), s)                  # last second before the gap
        last = np.flatnonzero(visible[:s])
        nxt = np.flatnonzero(visible[e:e + 10 * FPS])
        lx, ly = (bx[last[-1]], by[last[-1]]) if last.size else (np.nan, np.nan)
        rx, ry = (bx[e + nxt[0]], by[e + nxt[0]]) if nxt.size else (np.nan, np.nan)
        rows.append(dict(
            length=l,
            live_frac=live_tag[s:e].mean(),
            sp_near_end=sp_start[max(e - 10 * FPS, s):min(e + 5 * FPS, n)].any(),
            sp_inside=sp_start[s:e].any(),
            out_before=oob[before].any(),
            last_line=dist_to_line(lx, ly), last_box=bool(in_box(lx, ly)),
            reap_line=dist_to_line(rx, ry), reap_centre=bool(near_centre(rx, ry)),
            speed=np.nanmean(mean_speed[s:e]),
        ))

    if not rows:
        print("no long gaps")
        return
    L = np.array([r["length"] for r in rows])
    live = np.array([r["live_frac"] for r in rows])
    spe = np.array([r["sp_near_end"] for r in rows])
    spi = np.array([r["sp_inside"] for r in rows])
    outb = np.array([r["out_before"] for r in rows])
    gspeed = np.array([r["speed"] for r in rows])
    print("number of long gaps: {}   total frames: {}".format(len(rows), int(L.sum())))

    vis_g = visible & graded
    miss_g = ~visible & graded
    print("\nshare of frames covered by a phase-of-play tag (= sign the ball is still in play):")
    print("  ball visible: {}   ball missing: {}".format(
        pct(live_tag[vis_g].mean()), pct(live_tag[miss_g].mean())))
    print("mean player speed (m/s):")
    print("  ball visible: {}".format(percentiles(mean_speed[vis_g])))
    print("  ball missing: {}".format(percentiles(mean_speed[miss_g])))

    print("\nper gap (n = number of gaps, frames = share of gap frames):")
    print("  {:<52} {:>5} {:>8}".format("feature", "n", "frames"))

    def line(name, sel):
        print("  {:<52} {:>5} {:>8}".format(name, int(sel.sum()), pct(L[sel].sum() / L.sum())))

    line("set piece starts inside the gap", spi)
    line("set piece starts near the gap end (-10 s..+5 s)", spe)
    line("ball out of play in the 1 s before the gap", outb)
    line(">= 50% of the gap covered by a phase tag", live >= .5)
    line("< 10% of the gap covered by a phase tag", live < .1)
    line("players slow on average (< 1.5 m/s) during the gap", gspeed < 1.5)
    dead_like = spe | spi | outb
    line("HAS a dead-ball sign (set piece / ball out)", dead_like)
    line("NO dead-ball sign AND >= 50% covered by a tag", ~dead_like & (live >= .5))

    print("\nby gap length:")
    print("  {:<14} {:>4} {:>16} {:>16} {:>14}".format(
        "length", "n", "dead-ball sign", "tag >=50%", "players slow"))
    for lo, hi, name in ((125, 250, "5-10 s"), (251, 500, "10-20 s"), (501, 1500, "20-60 s"),
                         (1501, 10 ** 9, "> 60 s")):
        sel = (L >= lo) & (L <= hi)
        if sel.any():
            print("  {:<14} {:>4} {:>16} {:>16} {:>14}".format(
                name, int(sel.sum()), int(dead_like[sel].sum()), int((live[sel] >= .5).sum()),
                int((gspeed[sel] < 1.5).sum())))

    # -- C. before and after a gap ---------------------------------------------------------------
    header("C. BALL POSITION BEFORE AND AFTER A LONG GAP")
    last_line = np.array([r["last_line"] for r in rows])
    last_box = np.array([r["last_box"] for r in rows])
    reap_line = np.array([r["reap_line"] for r in rows])
    reap_centre = np.array([r["reap_centre"] for r in rows])
    print("last seen - distance to a line (m): " + percentiles(last_line))
    print("  within 2 m of a line / outside: {}   in a penalty box: {}".format(
        pct(np.nanmean(last_line < 2)), pct(np.mean(last_box))))
    print("reappears (within 10 s) - distance to a line (m): " + percentiles(reap_line))
    print("  within 2 m of a line / outside: {}   near the centre spot (kick-off): {}".format(
        pct(np.nanmean(reap_line < 2)), pct(np.mean(reap_centre))))
    print("  does not reappear within 10 s: {}".format(pct(np.mean(np.isnan(reap_line)))))

    print("\ndone. Every number above is a summary; safe to share with the team.")


if __name__ == "__main__":
    main()
