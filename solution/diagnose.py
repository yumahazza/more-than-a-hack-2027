"""Why does ours go wrong on the public 5 minutes? Mechanisms, not scores.

    .\\.venv\\Scripts\\python.exe solution\\diagnose.py

Uses the PUBLIC ground truth (frames 11185-18684, shipped in this repository) only to see WHICH
part of the logic fails: dead ball starting late, ending early, a touch credited to the wrong team.
It must never be used to pick a threshold value (RULES.md). Fixes are argued from football logic
and checked on the whole match.

Prints short summaries per ground-truth segment and per transition, nothing per frame.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, source_frames  # noqa: E402

import params as P  # noqa: E402
import tags as tagging  # noqa: E402
from atd import load_arrays  # noqa: E402
from deadball import close_holes, rolling_max, runs, sustained  # noqa: E402
from run import predict  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
CODE = {"H": 0, "A": 1, "D": 2, "X": -1}
NAME = {0: "H", 1: "A", 2: "D", -1: "X"}


def load_gt(n):
    labels = np.full(n, -1, dtype=np.int8)
    with open(GT_PATH) as handle:
        next(handle)
        for line in handle:
            frame, label = line.strip().split(",")
            labels[int(frame)] = CODE[label.strip().upper()]
    return labels


def segments(codes, lo, hi):
    out, s = [], lo
    for t in range(lo + 1, hi + 1):
        if t == hi or codes[t] != codes[s]:
            out.append((s, t, int(codes[s])))
            s = t
    return out


def phase_stats(f, tag_data, a, b):
    a, b = max(a, 0), min(b, f["n"])
    if b <= a:
        return "-"
    vis = f["vis"][a:b]
    spd = f["speed"][a:b][vis]
    out = f["outside"][a:b][vis]
    sp = [s["type"] or "?" for s in tag_data["setpieces"] if s["start"] < b and s["end"] > a]
    return "vis {:3.0f}%  ball {:>5}  out {:>5}  players {:4.2f}  {}".format(
        100 * vis.mean(),
        "{:.1f}".format(np.nanmedian(spd)) if spd.size and np.isfinite(spd).any() else "-",
        "{:+.1f}".format(np.nanmax(out)) if out.size else "-",
        np.nanmean(f["activity_s"][a:b]),
        ("SP:" + "/".join(sorted(set(sp)))) if sp else "")


def stoppage_flags(f, tag_data):
    """The per-frame conditions of the stoppage rule in deadball.py, so a miss can be explained."""
    slow = np.nan_to_num(f["activity_s"], nan=np.inf) < P.SLOW_PLAYERS
    passing = sustained(f["vis"] & (np.nan_to_num(f["speed"], nan=0.0) > P.PASS_SPEED), P.FAST_MIN)
    quiet_ball = ~rolling_max(passing, P.QUIET_WINDOW)
    quiet = close_holes(slow & quiet_ball, P.CLOSE_HOLES)
    return {"slow": slow, "passing": passing, "quiet_ball": quiet_ball, "quiet": quiet,
            "phase": tag_data["phase"]}


def why_dead_missed(gt, labels, f, tag_data, lo, hi):
    print("\n" + "=" * 100 + "\n1b. WHY THE STOPPAGE RULE DID OR DID NOT FIRE\n" + "=" * 100)
    print("share of the ground-truth D stretch where each condition holds:")
    print("  slow = players under {} m/s, pass = a sustained ball over {} m/s,".format(
        P.SLOW_PLAYERS, P.PASS_SPEED))
    print("  quiet = slow and no pass within 1 s, phase = covered by a phase-of-play tag")
    flags = stoppage_flags(f, tag_data)
    sec = P.FPS
    print("  {:<10} {:>6} {:>6} {:>6} {:>6} {:>7} {:>16}".format(
        "D length", "ours", "slow", "pass", "quiet", "phase", "longest quiet"))
    for s, e, code in segments(gt, lo, hi):
        if code != 2:
            continue
        q = flags["quiet"][s:e]
        longest = int(runs(q)[1].max()) if q.any() else 0
        print("  {:<10} {:>5.0f}% {:>5.0f}% {:>5.0f}% {:>5.0f}% {:>6.0f}% {:>14.1f} s".format(
            "{:.1f} s".format((e - s) / sec), 100 * np.mean(labels[s:e] == 2),
            100 * flags["slow"][s:e].mean(), 100 * flags["passing"][s:e].mean(),
            100 * q.mean(), 100 * flags["phase"][s:e].mean(), longest / sec))
    print("  rule: quiet for {} s with a set-piece tag / ball out nearby, or {} s with phase <= {:.0f}%".format(
        P.STOP_MIN // sec, P.STOP_ALONE // sec, 100 * P.LIVE_TAG_MAX))


def touches_near_changes(gt, labels, f, lo, hi):
    print("\n" + "=" * 100 + "\n2b. WHAT THE KICK DETECTOR AND ZONES SAW AROUND EACH TEAM CHANGE (+/- 2 s)\n" + "=" * 100)
    print("peak = local maximum of the Savitzky-Golay ball acceleration >= {} m/s^2.".format(P.KICK_ACCEL))
    print("kick = a peak with the nearest player within {} m and no duel (credited to his team);".format(
        P.R_POSSESSION))
    print("far = peak with nobody within {} m; duel = peak while both teams are within {} m.".format(
        P.R_POSSESSION, P.R_DUEL))
    print("zone = frames with the ball in that team's possession zone (evidence); duel% = duel frames.")
    sec = P.FPS
    vis, near_d, team, duel = f["vis"], f["near_d"], f["near_team"], f["duel"]
    a_ = np.nan_to_num(f["accel"], nan=0.0)
    peak = np.zeros(len(a_), dtype=bool)
    peak[1:-1] = (a_[1:-1] >= a_[:-2]) & (a_[1:-1] > a_[2:])
    peak &= vis & (a_ >= P.KICK_ACCEL)
    zone = vis & np.isfinite(near_d) & (near_d <= P.R_POSSESSION) & ~duel
    print("\n{:<8} {:>9} | {:^18} | {:^18} | {:>6} {:>8} {:>8}".format(
        "change", "ours", "NEW team", "OLD team", "far", "duel pk", "duel%"))
    print("{:<8} {:>9} | {:>8} {:>8} | {:>8} {:>8} |".format("", "", "kicks", "zone", "kicks", "zone"))
    for g in range(lo + 1, hi):
        if not (gt[g] in (0, 1) and gt[g - 1] in (0, 1) and gt[g] != gt[g - 1]):
            continue
        new = int(gt[g])
        a, b = g - 2 * sec, g + 2 * sec
        changed = [t for t in range(max(a, lo + 1), min(b, hi))
                   if labels[t] == new and labels[t - 1] == 1 - new]
        status = "{:+.1f} s".format((changed[0] - g) / sec) if changed else "MISSED"
        kt = f["kick_team"][a:b]
        cols = []
        for side in (new, 1 - new):
            cols += [int(np.sum(kt == side)), int(np.sum(zone[a:b] & (team[a:b] == side)))]
        far = int(np.sum(peak[a:b] & ~(np.nan_to_num(near_d[a:b], nan=99) <= P.R_POSSESSION)))
        dpk = int(np.sum(peak[a:b] & duel[a:b]))
        dshare = 100 * duel[a:b][vis[a:b]].mean() if vis[a:b].any() else 0
        print("{:<8} {:>9} | {:>8} {:>8} | {:>8} {:>8} | {:>6} {:>8} {:>7.0f}%".format(
            "-> " + NAME[new], status, *cols, far, dpk, dshare))


def main():
    n = source_frames("tactical", GAME)
    arrays = load_arrays()
    halves = half_video_blocks(GAME)
    tag_data = tagging.build(n)
    labels, f = predict(arrays, tag_data, halves)
    gt = load_gt(n)
    graded = np.flatnonzero(gt >= 0)
    lo, hi = int(graded[0]), int(graded[-1]) + 1
    sec = P.FPS

    print("public window: frames {}-{}".format(lo, hi - 1))
    print("columns: vis = % of frames with the ball visible, ball = median ball speed (m/s),")
    print("         out = furthest the ball is beyond a line (m, + = outside),")
    print("         players = player activity (m/s), SP = overlapping set-piece tags")

    # -- 1. dead-ball segments -------------------------------------------------------------------
    print("\n" + "=" * 100 + "\n1. DEAD BALL IN THE GROUND TRUTH vs ours\n" + "=" * 100)
    for s, e, code in segments(gt, lo, hi):
        if code != 2:
            continue
        ours = labels[s:e] == 2
        pred_d = np.flatnonzero(labels[max(s - 10 * sec, lo):min(e + 10 * sec, hi)] == 2)
        if pred_d.size:
            ps = max(s - 10 * sec, lo) + int(pred_d[0])
            pe = max(s - 10 * sec, lo) + int(pred_d[-1]) + 1
            offs = "ours D starts {:+.1f} s, ends {:+.1f} s".format((ps - s) / sec, (pe - e) / sec)
        else:
            offs = "ours gives no D anywhere near here"
        before = NAME[int(gt[s - 1])] if s > lo else "-"
        after = NAME[int(gt[e])] if e < hi else "-"
        print("\nD {:.1f} s  ({} -> D -> {})   ours marks {:.0f}% of it as D.  {}".format(
            (e - s) / sec, before, after, 100 * ours.mean(), offs))
        third = max((e - s) // 3, 1)
        for name, a, b in (("5 s before", s - 5 * sec, s), ("start of D", s, s + third),
                           ("middle of D", s + third, e - third), ("end of D", e - third, e),
                           ("5 s after", e, e + 5 * sec)):
            share = 100 * np.mean(labels[max(a, 0):b] == 2) if b > a else 0
            print("   {:<12} ours D {:3.0f}%   {}".format(name, share, phase_stats(f, tag_data, a, b)))

    why_dead_missed(gt, labels, f, tag_data, lo, hi)

    # -- 2. team changes in live play ------------------------------------------------------------
    print("\n" + "=" * 100 + "\n2. TEAM CHANGES (ball in play) IN THE GROUND TRUTH vs ours\n" + "=" * 100)
    pred_changes = [t for t in range(lo + 1, hi)
                    if labels[t] in (0, 1) and labels[t - 1] in (0, 1) and labels[t] != labels[t - 1]]
    gt_changes = [t for t in range(lo + 1, hi)
                  if gt[t] in (0, 1) and gt[t - 1] in (0, 1) and gt[t] != gt[t - 1]]
    print("ground truth: {}   ours: {}".format(len(gt_changes), len(pred_changes)))
    print("{:<12} {:>10} {:>14} {:>10} {:>10} {:>14}".format(
        "change", "ours offset", "ball visible", "nearest", "gap", "nearest team"))
    used = set()
    for g in gt_changes:
        to = NAME[int(gt[g])]
        near = [t for t in pred_changes if abs(t - g) <= 3 * sec and NAME[int(labels[t])] == to]
        best = min(near, key=lambda t: abs(t - g)) if near else None
        if best is not None:
            used.add(best)
        w = slice(g - sec // 2, g + sec // 2)
        vis = f["vis"][w]
        nd = f["near_d"][w][vis]
        mg = f["margin"][w][vis]
        nt = f["near_team"][w][vis]
        print("{:<12} {:>10} {:>13.0f}% {:>10} {:>10} {:>14}".format(
            "-> " + to,
            "{:+.1f} s".format((best - g) / sec) if best is not None else "MISSED",
            100 * vis.mean(),
            "{:.1f} m".format(np.nanmedian(nd)) if nd.size else "-",
            "{:.1f} m".format(np.nanmedian(mg[np.isfinite(mg)])) if np.isfinite(mg).any() else "-",
            "{:.0f}% {}".format(100 * np.mean(nt == CODE[to]), to) if nt.size else "-"))

    spurious = [t for t in pred_changes if t not in used]
    # what made the model change team there (v5 evidence types)
    kinds = {"a kick by the new team": 0, "possession-zone frames": 0, "ball lost (tags)": 0}
    half_s = P.FPS // 2
    for t in spurious:
        new = int(labels[t])
        if np.any(f["kick_team"][max(t - half_s, 0):t + half_s] == new):
            kinds["a kick by the new team"] += 1
        elif not f["vis"][t]:
            kinds["ball lost (tags)"] += 1
        else:
            kinds["possession-zone frames"] += 1
    print("\nours team changes not in the ground truth: {}".format(len(spurious)))
    for k, v in kinds.items():
        print("   triggered by {:<22} {}".format(k, v))

    touches_near_changes(gt, labels, f, lo, hi)

    # -- 3. restarts -----------------------------------------------------------------------------
    print("\n" + "=" * 100 + "\n3. AFTER A DEAD BALL: TEAM GIVEN BY ours\n" + "=" * 100)
    for s, e, code in segments(gt, lo, hi):
        if code == 2 and e < hi:
            first_live = next((t for t in range(e, min(e + 10 * sec, hi)) if labels[t] in (0, 1)), None)
            print("restart to {}: ours gives {}".format(
                NAME[int(gt[e])], NAME[int(labels[first_live])] if first_live else "-"))


if __name__ == "__main__":
    main()
