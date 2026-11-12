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
    kinds = {"touch (impulse)": 0, "dribble (control)": 0, "tag while ball lost": 0}
    for t in spurious:
        if not f["vis"][t]:
            kinds["tag while ball lost"] += 1
        elif np.nan_to_num(f["impulse"][t]) > P.IMPULSE:
            kinds["touch (impulse)"] += 1
        else:
            kinds["dribble (control)"] += 1
    print("\nours team changes not in the ground truth: {}".format(len(spurious)))
    for k, v in kinds.items():
        print("   triggered by {:<22} {}".format(k, v))

    # -- 3. restarts -----------------------------------------------------------------------------
    print("\n" + "=" * 100 + "\n3. AFTER A DEAD BALL: TEAM GIVEN BY ours\n" + "=" * 100)
    for s, e, code in segments(gt, lo, hi):
        if code == 2 and e < hi:
            first_live = next((t for t in range(e, min(e + 10 * sec, hi)) if labels[t] in (0, 1)), None)
            print("restart to {}: ours gives {}".format(
                NAME[int(gt[e])], NAME[int(labels[first_live])] if first_live else "-"))


if __name__ == "__main__":
    main()
