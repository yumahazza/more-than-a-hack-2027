"""Regenerate our submission from the provided data -- the one command a reviewer runs.

    .\\.venv\\Scripts\\python.exe solution\\run.py                 # -> evaluation/results/v12.csv
    .\\.venv\\Scripts\\python.exe solution\\run.py --name team     # -> evaluation/results/team.csv

Pipeline: tracking + Smart Tagging -> features -> dead ball -> possession -> CSV.
Prints only aggregate numbers (counts, shares, percentiles), never per-frame output.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402

import params as P  # noqa: E402
from atd import load_arrays  # noqa: E402
from deadball import detect, runs  # noqa: E402
from features import compute  # noqa: E402
from possession import DEAD, NONE, assign  # noqa: E402
import tags as tagging  # noqa: E402

LETTERS = {0: "H", 1: "A", DEAD: "D", NONE: "X"}


def predict(arrays, tag_data, halves):
    length, width = GAMES[GAME]["pitch"]
    f = compute(arrays, length, width, halves)
    dead = detect(f, tag_data, halves)
    labels = assign(f, dead, tag_data, halves)
    return labels, f


def write_csv(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, LETTERS[int(c)]) for i, c in enumerate(labels))


def summary(labels, f, halves):
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in halves:
        graded[first:last + 1] = True
    g = labels[graded]
    print("\nsummary (graded frames: {})".format(int(graded.sum())))
    for code in (0, 1, DEAD, NONE):
        print("  {}  {:6.1f}%".format(LETTERS[code], 100.0 * np.mean(g == code)))
    n_poss, durations, n_dead = 0, [], 0
    for _, first, last in halves:
        seg = labels[first:last + 1]
        for side in (0, 1):
            starts, lengths = runs(seg == side)
            n_poss += len(starts)
            durations += list(lengths / P.FPS)
        n_dead += len(runs(seg == DEAD)[0])
    print("  possessions: {}   mean {:.1f} s   median {:.1f} s".format(
        n_poss, np.mean(durations) if durations else 0, np.median(durations) if durations else 0))
    print("  dead-ball stretches (runs of D): {}".format(n_dead))
    minutes = graded.sum() / P.FPS / 60
    c = f["calib"]
    print("  calibrated: possession/duel zone {:.2f} m (p{} nearest {:.2f} m, cap = median second-nearest "
          "{:.2f} m; {} slow-ball live frames)".format(c["r_zone"], P.CAL_ZONE_PCT, c["r_zone_raw"],
                                                     c["r_zone_cap"], c["n_slow_live"]))
    print("  calibrated: kick threshold {:.1f} m/s^2 (p{} over graded frames), kick evidence {:.2f}".format(
        c["kick_accel"], P.CAL_KICK_PCT, P.KICK_EVIDENCE))
    g = f.get("gk", {})
    print("  v12 goalkeeper fix: {} goalkeeper slots, {} slot-frames credited to the other team".format(
        len(g.get("gk_slots", [])), g.get("reassigned_frames", 0)))
    print("  v12 ball out while lost: {} extra out-of-play events".format(f.get("oob_gap_events", 0)))
    tg = f.get("touch_gate")
    if tg is None:
        print("  v12 touch gate: not used")
    else:
        print("  v12 touch gate: live frames touch {:.1%} / ball away {:.1%} / lost {:.1%} -> change cost "
              "{:.2f} / {:.2f} / {:.2f}".format(tg["f_touch"], tg["f_away"], tg["f_lost"],
                                               tg["cost_touch"], tg["cost_away"], tg["cost_lost"]))
    m = f.get("shape_model")
    if m is None:
        print("  team shape model: not used")
    else:
        print("  team shape model: {} pseudo-labelled frames, slope {:+.3f} per m (+ = wider home -> home), "
              "agreement {:.1f}%, correlation time {:.1f} s".format(
                  m["n_train"], m["b"], 100 * m["agree"], m["tau"] / P.FPS))
    j = f.get("joint_model")
    if j is None:
        print("  joint H/A/D model: not used")
    else:
        print("  joint H/A/D model: {} dead + {} live pseudo-labelled frames, agreement {:.1f}%, "
              "correlation time {:.1f} s".format(j["n_dead"], j["n_live"], 100 * j["agree"],
                                                 j["tau"] / P.FPS))
        print("    first-pass mean live stretch {:.1f} s, mean dead stretch {:.1f} s -> "
              "cost to dead {:.2f}, from dead {:.2f}".format(
                  j["mean_alive_s"], j["mean_dead_s"], j["cost_to_dead"], j["cost_from_dead"]))
        print("    dead-ball cue weights (per standard deviation, + = more likely dead):")
        for name, w in zip(j["names"], j["w"][1:]):
            print("      {:<28} {:+.2f}".format(name, w))
    kicks = f["kick_team"][graded]
    vis_g = f["vis"][graded]
    print("  kicks credited: {:.1f} per minute (home {:.0f}%)   duel frames: {:.1f}% of ball-visible".format(
        np.sum(kicks >= 0) / minutes, 100 * np.mean(kicks[kicks >= 0] == 0) if np.any(kicks >= 0) else 0,
        100 * np.mean(f["duel"][graded][vis_g])))
    acc = f["accel"][graded][vis_g]
    print("  ball acceleration (m/s^2, Savitzky-Golay): p50={:.1f} p90={:.1f} p99={:.1f}".format(
        *np.nanpercentile(acc, [50, 90, 99])))
    act = f["activity_s"][graded]
    vis = f["vis"][graded]
    print("  player activity (m/s, computed) ball visible: p25={:.2f} p50={:.2f} p75={:.2f}".format(
        *np.nanpercentile(act[vis], [25, 50, 75])))
    print("  player activity (m/s, computed) ball missing: p25={:.2f} p50={:.2f} p75={:.2f}".format(
        *np.nanpercentile(act[~vis], [25, 50, 75])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v12", help="file name on the board (no .csv)")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = parser.parse_args()

    arrays = load_arrays()
    n = source_frames("tactical", GAME)
    assert len(arrays["ball"]) == n
    halves = half_video_blocks(GAME)
    labels, f = predict(arrays, tagging.build(n), halves)

    path = os.path.join(args.out, args.name + ".csv")
    write_csv(path, labels)
    summary(labels, f, halves)
    print("\nwritten:  {}".format(os.path.relpath(path, ROOT)))
    print("validate: .\\.venv\\Scripts\\python.exe evaluation\\validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))
    print("score:    .\\.venv\\Scripts\\python.exe evaluation\\run_evaluation.py")


if __name__ == "__main__":
    main()
