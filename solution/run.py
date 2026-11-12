"""Regenerate our submission from the provided data -- the one command a reviewer runs.

    .\\.venv\\Scripts\\python.exe solution\\run.py                 # -> evaluation/results/v1.csv
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
from possession import DEAD, NONE, assign, episodes, touch_frames  # noqa: E402
import tags as tagging  # noqa: E402

LETTERS = {0: "H", 1: "A", DEAD: "D", NONE: "X"}


def predict(arrays, tag_data, halves):
    length, width = GAMES[GAME]["pitch"]
    f = compute(arrays, length, width)
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
    print("\nringkasan (frame dinilai: {})".format(int(graded.sum())))
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
    print("  possession: {}   rata-rata {:.1f} s   median {:.1f} s".format(
        n_poss, np.mean(durations) if durations else 0, np.median(durations) if durations else 0))
    print("  bola mati (rangkaian D): {}".format(n_dead))
    eps = episodes(touch_frames(f))
    minutes = graded.sum() / P.FPS / 60
    print("  episode sentuhan: {:.1f} per menit".format(len(eps) / minutes))
    act = f["activity_s"][graded]
    vis = f["vis"][graded]
    print("  aktivitas pemain (m/s, dihitung) bola terlihat: p25={:.2f} p50={:.2f} p75={:.2f}".format(
        *np.nanpercentile(act[vis], [25, 50, 75])))
    print("  aktivitas pemain (m/s, dihitung) bola hilang:   p25={:.2f} p50={:.2f} p75={:.2f}".format(
        *np.nanpercentile(act[~vis], [25, 50, 75])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v1", help="file name on the board (no .csv)")
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
    print("\nditulis: {}".format(os.path.relpath(path, ROOT)))
    print("cek:     .\\.venv\\Scripts\\python.exe evaluation\\validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))
    print("skor:    .\\.venv\\Scripts\\python.exe evaluation\\run_evaluation.py")


if __name__ == "__main__":
    main()
