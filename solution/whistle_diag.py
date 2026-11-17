"""Why does whistle.py find so few whistles? Aggregates only, from data/derived/*_whistle.npz
(no re-decoding, no ground truth).

    .\\.venv\\Scripts\\python.exe solution\\whistle_diag.py

1. Where in the band the spectral peak sits, for all graded frames and for the most tonal ones.
   A referee uses one whistle all match, so true whistles pile up at one pitch; a steady
   background tone (a hum, a PA, a camera) shows up at one pitch on ALL frames.
2. How many runs survive at looser thresholds, with and without the pitch-stability filter.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402

import params as P  # noqa: E402
from deadball import close_holes, runs  # noqa: E402

PATH = os.path.join(ROOT, "data", "derived", "{}_whistle.npz".format(GAME))


def main():
    d = np.load(PATH)
    ton, peak, level = d["tonality"], d["peak"], d["level"]
    n = len(ton)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    ok = graded & np.isfinite(ton)
    t, p = ton[ok], peak[ok]
    minutes = ok.sum() / P.FPS / 60

    edges = np.arange(2000, 4600, 100)
    print("share of frames whose band peak is in each 100 Hz bin (graded frames)")
    print("  bin Hz    all   >p90   >p99  >p99.9")
    cols = []
    for q in (None, 90, 99, 99.9):
        sel = p if q is None else p[t > np.percentile(t, q)]
        h, _ = np.histogram(sel, bins=edges)
        cols.append(h / max(len(sel), 1))
    for i, lo in enumerate(edges[:-1]):
        row = [c[i] for c in cols]
        if max(row) >= 0.03:
            print("  {:>4}-{:<4} {:5.1%} {:6.1%} {:6.1%} {:6.1%}".format(lo, lo + 100, *row))

    med = np.median(t)
    mad = 1.4826 * np.median(np.abs(t - med))
    lv = level[ok]
    print("\ntonality median {:.1f} dB, robust sd {:.2f};  band level p10 {:.1f} p50 {:.1f} p90 {:.1f} dB".format(
        med, mad, *np.percentile(lv, [10, 50, 90])))
    print("\nruns >= {:.2f} s per minute at looser thresholds".format(P.WHISTLE_MIN_S))
    print("    K   thr dB   no pitch filter   pitch sd <= {:.0f} Hz   median pitch of kept".format(P.WHISTLE_PITCH_SD))
    tt = np.where(graded, np.nan_to_num(ton, nan=-np.inf), -np.inf)
    for k in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
        thr = med + k * mad
        above = close_holes(tt > thr, P.WHISTLE_GAP)
        allr, kept, pitches = 0, 0, []
        for s, l in zip(*runs(above)):
            if l < P.WHISTLE_MIN_S * P.FPS:
                continue
            allr += 1
            pk = peak[s:s + l][tt[s:s + l] > thr]
            if pk.size and np.std(pk) <= P.WHISTLE_PITCH_SD:
                kept += 1
                pitches.append(np.median(pk))
        print("  {:>3}   {:5.1f}   {:6.2f} ({:>5})      {:6.2f} ({:>5})         {}".format(
            k, thr, allr / minutes, allr, kept / minutes, kept,
            "{:.0f} Hz".format(np.median(pitches)) if pitches else "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
