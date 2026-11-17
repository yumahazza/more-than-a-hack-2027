"""Calibrate the team-change parameters on a DIFFERENT, labelled, public match (Metrica Sample
Game 3), never on the competition match.

    .\\.venv\\Scripts\\python.exe solution\\calibrate_external.py <folder with Sample_Game_3_*>

RULES.md: "Hand-labelled data from other games is fine, in any amount" to train or validate. The
competition's public 5 minutes are NOT used here; after the values are chosen they are only a
safety check, together with the full-match plausibility in run.py.

Only the three values that decide how readily the team changes are searched, all three stated
assumptions so far:
  EVIDENCE_CAP       evidence per frame with the ball in one player's possession zone
  MEAN_POSSESSION_S  mean possession length, which sets the change cost
  TOUCH_SHARE        share of changes happening with the ball within reach (touch gate)
Dead-ball parameters are left alone: the event-based dead ball of this match is defined too
differently from the competition's ground truth.

Criterion, fixed before running: highest spellF1 on this match, macroF1 as tie-break. The grid's
neighbourhood of the winner is printed, to see whether it sits on a plateau or a spike.
"""

import itertools
import sys

import numpy as np

import params as P
import tags as tagging
from deadball import detect
from external_validation import build_labels, load_tracking, to_codes, pm
from features import compute
from possession import assign

GRID = {
    "EVIDENCE_CAP": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "MEAN_POSSESSION_S": [8.0, 10.0, 12.0, 15.0],
    "TOUCH_SHARE": [0.8, 0.9, 0.95, 0.99],
}
V12 = {"USE_GK_FIX": True, "USE_OOB_GAP": True, "USE_TOUCH_GATE": True}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    folder = sys.argv[1]
    arrays, length, width, team_ids = load_tracking(folder)
    n = len(arrays["ball"])
    gt, halves = build_labels(folder, n, team_ids)
    gt_codes = to_codes(gt)
    for k, v in V12.items():
        setattr(P, k, v)
    tag_data = tagging.build(n, events=[])
    f = compute(arrays, length, width, halves)
    dead = detect(f, tag_data, halves)

    saved = {k: getattr(P, k) for k in list(GRID) + ["SWITCH_COST"]}
    rows = []
    try:
        for values in itertools.product(*GRID.values()):
            setting = dict(zip(GRID, values))
            for k, v in setting.items():
                setattr(P, k, v)
            P.SWITCH_COST = float(np.log(P.MEAN_POSSESSION_S * P.FPS))
            ff = dict(f)                      # assign() writes model info into the dict
            labels = assign(ff, dead, tag_data, halves)
            r = pm.evaluate(gt_codes, to_codes(labels), bootstrap_n=0)
            rows.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                         r["mean_spell_duration_s"], setting))
            print("  cap {:<5} T {:<5} touch {:<5} -> spellF1 {:.4f}  macroF1 {:.4f}  poss {:>4}".format(
                setting["EVIDENCE_CAP"], setting["MEAN_POSSESSION_S"], setting["TOUCH_SHARE"],
                r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]), flush=True)
    finally:
        for k, v in saved.items():
            setattr(P, k, v)

    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    current = next(r for r in rows if r[4] == {"EVIDENCE_CAP": saved["EVIDENCE_CAP"],
                                                 "MEAN_POSSESSION_S": saved["MEAN_POSSESSION_S"],
                                                 "TOUCH_SHARE": saved["TOUCH_SHARE"]})
    print("\ntruth: {} possessions of mean {:.1f} s".format(
        pm.evaluate(gt_codes, gt_codes, bootstrap_n=0)["n_gt_spells"],
        pm.evaluate(gt_codes, gt_codes, bootstrap_n=0)["gt_mean_spell_duration_s"]))
    print("current (v12): spellF1 {:.4f} macroF1 {:.4f} poss {} mean {:.1f} s  {}".format(*current))
    print("top 10:")
    for r in rows[:10]:
        print("  spellF1 {:.4f} macroF1 {:.4f} poss {:>4} mean {:>5.1f} s  {}".format(*r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
