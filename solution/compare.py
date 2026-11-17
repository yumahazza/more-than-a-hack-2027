"""Paired bootstrap: is submission B better than submission A on the public window, beyond noise?

    .\\.venv\\Scripts\\python.exe solution\\compare.py v9 v12

run_evaluation.py --bootstrap gives one interval per submission. Two intervals that overlap do not
mean the DIFFERENCE is noise: both submissions are scored on the same 17 possessions, so their
errors move together. The right question is the interval of the difference B - A, with A and B
resampled on the SAME draws (a paired bootstrap). This uses the scorer's own functions and its
resampling units (whole possessions for spellF1, whole segments for macroF1).

Only the public ground truth and our own submission files are read. It answers "is the gap real
on these 5 minutes"; it does not make 5 minutes representative of the match.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS = os.path.join(ROOT, "evaluation", "results")
DRAWS = 5000
SEED = 2027


def load(path):
    return pm.label_codes(pm.read_label_csv(path))


def spell_flags(gt, pred, runs):
    gt_spells, pred_spells = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), t) for t in pm.SPELL_IOUS])
    return flags, len(gt_spells), len(pred_spells)


def spell_score(flags, n_gt, n_pred, picks):
    f1s = []
    for row in flags:
        tp = int(row[picks].sum())
        fn = n_gt - tp
        fp = max(0, n_pred - tp)
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def segment_contrib(gt, pred, runs):
    segs = pm.segments(gt, runs)
    codes = np.array([c for _, _, c in segs], dtype=np.int64)
    contrib = np.zeros((len(segs), pm.N_LABELS), dtype=np.int64)
    for i, (s, e, _) in enumerate(segs):
        contrib[i] = np.bincount(pred[s:e].astype(np.int64), minlength=pm.N_LABELS)
    return codes, contrib


def macro_score(codes, contrib, picks):
    cm3 = np.zeros((pm.N_SCORED, pm.N_LABELS), dtype=np.int64)
    np.add.at(cm3, codes[picks], contrib[picks])
    return pm.macro_f1(cm3)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    a_name, b_name = sys.argv[1], sys.argv[2]
    gt = load(GT)
    runs = pm.valid_runs(gt)
    a = load(os.path.join(RESULTS, a_name + ".csv"))
    b = load(os.path.join(RESULTS, b_name + ".csv"))
    rng = np.random.RandomState(SEED)

    fa, n_gt, npa = spell_flags(gt, a, runs)
    fb, _, npb = spell_flags(gt, b, runs)
    ca_codes, ca = segment_contrib(gt, a, runs)
    _, cb = segment_contrib(gt, b, runs)
    n_seg = len(ca_codes)

    d_spell, d_macro = np.empty(DRAWS), np.empty(DRAWS)
    for k in range(DRAWS):
        p = rng.randint(0, n_gt, n_gt)
        d_spell[k] = spell_score(fb, n_gt, npb, p) - spell_score(fa, n_gt, npa, p)
        q = rng.randint(0, n_seg, n_seg)
        d_macro[k] = macro_score(ca_codes, cb, q) - macro_score(ca_codes, ca, q)

    full = np.arange(n_gt)
    seg_all = np.arange(n_seg)
    print("paired bootstrap, {} draws, public window ({} possessions, {} segments)".format(DRAWS, n_gt, n_seg))
    for name, d, point in (
            ("spellF1", d_spell, spell_score(fb, n_gt, npb, full) - spell_score(fa, n_gt, npa, full)),
            ("macroF1", d_macro, macro_score(ca_codes, cb, seg_all) - macro_score(ca_codes, ca, seg_all))):
        lo, hi = np.percentile(d, [5, 95])
        print("  {} {} - {}: {:+.4f}   5-95%: {:+.4f} .. {:+.4f}   P(diff <= 0) = {:.1%}".format(
            name, b_name, a_name, point, lo, hi, np.mean(d <= 0)))
    print("  (an interval entirely above 0 means {} is better than {} beyond resampling noise)".format(
        b_name, a_name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
