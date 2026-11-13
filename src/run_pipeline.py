"""End-to-end entry point for the v1 modular possession inference pipeline.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --name my_v1 --no-cache
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME  # noqa: E402
from src.atd_features import extract_atd_features  # noqa: E402
from src.atd_loader import load_atd_arrays  # noqa: E402
from src.possession_model import predict_possession  # noqa: E402
from src.smart_tagging_features import extract_smart_tagging_features  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def write_submission(path, labels):
    """Write frame,label CSV matching the competition submission contract."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="v1_physics_viterbi",
                        help="output submission stem (without .csv)")
    parser.add_argument("--out", default=RESULTS_DIR, help="output directory")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore and bypass cached ATD npz")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("[1/4] Loading ATD arrays...")
    atd = load_atd_arrays(game=GAME, use_cache=not args.no_cache)
    print(f"      done in {time.perf_counter() - t0:.2f}s ({atd['n_frames']} frames)")

    t1 = time.perf_counter()
    print("[2/4] Extracting ATD physics & proximity features...")
    atd_feats = extract_atd_features(atd)
    print(f"      done in {time.perf_counter() - t1:.2f}s")

    t2 = time.perf_counter()
    print("[3/4] Extracting Smart Tagging tactical priors & set-piece anchors...")
    st_feats = extract_smart_tagging_features(atd["n_frames"], game=GAME)
    print(f"      done in {time.perf_counter() - t2:.2f}s ({len(st_feats['set_pieces'])} set pieces)")

    t3 = time.perf_counter()
    print("[4/4] Running dead-ball scoring, passing-bridge & 3-state Viterbi decoder...")
    labels = predict_possession(atd_feats, st_feats, game=GAME)
    out_path = write_submission(os.path.join(args.out, f"{args.name}.csv"), labels)
    print(f"      done in {time.perf_counter() - t3:.2f}s")

    counts = {lbl: int(np.sum(labels == lbl)) for lbl in ("H", "A", "D", "X")}
    print(f"\nWrote {os.path.relpath(out_path, ROOT)}  (total {time.perf_counter() - t0:.2f}s)")
    print("Label counts: H={H}  A={A}  D={D}  X={X}".format(**counts))
    print(f"\nNext:")
    print(f"  python evaluation/validate_submission.py {os.path.relpath(out_path, ROOT)}")
    print(f"  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
