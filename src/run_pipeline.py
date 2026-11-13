"""End-to-end entry point for the modular possession inference pipelines (v1, v2, v3).

Usage:
    python src/run_pipeline.py                  # runs v1_physics_viterbi
    python src/run_pipeline.py --version v2     # runs v2_physics_viterbi
    python src/run_pipeline.py --version v3     # runs v3_physics_viterbi
    python src/run_pipeline_v3.py               # shortcut for v3_physics_viterbi
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
from src.possession_model_v2 import live_half_blocks, predict_possession_v2  # noqa: E402
from src.possession_model_v3 import predict_possession_v3  # noqa: E402
from src.smart_tagging_features import extract_smart_tagging_features  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
VERSION_DEFAULTS = {
    "v1": "v1_physics_viterbi",
    "v2": "v2_physics_viterbi",
    "v3": "v3_physics_viterbi",
}


def write_submission(path, labels):
    """Write frame,label CSV matching the competition submission contract."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return path


def run_pipeline(version="v1", name=None, out_dir=RESULTS_DIR, use_cache=True):
    """Load tracking/tagging data, extract features, decode possession, and write submission CSV."""
    stem = name or VERSION_DEFAULTS[version]
    t0 = time.perf_counter()

    print("[1/4] Loading ATD arrays...")
    atd = load_atd_arrays(game=GAME, use_cache=use_cache)
    print(f"      done in {time.perf_counter() - t0:.2f}s ({atd['n_frames']} frames)")

    t1 = time.perf_counter()
    print("[2/4] Extracting Smart Tagging tactical priors & set-piece anchors...")
    st_feats = extract_smart_tagging_features(atd["n_frames"], game=GAME)
    print(f"      done in {time.perf_counter() - t1:.2f}s ({len(st_feats['set_pieces'])} set pieces)")

    t2 = time.perf_counter()
    print(f"[3/4] Extracting ATD physics & proximity features ({version})...")
    blocks = live_half_blocks(st_feats, atd["n_frames"], game=GAME) if version in ("v2", "v3") else None
    atd_feats = extract_atd_features(atd, active_blocks=blocks)
    print(f"      done in {time.perf_counter() - t2:.2f}s")

    t3 = time.perf_counter()
    print(f"[4/4] Running {version} possession decoder & writing {stem}.csv...")
    if version == "v3":
        labels = predict_possession_v3(atd_feats, st_feats, game=GAME)
    elif version == "v2":
        labels = predict_possession_v2(atd_feats, st_feats, game=GAME)
    else:
        labels = predict_possession(atd_feats, st_feats, game=GAME)
    out_path = write_submission(os.path.join(out_dir, f"{stem}.csv"), labels)
    print(f"      done in {time.perf_counter() - t3:.2f}s")

    counts = {lbl: int(np.sum(labels == lbl)) for lbl in ("H", "A", "D", "X")}
    try:
        rel_out = os.path.relpath(out_path, ROOT)
    except ValueError:
        rel_out = out_path
    print(f"\nWrote {rel_out}  (total {time.perf_counter() - t0:.2f}s)")
    print("Label counts: H={H}  A={A}  D={D}  X={X}".format(**counts))
    print("\nNext:")
    print(f"  python evaluation/validate_submission.py {rel_out}")
    print("  python evaluation/run_evaluation.py")
    return out_path


def main(default_version="v1"):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", choices=("v1", "v2", "v3"), default=default_version,
                        help="model version to run")
    parser.add_argument("--name", default=None,
                        help="output submission stem (without .csv)")
    parser.add_argument("--out", default=RESULTS_DIR, help="output directory")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore and bypass cached ATD npz")
    args = parser.parse_args()
    run_pipeline(
        version=args.version,
        name=args.name,
        out_dir=args.out,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
