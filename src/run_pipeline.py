"""End-to-end possession inference pipeline (supports v1, v2, v3, and v4).

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --version v3 --output evaluation/results/v3_physics_viterbi.csv
    python src/run_pipeline.py --version v4 --output evaluation/results/v4_physics_viterbi.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_GAME = "BAR-ATH"
from src.atd_features import extract_atd_features  # noqa: E402
from src.atd_loader import load_atd_arrays  # noqa: E402
from src.possession_model import predict_possession  # noqa: E402
from src.possession_model_v2 import live_half_blocks, predict_possession_v2  # noqa: E402
from src.possession_model_v3 import predict_possession_v3  # noqa: E402
from src.possession_model_v4 import predict_possession_v4  # noqa: E402
from src.smart_tagging_features import extract_smart_tagging_features  # noqa: E402

DEFAULT_OUTPUTS = {
    "v1": os.path.join(ROOT, "evaluation", "results", "v1_physics_viterbi.csv"),
    "v2": os.path.join(ROOT, "evaluation", "results", "v2_physics_viterbi.csv"),
    "v3": os.path.join(ROOT, "evaluation", "results", "v3_physics_viterbi.csv"),
    "v4": os.path.join(ROOT, "evaluation", "results", "v4_physics_viterbi.csv"),
}


def run_pipeline(
    game=DEFAULT_GAME, data_dir=None, source="tactical", output_path=None, version="v4"
):
    t0 = time.time()
    print(f"[1/4] Loading ATD tracking data for {game} ...")
    atd = load_atd_arrays(game=game, data_dir=data_dir, source=source)
    n_frames = len(atd["ball_x"])
    print(
        f"      Loaded {n_frames:,} frames ({n_frames / 25.0 / 60.0:.1f} min) "
        f"in {time.time() - t0:.1f}s"
    )

    t1 = time.time()
    print(f"[2/4] Extracting Smart Tagging features for {game} ...")
    st_feats = extract_smart_tagging_features(
        n_frames=n_frames, game=game, data_dir=data_dir
    )
    active_blocks = live_half_blocks(st_feats, n_frames) if version in ("v2", "v3", "v4") else None
    print(f"      Done in {time.time() - t1:.1f}s")

    t2 = time.time()
    print(f"[3/4] Extracting physics & geometry features from ATD (mode={version}) ...")
    atd_feats = extract_atd_features(atd, active_blocks=active_blocks)
    vis_pct = 100.0 * np.mean(atd_feats["ball_visible"])
    print(
        f"      Ball visible on {vis_pct:.1f}% of frames | "
        f"done in {time.time() - t2:.1f}s"
    )

    t3 = time.time()
    print(f"[4/4] Running 3-state Viterbi possession decoder ({version}) ...")
    if version == "v1":
        labels, _, _, _ = predict_possession(atd_feats, st_feats)
    elif version == "v2":
        labels, _, _, _ = predict_possession_v2(atd_feats, st_feats)
    elif version == "v3":
        labels, _, _, _ = predict_possession_v3(atd_feats, st_feats)
    else:
        labels, _, _, _ = predict_possession_v4(atd_feats, st_feats, atd_raw=atd)
    print(f"      Decoded in {time.time() - t3:.1f}s")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_df = pd.DataFrame({"frame": np.arange(n_frames, dtype=np.int64), "label": labels})
    out_df.to_csv(output_path, index=False)

    unique, counts = np.unique(labels, return_counts=True)
    dist_str = ", ".join(
        f"{k}: {v:,} ({100.0 * v / n_frames:.1f}%)" for k, v in zip(unique, counts)
    )
    print(f"\nSaved {len(out_df):,} predictions -> {output_path}")
    print(f"Label distribution: {dist_str}")
    print(f"Total pipeline time: {time.time() - t0:.1f}s")


def main(default_version="v4"):
    parser = argparse.ArgumentParser(description="Run possession inference pipeline.")
    parser.add_argument("--game", default=DEFAULT_GAME, help="Game identifier (default: BAR-ATH)")
    parser.add_argument("--data-dir", default=None, help="Optional data directory override")
    parser.add_argument("--source", default="tactical", help="Optional data source override")
    parser.add_argument(
        "--version",
        choices=("v1", "v2", "v3", "v4"),
        default=default_version,
        help="Model version to run (v1, v2, v3, or v4)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output submission CSV path (defaults to evaluation/results/<version>_physics_viterbi.csv)",
    )
    args = parser.parse_args()
    out_path = args.output or DEFAULT_OUTPUTS[args.version]
    run_pipeline(
        game=args.game,
        data_dir=args.data_dir,
        source=args.source,
        output_path=out_path,
        version=args.version,
    )


if __name__ == "__main__":
    main()
