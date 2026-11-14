"""Entry point for running the v3 possession inference pipeline."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.run_pipeline import main  # noqa: E402

if __name__ == "__main__":
    main(default_version="v3")
