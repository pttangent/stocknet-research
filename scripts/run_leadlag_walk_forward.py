from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.backtest.walk_forward import build_walk_forward_splits, write_walk_forward_splits
from stocknet_alpha.config import AlphaPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate walk-forward split metadata for lead-lag alpha research.")
    parser.add_argument("--start-month", required=True, help="Inclusive start month in YYYY-MM format.")
    parser.add_argument("--end-month", required=True, help="Inclusive end month in YYYY-MM format.")
    parser.add_argument("--train-months", type=int, default=5)
    parser.add_argument("--valid-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--feature-version", default="causal_leadlag_v1")
    parser.add_argument("--model-version", default="baseline_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AlphaPaths()
    splits = build_walk_forward_splits(
        start_month=args.start_month,
        end_month=args.end_month,
        train_months=args.train_months,
        valid_months=args.valid_months,
        test_months=args.test_months,
        feature_version=args.feature_version,
        model_version=args.model_version,
    )
    output_path = write_walk_forward_splits(splits, paths)
    print(f"Wrote {len(splits)} walk-forward splits to {output_path}")


if __name__ == "__main__":
    main()
