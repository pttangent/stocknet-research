from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.backtest.walk_forward_strategy import run_walk_forward_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize realized OOS trades from walk-forward rule selection.")
    parser.add_argument("--evaluated-trades", required=True, help="Source evaluated_trades.parquet path.")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--valid-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--min-train-count", type=int, default=200)
    parser.add_argument("--min-valid-count", type=int, default=0)
    parser.add_argument("--leadlag-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluated_path = Path(args.evaluated_trades).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else evaluated_path.parent / "walk_forward_strategy_backtest"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluated = pd.read_parquet(evaluated_path)
    selections, strategy_trades, split_summary, selection_summary, report = run_walk_forward_strategy(
        evaluated,
        train_months=args.train_months,
        valid_months=args.valid_months,
        test_months=args.test_months,
        min_train_count=args.min_train_count,
        min_valid_count=args.min_valid_count,
        leadlag_threshold=args.leadlag_threshold,
    )

    selections.to_csv(output_dir / "walk_forward_rule_selection.csv", index=False)
    strategy_trades.to_parquet(output_dir / "walk_forward_strategy_trades.parquet", index=False)
    split_summary.to_csv(output_dir / "walk_forward_strategy_split_summary.csv", index=False)
    (output_dir / "walk_forward_strategy_summary.json").write_text(
        json.dumps(selection_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "walk_forward_strategy_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote walk-forward strategy artifacts to {output_dir}")


if __name__ == "__main__":
    main()
