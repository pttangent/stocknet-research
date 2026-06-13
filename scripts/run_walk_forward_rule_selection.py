from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.backtest.rule_selection import (
    build_rule_monthly_summary,
    build_walk_forward_report,
    expand_strategy_rule_trades,
    select_walk_forward_rules,
    summarize_walk_forward_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward rule selection on evaluated lead-lag trades.")
    parser.add_argument("--evaluated-trades", required=True, help="Path to evaluated_trades.parquet.")
    parser.add_argument("--output-dir", default="", help="Optional directory for CSV/MD outputs.")
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
        else evaluated_path.parent / "walk_forward_rule_selection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluated = pd.read_parquet(evaluated_path)
    expanded = expand_strategy_rule_trades(evaluated, leadlag_threshold=args.leadlag_threshold)
    monthly = build_rule_monthly_summary(expanded)
    selections = select_walk_forward_rules(
        monthly,
        train_months=args.train_months,
        valid_months=args.valid_months,
        test_months=args.test_months,
        min_train_count=args.min_train_count,
        min_valid_count=args.min_valid_count,
    )
    summary = summarize_walk_forward_results(selections)
    report = build_walk_forward_report(selections, summary)

    monthly.to_csv(output_dir / "rule_monthly_summary.csv", index=False)
    selections.to_csv(output_dir / "walk_forward_rule_selection.csv", index=False)
    (output_dir / "walk_forward_rule_selection.md").write_text(report, encoding="utf-8")
    (output_dir / "walk_forward_rule_selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote walk-forward rule selection outputs to {output_dir}")


if __name__ == "__main__":
    main()
