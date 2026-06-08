#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.confirmation_backtest import run_confirmation_backtest
from stocknetwork.run_metadata import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Confirmation Backtest over lifecycle communities.")
    parser.add_argument("--parquet-root", required=True, help="Path to the symbol-partitioned parquet database.")
    parser.add_argument("--rotation-dir", required=True, help="Directory containing rotation/community_timeseries artifacts.")
    parser.add_argument("--output", required=True, help="Output directory for confirmation backtest artifacts.")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark symbol for report context.")
    parser.add_argument("--top-themes", type=int, default=3, help="Number of active themes to hold.")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Per-turnover transaction cost assumption.")
    parser.add_argument("--exit-lag-days", type=int, default=1, help="Number of trade days between exit signal and execution.")
    parser.add_argument(
        "--signal-mode",
        choices=["causal", "full_info"],
        default="causal",
        help="Use causal observable-only features or the full descriptive post-hoc feature set.",
    )
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier.")
    parser.add_argument("--run-label", default="", help="Optional short label for this run.")
    parser.add_argument("--run-notes", default="", help="Optional notes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parquet_root = Path(args.parquet_root).expanduser().resolve()
    rotation_dir = Path(args.rotation_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="run_confirmation_backtest",
        output_dir=output_dir,
        args=args,
        inputs={"parquet_root": parquet_root, "rotation_dir": rotation_dir},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    result = run_confirmation_backtest(
        parquet_root=parquet_root,
        rotation_dir=rotation_dir,
        output_dir=output_dir,
        benchmark_symbol=args.benchmark,
        top_themes=args.top_themes,
        transaction_cost_bps=args.cost_bps,
        signal_mode=args.signal_mode,
        exit_lag_days=args.exit_lag_days,
    )

    run_context.write_validation(
        {
            "benchmark": args.benchmark,
            "top_themes": args.top_themes,
            "cost_bps": args.cost_bps,
            "signal_mode": args.signal_mode,
            "exit_lag_days": args.exit_lag_days,
        }
    )
    run_context.write_artifacts(
        {
            "results_csv": output_dir / "entry_exit_holding_results.csv",
            "trade_log_csv": output_dir / "trade_log.csv",
            "leaderboard_csv": output_dir / "strategy_leaderboard.csv",
            "report_md": output_dir / "confirmation_standard_report.md",
        }
    )
    run_context.write_summary({"status": "completed", **result})
    print(f"Confirmation backtest complete in {output_dir}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
