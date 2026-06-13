from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.backtest.confirmation_relabel import relabel_evaluated_trades_file
from stocknet_alpha.backtest.final_pipeline import run_final_leadlag_pipeline
from stocknet_alpha.config import AlphaPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final causal lead-lag pipeline: relabel, OOS strategy, robustness, self-audit.")
    parser.add_argument("--evaluated-trades", required=True, help="Source evaluated_trades.parquet path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for final artifacts.")
    parser.add_argument("--relabel-confirmations", action="store_true", help="Rebuild confirmation flags before running strategy.")
    parser.add_argument("--start-date", default="", help="Required with --relabel-confirmations.")
    parser.add_argument("--end-date", default="", help="Required with --relabel-confirmations.")
    parser.add_argument("--lookback-bars", type=int, default=6)
    parser.add_argument("--top-symbols", type=int, default=60)
    parser.add_argument("--min-members", type=int, default=3)
    parser.add_argument("--min-theme-score", type=float, default=0.20)
    parser.add_argument("--min-pair-corr", type=float, default=0.55)
    parser.add_argument("--theme-path-score-method", default="jaccard", choices=["jaccard", "overlap_small"])
    parser.add_argument("--theme-path-min-overlap", type=float, default=0.40)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--valid-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--min-train-count", type=int, default=200)
    parser.add_argument("--min-valid-count", type=int, default=0)
    parser.add_argument("--leadlag-threshold", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.evaluated_trades).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_summary_path = source_path.parent / "audit_summary.json"
    audit_evidence: dict[str, object] = {}
    if audit_summary_path.exists():
        audit_evidence = json.loads(audit_summary_path.read_text(encoding="utf-8"))

    evaluated_path = source_path
    if args.relabel_confirmations:
        if not args.start_date or not args.end_date:
            raise SystemExit("--start-date and --end-date are required with --relabel-confirmations")
        relabeled_path = output_dir / "evaluated_trades_relabeled.parquet"
        relabel_evaluated_trades_file(
            source_path,
            output_path=relabeled_path,
            paths=AlphaPaths(),
            start_date=args.start_date,
            end_date=args.end_date,
            lookback_bars=args.lookback_bars,
            top_symbols=args.top_symbols,
            min_members=args.min_members,
            min_theme_score=args.min_theme_score,
            min_pair_corr=args.min_pair_corr,
            theme_path_score_method=args.theme_path_score_method,
            theme_path_min_overlap=args.theme_path_min_overlap,
        )
        evaluated_path = relabeled_path

    evaluated = pd.read_parquet(evaluated_path)
    bundle = run_final_leadlag_pipeline(
        evaluated,
        train_months=args.train_months,
        valid_months=args.valid_months,
        test_months=args.test_months,
        min_train_count=args.min_train_count,
        min_valid_count=args.min_valid_count,
        leadlag_threshold=args.leadlag_threshold,
        audit_evidence=audit_evidence,
    )

    bundle["selections"].to_csv(output_dir / "walk_forward_rule_selection.csv", index=False)
    bundle["strategy_trades"].to_parquet(output_dir / "walk_forward_strategy_trades.parquet", index=False)
    bundle["split_summary"].to_csv(output_dir / "walk_forward_strategy_split_summary.csv", index=False)
    bundle["robustness_scan"].to_csv(output_dir / "strategy_robustness_scan.csv", index=False)
    (output_dir / "walk_forward_strategy_summary.json").write_text(
        json.dumps(bundle["selection_summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "strategy_robustness_summary.json").write_text(
        json.dumps(bundle["robustness_summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "walk_forward_strategy_report.md").write_text(bundle["strategy_report"], encoding="utf-8")
    (output_dir / "walk_forward_strategy_self_audit.md").write_text(bundle["self_audit_report"], encoding="utf-8")
    (output_dir / "pipeline_manifest.json").write_text(
        json.dumps(
            {
                "evaluated_trades": str(source_path),
                "effective_evaluated_trades": str(evaluated_path),
                "relabel_confirmations": bool(args.relabel_confirmations),
                "train_months": args.train_months,
                "valid_months": args.valid_months,
                "test_months": args.test_months,
                "min_train_count": args.min_train_count,
                "min_valid_count": args.min_valid_count,
                "leadlag_threshold": args.leadlag_threshold,
                "source_audit_summary": str(audit_summary_path) if audit_summary_path.exists() else None,
                "audit_checks": bundle["audit_checks"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote final lead-lag pipeline artifacts to {output_dir}")


if __name__ == "__main__":
    main()
