from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.backtest.backtest_signals import summarize_signal_backtest
from stocknet_alpha.backtest.audit import (
    HISTORICAL_AUDIT_LABELS,
    all_audit_checks_pass,
    evaluate_cost_realism,
    evaluate_historical_logic,
    evaluate_historical_robustness,
    evaluate_historical_survivorship,
    evaluate_temporal_causality,
)
from stocknet_alpha.backtest.historical_leadlag import (
    aggregate_evaluated_trades,
    build_self_audit_report,
    discover_trade_dates,
)
from stocknet_alpha.backtest.robustness import summarize_robustness
from stocknet_alpha.config import AlphaPaths
from stocknet_alpha.leadlag.evaluate_edges import evaluate_leadlag_signals
from stocknet_alpha.leadlag.generate_signals import generate_leadlag_signals
from stocknet_alpha.theme.build_historical_theme_candidates import load_daily_market_inputs
from stocknet_alpha.theme.historical_candidates import build_theme_candidates_from_market_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run historical causal lead-lag backtests over a date range.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="Inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--lookback-bars", type=int, default=6)
    parser.add_argument("--top-symbols", type=int, default=60)
    parser.add_argument("--min-members", type=int, default=3)
    parser.add_argument("--min-theme-score", type=float, default=0.20)
    parser.add_argument("--min-pair-corr", type=float, default=0.55)
    parser.add_argument("--theme-path-score-method", default="jaccard", choices=["jaccard", "overlap_small"])
    parser.add_argument("--theme-path-min-overlap", type=float, default=0.40)
    parser.add_argument("--lookback-minutes", type=int, default=60)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--top-followers", type=int, default=1)
    parser.add_argument("--min-leadlag-score", type=float, default=0.40)
    parser.add_argument("--commission-bps", type=float, default=0.5)
    parser.add_argument("--fees-bps", type=float, default=0.5)
    parser.add_argument("--slippage-bps", type=float, default=1.5)
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AlphaPaths()
    dates = discover_trade_dates(paths.trade_flow_1m_root, start_date=args.start_date, end_date=args.end_date)
    if not dates:
        raise SystemExit("No trade dates found in the requested range.")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else paths.backtest_root / f"historical_run_{args.start_date}_{args.end_date}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_rows: list[dict[str, object]] = []
    evaluated_frames: list[pd.DataFrame] = []

    for trade_date in dates:
        try:
            bars_5m, flow_1m = load_daily_market_inputs(paths, trade_date)
            bars_1m = pd.read_parquet(paths.raw_1m_path(trade_date))
        except Exception as exc:  # keep the batch moving
            daily_rows.append({"trade_date": trade_date, "status": "load_failed", "error": str(exc)})
            continue

        candidates = build_theme_candidates_from_market_data(
            bars_5m,
            flow_1m,
            trade_date=trade_date,
            lookback_bars=args.lookback_bars,
            top_symbols=args.top_symbols,
            min_members=args.min_members,
            min_theme_score=args.min_theme_score,
            min_pair_corr=args.min_pair_corr,
            theme_path_score_method=args.theme_path_score_method,
            theme_path_min_overlap=args.theme_path_min_overlap,
        )
        signals = generate_leadlag_signals(
            bars_1m,
            candidates,
            lookback_minutes=args.lookback_minutes,
            max_lag=args.max_lag,
            top_followers=args.top_followers,
        )
        if not signals.empty:
            signals = signals[signals["leadlag_score"] >= float(args.min_leadlag_score)].copy()
        evaluated = evaluate_leadlag_signals(
            signals,
            bars_1m,
            commission_bps=args.commission_bps,
            fees_bps=args.fees_bps,
            slippage_bps=args.slippage_bps,
        )
        summary = summarize_signal_backtest(evaluated)
        best_net = float(summary["avg_net_return"].max()) if not summary.empty else float("nan")
        best_horizon = int(summary.sort_values("avg_net_return", ascending=False).iloc[0]["horizon_minutes"]) if not summary.empty else -1
        daily_rows.append(
            {
                "trade_date": trade_date,
                "status": "ok",
                "candidate_rows": len(candidates),
                "signal_rows": len(signals),
                "evaluated_rows": len(evaluated),
                "best_horizon": best_horizon,
                "best_avg_net_return": best_net,
            }
        )
        if not evaluated.empty:
            evaluated_frames.append(evaluated)

    daily_df = pd.DataFrame(daily_rows)
    all_evaluated = pd.concat(evaluated_frames, ignore_index=True) if evaluated_frames else pd.DataFrame()
    aggregate_df = aggregate_evaluated_trades(all_evaluated)

    robustness_inputs = daily_df.loc[daily_df["status"].eq("ok"), ["trade_date", "best_avg_net_return"]].copy()
    robustness_inputs = robustness_inputs.rename(columns={"trade_date": "variant_id", "best_avg_net_return": "avg_net_return"})
    if not robustness_inputs.empty:
        baseline_value = float(robustness_inputs["avg_net_return"].mean())
        robustness_frame = pd.concat(
            [
                pd.DataFrame([{"variant_id": "baseline", "avg_net_return": baseline_value}]),
                robustness_inputs,
            ],
            ignore_index=True,
        )
        robustness_summary = summarize_robustness(robustness_frame, metric_col="avg_net_return")
    else:
        robustness_summary = {"baseline_metric": None, "same_sign_rate": None, "max_abs_deviation": None, "is_robust": False}

    audit_checks = {
        "lookahead_guard": evaluate_temporal_causality(all_evaluated),
        "survivorship_bias": evaluate_historical_survivorship(dates, daily_df),
        "robustness": evaluate_historical_robustness(robustness_summary),
        "logic_explainability": evaluate_historical_logic(daily_df),
        "cost_realism": evaluate_cost_realism(all_evaluated),
    }
    metadata = {
        "audit_checks": audit_checks,
        "notes": [
            "signals are generated only from same-day historical bars_5m and trade_flow_1m",
            "historical route scans all symbols present in raw daily partitions instead of a current survivor list",
            f"costs: commission={args.commission_bps}bps/side fees={args.fees_bps}bps/side slippage={args.slippage_bps}bps/side",
            f"theme persistence: method={args.theme_path_score_method} min_overlap={args.theme_path_min_overlap}",
            f"robustness summary: {json.dumps(robustness_summary, ensure_ascii=False)}",
        ],
    }
    audit_report = build_self_audit_report(metadata, aggregate_df)

    daily_df.to_csv(output_dir / "daily_backtest_results.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_backtest_summary.csv", index=False)
    if not all_evaluated.empty:
        all_evaluated.to_parquet(output_dir / "evaluated_trades.parquet", index=False)
    (output_dir / "self_audit_report.md").write_text(audit_report, encoding="utf-8")
    (output_dir / "audit_summary.json").write_text(
        json.dumps({"audit_checks": audit_checks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "run_params.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    if not all_audit_checks_pass(audit_checks, required_keys=tuple(HISTORICAL_AUDIT_LABELS)):
        failed = [key for key in HISTORICAL_AUDIT_LABELS if audit_checks[key]["status"] != "PASS"]
        raise SystemExit(f"historical audit failed: {', '.join(failed)}")
    print(f"Wrote historical lead-lag results to {output_dir}")


if __name__ == "__main__":
    main()
