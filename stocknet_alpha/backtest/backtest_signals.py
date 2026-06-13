from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.config import AlphaPaths
from stocknet_alpha.data.resample_bars import load_raw_1m_bars
from stocknet_alpha.leadlag.evaluate_edges import evaluate_leadlag_signals


def summarize_signal_backtest(
    evaluated_trades: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5, 10, 15),
) -> pd.DataFrame:
    """Collapse realized lead-lag trades into horizon-level performance stats."""

    if evaluated_trades.empty:
        return pd.DataFrame(
            columns=[
                "horizon_minutes",
                "signal_count",
                "avg_gross_return",
                "avg_net_return",
                "median_net_return",
                "hit_rate",
                "confirmed_signal_count",
                "confirmed_avg_net_return",
            ]
        )

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        trades = evaluated_trades[evaluated_trades["horizon_minutes"] == int(horizon)].copy()
        if trades.empty:
            continue
        net_series = trades["net_return"].dropna()
        gross_series = trades["gross_return"].dropna()
        confirmed = trades.loc[trades["confirmed_on_15m"].fillna(False), "net_return"].dropna()
        if net_series.empty:
            continue
        rows.append(
            {
                "horizon_minutes": int(horizon),
                "signal_count": int(net_series.shape[0]),
                "avg_gross_return": float(gross_series.mean()) if not gross_series.empty else 0.0,
                "avg_net_return": float(net_series.mean()),
                "median_net_return": float(net_series.median()),
                "hit_rate": float((net_series > 0).mean()),
                "confirmed_signal_count": int(confirmed.shape[0]),
                "confirmed_avg_net_return": float(confirmed.mean()) if not confirmed.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_minutes").reset_index(drop=True)


def write_backtest_artifacts(
    evaluated_trades: pd.DataFrame,
    summary: pd.DataFrame,
    paths: AlphaPaths,
    trade_date: str,
    *,
    commission_bps: float = 0.0,
    fees_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[Path, Path]:
    summary_path = paths.ensure_parent(paths.backtest_summary_path(trade_date))
    report_path = paths.ensure_parent(paths.backtest_report_path(trade_date))
    summary.to_csv(summary_path, index=False)
    report_path.write_text(
        _build_report(
            evaluated_trades,
            summary,
            trade_date,
            commission_bps=commission_bps,
            fees_bps=fees_bps,
            slippage_bps=slippage_bps,
        ),
        encoding="utf-8",
    )
    return summary_path, report_path


def load_signals(paths: AlphaPaths, trade_date: str, input_path: Path | str | None = None) -> pd.DataFrame:
    source = Path(input_path).expanduser().resolve() if input_path else paths.signals_path(trade_date)
    if not source.exists():
        return pd.DataFrame()
    return pd.read_parquet(source)


def _build_report(
    evaluated_trades: pd.DataFrame,
    summary: pd.DataFrame,
    trade_date: str,
    *,
    commission_bps: float = 0.0,
    fees_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> str:
    lines = [
        "# Lead-Lag Alpha Review",
        "",
        f"- Trade date: `{trade_date}`",
        f"- Realized trades: `{len(evaluated_trades)}`",
        f"- Confirmed 15m trades: `{int(evaluated_trades['confirmed_on_15m'].fillna(False).sum()) if not evaluated_trades.empty else 0}`",
        f"- Commission (bps / side): `{commission_bps}`",
        f"- Fees (bps / side): `{fees_bps}`",
        f"- Slippage (bps / side): `{slippage_bps}`",
        "",
        "## Horizon Summary",
        "",
    ]
    lines.extend(_markdown_table(summary))
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest previously generated lead-lag signals.")
    parser.add_argument("--date", required=True, help="Trade date partition in YYYY-MM-DD format.")
    parser.add_argument("--signals", default="", help="Optional override path for the signals parquet.")
    parser.add_argument("--commission-bps", type=float, default=0.0, help="Per-side commission in basis points.")
    parser.add_argument("--fees-bps", type=float, default=0.0, help="Per-side additional fees or taxes in basis points.")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Per-side slippage in basis points.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AlphaPaths()
    signals = load_signals(paths, args.date, input_path=args.signals or None)
    bars_1m = load_raw_1m_bars(paths, args.date)
    evaluated_trades = evaluate_leadlag_signals(
        signals,
        bars_1m,
        commission_bps=args.commission_bps,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
    )
    summary = summarize_signal_backtest(evaluated_trades)
    summary_path, report_path = write_backtest_artifacts(
        evaluated_trades,
        summary,
        paths,
        args.date,
        commission_bps=args.commission_bps,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
    )
    print(f"Wrote backtest summary to {summary_path} and report to {report_path}")


if __name__ == "__main__":
    main()
