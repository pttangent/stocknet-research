from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from stocknet_alpha.config import AlphaPaths


def summarize_signal_backtest(
    signals: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5, 10, 15),
) -> pd.DataFrame:
    """Collapse lead-lag signals into horizon-level performance stats."""

    if signals.empty:
        return pd.DataFrame(
            columns=[
                "horizon_minutes",
                "signal_count",
                "avg_return",
                "median_return",
                "hit_rate",
                "confirmed_signal_count",
                "confirmed_avg_return",
            ]
        )

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        column = f"forward_return_{int(horizon)}m"
        if column not in signals.columns:
            continue
        series = signals[column].dropna()
        confirmed = signals.loc[signals["confirmed_on_15m"].fillna(False), column].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "horizon_minutes": int(horizon),
                "signal_count": int(series.shape[0]),
                "avg_return": float(series.mean()),
                "median_return": float(series.median()),
                "hit_rate": float((series > 0).mean()),
                "confirmed_signal_count": int(confirmed.shape[0]),
                "confirmed_avg_return": float(confirmed.mean()) if not confirmed.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_minutes").reset_index(drop=True)


def write_backtest_artifacts(
    signals: pd.DataFrame,
    summary: pd.DataFrame,
    paths: AlphaPaths,
    trade_date: str,
) -> tuple[Path, Path]:
    summary_path = paths.ensure_parent(paths.backtest_summary_path(trade_date))
    report_path = paths.ensure_parent(paths.backtest_report_path(trade_date))
    summary.to_csv(summary_path, index=False)
    report_path.write_text(_build_report(signals, summary, trade_date), encoding="utf-8")
    return summary_path, report_path


def load_signals(paths: AlphaPaths, trade_date: str, input_path: Path | str | None = None) -> pd.DataFrame:
    source = Path(input_path).expanduser().resolve() if input_path else paths.signals_path(trade_date)
    return pd.read_parquet(source)


def _build_report(signals: pd.DataFrame, summary: pd.DataFrame, trade_date: str) -> str:
    lines = [
        "# Lead-Lag Alpha Review",
        "",
        f"- Trade date: `{trade_date}`",
        f"- Signals: `{len(signals)}`",
        f"- Confirmed 15m signals: `{int(signals['confirmed_on_15m'].fillna(False).sum()) if not signals.empty else 0}`",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AlphaPaths()
    signals = load_signals(paths, args.date, input_path=args.signals or None)
    summary = summarize_signal_backtest(signals)
    summary_path, report_path = write_backtest_artifacts(signals, summary, paths, args.date)
    print(f"Wrote backtest summary to {summary_path} and report to {report_path}")


if __name__ == "__main__":
    main()
