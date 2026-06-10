from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stocknet_alpha.config import AlphaPaths, load_universe_symbols


RESAMPLE_RULES = {
    "5m": "5min",
    "15m": "15min",
}


def resample_ohlcv_bars(bars: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """Aggregate 1m OHLCV bars into a higher interval."""

    if target_interval not in RESAMPLE_RULES:
        raise ValueError(f"Unsupported target interval: {target_interval}")
    if bars.empty:
        return bars.copy()

    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["symbol", "timestamp"])

    result_frames: list[pd.DataFrame] = []
    rule = RESAMPLE_RULES[target_interval]
    for symbol, group in frame.groupby("symbol", sort=True):
        indexed = group.set_index("timestamp")
        aggregated = indexed.resample(rule, label="right", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        aggregated = aggregated.dropna(subset=["open", "high", "low", "close"])
        aggregated["symbol"] = symbol
        if "vwap" in group.columns:
            weighted = (indexed["close"] * indexed["volume"]).resample(rule, label="right", closed="left").sum()
            volume = indexed["volume"].resample(rule, label="right", closed="left").sum().replace(0, pd.NA)
            aggregated["vwap"] = (weighted / volume).astype(float)
        if "source" in group.columns:
            aggregated["source"] = indexed["source"].resample(rule, label="right", closed="left").last()
        result_frames.append(aggregated.reset_index())

    if not result_frames:
        return pd.DataFrame(columns=list(frame.columns))
    result = pd.concat(result_frames, ignore_index=True)
    return result.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def write_resampled_bars(bars: pd.DataFrame, paths: AlphaPaths, trade_date: str, target_interval: str) -> Path:
    output_path = paths.ensure_parent(paths.bars_path(trade_date, target_interval))
    bars.to_parquet(output_path, index=False)
    return output_path


def load_raw_1m_bars(paths: AlphaPaths, trade_date: str, input_path: Path | str | None = None) -> pd.DataFrame:
    source_path = Path(input_path).expanduser().resolve() if input_path else paths.raw_1m_path(trade_date)
    if not source_path.exists() and input_path is None:
        fallback = (
            paths.repo_root
            / "realtime_dashboard"
            / "data"
            / "archive_5m"
            / f"date={trade_date}"
            / "5m_bars.parquet"
        )
        if fallback.exists():
            frame = pd.read_parquet(fallback)
            if "archive_interval" in frame.columns:
                frame = frame[frame["archive_interval"].astype(str) == "1m"].copy()
            if not frame.empty:
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
                cached_path = paths.ensure_parent(paths.raw_1m_path(trade_date))
                frame.to_parquet(cached_path, index=False)
                return frame
    frame = pd.read_parquet(source_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample 1m bars into 5m or 15m alpha research panels.")
    parser.add_argument("--date", required=True, help="Trade date partition in YYYY-MM-DD format.")
    parser.add_argument("--from", dest="source_interval", choices=["1m"], required=True)
    parser.add_argument("--to", dest="target_interval", choices=sorted(RESAMPLE_RULES), required=True)
    parser.add_argument("--input", default="", help="Optional override path for the source parquet.")
    parser.add_argument("--screen-csv", default="", help="Optional override for the stock universe CSV.")
    parser.add_argument("--exclude-csv", default="", help="Optional override for the ETF/CEF exclusion CSV.")
    parser.add_argument("--keep-benchmark", action="append", default=[], help="Benchmark symbol to keep even if excluded.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AlphaPaths()
    raw_bars = load_raw_1m_bars(paths, args.date, input_path=args.input or None)
    universe_symbols, excluded = load_universe_symbols(
        args.screen_csv or paths.universe_csv,
        args.exclude_csv or paths.exclude_symbol_csv,
        keep_benchmarks=args.keep_benchmark,
    )
    filtered = raw_bars[raw_bars["symbol"].astype(str).isin(universe_symbols)].copy()
    resampled = resample_ohlcv_bars(filtered, args.target_interval)
    output_path = write_resampled_bars(resampled, paths, args.date, args.target_interval)
    print(
        f"Resampled {len(filtered)} raw bars into {len(resampled)} {args.target_interval} rows | "
        f"symbols={filtered['symbol'].nunique()} | excluded={len(excluded)} | output={output_path}"
    )


if __name__ == "__main__":
    main()
