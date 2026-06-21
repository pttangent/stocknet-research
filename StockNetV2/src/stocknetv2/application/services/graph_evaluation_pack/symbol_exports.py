from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import _BENCHMARK_PROXY_PRICE_METHOD
from .io_utils import (
    _ensure_columns,
    _merge_latest_available,
    _prepare_active_snapshot_frame,
    _prepare_feature_review_frame,
    _prepare_label_review_frame,
    _prepare_trade_flow_review_frame,
    _read_partition_parquet,
    _write_parquet_dataframe,
)


def _export_symbol_snapshot_feature_shards(
    trade_dates: list[str],
    active_snapshot_key_dir: Path,
    output_dir: Path,
    symbol_master_frame: pd.DataFrame,
    market_data_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for trade_date in trade_dates:
        active_day = pd.read_parquet(active_snapshot_key_dir / f"{trade_date}.parquet")
        if active_day.empty:
            _write_parquet_dataframe(active_day, output_dir / f"{trade_date}.parquet")
            continue
        active_day = _prepare_active_snapshot_frame(active_day)
        bars_full_day = _read_partition_parquet(
            market_data_root,
            "bars_5m",
            trade_date,
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "date"],
        )
        if not bars_full_day.empty:
            bars_full_day = bars_full_day.sort_values(["symbol", "timestamp"]).copy()
            bars_full_day["bar_ret_5m_past"] = bars_full_day.groupby("symbol")["close"].pct_change(1)
            bars_full_day["bar_ret_15m_past"] = bars_full_day.groupby("symbol")["close"].pct_change(3)
            bars_full_day["bar_dollar_volume"] = bars_full_day["close"] * bars_full_day["volume"]
            bars_full_day["bar_volume_cs_z"] = bars_full_day.groupby("timestamp")["volume"].transform(
                lambda series: (series - series.mean()) / series.std(ddof=0) if series.std(ddof=0) not in (0, None) else 0.0
            )
            bars_full_day["bar_dollar_volume_cs_z"] = bars_full_day.groupby("timestamp")["bar_dollar_volume"].transform(
                lambda series: (series - series.mean()) / series.std(ddof=0) if series.std(ddof=0) not in (0, None) else 0.0
            )
            bars_day = bars_full_day.rename(
                columns={
                    "open": "bar_open",
                    "high": "bar_high",
                    "low": "bar_low",
                    "close": "bar_close",
                    "volume": "bar_volume",
                    "vwap": "bar_vwap",
                }
            )
        else:
            bars_day = bars_full_day
        if not bars_day.empty:
            bars_day["bar_timestamp"] = bars_day["timestamp"]
            bars_day["bar_available_time"] = bars_day["timestamp"]

        features_day = _read_partition_parquet(
            market_data_root,
            "features_1m",
            trade_date,
            columns=[
                "symbol",
                "timestamp",
                "available_time",
                "bar_end",
                "date",
                "close",
                "volume",
                "dollar_volume",
                "trade_count",
                "imbalance_proxy",
                "large_trade_count",
                "large_trade_dollar_volume",
                "ret_1m",
                "ret_1m_past",
                "ret_3m_past",
                "ret_5m_past",
                "ret_15m_past",
                "large_trade_ratio",
                "large_trade_ratio_z",
                "volume_z_12",
                "volume_z_proxy",
                "flow_impulse_score",
            ],
        )
        features_day = _prepare_feature_review_frame(features_day)

        trade_flow_day = _read_partition_parquet(
            market_data_root,
            "trade_flow_1m",
            trade_date,
            columns=[
                "ticker",
                "minute",
                "trade_count",
                "volume",
                "dollar_volume",
                "imbalance_proxy",
                "large_trade_count",
                "large_trade_dollar_volume",
                "off_exchange_volume",
            ],
        ).rename(
            columns={
                "trade_count": "flow_trade_count",
                "volume": "flow_volume",
                "dollar_volume": "flow_dollar_volume",
                "imbalance_proxy": "flow_imbalance_proxy",
                "large_trade_count": "flow_large_trade_count",
                "large_trade_dollar_volume": "flow_large_trade_dollar_volume",
            }
        )
        trade_flow_day = _prepare_trade_flow_review_frame(trade_flow_day)

        merged = _merge_latest_available(active_day, bars_day, right_time_column="bar_available_time")
        merged = _merge_latest_available(merged, features_day, right_time_column="graph_input_available_time")
        merged = _merge_latest_available(merged, trade_flow_day, right_time_column="flow_available_time")
        merged = merged.drop(columns=[column for column in ("ticker", "minute") if column in merged.columns])
        merged = merged.merge(symbol_master_frame, how="left", on="symbol")
        merged["feature_source"] = "causality_safe_graph_inputs_plus_context"
        _write_parquet_dataframe(merged, output_dir / f"{trade_date}.parquet")


def _export_symbol_forward_label_shards(
    trade_dates: list[str],
    active_snapshot_key_dir: Path,
    output_dir: Path,
    primary_benchmark: str,
    market_data_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_columns = [
        "label_source_timestamp",
        "label_available_time",
        "future_ret_1m",
        "future_ret_5m",
        "future_ret_15m",
        "future_ret_30m",
    ]
    benchmark_columns = [
        "benchmark_label_source_timestamp",
        "benchmark_label_available_time",
        "benchmark_future_ret_1m",
        "benchmark_future_ret_5m",
        "benchmark_future_ret_15m",
        "benchmark_future_ret_30m",
        "benchmark_label_source",
        "benchmark_proxy_price_method",
    ]
    for trade_date in trade_dates:
        active_day = pd.read_parquet(active_snapshot_key_dir / f"{trade_date}.parquet")
        if active_day.empty:
            _write_parquet_dataframe(active_day, output_dir / f"{trade_date}.parquet")
            continue
        active_day = _prepare_active_snapshot_frame(active_day)
        labels_day = _read_partition_parquet(
            market_data_root,
            "labels_1m",
            trade_date,
            columns=["symbol", "timestamp", "future_ret_1m", "future_ret_5m", "future_ret_15m", "future_ret_30m"],
        )
        labels_day = _prepare_label_review_frame(labels_day)
        benchmark_day = _build_benchmark_label_frame(
            trade_date=trade_date,
            primary_benchmark=primary_benchmark,
            labels_day=labels_day,
            market_data_root=market_data_root,
        )
        merged = _merge_latest_available(active_day, labels_day, right_time_column="label_available_time")
        _ensure_columns(merged, label_columns)
        merged.insert(5, "benchmark_symbol", primary_benchmark)
        merged = _merge_latest_available(
            merged,
            benchmark_day,
            by_column="benchmark_symbol",
            right_time_column="benchmark_label_available_time",
        )
        _ensure_columns(merged, benchmark_columns)
        merged["benchmark_label_source"] = merged["benchmark_label_source"].fillna("missing")
        merged["excess_future_ret_1m"] = merged["future_ret_1m"] - merged["benchmark_future_ret_1m"]
        merged["excess_future_ret_5m"] = merged["future_ret_5m"] - merged["benchmark_future_ret_5m"]
        merged["excess_future_ret_15m"] = merged["future_ret_15m"] - merged["benchmark_future_ret_15m"]
        merged["excess_future_ret_30m"] = merged["future_ret_30m"] - merged["benchmark_future_ret_30m"]
        _write_parquet_dataframe(merged, output_dir / f"{trade_date}.parquet")


def _export_benchmark_series_shards(
    trade_dates: list[str],
    output_dir: Path,
    benchmark_symbols: tuple[str, ...],
    market_data_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for trade_date in trade_dates:
        frame = _read_partition_parquet(
            market_data_root,
            "bars_5m",
            trade_date,
            columns=["timestamp", "open", "high", "low", "close", "volume", "symbol", "vwap", "source", "date"],
        )
        frame = frame.loc[frame["symbol"].isin(benchmark_symbols)].copy()
        if not frame.empty:
            frame = frame.sort_values(["symbol", "timestamp"])
            frame["ret_5m"] = frame.groupby("symbol")["close"].pct_change(1)
        _write_parquet_dataframe(frame, output_dir / f"{trade_date}.parquet")


def _build_benchmark_label_frame(
    *,
    trade_date: str,
    primary_benchmark: str,
    labels_day: pd.DataFrame,
    market_data_root: Path,
) -> pd.DataFrame:
    benchmark_day = labels_day.loc[
        labels_day["symbol"] == primary_benchmark,
        [
            "symbol",
            "label_source_timestamp",
            "label_available_time",
            "future_ret_1m",
            "future_ret_5m",
            "future_ret_15m",
            "future_ret_30m",
        ],
    ].copy()
    if benchmark_day.empty:
        benchmark_day = _synthesize_benchmark_labels_from_trade_flow(
            trade_date=trade_date,
            benchmark_symbol=primary_benchmark,
            market_data_root=market_data_root,
        )
    else:
        benchmark_day["benchmark_label_source"] = "labels_1m"
        benchmark_day["benchmark_proxy_price_method"] = pd.NA
    return benchmark_day.rename(
        columns={
            "symbol": "benchmark_symbol",
            "label_source_timestamp": "benchmark_label_source_timestamp",
            "label_available_time": "benchmark_label_available_time",
            "future_ret_1m": "benchmark_future_ret_1m",
            "future_ret_5m": "benchmark_future_ret_5m",
            "future_ret_15m": "benchmark_future_ret_15m",
            "future_ret_30m": "benchmark_future_ret_30m",
        }
    )


def _synthesize_benchmark_labels_from_trade_flow(
    *,
    trade_date: str,
    benchmark_symbol: str,
    market_data_root: Path,
) -> pd.DataFrame:
    trade_flow_day = _read_partition_parquet(
        market_data_root,
        "trade_flow_1m",
        trade_date,
        columns=["ticker", "minute", "volume", "dollar_volume", "date"],
    )
    if trade_flow_day.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "label_source_timestamp",
                "label_available_time",
                "future_ret_1m",
                "future_ret_5m",
                "future_ret_15m",
                "future_ret_30m",
                "benchmark_label_source",
                "benchmark_proxy_price_method",
            ]
        )
    benchmark_flow = trade_flow_day.loc[trade_flow_day["ticker"] == benchmark_symbol].copy()
    if benchmark_flow.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "label_source_timestamp",
                "label_available_time",
                "future_ret_1m",
                "future_ret_5m",
                "future_ret_15m",
                "future_ret_30m",
                "benchmark_label_source",
                "benchmark_proxy_price_method",
            ]
        )
    benchmark_flow["volume"] = pd.to_numeric(benchmark_flow["volume"], errors="coerce")
    benchmark_flow["dollar_volume"] = pd.to_numeric(benchmark_flow["dollar_volume"], errors="coerce")
    benchmark_flow["proxy_price"] = benchmark_flow["dollar_volume"] / benchmark_flow["volume"].replace(0.0, pd.NA)
    benchmark_flow["timestamp"] = pd.to_datetime(benchmark_flow["minute"])
    benchmark_flow = benchmark_flow.dropna(subset=["timestamp", "proxy_price"]).sort_values("timestamp").reset_index(drop=True).copy()
    if benchmark_flow.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "label_source_timestamp",
                "label_available_time",
                "future_ret_1m",
                "future_ret_5m",
                "future_ret_15m",
                "future_ret_30m",
                "benchmark_label_source",
                "benchmark_proxy_price_method",
            ]
        )
    labels = pd.DataFrame({"symbol": benchmark_symbol, "timestamp": benchmark_flow["timestamp"].to_numpy()})
    price_series = benchmark_flow["proxy_price"].astype(float).reset_index(drop=True)
    for horizon in (1, 5, 15, 30):
        labels[f"future_ret_{horizon}m"] = (price_series.shift(-horizon) / price_series) - 1.0
    labels = _prepare_label_review_frame(labels)
    labels["benchmark_label_source"] = "trade_flow_proxy"
    labels["benchmark_proxy_price_method"] = _BENCHMARK_PROXY_PRICE_METHOD
    return labels
