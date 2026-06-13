from __future__ import annotations

import gzip
import hashlib
import io
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyzipper

from stocknet_alpha.config import AlphaPaths
from stocknet_alpha.data.resample_bars import resample_ohlcv_bars


ZIP_PASSWORD_SALT = "vvtr123!@#qwe"
OFF_EXCHANGE_IDS = {201, 202, 203}
TRADE_FLOW_COLUMNS = [
    "ticker",
    "minute",
    "trade_count",
    "volume",
    "dollar_volume",
    "vwap",
    "avg_trade_size",
    "median_trade_size",
    "max_trade_size",
    "buy_vol_proxy",
    "sell_vol_proxy",
    "buy_trade_count_proxy",
    "sell_trade_count_proxy",
    "uptick_count",
    "downtick_count",
    "zero_tick_count",
    "imbalance_proxy",
    "trade_count_imbalance_proxy",
    "large_trade_count",
    "large_trade_volume",
    "large_trade_dollar_volume",
    "large_trade_buy_vol_proxy",
    "large_trade_sell_vol_proxy",
    "odd_lot_trade_count",
    "odd_lot_volume",
    "block_trade_count",
    "block_trade_volume",
    "lit_volume",
    "off_exchange_volume",
    "lit_trade_count",
    "off_exchange_trade_count",
    "correction_count",
    "unique_exchange_count",
    "avg_report_lag_ns",
    "max_report_lag_ns",
]
TRADE_FLOW_NUMERIC_COLUMNS = [column for column in TRADE_FLOW_COLUMNS if column not in {"ticker", "minute"}]


def generate_zip_password(filename: str, salt: str = ZIP_PASSWORD_SALT) -> str:
    return hashlib.sha256(f"{filename}{salt}".encode("utf-8")).hexdigest()


def read_encrypted_zip_csv(
    zip_path: Path | str,
    member_name: str | None = None,
    *,
    usecols: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    source = Path(zip_path).expanduser().resolve()
    password = generate_zip_password(source.name).encode("utf-8")
    with pyzipper.AESZipFile(source) as archive:
        archive.setpassword(password)
        target_name = member_name or archive.namelist()[0]
        with archive.open(target_name) as handle:
            return pd.read_csv(handle, usecols=usecols, nrows=nrows)


def iter_encrypted_zip_csv_chunks(
    zip_path: Path | str,
    *,
    member_name: str | None = None,
    usecols: Sequence[str] | None = None,
    chunksize: int = 1_000_000,
) -> Iterator[pd.DataFrame]:
    source = Path(zip_path).expanduser().resolve()
    password = generate_zip_password(source.name).encode("utf-8")
    with pyzipper.AESZipFile(source) as archive:
        archive.setpassword(password)
        target_name = member_name or archive.namelist()[0]
        with archive.open(target_name) as handle:
            for chunk in pd.read_csv(handle, usecols=usecols, chunksize=chunksize):
                yield chunk


def normalize_vendor_1m_bars(raw: pd.DataFrame, source: str = "vendor_1m") -> pd.DataFrame:
    required = {
        "exchange",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "amount",
        "volume",
        "bob",
        "eob",
        "type",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required 1m bar columns: {sorted(missing)}")

    frame = raw.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame["exchange"] = frame["exchange"].astype(str).str.upper().str.strip()
    frame = frame[frame["symbol"].ne("") & frame["symbol"].ne("NAN")].copy()
    frame["timestamp"] = pd.to_datetime(frame["bob"], utc=True)
    frame["bar_end"] = pd.to_datetime(frame["eob"], utc=True)
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0).astype(float)
    frame["dollar_volume"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0).astype(float)
    frame["vwap"] = np.where(frame["volume"] > 0, frame["dollar_volume"] / frame["volume"], np.nan)
    frame["source"] = source
    frame["bar_type"] = frame["type"]
    if "sequence" not in frame.columns:
        frame["sequence"] = 0
    frame["sequence"] = pd.to_numeric(frame["sequence"], errors="coerce").fillna(0).astype("int64")

    columns = [
        "symbol",
        "timestamp",
        "bar_end",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_volume",
        "vwap",
        "exchange",
        "bar_type",
        "sequence",
        "source",
    ]
    return frame[columns].sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def normalize_vendor_legacy_gzip_bars(raw: pd.DataFrame, source: str = "vendor_1m_legacy_gz") -> pd.DataFrame:
    required = {"ticker", "volume", "open", "close", "high", "low", "window_start"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required legacy 1m bar columns: {sorted(missing)}")

    frame = raw.copy()
    frame["symbol"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame = frame[frame["symbol"].ne("") & frame["symbol"].ne("NAN")].copy()
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["window_start"], errors="coerce"), unit="ns", utc=True)
    frame["bar_end"] = frame["timestamp"] + pd.Timedelta(minutes=1)
    typical_price = frame[["open", "high", "low", "close"]].mean(axis=1)
    frame["dollar_volume"] = typical_price * frame["volume"].fillna(0.0)
    frame["vwap"] = np.where(frame["volume"].fillna(0.0) > 0, typical_price, np.nan)
    frame["exchange"] = "UNK"
    frame["bar_type"] = "legacy_gz"
    frame["sequence"] = pd.to_numeric(frame.get("transactions", 0), errors="coerce").fillna(0).astype("int64")
    frame["source"] = source

    columns = [
        "symbol",
        "timestamp",
        "bar_end",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_volume",
        "vwap",
        "exchange",
        "bar_type",
        "sequence",
        "source",
    ]
    return frame[columns].sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def empty_trade_flow_frame() -> pd.DataFrame:
    frame = {
        "ticker": pd.Series(dtype="string"),
        "minute": pd.Series(dtype="datetime64[ns, UTC]"),
    }
    frame.update({column: pd.Series(dtype="float64") for column in TRADE_FLOW_NUMERIC_COLUMNS})
    return pd.DataFrame(frame)


def ensure_trade_flow_schema(trade_flow_1m: pd.DataFrame | None) -> pd.DataFrame:
    if trade_flow_1m is None:
        return empty_trade_flow_frame()

    frame = trade_flow_1m.copy()
    for column in TRADE_FLOW_COLUMNS:
        if column not in frame.columns:
            if column == "ticker":
                frame[column] = pd.Series(pd.array([pd.NA] * len(frame), dtype="string"))
            elif column == "minute":
                frame[column] = pd.NaT
            else:
                frame[column] = np.nan

    frame["ticker"] = frame["ticker"].astype("string").str.upper()
    frame["minute"] = pd.to_datetime(frame["minute"], utc=True, errors="coerce")
    for column in TRADE_FLOW_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[TRADE_FLOW_COLUMNS]


def summarize_trade_chunk(
    chunk: pd.DataFrame,
    *,
    state: dict[str, dict[str, float | int]] | None = None,
    large_trade_dollar_threshold: float = 50_000.0,
    block_trade_share_threshold: float = 10_000.0,
    block_trade_dollar_threshold: float = 200_000.0,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    if chunk.empty:
        return pd.DataFrame(), state or {}

    prior_state = {str(key): dict(value) for key, value in (state or {}).items()}
    frame = chunk.copy()
    frame = frame.dropna(subset=["ticker", "participant_timestamp", "price", "size"])
    if frame.empty:
        return pd.DataFrame(), prior_state

    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["participant_timestamp"] = pd.to_numeric(frame["participant_timestamp"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
    frame["sequence_number"] = pd.to_numeric(frame.get("sequence_number", 0), errors="coerce").fillna(0).astype("int64")
    frame["exchange"] = pd.to_numeric(frame.get("exchange", 0), errors="coerce").fillna(0).astype("int64")
    frame["trf_id"] = pd.to_numeric(frame.get("trf_id", 0), errors="coerce").fillna(0).astype("int64")
    frame["correction"] = pd.to_numeric(frame.get("correction", 0), errors="coerce").fillna(0).astype("int64")
    frame["sip_timestamp"] = pd.to_numeric(frame.get("sip_timestamp", 0), errors="coerce").fillna(0).astype("int64")
    frame["trf_timestamp"] = pd.to_numeric(frame.get("trf_timestamp", 0), errors="coerce").fillna(0).astype("int64")
    frame = frame.sort_values(["ticker", "participant_timestamp", "sequence_number"]).reset_index(drop=True)

    frame["minute"] = pd.to_datetime(frame["participant_timestamp"], unit="ns", utc=True).dt.floor("1min")
    frame["dollar"] = frame["price"] * frame["size"]
    frame["report_lag_ns"] = (frame["sip_timestamp"] - frame["participant_timestamp"]).clip(lower=0)
    frame["is_off_exchange"] = frame["trf_id"].gt(0) | frame["exchange"].isin(OFF_EXCHANGE_IDS)
    frame["is_large_trade"] = frame["dollar"] >= large_trade_dollar_threshold
    frame["is_odd_lot"] = frame["size"] < 100
    frame["is_block_trade"] = (frame["size"] >= block_trade_share_threshold) | (frame["dollar"] >= block_trade_dollar_threshold)

    seed_price = frame["ticker"].map(lambda ticker: prior_state.get(ticker, {}).get("last_price", np.nan)).astype(float)
    seed_side = frame["ticker"].map(lambda ticker: prior_state.get(ticker, {}).get("last_side", np.nan))

    frame["prev_price"] = frame.groupby("ticker")["price"].shift(1)
    frame["prev_price"] = frame["prev_price"].where(frame["prev_price"].notna(), seed_price)

    frame["side"] = np.select(
        [frame["price"] > frame["prev_price"], frame["price"] < frame["prev_price"]],
        [1, -1],
        default=0,
    )
    frame["side"] = frame["side"].replace(0, np.nan)
    frame["side"] = frame.groupby("ticker")["side"].ffill()
    frame["side"] = frame["side"].where(frame["side"].notna(), seed_side)
    frame["side"] = frame["side"].fillna(0).astype("int8")

    frame["buy_vol_proxy"] = np.where(frame["side"] > 0, frame["size"], 0.0)
    frame["sell_vol_proxy"] = np.where(frame["side"] < 0, frame["size"], 0.0)
    frame["buy_trade_count_proxy"] = np.where(frame["side"] > 0, 1, 0)
    frame["sell_trade_count_proxy"] = np.where(frame["side"] < 0, 1, 0)
    frame["uptick_count"] = np.where(frame["side"] > 0, 1, 0)
    frame["downtick_count"] = np.where(frame["side"] < 0, 1, 0)
    frame["zero_tick_count"] = np.where(frame["side"] == 0, 1, 0)
    frame["large_trade_volume"] = np.where(frame["is_large_trade"], frame["size"], 0.0)
    frame["large_trade_dollar_volume"] = np.where(frame["is_large_trade"], frame["dollar"], 0.0)
    frame["large_trade_buy_vol_proxy"] = np.where(frame["is_large_trade"] & frame["side"].gt(0), frame["size"], 0.0)
    frame["large_trade_sell_vol_proxy"] = np.where(frame["is_large_trade"] & frame["side"].lt(0), frame["size"], 0.0)
    frame["odd_lot_trade_count"] = np.where(frame["is_odd_lot"], 1, 0)
    frame["odd_lot_volume"] = np.where(frame["is_odd_lot"], frame["size"], 0.0)
    frame["block_trade_count"] = np.where(frame["is_block_trade"], 1, 0)
    frame["block_trade_volume"] = np.where(frame["is_block_trade"], frame["size"], 0.0)
    frame["lit_volume"] = np.where(frame["is_off_exchange"], 0.0, frame["size"])
    frame["off_exchange_volume"] = np.where(frame["is_off_exchange"], frame["size"], 0.0)
    frame["lit_trade_count"] = np.where(frame["is_off_exchange"], 0, 1)
    frame["off_exchange_trade_count"] = np.where(frame["is_off_exchange"], 1, 0)

    grouped = (
        frame.groupby(["ticker", "minute"], sort=False)
        .agg(
            trade_count=("price", "count"),
            volume=("size", "sum"),
            dollar_volume=("dollar", "sum"),
            vwap_num=("dollar", "sum"),
            avg_trade_size_num=("size", "sum"),
            median_trade_size=("size", "median"),
            max_trade_size=("size", "max"),
            buy_vol_proxy=("buy_vol_proxy", "sum"),
            sell_vol_proxy=("sell_vol_proxy", "sum"),
            buy_trade_count_proxy=("buy_trade_count_proxy", "sum"),
            sell_trade_count_proxy=("sell_trade_count_proxy", "sum"),
            uptick_count=("uptick_count", "sum"),
            downtick_count=("downtick_count", "sum"),
            zero_tick_count=("zero_tick_count", "sum"),
            large_trade_count=("is_large_trade", "sum"),
            large_trade_volume=("large_trade_volume", "sum"),
            large_trade_dollar_volume=("large_trade_dollar_volume", "sum"),
            large_trade_buy_vol_proxy=("large_trade_buy_vol_proxy", "sum"),
            large_trade_sell_vol_proxy=("large_trade_sell_vol_proxy", "sum"),
            odd_lot_trade_count=("odd_lot_trade_count", "sum"),
            odd_lot_volume=("odd_lot_volume", "sum"),
            block_trade_count=("block_trade_count", "sum"),
            block_trade_volume=("block_trade_volume", "sum"),
            lit_volume=("lit_volume", "sum"),
            off_exchange_volume=("off_exchange_volume", "sum"),
            lit_trade_count=("lit_trade_count", "sum"),
            off_exchange_trade_count=("off_exchange_trade_count", "sum"),
            correction_count=("correction", lambda series: int((series > 0).sum())),
            unique_exchange_count=("exchange", "nunique"),
            avg_report_lag_ns=("report_lag_ns", "mean"),
            max_report_lag_ns=("report_lag_ns", "max"),
        )
        .reset_index()
    )

    for ticker, group in frame.groupby("ticker", sort=False):
        last_row = group.iloc[-1]
        prior_state[ticker] = {
            "last_price": float(last_row["price"]),
            "last_side": int(last_row["side"]),
            "last_timestamp": int(last_row["participant_timestamp"]),
            "last_sequence_number": int(last_row["sequence_number"]),
        }

    return grouped, prior_state


def finalize_trade_flow(partials: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in partials if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    numeric_sum_columns = [
        "trade_count",
        "volume",
        "dollar_volume",
        "vwap_num",
        "avg_trade_size_num",
        "buy_vol_proxy",
        "sell_vol_proxy",
        "buy_trade_count_proxy",
        "sell_trade_count_proxy",
        "uptick_count",
        "downtick_count",
        "zero_tick_count",
        "large_trade_count",
        "large_trade_volume",
        "large_trade_dollar_volume",
        "large_trade_buy_vol_proxy",
        "large_trade_sell_vol_proxy",
        "odd_lot_trade_count",
        "odd_lot_volume",
        "block_trade_count",
        "block_trade_volume",
        "lit_volume",
        "off_exchange_volume",
        "lit_trade_count",
        "off_exchange_trade_count",
        "correction_count",
    ]

    aggregations: dict[str, str] = {column: "sum" for column in numeric_sum_columns}
    aggregations.update(
        {
            "median_trade_size": "mean",
            "max_trade_size": "max",
            "unique_exchange_count": "max",
            "avg_report_lag_ns": "mean",
            "max_report_lag_ns": "max",
        }
    )
    grouped = combined.groupby(["ticker", "minute"], sort=False).agg(aggregations).reset_index()
    grouped["vwap"] = np.where(grouped["volume"] > 0, grouped["vwap_num"] / grouped["volume"], np.nan)
    grouped["avg_trade_size"] = np.where(grouped["trade_count"] > 0, grouped["avg_trade_size_num"] / grouped["trade_count"], np.nan)
    grouped["imbalance_proxy"] = np.where(
        grouped["volume"] > 0,
        (grouped["buy_vol_proxy"] - grouped["sell_vol_proxy"]) / grouped["volume"],
        0.0,
    )
    grouped["trade_count_imbalance_proxy"] = np.where(
        grouped["trade_count"] > 0,
        (grouped["buy_trade_count_proxy"] - grouped["sell_trade_count_proxy"]) / grouped["trade_count"],
        0.0,
    )
    grouped = grouped.drop(columns=["vwap_num", "avg_trade_size_num"])

    return grouped[TRADE_FLOW_COLUMNS].sort_values(["minute", "ticker"]).reset_index(drop=True)


def build_split_adjustment_factors(splits: pd.DataFrame, trading_dates: Sequence[str]) -> pd.DataFrame:
    if splits.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "cumulative_split_multiplier", "future_split_multiplier", "price_adjustment_factor", "volume_adjustment_factor"])

    frame = splits.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if "split_date" not in frame.columns and "date" in frame.columns:
        frame = frame.rename(columns={"date": "split_date"})
    frame["split_date"] = pd.to_datetime(frame["split_date"]).dt.date
    frame["split_multiplier"] = pd.to_numeric(frame["to"], errors="coerce") / pd.to_numeric(frame["from"], errors="coerce")
    unique_dates = [pd.Timestamp(date).date() for date in trading_dates]

    rows: list[dict[str, object]] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        group = group.sort_values("split_date")
        for trade_date in unique_dates:
            past = group.loc[group["split_date"] <= trade_date, "split_multiplier"]
            future = group.loc[group["split_date"] > trade_date, "split_multiplier"]
            cumulative_split_multiplier = float(past.prod()) if not past.empty else 1.0
            future_split_multiplier = float(future.prod()) if not future.empty else 1.0
            price_adjustment_factor = 1.0 / future_split_multiplier
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date.isoformat(),
                    "cumulative_split_multiplier": cumulative_split_multiplier,
                    "future_split_multiplier": future_split_multiplier,
                    "price_adjustment_factor": price_adjustment_factor,
                    "volume_adjustment_factor": 1.0 / price_adjustment_factor,
                }
            )
    return pd.DataFrame(rows)


def load_splits_csv(path: Path | str) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    frame = pd.read_csv(source)
    frame = frame.rename(columns={"date": "split_date"})
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["split_date"] = pd.to_datetime(frame["split_date"]).dt.date
    frame["from"] = pd.to_numeric(frame["from"], errors="coerce")
    frame["to"] = pd.to_numeric(frame["to"], errors="coerce")
    frame["split_multiplier"] = frame["to"] / frame["from"]
    return frame.sort_values(["symbol", "split_date"]).reset_index(drop=True)


def build_intraday_features(bars_1m: pd.DataFrame, trade_flow_1m: pd.DataFrame) -> pd.DataFrame:
    bars = bars_1m.copy()
    flow = ensure_trade_flow_schema(trade_flow_1m)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    flow["minute"] = pd.to_datetime(flow["minute"], utc=True)

    merged = bars.merge(
        flow,
        left_on=["symbol", "timestamp"],
        right_on=["ticker", "minute"],
        how="left",
        suffixes=("", "_flow"),
    )
    merged = merged.drop(columns=[column for column in ["ticker", "minute"] if column in merged.columns])
    merged = merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    by_symbol = merged.groupby("symbol", sort=False)
    for lag in (1, 3, 5, 15):
        merged[f"ret_{lag}m_past"] = by_symbol["close"].pct_change(lag)

    merged["vwap_distance"] = np.where(merged["vwap"] > 0, merged["close"] / merged["vwap"] - 1, np.nan)
    merged["large_trade_ratio"] = np.where(
        merged["dollar_volume"].fillna(0) > 0,
        merged["large_trade_dollar_volume"].fillna(0) / merged["dollar_volume"].fillna(0),
        0.0,
    )
    merged["off_exchange_ratio"] = np.where(
        merged["volume"].fillna(0) > 0,
        merged["off_exchange_volume"].fillna(0) / merged["volume"].fillna(0),
        0.0,
    )
    merged["price_impact_proxy"] = np.where(
        merged["dollar_volume"].fillna(0) > 0,
        (merged["close"] - merged["open"]).abs() / merged["dollar_volume"].replace(0, np.nan),
        np.nan,
    )

    for base, output in [
        ("volume", "volume_z_proxy"),
        ("trade_count", "trade_count_z_proxy"),
        ("dollar_volume", "dollar_volume_z_proxy"),
    ]:
        source_column = base if base in merged.columns else f"{base}_flow"
        rolling_mean = merged.groupby("symbol", sort=False)[source_column].transform(
            lambda series: series.shift(1).rolling(30, min_periods=1).mean()
        )
        merged[output] = np.where(rolling_mean > 0, merged[source_column] / rolling_mean - 1, np.nan)

    for source_column, output in [
        ("imbalance_proxy", "imbalance_z"),
        ("large_trade_ratio", "large_trade_ratio_z"),
        ("off_exchange_ratio", "off_exchange_ratio_z"),
    ]:
        rolling_mean = merged.groupby("symbol", sort=False)[source_column].transform(
            lambda series: series.shift(1).rolling(30, min_periods=3).mean()
        )
        rolling_std = merged.groupby("symbol", sort=False)[source_column].transform(
            lambda series: series.shift(1).rolling(30, min_periods=3).std(ddof=0)
        )
        merged[output] = ((merged[source_column] - rolling_mean) / rolling_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    merged["flow_impulse_score"] = (
        0.35 * merged["dollar_volume_z_proxy"].fillna(0.0)
        + 0.25 * merged["trade_count_z_proxy"].fillna(0.0)
        + 0.25 * merged["imbalance_z"].fillna(0.0)
        + 0.15 * merged["large_trade_ratio_z"].fillna(0.0)
    )

    return merged


def build_labels(bars_1m: pd.DataFrame) -> pd.DataFrame:
    bars = bars_1m.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    grouped = bars.groupby("symbol", sort=False)["close"]

    output = bars[["symbol", "timestamp"]].copy()
    for horizon in (1, 5, 15, 30):
        output[f"future_ret_{horizon}m"] = grouped.shift(-horizon) / bars["close"] - 1
    return output


def register_market_data_views(
    con: duckdb.DuckDBPyConnection,
    paths: AlphaPaths,
    *,
    name_overrides: dict[str, str] | None = None,
    selected_views: Sequence[str] | None = None,
) -> None:
    mapping = {
        "bars_1m": paths.raw_1m_root,
        "trade_flow_1m": paths.trade_flow_1m_root,
        "bars_5m": paths.bars_5m_root,
        "bars_15m": paths.bars_15m_root,
        "features_1m": paths.features_1m_root,
        "labels_1m": paths.labels_1m_root,
    }
    timestamp_columns = {
        "bars_1m": ["timestamp", "bar_end"],
        "trade_flow_1m": ["minute"],
        "bars_5m": ["timestamp"],
        "bars_15m": ["timestamp"],
        "features_1m": ["timestamp"],
        "labels_1m": ["timestamp"],
    }
    selected = set(selected_views) if selected_views else None
    for view_name, root in mapping.items():
        if selected is not None and view_name not in selected:
            continue
        actual_name = name_overrides.get(view_name, view_name) if name_overrides else view_name
        parquet_glob = Path(root) / "date=*" / "*.parquet"
        normalized_glob = parquet_glob.as_posix()
        if not root.exists():
            continue
        replacements = ", ".join(
            f"({column} AT TIME ZONE 'UTC') AS {column}" for column in timestamp_columns.get(view_name, [])
        )
        if replacements:
            query = (
                f"CREATE OR REPLACE VIEW {actual_name} AS "
                f"SELECT * REPLACE ({replacements}) "
                f"FROM read_parquet('{normalized_glob}', hive_partitioning=1, union_by_name=1)"
            )
        else:
            query = (
                f"CREATE OR REPLACE VIEW {actual_name} AS "
                f"SELECT * FROM read_parquet('{normalized_glob}', hive_partitioning=1, union_by_name=1)"
            )
        con.execute(query)


def load_vendor_1m_zip(zip_path: Path | str, *, symbols: Sequence[str] | None = None, source: str = "vendor_1m") -> pd.DataFrame:
    selected = {symbol.upper() for symbol in symbols} if symbols else None
    source_path = Path(zip_path).expanduser().resolve()
    if source_path.suffix.lower() == ".gz":
        with gzip.open(source_path, "rt", encoding="utf-8") as handle:
            raw = pd.read_csv(handle)
        normalized = normalize_vendor_legacy_gzip_bars(raw, source=f"{source}_legacy_gz")
        if selected is not None:
            normalized = normalized[normalized["symbol"].isin(selected)].copy()
        return normalized.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    password = generate_zip_password(source_path.name).encode("utf-8")
    frames: list[pd.DataFrame] = []

    with pyzipper.AESZipFile(source_path) as archive:
        archive.setpassword(password)
        names = archive.namelist()
        for name in names:
            symbol = Path(name).stem.upper()
            if selected is not None and symbol not in selected:
                continue
            with archive.open(name) as handle:
                raw = pd.read_csv(handle)
            if not raw.empty:
                frames.append(normalize_vendor_1m_bars(raw, source=source))

    if not frames:
        return pd.DataFrame(columns=["symbol", "timestamp", "bar_end", "open", "high", "low", "close", "volume", "dollar_volume", "vwap", "exchange", "bar_type", "sequence", "source"])
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def ingest_trade_zip_to_flow(
    zip_path: Path | str,
    *,
    chunksize: int = 2_000_000,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    selected_columns = usecols or [
        "ticker",
        "conditions",
        "correction",
        "exchange",
        "participant_timestamp",
        "price",
        "sequence_number",
        "sip_timestamp",
        "size",
        "tape",
        "trf_id",
        "trf_timestamp",
    ]
    state: dict[str, dict[str, float | int]] = {}
    con = duckdb.connect()
    partial_table_created = False
    for chunk in iter_encrypted_zip_csv_chunks(zip_path, usecols=selected_columns, chunksize=chunksize):
        partial, state = summarize_trade_chunk(chunk, state=state)
        if not partial.empty:
            con.register("partial_chunk", partial)
            if not partial_table_created:
                con.execute("CREATE TABLE trade_flow_partials AS SELECT * FROM partial_chunk")
                partial_table_created = True
            else:
                con.execute("INSERT INTO trade_flow_partials SELECT * FROM partial_chunk")
            con.unregister("partial_chunk")
    if not partial_table_created:
        con.close()
        return empty_trade_flow_frame()
    final = con.execute(
        """
        SELECT
            ticker,
            minute,
            SUM(trade_count) AS trade_count,
            SUM(volume) AS volume,
            SUM(dollar_volume) AS dollar_volume,
            SUM(vwap_num) / NULLIF(SUM(volume), 0) AS vwap,
            SUM(avg_trade_size_num) / NULLIF(SUM(trade_count), 0) AS avg_trade_size,
            AVG(median_trade_size) AS median_trade_size,
            MAX(max_trade_size) AS max_trade_size,
            SUM(buy_vol_proxy) AS buy_vol_proxy,
            SUM(sell_vol_proxy) AS sell_vol_proxy,
            SUM(buy_trade_count_proxy) AS buy_trade_count_proxy,
            SUM(sell_trade_count_proxy) AS sell_trade_count_proxy,
            SUM(uptick_count) AS uptick_count,
            SUM(downtick_count) AS downtick_count,
            SUM(zero_tick_count) AS zero_tick_count,
            (SUM(buy_vol_proxy) - SUM(sell_vol_proxy)) / NULLIF(SUM(volume), 0) AS imbalance_proxy,
            (SUM(buy_trade_count_proxy) - SUM(sell_trade_count_proxy)) / NULLIF(SUM(trade_count), 0) AS trade_count_imbalance_proxy,
            SUM(large_trade_count) AS large_trade_count,
            SUM(large_trade_volume) AS large_trade_volume,
            SUM(large_trade_dollar_volume) AS large_trade_dollar_volume,
            SUM(large_trade_buy_vol_proxy) AS large_trade_buy_vol_proxy,
            SUM(large_trade_sell_vol_proxy) AS large_trade_sell_vol_proxy,
            SUM(odd_lot_trade_count) AS odd_lot_trade_count,
            SUM(odd_lot_volume) AS odd_lot_volume,
            SUM(block_trade_count) AS block_trade_count,
            SUM(block_trade_volume) AS block_trade_volume,
            SUM(lit_volume) AS lit_volume,
            SUM(off_exchange_volume) AS off_exchange_volume,
            SUM(lit_trade_count) AS lit_trade_count,
            SUM(off_exchange_trade_count) AS off_exchange_trade_count,
            SUM(correction_count) AS correction_count,
            MAX(unique_exchange_count) AS unique_exchange_count,
            AVG(avg_report_lag_ns) AS avg_report_lag_ns,
            MAX(max_report_lag_ns) AS max_report_lag_ns
        FROM trade_flow_partials
        GROUP BY ticker, minute
        ORDER BY minute, ticker
        """
    ).df()
    con.close()
    return ensure_trade_flow_schema(final)


def _feature_view_sql(target_view: str, bars_view: str, trade_flow_view: str) -> str:
    return f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT
            b.symbol,
            b.timestamp,
            b.bar_end,
            b.open,
            b.high,
            b.low,
            b.close,
            b.volume,
            b.dollar_volume,
            b.vwap,
            b.exchange,
            b.bar_type,
            b.sequence,
            b.source,
            b.date,
            f.trade_count,
            f.buy_vol_proxy,
            f.sell_vol_proxy,
            f.buy_trade_count_proxy,
            f.sell_trade_count_proxy,
            f.uptick_count,
            f.downtick_count,
            f.zero_tick_count,
            f.imbalance_proxy,
            f.trade_count_imbalance_proxy,
            f.large_trade_count,
            f.large_trade_volume,
            f.large_trade_dollar_volume,
            f.large_trade_buy_vol_proxy,
            f.large_trade_sell_vol_proxy,
            f.odd_lot_trade_count,
            f.odd_lot_volume,
            f.block_trade_count,
            f.block_trade_volume,
            f.lit_volume,
            f.off_exchange_volume,
            f.lit_trade_count,
            f.off_exchange_trade_count,
            f.correction_count,
            f.unique_exchange_count,
            f.avg_report_lag_ns,
            f.max_report_lag_ns,
            b.close / NULLIF(LAG(b.close, 1) OVER w, 0) - 1 AS ret_1m_past,
            b.close / NULLIF(LAG(b.close, 3) OVER w, 0) - 1 AS ret_3m_past,
            b.close / NULLIF(LAG(b.close, 5) OVER w, 0) - 1 AS ret_5m_past,
            b.close / NULLIF(LAG(b.close, 15) OVER w, 0) - 1 AS ret_15m_past,
            b.close / NULLIF(b.vwap, 0) - 1 AS vwap_distance,
            f.large_trade_dollar_volume / NULLIF(f.dollar_volume, 0) AS large_trade_ratio,
            ABS(b.close - b.open) / NULLIF(f.dollar_volume, 0) AS price_impact_proxy,
            b.volume / NULLIF(AVG(b.volume) OVER w30, 0) - 1 AS volume_z_proxy,
            f.trade_count / NULLIF(AVG(f.trade_count) OVER w30, 0) - 1 AS trade_count_z_proxy,
            f.dollar_volume / NULLIF(AVG(f.dollar_volume) OVER w30, 0) - 1 AS dollar_volume_z_proxy
        FROM {bars_view} b
        LEFT JOIN {trade_flow_view} f
          ON b.symbol = f.ticker
         AND b.timestamp = f.minute
        WINDOW
            w AS (PARTITION BY b.symbol ORDER BY b.timestamp),
            w30 AS (
                PARTITION BY b.symbol
                ORDER BY b.timestamp
                ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
            )
    """


def _label_view_sql(target_view: str, bars_view: str, *, shift_minutes: int = 0) -> str:
    shift_expr = f" + INTERVAL {shift_minutes} MINUTE" if shift_minutes else ""
    return f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT
            symbol,
            timestamp{shift_expr} AS timestamp,
            LEAD(close, 1) OVER w / NULLIF(close, 0) - 1 AS future_ret_1m,
            LEAD(close, 5) OVER w / NULLIF(close, 0) - 1 AS future_ret_5m,
            LEAD(close, 15) OVER w / NULLIF(close, 0) - 1 AS future_ret_15m,
            LEAD(close, 30) OVER w / NULLIF(close, 0) - 1 AS future_ret_30m,
            date
        FROM {bars_view}
        WINDOW w AS (PARTITION BY symbol ORDER BY timestamp)
    """


def _adjusted_bars_view_sql(target_view: str, bars_view: str) -> str:
    return f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT
            b.*,
            COALESCE(f.cumulative_split_multiplier, 1.0) AS forward_adjustment_factor,
            b.open * COALESCE(f.cumulative_split_multiplier, 1.0) AS open_forward_adjusted,
            b.high * COALESCE(f.cumulative_split_multiplier, 1.0) AS high_forward_adjusted,
            b.low * COALESCE(f.cumulative_split_multiplier, 1.0) AS low_forward_adjusted,
            b.close * COALESCE(f.cumulative_split_multiplier, 1.0) AS close_forward_adjusted,
            b.volume / NULLIF(COALESCE(f.cumulative_split_multiplier, 1.0), 0.0) AS volume_forward_adjusted
        FROM {bars_view} b
        LEFT JOIN split_adjustment_factors f
          ON b.symbol = f.symbol
         AND CAST(b.timestamp AS DATE) = CAST(f.trade_date AS DATE)
    """


def _empty_trade_flow_view_sql(target_view: str) -> str:
    numeric_columns = ",\n            ".join(
        f"CAST(NULL AS DOUBLE) AS {column}" for column in TRADE_FLOW_NUMERIC_COLUMNS
    )
    return f"""
        CREATE OR REPLACE VIEW {target_view} AS
        SELECT
            CAST(NULL AS VARCHAR) AS ticker,
            CAST(NULL AS TIMESTAMP) AS minute,
            {numeric_columns},
            CAST(NULL AS DATE) AS date
        WHERE 1 = 0
    """


def initialize_market_database(paths: AlphaPaths) -> Path:
    db_path = paths.ensure_parent(paths.market_db_file)
    con = duckdb.connect(str(db_path))
    try:
        for schema_name in ("raw", "research", "backtest"):
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

        register_market_data_views(
            con,
            paths,
            name_overrides={
                "bars_1m": "raw.bars_1m",
                "trade_flow_1m": "raw.trade_flow_1m",
                "bars_5m": "raw.bars_5m",
                "bars_15m": "raw.bars_15m",
            },
            selected_views=("bars_1m", "trade_flow_1m", "bars_5m", "bars_15m"),
        )
        if not paths.trade_flow_1m_root.exists():
            con.execute(_empty_trade_flow_view_sql("raw.trade_flow_1m"))

        if paths.labels_1m_root.exists():
            register_market_data_views(
                con,
                paths,
                name_overrides={"labels_1m": "research.labels_1m_materialized"},
                selected_views=("labels_1m",),
            )

        if paths.splits_path().exists():
            con.execute(
                f"CREATE OR REPLACE VIEW splits AS SELECT * FROM read_parquet('{paths.splits_path().as_posix()}')"
            )
        if paths.split_factors_path().exists():
            con.execute(
                f"CREATE OR REPLACE VIEW split_adjustment_factors AS SELECT * FROM read_parquet('{paths.split_factors_path().as_posix()}')"
            )

        if paths.labels_1m_root.exists():
            con.execute("CREATE OR REPLACE VIEW research.labels_1m AS SELECT * FROM research.labels_1m_materialized")
        else:
            con.execute(_label_view_sql("research.labels_1m", "raw.bars_1m"))

        con.execute(_feature_view_sql("research.features_1m", "raw.bars_1m", "raw.trade_flow_1m"))
        if paths.split_factors_path().exists():
            con.execute(_adjusted_bars_view_sql("research.bars_1m_forward_adjusted", "raw.bars_1m"))
        else:
            con.execute(
                """
                CREATE OR REPLACE VIEW research.bars_1m_forward_adjusted AS
                SELECT
                    b.*,
                    1.0 AS forward_adjustment_factor,
                    b.open AS open_forward_adjusted,
                    b.high AS high_forward_adjusted,
                    b.low AS low_forward_adjusted,
                    b.close AS close_forward_adjusted,
                    b.volume AS volume_forward_adjusted
                FROM raw.bars_1m b
                """
            )

        con.execute(
            """
            CREATE OR REPLACE VIEW backtest.bars_1m AS
            SELECT
                symbol,
                timestamp + INTERVAL 1 MINUTE AS timestamp,
                bar_end + INTERVAL 1 MINUTE AS bar_end,
                open,
                high,
                low,
                close,
                volume,
                dollar_volume,
                vwap,
                exchange,
                bar_type,
                sequence,
                source,
                date
            FROM raw.bars_1m
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW backtest.trade_flow_1m AS
            SELECT
                ticker,
                minute + INTERVAL 1 MINUTE AS minute,
                trade_count,
                volume,
                dollar_volume,
                vwap,
                avg_trade_size,
                median_trade_size,
                max_trade_size,
                buy_vol_proxy,
                sell_vol_proxy,
                buy_trade_count_proxy,
                sell_trade_count_proxy,
                uptick_count,
                downtick_count,
                zero_tick_count,
                imbalance_proxy,
                trade_count_imbalance_proxy,
                large_trade_count,
                large_trade_volume,
                large_trade_dollar_volume,
                large_trade_buy_vol_proxy,
                large_trade_sell_vol_proxy,
                odd_lot_trade_count,
                odd_lot_volume,
                block_trade_count,
                block_trade_volume,
                lit_volume,
                off_exchange_volume,
                lit_trade_count,
                off_exchange_trade_count,
                correction_count,
                unique_exchange_count,
                avg_report_lag_ns,
                max_report_lag_ns,
                date
            FROM raw.trade_flow_1m
            """
        )
        con.execute(_feature_view_sql("backtest.features_1m", "backtest.bars_1m", "backtest.trade_flow_1m"))
        con.execute(
            """
            CREATE OR REPLACE VIEW backtest.labels_1m AS
            SELECT
                symbol,
                timestamp + INTERVAL 1 MINUTE AS timestamp,
                future_ret_1m,
                future_ret_5m,
                future_ret_15m,
                future_ret_30m,
                date
            FROM research.labels_1m
            """
        )
        if paths.split_factors_path().exists():
            con.execute(_adjusted_bars_view_sql("backtest.bars_1m_forward_adjusted", "backtest.bars_1m"))
        else:
            con.execute(
                """
                CREATE OR REPLACE VIEW backtest.bars_1m_forward_adjusted AS
                SELECT
                    b.*,
                    1.0 AS forward_adjustment_factor,
                    b.open AS open_forward_adjusted,
                    b.high AS high_forward_adjusted,
                    b.low AS low_forward_adjusted,
                    b.close AS close_forward_adjusted,
                    b.volume AS volume_forward_adjusted
                FROM backtest.bars_1m b
                """
            )

        compatibility_aliases = {
            "bars_1m_raw": "raw.bars_1m",
            "trade_flow_1m_raw": "raw.trade_flow_1m",
            "features_1m_raw": "research.features_1m",
            "labels_1m_raw": "research.labels_1m",
            "bars_1m": "backtest.bars_1m",
            "trade_flow_1m": "backtest.trade_flow_1m",
            "features_1m": "backtest.features_1m",
            "labels_1m": "backtest.labels_1m",
            "bars_1m_forward_adjusted": "backtest.bars_1m_forward_adjusted",
        }
        if paths.bars_5m_root.exists():
            compatibility_aliases["bars_5m_raw"] = "raw.bars_5m"
        if paths.bars_15m_root.exists():
            compatibility_aliases["bars_15m_raw"] = "raw.bars_15m"
        for alias, source in compatibility_aliases.items():
            try:
                con.execute(f"CREATE OR REPLACE VIEW {alias} AS SELECT * FROM {source}")
            except duckdb.Error:
                continue
    finally:
        con.close()
    return db_path


def initialize_backtest_database(paths: AlphaPaths, db_path: Path | str | None = None) -> Path:
    return initialize_market_database(paths)


def discover_month_dates(
    bars_month_dir: Path | str,
    trades_month_dir: Path | str,
    *,
    selected_dates: Sequence[str] | None = None,
) -> list[str]:
    bars_dir = Path(bars_month_dir).expanduser().resolve()
    trades_dir = Path(trades_month_dir).expanduser().resolve()
    bar_dates = {path.stem for path in bars_dir.glob("*.zip")}
    trade_dates = {path.stem for path in trades_dir.glob("*.zip")}
    common = sorted(bar_dates & trade_dates)
    if selected_dates:
        selected = {date.replace("-", "") for date in selected_dates}
        common = [date for date in common if date in selected]
    return [f"{date[:4]}-{date[4:6]}-{date[6:8]}" for date in common]


def discover_bars_only_zip_paths(
    bars_roots: Sequence[Path | str],
    *,
    selected_dates: Sequence[str] | None = None,
) -> list[tuple[str, Path]]:
    selected = {date.replace("-", "") for date in selected_dates} if selected_dates else None
    discovered: dict[str, Path] = {}

    for root in bars_roots:
        base = Path(root).expanduser().resolve()
        if base.is_file():
            paths = [base]
        else:
            paths = sorted(path for pattern in ("*.zip", "*.gz") for path in base.rglob(pattern))
        for archive_path in paths:
            stem = archive_path.stem
            if len(stem) != 8 or not stem.isdigit():
                continue
            if selected and stem not in selected:
                continue
            trade_date = f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"
            resolved = archive_path.resolve()
            existing = discovered.get(trade_date)
            if existing is not None and existing != resolved:
                raise ValueError(f"Duplicate bars zip found for {trade_date}: {existing} and {resolved}")
            discovered[trade_date] = resolved

    return sorted(discovered.items(), key=lambda item: item[0])


def write_reference_tables(
    paths: AlphaPaths,
    splits_csv_path: Path | str,
    trade_dates: Sequence[str],
) -> tuple[Path, Path]:
    splits = load_splits_csv(splits_csv_path)
    split_factors = build_split_adjustment_factors(splits, trade_dates)
    splits_path = paths.ensure_parent(paths.splits_path())
    split_factors_path = paths.ensure_parent(paths.split_factors_path())
    splits.to_parquet(splits_path, index=False)
    split_factors.to_parquet(split_factors_path, index=False)
    return splits_path, split_factors_path


def build_daily_bars_only_artifacts(
    trade_date: str,
    *,
    paths: AlphaPaths,
    bars_zip_path: Path | str,
    skip_existing: bool = False,
) -> dict[str, object]:
    raw_output = paths.raw_1m_path(trade_date)
    bars_5m_output = paths.bars_path(trade_date, "5m")
    bars_15m_output = paths.bars_path(trade_date, "15m")
    labels_output = paths.labels_1m_path(trade_date)
    required_outputs = [raw_output, bars_5m_output, bars_15m_output, labels_output]
    if skip_existing and all(path.exists() for path in required_outputs):
        return {
            "trade_date": trade_date,
            "status": "skipped",
            "bars_path": str(raw_output),
            "labels_path": str(labels_output),
        }

    bars_1m = load_vendor_1m_zip(bars_zip_path)
    labels_1m = build_labels(bars_1m)
    bars_5m = resample_ohlcv_bars(bars_1m, "5m")
    bars_15m = resample_ohlcv_bars(bars_1m, "15m")

    bars_1m.to_parquet(paths.ensure_parent(raw_output), index=False)
    bars_5m.to_parquet(paths.ensure_parent(bars_5m_output), index=False)
    bars_15m.to_parquet(paths.ensure_parent(bars_15m_output), index=False)
    labels_1m.to_parquet(paths.ensure_parent(labels_output), index=False)

    return {
        "trade_date": trade_date,
        "status": "built",
        "bars_rows": len(bars_1m),
        "features_rows": len(bars_1m),
        "labels_rows": len(labels_1m),
        "bars_path": str(raw_output),
        "labels_path": str(labels_output),
    }


def build_daily_market_artifacts(
    trade_date: str,
    *,
    paths: AlphaPaths,
    bars_zip_path: Path | str,
    trades_zip_path: Path | str,
    trade_chunksize: int = 2_000_000,
    skip_existing: bool = False,
) -> dict[str, object]:
    raw_output = paths.raw_1m_path(trade_date)
    flow_output = paths.trade_flow_1m_path(trade_date)
    labels_output = paths.labels_1m_path(trade_date)
    if skip_existing and raw_output.exists() and flow_output.exists() and labels_output.exists():
        return {
            "trade_date": trade_date,
            "status": "skipped",
            "bars_path": str(raw_output),
            "trade_flow_path": str(flow_output),
        }

    bars_1m = load_vendor_1m_zip(bars_zip_path)
    flow_1m = ensure_trade_flow_schema(ingest_trade_zip_to_flow(trades_zip_path, chunksize=trade_chunksize))
    labels_1m = build_labels(bars_1m)
    bars_5m = resample_ohlcv_bars(bars_1m, "5m")
    bars_15m = resample_ohlcv_bars(bars_1m, "15m")

    bars_1m.to_parquet(paths.ensure_parent(raw_output), index=False)
    flow_1m.to_parquet(paths.ensure_parent(flow_output), index=False)
    bars_5m.to_parquet(paths.ensure_parent(paths.bars_path(trade_date, "5m")), index=False)
    bars_15m.to_parquet(paths.ensure_parent(paths.bars_path(trade_date, "15m")), index=False)
    labels_1m.to_parquet(paths.ensure_parent(labels_output), index=False)

    return {
        "trade_date": trade_date,
        "status": "built",
        "bars_rows": len(bars_1m),
        "trade_flow_rows": len(flow_1m),
        "features_rows": len(bars_1m),
        "labels_rows": len(labels_1m),
        "bars_path": str(raw_output),
        "trade_flow_path": str(flow_output),
    }
