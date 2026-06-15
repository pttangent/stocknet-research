from __future__ import annotations

import math

import pandas as pd


def select_time_window(frame: pd.DataFrame, *, snapshot_time: pd.Timestamp, minutes: int | None = None) -> pd.DataFrame:
    window = frame[frame["timestamp"] <= snapshot_time].copy()
    if minutes is not None:
        start = snapshot_time - pd.Timedelta(minutes=minutes - 1)
        window = window[window["timestamp"] >= start].copy()
    return window


def zscore_series(values: list[float]) -> list[float]:
    if not values:
        return []
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = math.sqrt(variance)
    if std_value < 1e-12:
        if mean_value > 0:
            return [1.0 for _ in values]
        if mean_value < 0:
            return [-1.0 for _ in values]
        return [0.0 for _ in values]
    return [(value - mean_value) / std_value for value in values]


def safe_correlation(left: pd.Series, right: pd.Series) -> float:
    joined = pd.DataFrame({"left": left, "right": right}).dropna()
    if joined.empty:
        return 0.0

    left_std = float(joined["left"].std(ddof=0))
    right_std = float(joined["right"].std(ddof=0))
    if left_std < 1e-12 and right_std < 1e-12:
        left_mean = float(joined["left"].mean())
        right_mean = float(joined["right"].mean())
        if left_mean == 0.0 and right_mean == 0.0:
            return 1.0
        if (left_mean > 0 and right_mean > 0) or (left_mean < 0 and right_mean < 0):
            return 1.0
        if (left_mean > 0 > right_mean) or (left_mean < 0 < right_mean):
            return -1.0
        return 0.0

    correlation = float(joined["left"].corr(joined["right"]))
    if pd.isna(correlation):
        return 0.0
    return correlation


def build_symbol_series(
    frame: pd.DataFrame,
    *,
    symbol: str,
    value_column: str,
) -> pd.Series:
    if value_column not in frame.columns:
        return pd.Series(dtype=float)

    symbol_frame = frame.loc[frame["symbol"] == symbol, ["timestamp", value_column]].dropna()
    if symbol_frame.empty:
        return pd.Series(dtype=float)

    series = pd.Series(symbol_frame[value_column].astype(float).to_numpy(), index=symbol_frame["timestamp"])
    return series.groupby(level=0).mean().sort_index()
