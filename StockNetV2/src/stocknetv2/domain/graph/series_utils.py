from __future__ import annotations

import math

import numpy as np
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


def build_pivot_matrix(
    frame: pd.DataFrame,
    *,
    value_column: str,
    snapshot_time: pd.Timestamp,
    minutes: int | None = None,
) -> pd.DataFrame:
    window = select_time_window(frame, snapshot_time=snapshot_time, minutes=minutes)
    if window.empty or value_column not in window.columns:
        return pd.DataFrame()
    return (
        window.pivot_table(index="timestamp", columns="symbol", values=value_column, aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )


def zscore_frame_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return matrix.copy()

    normalized = matrix.copy().astype(float)
    for column in normalized.columns:
        series = normalized[column].dropna().tolist()
        if not series:
            continue
        zscored = zscore_series(series)
        normalized.loc[normalized[column].notna(), column] = zscored
    return normalized


def compute_overlap_counts(matrix: pd.DataFrame) -> np.ndarray:
    if matrix.empty:
        return np.zeros((0, 0), dtype=int)
    mask = (~matrix.isna()).to_numpy(dtype=np.int32)
    return mask.T @ mask


def compute_same_direction_ratio(matrix: pd.DataFrame) -> np.ndarray:
    if matrix.empty:
        return np.zeros((0, 0), dtype=float)

    values = matrix.to_numpy(dtype=float)
    mask = ~np.isnan(values)
    positive = ((values >= 0) & mask).astype(np.int32)
    negative = ((values < 0) & mask).astype(np.int32)
    same_direction = positive.T @ positive + negative.T @ negative
    overlap = compute_overlap_counts(matrix)
    return np.divide(
        same_direction,
        overlap,
        out=np.zeros_like(same_direction, dtype=float),
        where=overlap > 0,
    )


def compute_above_threshold_ratio(matrix: pd.DataFrame, threshold: float) -> np.ndarray:
    if matrix.empty:
        return np.zeros((0, 0), dtype=float)

    values = matrix.to_numpy(dtype=float)
    mask = ~np.isnan(values)
    above_threshold = ((values > threshold) & mask).astype(np.int32)
    overlap = compute_overlap_counts(matrix)
    co_occurrence = above_threshold.T @ above_threshold
    return np.divide(
        co_occurrence,
        overlap,
        out=np.zeros_like(co_occurrence, dtype=float),
        where=overlap > 0,
    )


def compute_pairwise_correlation_matrix(matrix: pd.DataFrame) -> np.ndarray:
    if matrix.empty:
        return np.zeros((0, 0), dtype=float)
    correlation = matrix.corr(min_periods=2).fillna(0.0).to_numpy(dtype=float).copy()
    values = matrix.to_numpy(dtype=float)
    means = np.nanmean(values, axis=0)
    stds = np.nanstd(values, axis=0)
    constant_mask = stds < 1e-12

    if not constant_mask.any():
        return correlation

    constant_indices = np.where(constant_mask)[0]
    for left_index in constant_indices:
        for right_index in constant_indices:
            left_mean = float(means[left_index])
            right_mean = float(means[right_index])
            if left_mean == 0.0 and right_mean == 0.0:
                correlation[left_index, right_index] = 1.0
            elif (left_mean > 0 and right_mean > 0) or (left_mean < 0 and right_mean < 0):
                correlation[left_index, right_index] = 1.0
            elif (left_mean > 0 > right_mean) or (left_mean < 0 < right_mean):
                correlation[left_index, right_index] = -1.0
            else:
                correlation[left_index, right_index] = 0.0
    return correlation


def select_topk_pair_indices(
    score_matrix: np.ndarray,
    *,
    min_score: float,
    top_k_per_symbol: int,
) -> set[tuple[int, int]]:
    pair_indices: set[tuple[int, int]] = set()
    if score_matrix.size == 0:
        return pair_indices

    size = score_matrix.shape[0]
    if top_k_per_symbol <= 0:
        rows, cols = np.where(np.triu(score_matrix, 1) >= min_score)
        return {(int(row), int(col)) for row, col in zip(rows, cols, strict=False)}

    for row_index in range(size):
        row = score_matrix[row_index].copy()
        row[row_index] = -np.inf
        candidate_count = min(top_k_per_symbol, max(size - 1, 0))
        if candidate_count <= 0:
            continue
        top_indices = np.argpartition(row, -candidate_count)[-candidate_count:]
        for column_index in top_indices:
            score = float(row[column_index])
            if score < min_score:
                continue
            left = min(row_index, int(column_index))
            right = max(row_index, int(column_index))
            if left != right:
                pair_indices.add((left, right))
    return pair_indices
