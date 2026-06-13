from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocknet_alpha.config import AlphaPaths


SPLIT_COLUMNS = [
    "split_id",
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "test_start",
    "test_end",
    "feature_version",
    "model_version",
]


def build_walk_forward_splits(
    start_month: str,
    end_month: str,
    *,
    train_months: int,
    valid_months: int,
    test_months: int,
    feature_version: str = "",
    model_version: str = "",
) -> pd.DataFrame:
    """Build chronological month-based walk-forward splits."""

    months = pd.period_range(start=start_month, end=end_month, freq="M")
    total_window = int(train_months) + int(valid_months) + int(test_months)
    if len(months) < total_window:
        return pd.DataFrame(columns=SPLIT_COLUMNS)

    rows: list[dict[str, str]] = []
    split_number = 1
    for start_idx in range(0, len(months) - total_window + 1):
        train_slice = months[start_idx : start_idx + train_months]
        valid_slice = months[start_idx + train_months : start_idx + train_months + valid_months]
        test_slice = months[start_idx + train_months + valid_months : start_idx + total_window]
        rows.append(
            {
                "split_id": f"wf_{split_number:04d}",
                "train_start": str(train_slice[0]),
                "train_end": str(train_slice[-1]),
                "valid_start": str(valid_slice[0]),
                "valid_end": str(valid_slice[-1]),
                "test_start": str(test_slice[0]),
                "test_end": str(test_slice[-1]),
                "feature_version": feature_version,
                "model_version": model_version,
            }
        )
        split_number += 1

    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def write_walk_forward_splits(splits: pd.DataFrame, paths: AlphaPaths) -> Path:
    output_path = paths.ensure_parent(paths.walk_forward_splits_path())
    splits.to_parquet(output_path, index=False)
    return output_path
