from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocknet_alpha.backtest.walk_forward import build_walk_forward_splits, write_walk_forward_splits
from stocknet_alpha.config import AlphaPaths


def test_build_walk_forward_splits_rolls_month_windows_without_overlap():
    splits = build_walk_forward_splits(
        start_month="2025-09",
        end_month="2026-05",
        train_months=5,
        valid_months=1,
        test_months=1,
        feature_version="v1",
        model_version="baseline_a",
    )

    assert len(splits) == 3
    assert list(splits["split_id"]) == ["wf_0001", "wf_0002", "wf_0003"]

    first = splits.iloc[0]
    assert first["train_start"] == "2025-09"
    assert first["train_end"] == "2026-01"
    assert first["valid_start"] == "2026-02"
    assert first["valid_end"] == "2026-02"
    assert first["test_start"] == "2026-03"
    assert first["test_end"] == "2026-03"

    second = splits.iloc[1]
    assert second["train_start"] == "2025-10"
    assert second["train_end"] == "2026-02"
    assert second["valid_start"] == "2026-03"
    assert second["test_start"] == "2026-04"
    assert (splits["feature_version"] == "v1").all()
    assert (splits["model_version"] == "baseline_a").all()


def test_build_walk_forward_splits_returns_empty_when_window_does_not_fit():
    splits = build_walk_forward_splits(
        start_month="2025-09",
        end_month="2025-12",
        train_months=5,
        valid_months=1,
        test_months=1,
    )

    assert splits.empty
    assert list(splits.columns) == [
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


def test_write_walk_forward_splits_persists_metadata(tmp_path: Path):
    paths = AlphaPaths(repo_root=tmp_path)
    splits = build_walk_forward_splits(
        start_month="2025-09",
        end_month="2026-05",
        train_months=5,
        valid_months=1,
        test_months=1,
        feature_version="feat_v2",
        model_version="model_v3",
    )

    output_path = write_walk_forward_splits(splits, paths)

    assert output_path.exists()
    loaded = pd.read_parquet(output_path)
    assert len(loaded) == 3
    assert loaded.iloc[0]["feature_version"] == "feat_v2"
    assert loaded.iloc[0]["model_version"] == "model_v3"
