from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import pickle

import pandas as pd

from stocknetwork.graph_snapshots import build_snapshot_dataset
from stocknetwork.graph_snapshots import compute_symbol_features
from stocknetwork.features import compute_residual_returns, compute_rolling_volume_zscore


def _write_manifest(parquet_root: Path, symbols: list[str]) -> None:
    rows = []
    for symbol in symbols:
        rows.append(
            {
                "source_symbol": symbol,
                "symbol": symbol,
                "status": "success",
                "row_count": 8,
                "min_timestamp": "2026-06-02T13:30:00Z",
                "max_timestamp": "2026-06-03T14:15:00Z",
                "exchange_timezone": "America/New_York",
                "last_fetch_at": "2026-06-07T00:00:00Z",
                "error_message": "",
            }
        )
    pd.DataFrame(rows).to_csv(parquet_root / "_manifest.csv", index=False)


def _write_symbol_partition(parquet_root: Path, symbol: str, closes: list[float], volumes: list[int]) -> None:
    timestamps = [
        datetime(2026, 6, 2, 13, 30, tzinfo=UTC),
        datetime(2026, 6, 2, 13, 45, tzinfo=UTC),
        datetime(2026, 6, 2, 14, 0, tzinfo=UTC),
        datetime(2026, 6, 2, 14, 15, tzinfo=UTC),
        datetime(2026, 6, 3, 13, 30, tzinfo=UTC),
        datetime(2026, 6, 3, 13, 45, tzinfo=UTC),
        datetime(2026, 6, 3, 14, 0, tzinfo=UTC),
        datetime(2026, 6, 3, 14, 15, tzinfo=UTC),
    ]
    frame = pd.DataFrame(
        {
            "source_symbol": symbol,
            "symbol": symbol,
            "timestamp": timestamps,
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": volumes,
            "exchange_timezone": "America/New_York",
            "fetch_date": [timestamps[-1] + timedelta(days=1)] * len(timestamps),
        }
    )
    partition = parquet_root / f"symbol={symbol}"
    partition.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(partition / "part-000.parquet", index=False)


def test_build_snapshot_dataset_writes_manifest_and_tensor_snapshots(tmp_path):
    parquet_root = tmp_path / "parquet"
    output_dir = tmp_path / "snapshots"
    parquet_root.mkdir()

    symbols = ["SPY", "AAA", "BBB"]
    _write_manifest(parquet_root, symbols)
    _write_symbol_partition(parquet_root, "SPY", [100, 101, 102, 103, 104, 105, 106, 107], [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
    _write_symbol_partition(parquet_root, "AAA", [50, 51, 52, 53, 54, 55, 56, 57], [200, 210, 220, 230, 240, 250, 260, 270])
    _write_symbol_partition(parquet_root, "BBB", [80, 79, 80, 81, 82, 83, 84, 85], [300, 290, 295, 305, 315, 320, 330, 340])

    result = build_snapshot_dataset(
        parquet_root=parquet_root,
        output_dir=output_dir,
        benchmark_symbol="SPY",
        window_bars=4,
        min_history_bars=4,
        top_k=2,
        edge_threshold=0.0,
    )

    manifest = pd.read_csv(output_dir / "snapshot_manifest.csv")
    assert result["snapshot_count"] == len(manifest)
    assert len(manifest) > 0
    assert set(manifest.columns) >= {
        "snapshot_id",
        "timestamp",
        "num_nodes",
        "num_edges",
        "feature_dim",
        "edge_feature_dim",
        "path",
    }

    first_snapshot_path = output_dir / manifest.iloc[0]["path"]
    with first_snapshot_path.open("rb") as handle:
        payload = pickle.load(handle)

    assert tuple(payload["x"].shape)[0] == 2
    assert payload["feature_names"] == [
        "log_return",
        "residual_return",
        "volume_zscore",
        "intraday_range",
        "rolling_volatility",
        "liquidity_score",
        "degree_centrality",
        "pagerank",
        "community_confidence",
    ]
    assert payload["edge_feature_names"] == [
        "return_corr",
        "residual_corr",
        "volume_corr",
        "edge_strength",
        "edge_persistence",
    ]
    assert payload["symbols"] == ["AAA", "BBB"]
    assert payload["community_ids"].shape[0] == 2
    assert payload["edge_index"].shape[0] == 2
    assert payload["edge_attr"].shape[1] == 5


def test_build_snapshot_dataset_raises_when_benchmark_missing(tmp_path):
    parquet_root = tmp_path / "parquet"
    output_dir = tmp_path / "snapshots"
    parquet_root.mkdir()

    _write_manifest(parquet_root, ["AAA"])
    _write_symbol_partition(parquet_root, "AAA", [50, 51, 52, 53, 54, 55, 56, 57], [200, 210, 220, 230, 240, 250, 260, 270])

    try:
        build_snapshot_dataset(
            parquet_root=parquet_root,
            output_dir=output_dir,
            benchmark_symbol="SPY",
            window_bars=4,
            min_history_bars=4,
        )
    except RuntimeError as exc:
        assert "Benchmark symbol SPY" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when benchmark is missing")


def test_compute_symbol_features_uses_shared_feature_module_logic():
    timestamps = []
    for day in range(1, 13):
        timestamps.extend(
            [
                datetime(2026, 6, day, 13, 30, tzinfo=UTC),
                datetime(2026, 6, day, 13, 45, tzinfo=UTC),
            ]
        )

    def _symbol_rows(symbol: str, closes: list[float], volumes: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": [symbol] * len(timestamps),
                "open": closes,
                "high": [value + 1 for value in closes],
                "low": [value - 1 for value in closes],
                "close": closes,
                "volume": volumes,
            }
        )

    panel = pd.concat(
        [
            _symbol_rows("SPY", [100 + i for i in range(24)], [1000 + 5 * i for i in range(24)]),
            _symbol_rows("QQQ", [200 + i for i in range(24)], [2000 + 5 * i for i in range(24)]),
            _symbol_rows("IWM", [300 + i for i in range(24)], [3000 + 5 * i for i in range(24)]),
            _symbol_rows(
                "AAA",
                [50 + i for i in range(24)],
                [100, 500, 110, 520, 120, 540, 130, 560, 140, 580, 150, 600, 160, 620, 170, 640, 180, 660, 190, 680, 200, 700, 210, 720],
            ),
        ],
        ignore_index=True,
    )

    features = compute_symbol_features(panel, benchmark_symbol="SPY")
    close_df = panel.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    volume_df = panel.pivot(index="timestamp", columns="symbol", values="volume").sort_index()
    expected_residuals = compute_residual_returns(close_df)
    expected_volume_z = compute_rolling_volume_zscore(volume_df)

    feature_row = features[(features["symbol"] == "AAA") & (features["timestamp"] == timestamps[-2])].iloc[0]
    assert abs(feature_row["residual_return"] - expected_residuals.loc[timestamps[-2], "AAA"]) < 1e-9
    assert abs(feature_row["volume_zscore"] - expected_volume_z.loc[timestamps[-2], "AAA"]) < 1e-9
