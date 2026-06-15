from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from stocknetv2.infrastructure.repositories.market_read_repository import (
    LegacySourceLayout,
    MarketReadRepository,
)


def _write_partition(root, relative_dir: str, filename: str, frame: pd.DataFrame) -> None:
    target_dir = root / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target_dir / filename, index=False)


def test_market_read_repository_lists_available_trade_dates_and_loads_inputs(tmp_path):
    bars_5m = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 2, 14, 35, tzinfo=UTC),
                datetime(2026, 1, 2, 14, 40, tzinfo=UTC),
            ],
            "symbol": ["AAA", "BBB"],
            "close": [10.0, 20.0],
        }
    )
    trade_flow_1m = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "symbol": ["AAA"],
            "flow_impulse_score": [1.2],
        }
    )
    features_1m = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "symbol": ["AAA"],
            "ret_1m": [0.01],
        }
    )

    _write_partition(tmp_path, "bars_5m/date=2026-01-02", "bars_5m.parquet", bars_5m)
    _write_partition(tmp_path, "trade_flow_1m/date=2026-01-02", "trade_flow_1m.parquet", trade_flow_1m)
    _write_partition(tmp_path, "features_1m/date=2026-01-02", "features_1m.parquet", features_1m)

    repository = MarketReadRepository(LegacySourceLayout(data_root=tmp_path))

    assert repository.list_available_trade_dates("bars_5m") == ["2026-01-02"]

    inputs = repository.load_trade_date_inputs("2026-01-02")

    assert list(inputs.bars_5m["symbol"]) == ["AAA", "BBB"]
    assert list(inputs.trade_flow_1m["symbol"]) == ["AAA"]
    assert list(inputs.features_1m["symbol"]) == ["AAA"]
    assert inputs.data_version == "bars_5m:2026-01-02|trade_flow_1m:2026-01-02|features_1m:2026-01-02"


def test_market_read_repository_returns_empty_optional_frames_when_missing(tmp_path):
    bars_5m = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 2, 14, 35, tzinfo=UTC)],
            "symbol": ["AAA"],
            "close": [10.0],
        }
    )
    _write_partition(tmp_path, "bars_5m/date=2026-01-02", "bars_5m.parquet", bars_5m)

    repository = MarketReadRepository(LegacySourceLayout(data_root=tmp_path))
    inputs = repository.load_trade_date_inputs("2026-01-02")

    assert not inputs.bars_5m.empty
    assert inputs.trade_flow_1m.empty
    assert inputs.features_1m.empty
