from __future__ import annotations

from datetime import UTC, datetime
from concurrent.futures import Future

import pandas as pd

from stocknetv2.application.services.layer_execution_service import LayerExecutionService
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs


def _build_trade_date_inputs() -> TradeDateInputs:
    minute_timestamps = list(pd.date_range("2026-01-02T14:31:00Z", periods=20, freq="1min"))
    bars_timestamps = [
        datetime(2026, 1, 2, 14, 35, tzinfo=UTC),
        datetime(2026, 1, 2, 14, 40, tzinfo=UTC),
        datetime(2026, 1, 2, 14, 45, tzinfo=UTC),
        datetime(2026, 1, 2, 14, 50, tzinfo=UTC),
    ]

    bars_5m = pd.DataFrame(
        {
            "timestamp": bars_timestamps * 3,
            "symbol": ["AAA"] * 4 + ["BBB"] * 4 + ["CCC"] * 4,
            "close": [10.0, 10.1, 10.2, 10.3] + [20.0, 20.2, 20.4, 20.6] + [30.0, 29.7, 29.4, 29.1],
        }
    )
    features_1m = pd.DataFrame(
        {
            "timestamp": minute_timestamps * 3,
            "symbol": ["AAA"] * 20 + ["BBB"] * 20 + ["CCC"] * 20,
            "ret_1m": [0.01] * 20 + [0.011] * 20 + [-0.02] * 20,
            "volume_z_12": [2.0] * 20 + [2.1] * 20 + [0.1] * 20,
            "large_trade_ratio_z": [1.5] * 20 + [1.51] * 20 + [0.1] * 20,
        }
    )
    trade_flow_1m = pd.DataFrame(
        {
            "timestamp": minute_timestamps * 3,
            "symbol": ["AAA"] * 20 + ["BBB"] * 20 + ["CCC"] * 20,
            "flow_impulse_score": [1.0] * 20 + [1.01] * 20 + [-1.0] * 20,
            "imbalance_z": [0.5] * 20 + [0.49] * 20 + [-0.5] * 20,
            "large_trade_ratio_z": [0.2] * 20 + [0.21] * 20 + [1.0] * 20,
        }
    )
    return TradeDateInputs(
        trade_date="2026-01-02",
        bars_5m=bars_5m,
        trade_flow_1m=trade_flow_1m,
        features_1m=features_1m,
        data_version="test-data-version",
    )


def test_layer_execution_service_builds_all_six_layer_outputs():
    service = LayerExecutionService()
    inputs = _build_trade_date_inputs()

    result = service.execute_for_snapshot(
        inputs=inputs,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
    )

    assert set(result.layer_edges) == {
        "return_corr_graph",
        "dtw_return_similarity_graph",
        "flow_alignment_graph",
        "dtw_trade_flow_similarity_graph",
        "volume_expansion_graph",
        "large_trade_alignment_graph",
    }
    assert all(len(edges) >= 1 for edges in result.layer_edges.values())


class _InlineExecutor:
    def __init__(self) -> None:
        self.submitted_layer_names: list[str] = []

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def submit(self, fn, *args, **kwargs):
        self.submitted_layer_names.append(args[0])
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


def test_layer_execution_service_can_dispatch_layers_through_executor():
    executor = _InlineExecutor()
    service = LayerExecutionService(
        parallel_workers=3,
        executor_factory=lambda max_workers: executor,
    )
    inputs = _build_trade_date_inputs()

    result = service.execute_for_snapshot(
        inputs=inputs,
        snapshot_time=pd.Timestamp("2026-01-02T14:50:00Z"),
        session_open=pd.Timestamp("2026-01-02T14:30:00Z"),
    )

    assert executor.submitted_layer_names == [
        "return_corr_graph",
        "dtw_return_similarity_graph",
        "flow_alignment_graph",
        "dtw_trade_flow_similarity_graph",
        "volume_expansion_graph",
        "large_trade_alignment_graph",
    ]
    assert set(result.layer_edges) == set(executor.submitted_layer_names)
