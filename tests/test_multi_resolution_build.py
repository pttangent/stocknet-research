from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocknetwork.multi_resolution import resample_5m_to_higher


def test_resample_5m_to_15m_uses_ohlcv_aggregation():
    panel_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-06-02T13:30:00Z",
                    "2026-06-02T13:35:00Z",
                    "2026-06-02T13:40:00Z",
                    "2026-06-02T13:45:00Z",
                    "2026-06-02T13:50:00Z",
                    "2026-06-02T13:55:00Z",
                ],
                utc=True,
            ),
            "symbol": ["AAA"] * 6,
            "open": [10, 11, 12, 20, 21, 22],
            "high": [15, 16, 17, 25, 26, 27],
            "low": [9, 10, 11, 19, 20, 21],
            "close": [14, 15, 16, 24, 25, 26],
            "volume": [100, 110, 120, 200, 210, 220],
        }
    )

    result = resample_5m_to_higher(panel_5m, "15m").sort_values("timestamp").reset_index(drop=True)

    assert len(result) == 2
    first = result.iloc[0]
    second = result.iloc[1]

    assert first["open"] == 10
    assert first["high"] == 17
    assert first["low"] == 9
    assert first["close"] == 16
    assert first["volume"] == 330

    assert second["open"] == 20
    assert second["high"] == 27
    assert second["low"] == 19
    assert second["close"] == 26
    assert second["volume"] == 630
