from __future__ import annotations

import pandas as pd


def compute_effective_dtw_window(
    *,
    snapshot_time: pd.Timestamp,
    session_open: pd.Timestamp,
    min_minutes: int = 10,
    max_minutes: int = 30,
) -> dict[str, int | float | bool | str]:
    available = int((snapshot_time - session_open).total_seconds() // 60)

    if available < min_minutes:
        return {
            "enabled": False,
            "effective_lookback_minutes": max(available, 0),
            "dtw_mode": "warmup",
            "window_confidence": 0.0,
        }

    effective = min(available, max_minutes)

    if effective < max_minutes:
        confidence = 0.5 + 0.5 * ((effective - min_minutes) / (max_minutes - min_minutes))
        mode = "early"
    else:
        confidence = 1.0
        mode = "full"

    return {
        "enabled": True,
        "effective_lookback_minutes": effective,
        "dtw_mode": mode,
        "window_confidence": round(confidence, 10),
    }
