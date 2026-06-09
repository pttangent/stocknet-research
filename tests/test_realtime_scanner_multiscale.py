import os
import sys
import pandas as pd

ROOT = r"D:\DEV\stocknetwork\StockNet"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from realtime_dashboard.scripts import run_realtime_scanner as scanner


def test_build_state_payload_includes_5m_section():
    one = pd.DataFrame([
        {"timestamp": "2026-06-09 15:59:00+00:00", "radar_score": 0.8, "community_id": "C001"}
    ])
    five = pd.DataFrame([
        {"timestamp": "2026-06-09 16:00:00+00:00", "radar_score": 0.7, "community_id": "C002"}
    ])
    fifteen = pd.DataFrame([
        {"timestamp": "2026-06-09 16:00:00+00:00", "radar_score": 0.9, "community_id": "C003"}
    ])

    payload = scanner.build_state_payload(
        universe="core_500",
        scan_mode="full_parallel",
        symbols=500,
        scan_number=1,
        latest_1m_snapshots=one,
        alerts_1m=[],
        latest_5m_snapshots=five,
        alerts_5m=[],
        latest_15m_snapshots=fifteen,
        alerts_15m=[],
    )

    assert payload["one_minute"]["community_count"] == 1
    assert payload["five_minute"]["community_count"] == 1
    assert payload["fifteen_minute"]["community_count"] == 1
