from __future__ import annotations

import pandas as pd

from stocknet_alpha.theme.historical_candidates import (
    aggregate_trade_flow_to_5m,
    build_theme_candidates_from_market_data,
)


def test_aggregate_trade_flow_to_5m_aligns_to_bar_close():
    flow = pd.DataFrame(
        [
            {"ticker": "AAA", "minute": "2026-06-09T14:00:00Z", "imbalance_proxy": 10.0, "dollar_volume": 100.0, "buy_vol_proxy": 60.0, "sell_vol_proxy": 40.0},
            {"ticker": "AAA", "minute": "2026-06-09T14:01:00Z", "imbalance_proxy": 5.0, "dollar_volume": 110.0, "buy_vol_proxy": 70.0, "sell_vol_proxy": 40.0},
            {"ticker": "AAA", "minute": "2026-06-09T14:04:00Z", "imbalance_proxy": 3.0, "dollar_volume": 120.0, "buy_vol_proxy": 75.0, "sell_vol_proxy": 45.0},
            {"ticker": "AAA", "minute": "2026-06-09T14:05:00Z", "imbalance_proxy": 2.0, "dollar_volume": 130.0, "buy_vol_proxy": 66.0, "sell_vol_proxy": 64.0},
        ]
    )

    aggregated = aggregate_trade_flow_to_5m(flow)

    assert list(aggregated["timestamp"].dt.strftime("%H:%M:%S")) == ["14:05:00", "14:10:00"]
    assert list(aggregated["imbalance_proxy"]) == [18.0, 2.0]
    assert list(aggregated["dollar_volume"]) == [330.0, 130.0]


def test_build_theme_candidates_from_market_data_finds_persistent_cluster():
    timestamps = pd.date_range("2026-06-09T14:05:00Z", periods=10, freq="5min")
    rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []

    close_paths = {
        "AAA": [10.00, 10.10, 10.20, 10.35, 10.50, 10.65, 10.80, 10.95, 11.10, 11.25],
        "BBB": [20.00, 20.15, 20.30, 20.50, 20.65, 20.80, 20.95, 21.10, 21.25, 21.40],
        "CCC": [30.00, 30.18, 30.35, 30.55, 30.72, 30.90, 31.06, 31.22, 31.38, 31.55],
        "ZZZ": [15.00, 14.98, 15.01, 15.00, 14.99, 15.01, 15.00, 15.02, 15.01, 15.03],
    }

    for symbol, closes in close_paths.items():
        prev = closes[0]
        for ts, close in zip(timestamps, closes):
            rows.append(
                {
                    "timestamp": ts,
                    "open": prev,
                    "high": max(prev, close),
                    "low": min(prev, close),
                    "close": close,
                    "volume": 1000.0 if symbol != "ZZZ" else 200.0,
                    "symbol": symbol,
                    "vwap": (prev + close) / 2.0,
                    "source": "synthetic",
                    "date": ts.date(),
                }
            )
            bar_open = ts - pd.Timedelta(minutes=5)
            for minute_offset in range(5):
                minute = bar_open + pd.Timedelta(minutes=minute_offset)
                flow_rows.append(
                    {
                        "ticker": symbol,
                        "minute": minute,
                        "imbalance_proxy": 8.0 if symbol in {"AAA", "BBB", "CCC"} else 0.5,
                        "dollar_volume": 2000.0 if symbol != "ZZZ" else 100.0,
                        "buy_vol_proxy": 1200.0 if symbol in {"AAA", "BBB", "CCC"} else 50.0,
                        "sell_vol_proxy": 800.0 if symbol in {"AAA", "BBB", "CCC"} else 50.0,
                        "large_trade_dollar_volume": 600.0 if symbol in {"AAA", "BBB", "CCC"} else 0.0,
                        "off_exchange_volume": 150.0 if symbol in {"AAA", "BBB", "CCC"} else 10.0,
                        "volume": 300.0 if symbol != "ZZZ" else 20.0,
                        "trade_count": 12.0 if symbol != "ZZZ" else 2.0,
                    }
                )
            prev = close

    bars_5m = pd.DataFrame(rows)
    trade_flow_1m = pd.DataFrame(flow_rows)

    candidates = build_theme_candidates_from_market_data(
        bars_5m,
        trade_flow_1m,
        trade_date="2026-06-09",
        lookback_bars=4,
        top_symbols=4,
        min_members=3,
        min_theme_score=0.0,
        min_pair_corr=-1.0,
    )

    assert not candidates.empty
    members = set(candidates.iloc[-1]["members"].split(","))
    assert {"AAA", "BBB", "CCC"}.issubset(members)
    assert candidates["theme_path_id"].nunique() >= 1
    assert candidates["theme_path_id"].iloc[-1] == candidates["theme_path_id"].iloc[-2]
    assert (candidates["signal_timestamp"].diff().dropna() >= pd.Timedelta(minutes=0)).all()
    assert bool(candidates["confirmed_by_15m_graph"].any())
    confirmed = candidates.loc[candidates["confirmed_by_15m_graph"]].iloc[-1]
    assert confirmed["confirmation_source"] == "15m_graph"
    assert "price_theme_score" in candidates.columns
    assert "flow_theme_score" in candidates.columns
    assert "confirmed_by_flow" in candidates.columns
    assert float(candidates["flow_theme_score"].iloc[-1]) > 0.0


def test_build_theme_candidates_can_use_overlap_small_confirmation():
    timestamps = pd.date_range("2026-06-09T14:05:00Z", periods=6, freq="5min")
    bars_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []

    members_by_bar = [
        {"AAA", "BBB", "CCC", "DDD", "EEE"},
        {"AAA", "BBB", "FFF", "GGG", "HHH"},
        {"AAA", "BBB", "III", "JJJ", "KKK"},
        {"AAA", "BBB", "LLL", "MMM", "NNN"},
        {"AAA", "BBB", "OOO", "PPP", "QQQ"},
        {"AAA", "BBB", "RRR", "SSS", "TTT"},
    ]
    price_paths = {
        symbol: [10.0 + idx + step * 0.2 for step in range(len(timestamps))]
        for idx, symbol in enumerate(sorted({symbol for members in members_by_bar for symbol in members}))
    }

    for symbol, closes in price_paths.items():
        prev = closes[0]
        for ts, close in zip(timestamps, closes):
            bars_rows.append(
                {
                    "timestamp": ts,
                    "open": prev,
                    "high": max(prev, close),
                    "low": min(prev, close),
                    "close": close,
                    "volume": 5000.0 if symbol in {"AAA", "BBB"} else 1200.0,
                    "symbol": symbol,
                    "vwap": (prev + close) / 2.0,
                    "source": "synthetic",
                    "date": ts.date(),
                }
            )
            bar_open = ts - pd.Timedelta(minutes=5)
            active_members = members_by_bar[list(timestamps).index(ts)]
            for minute_offset in range(5):
                flow_rows.append(
                    {
                        "ticker": symbol,
                        "minute": bar_open + pd.Timedelta(minutes=minute_offset),
                        "imbalance_proxy": 8.0 if symbol in active_members else 0.1,
                        "dollar_volume": 2500.0 if symbol in active_members else 100.0,
                        "buy_vol_proxy": 1500.0 if symbol in active_members else 50.0,
                        "sell_vol_proxy": 1000.0 if symbol in active_members else 50.0,
                        "large_trade_dollar_volume": 700.0 if symbol in active_members else 0.0,
                        "off_exchange_volume": 120.0 if symbol in active_members else 10.0,
                        "volume": 400.0 if symbol in active_members else 20.0,
                        "trade_count": 15.0 if symbol in active_members else 2.0,
                    }
                )
            prev = close

    candidates = build_theme_candidates_from_market_data(
        pd.DataFrame(bars_rows),
        pd.DataFrame(flow_rows),
        trade_date="2026-06-09",
        lookback_bars=3,
        top_symbols=5,
        min_members=3,
        min_theme_score=0.0,
        min_pair_corr=-1.0,
        theme_path_score_method="overlap_small",
        theme_path_min_overlap=0.40,
    )

    assert not candidates.empty
    assert candidates["theme_path_id"].nunique() == 1
    assert list(candidates["age_bars"].tail(3)) == [2, 3, 4]
    assert bool(candidates.iloc[-1]["confirmed_on_15m"])
    assert bool(candidates.iloc[-1]["confirmed_by_age_3x5m"])
    assert not bool(candidates.iloc[-1]["confirmed_by_15m_graph"])
    assert candidates.iloc[-1]["confirmation_source"] == "age_3x5m"
    assert pd.Timestamp(candidates.iloc[-1]["confirmation_timestamp"]) == pd.Timestamp(candidates.iloc[-1]["signal_timestamp"])


def test_build_theme_candidates_can_reset_paths_after_large_gap():
    timestamps = [
        pd.Timestamp("2026-06-09T14:05:00Z"),
        pd.Timestamp("2026-06-09T14:10:00Z"),
        pd.Timestamp("2026-06-09T14:30:00Z"),
        pd.Timestamp("2026-06-09T14:35:00Z"),
    ]
    rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []

    close_paths = {
        "AAA": [10.0, 10.2, 10.4, 10.6],
        "BBB": [20.0, 20.2, 20.4, 20.6],
        "CCC": [30.0, 30.2, 30.4, 30.6],
    }

    for symbol, closes in close_paths.items():
        prev = closes[0]
        for ts, close in zip(timestamps, closes):
            rows.append(
                {
                    "timestamp": ts,
                    "open": prev,
                    "high": max(prev, close),
                    "low": min(prev, close),
                    "close": close,
                    "volume": 1000.0,
                    "symbol": symbol,
                    "vwap": (prev + close) / 2.0,
                    "source": "synthetic",
                    "date": ts.date(),
                }
            )
            bar_open = ts - pd.Timedelta(minutes=5)
            for minute_offset in range(5):
                flow_rows.append(
                    {
                        "ticker": symbol,
                        "minute": bar_open + pd.Timedelta(minutes=minute_offset),
                        "imbalance_proxy": 8.0,
                        "dollar_volume": 2000.0,
                        "buy_vol_proxy": 1200.0,
                        "sell_vol_proxy": 800.0,
                        "large_trade_dollar_volume": 500.0,
                        "off_exchange_volume": 100.0,
                        "volume": 300.0,
                        "trade_count": 10.0,
                    }
                )
            prev = close

    candidates = build_theme_candidates_from_market_data(
        pd.DataFrame(rows),
        pd.DataFrame(flow_rows),
        trade_date="2026-06-09",
        lookback_bars=2,
        top_symbols=3,
        min_members=3,
        min_theme_score=0.0,
        min_pair_corr=-1.0,
        theme_path_max_gap="10min",
    )

    assert not candidates.empty
    assert list(candidates["theme_path_id"].unique()) == ["T0001", "T0002"]
