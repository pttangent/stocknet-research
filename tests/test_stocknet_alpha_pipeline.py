from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stocknet_alpha.backtest.backtest_signals import summarize_signal_backtest
from stocknet_alpha.config import AlphaPaths, load_universe_symbols
from stocknet_alpha.data.resample_bars import resample_ohlcv_bars, write_resampled_bars
from stocknet_alpha.data.us_market_data import build_intraday_features
from stocknet_alpha.leadlag.evaluate_edges import evaluate_leadlag_signals
from stocknet_alpha.leadlag.generate_signals import generate_leadlag_signals
from stocknet_alpha.theme.build_theme_candidates import build_theme_candidates_from_state


def test_load_universe_symbols_uses_screen_file_and_etf_exclusions(tmp_path: Path):
    screen_csv = tmp_path / "P123_Screen_0_20260606.csv"
    screen_csv.write_text(
        "\n".join(
            [
                "New Stock Screen",
                "2026-06-06",
                "",
                "Ticker,Name,Last",
                "AAA,Alpha,10",
                "BBB,Beta,11",
                "SPY,SPDR,500",
            ]
        ),
        encoding="utf-8",
    )
    etf_csv = tmp_path / "P123_ETFCEF.csv"
    etf_csv.write_text(
        "\n".join(
            [
                "New ETF Screen",
                "2026-06-08",
                "",
                "Ticker,Name,Last",
                "BBB,Beta ETF,11",
            ]
        ),
        encoding="utf-8",
    )

    symbols, excluded = load_universe_symbols(screen_csv, etf_csv, keep_benchmarks={"SPY"})

    assert symbols == ["AAA", "SPY"]
    assert excluded == ["BBB"]


def test_resample_ohlcv_bars_rolls_1m_into_5m(tmp_path: Path):
    timestamps = pd.date_range("2026-06-09T13:30:00Z", periods=10, freq="1min")
    raw_bars = pd.DataFrame(
        {
            "symbol": ["AAA"] * 10,
            "timestamp": timestamps,
            "open": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            "high": [15, 16, 17, 18, 19, 25, 26, 27, 28, 29],
            "low": [9, 10, 11, 12, 13, 19, 20, 21, 22, 23],
            "close": [14, 15, 16, 17, 18, 24, 25, 26, 27, 28],
            "volume": [100, 110, 120, 130, 140, 200, 210, 220, 230, 240],
            "vwap": [13.0, 14.0, 15.0, 16.0, 17.0, 23.0, 24.0, 25.0, 26.0, 27.0],
            "source": ["synthetic"] * 10,
        }
    )

    resampled = resample_ohlcv_bars(raw_bars, "5m").sort_values("timestamp").reset_index(drop=True)

    assert len(resampled) == 2
    assert resampled.loc[0, "open"] == 10
    assert resampled.loc[0, "high"] == 19
    assert resampled.loc[0, "low"] == 9
    assert resampled.loc[0, "close"] == 18
    assert resampled.loc[0, "volume"] == 600
    assert resampled.loc[0, "source"] == "synthetic"

    alpha_paths = AlphaPaths(repo_root=tmp_path)
    output_path = write_resampled_bars(resampled, alpha_paths, "2026-06-09", "5m")
    assert output_path.exists()


def test_build_theme_candidates_marks_15m_confirmation():
    current_state = {
        "five_minute": {
            "top_communities": [
                {
                    "timestamp": "2026-06-09T14:00:00Z",
                    "theme_path_id": "T001",
                    "community_id": "C001",
                    "members": "AAA,BBB,CCC",
                    "top_members": "AAA,BBB,CCC",
                    "member_count": 3,
                    "radar_score": 0.8,
                    "confirmation_score": 0.7,
                    "coherence": 0.6,
                    "breadth": 0.5,
                    "volume_expansion": 1.2,
                    "relative_return": 0.03,
                },
                {
                    "timestamp": "2026-06-09T14:05:00Z",
                    "theme_path_id": "T002",
                    "community_id": "C002",
                    "members": "DDD,EEE,FFF",
                    "top_members": "DDD,EEE,FFF",
                    "member_count": 3,
                    "radar_score": 0.5,
                    "confirmation_score": 0.4,
                    "coherence": 0.4,
                    "breadth": 0.3,
                    "volume_expansion": 0.8,
                    "relative_return": 0.01,
                },
            ]
        },
        "fifteen_minute": {
            "top_communities": [
                {
                    "timestamp": "2026-06-09T14:00:00Z",
                    "theme_path_id": "T001",
                    "community_id": "C101",
                    "members": "AAA,BBB,CCC",
                    "top_members": "AAA,BBB,CCC",
                    "member_count": 3,
                    "radar_score": 0.85,
                    "confirmation_score": 0.82,
                    "coherence": 0.7,
                    "breadth": 0.6,
                    "volume_expansion": 1.1,
                    "relative_return": 0.04,
                }
            ]
        },
    }

    candidates = build_theme_candidates_from_state(current_state, "2026-06-09")

    assert list(candidates["theme_path_id"]) == ["T001", "T002"]
    assert bool(candidates.loc[candidates["theme_path_id"] == "T001", "confirmed_on_15m"].iloc[0])
    assert not bool(candidates.loc[candidates["theme_path_id"] == "T002", "confirmed_on_15m"].iloc[0])


def test_generate_leadlag_signals_and_backtest_summary(tmp_path: Path):
    timestamps = pd.date_range("2026-06-09T14:00:00Z", periods=20, freq="1min")
    aaa_close = [10.0, 10.2, 10.4, 10.7, 10.9, 11.1, 11.3, 11.6, 11.8, 12.0, 12.2, 12.4, 12.5, 12.6, 12.7, 12.8, 12.8, 12.8, 12.8, 12.8]
    bbb_close = [10.0, 10.0, 10.1, 10.3, 10.5, 10.8, 11.0, 11.2, 11.5, 11.7, 11.9, 12.1, 12.3, 12.5, 12.7, 12.9, 13.0, 13.1, 13.1, 13.1]
    ccc_close = [10.0, 10.0, 10.0, 10.1, 10.2, 10.4, 10.5, 10.7, 10.8, 10.9, 11.0, 11.1, 11.1, 11.2, 11.3, 11.4, 11.4, 11.4, 11.4, 11.4]

    rows: list[dict[str, object]] = []
    for symbol, closes in {"AAA": aaa_close, "BBB": bbb_close, "CCC": ccc_close}.items():
        previous = closes[0]
        for ts, close in zip(timestamps, closes):
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": previous,
                    "high": max(previous, close),
                    "low": min(previous, close),
                    "close": close,
                    "volume": 1000,
                    "source": "synthetic",
                }
            )
            previous = close
    bars_1m = pd.DataFrame(rows)
    candidates = pd.DataFrame(
        [
                {
                    "signal_timestamp": pd.Timestamp("2026-06-09T14:12:00Z"),
                    "theme_path_id": "T001",
                    "community_id": "C001",
                    "members": "AAA,BBB,CCC",
                "member_count": 3,
                "confirmed_on_15m": True,
                "theme_score": 0.81,
            }
        ]
    )

    signals = generate_leadlag_signals(
        bars_1m,
        candidates,
        lookback_minutes=10,
        max_lag=2,
        top_followers=2,
    )

    assert not signals.empty
    assert (signals["leadlag_score"] > 0).all()
    assert (signals["lag_minutes"] >= 1).all()
    assert "forward_return_1m" not in signals.columns
    assert "feature_max_timestamp" in signals.columns
    assert "decision_timestamp" in signals.columns
    assert "execution_timestamp" in signals.columns
    assert (pd.to_datetime(signals["feature_max_timestamp"], utc=True) <= pd.to_datetime(signals["decision_timestamp"], utc=True)).all()
    assert (signals["execution_timestamp"] > signals["decision_timestamp"]).all()

    evaluated = evaluate_leadlag_signals(
        signals,
        bars_1m,
        horizons=(1, 3),
    )
    assert not evaluated.empty
    assert set(evaluated["horizon_minutes"]) == {1, 3}
    assert (evaluated["gross_return"] > 0).all()

    summary = summarize_signal_backtest(evaluated, horizons=(1, 3))

    assert set(summary["horizon_minutes"]) == {1, 3}
    assert (summary["signal_count"] > 0).all()
    assert "avg_net_return" in summary.columns


def test_generate_leadlag_signals_handles_duplicate_symbol_timestamps():
    bars_1m = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2026-06-09T14:00:00Z", "open": 10.0, "high": 10.1, "low": 10.0, "close": 10.1, "volume": 100.0, "source": "synthetic"},
            {"symbol": "AAA", "timestamp": "2026-06-09T14:01:00Z", "open": 10.1, "high": 10.2, "low": 10.1, "close": 10.2, "volume": 100.0, "source": "synthetic"},
            {"symbol": "AAA", "timestamp": "2026-06-09T14:01:00Z", "open": 10.1, "high": 10.2, "low": 10.1, "close": 10.2, "volume": 100.0, "source": "synthetic"},
            {"symbol": "BBB", "timestamp": "2026-06-09T14:00:00Z", "open": 20.0, "high": 20.0, "low": 19.9, "close": 19.9, "volume": 100.0, "source": "synthetic"},
            {"symbol": "BBB", "timestamp": "2026-06-09T14:01:00Z", "open": 19.9, "high": 20.1, "low": 19.9, "close": 20.1, "volume": 100.0, "source": "synthetic"},
            {"symbol": "CCC", "timestamp": "2026-06-09T14:00:00Z", "open": 30.0, "high": 30.1, "low": 30.0, "close": 30.1, "volume": 100.0, "source": "synthetic"},
            {"symbol": "CCC", "timestamp": "2026-06-09T14:01:00Z", "open": 30.1, "high": 30.2, "low": 30.1, "close": 30.2, "volume": 100.0, "source": "synthetic"},
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "signal_timestamp": pd.Timestamp("2026-06-09T14:01:00Z"),
                "theme_path_id": "T001",
                "community_id": "C001",
                "members": "AAA,BBB,CCC",
                "member_count": 3,
                "confirmed_on_15m": False,
                "theme_score": 0.5,
            }
        ]
    )

    signals = generate_leadlag_signals(
        bars_1m,
        candidates,
        lookback_minutes=5,
        max_lag=1,
        top_followers=1,
    )

    assert isinstance(signals, pd.DataFrame)


def test_generate_leadlag_signals_emits_flow_to_return_signals_when_features_are_available():
    timestamps = pd.date_range("2026-06-09T14:00:00Z", periods=12, freq="1min")
    bars_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []

    closes_by_symbol = {
        "AAA": [10.00, 10.10, 10.10, 10.25, 10.25, 10.40, 10.40, 10.55, 10.55, 10.70, 10.70, 10.85],
        "BBB": [20.00, 20.00, 20.05, 20.05, 20.20, 20.20, 20.35, 20.35, 20.50, 20.50, 20.65, 20.65],
        "CCC": [30.00] * 12,
    }
    volume_by_symbol = {"AAA": 1000.0, "BBB": 900.0, "CCC": 300.0}
    imbalance_by_symbol = {"AAA": 0.9, "BBB": 0.2, "CCC": 0.0}

    for symbol, closes in closes_by_symbol.items():
        previous = closes[0]
        for idx, (ts, close) in enumerate(zip(timestamps, closes)):
            bars_rows.append(
                {
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": previous,
                    "high": max(previous, close),
                    "low": min(previous, close),
                    "close": close,
                    "volume": volume_by_symbol[symbol],
                    "dollar_volume": close * volume_by_symbol[symbol],
                    "vwap": (previous + close) / 2.0,
                    "source": "synthetic",
                }
            )
            flow_rows.append(
                {
                    "ticker": symbol,
                    "minute": ts,
                    "trade_count": 20 if symbol != "CCC" else 5,
                    "volume": volume_by_symbol[symbol],
                    "dollar_volume": close * volume_by_symbol[symbol],
                    "vwap": (previous + close) / 2.0,
                    "avg_trade_size": 50.0,
                    "median_trade_size": 50.0,
                    "max_trade_size": 200.0,
                    "buy_vol_proxy": volume_by_symbol[symbol] * (0.5 + imbalance_by_symbol[symbol] / 2.0),
                    "sell_vol_proxy": volume_by_symbol[symbol] * (0.5 - imbalance_by_symbol[symbol] / 2.0),
                    "buy_trade_count_proxy": 12,
                    "sell_trade_count_proxy": 8,
                    "uptick_count": 10,
                    "downtick_count": 4,
                    "zero_tick_count": 0,
                    "imbalance_proxy": imbalance_by_symbol[symbol],
                    "trade_count_imbalance_proxy": 0.2,
                    "large_trade_count": 2 if symbol == "AAA" else 0,
                    "large_trade_volume": 250.0 if symbol == "AAA" else 0.0,
                    "large_trade_dollar_volume": close * 250.0 if symbol == "AAA" else 0.0,
                    "large_trade_buy_vol_proxy": 250.0 if symbol == "AAA" else 0.0,
                    "large_trade_sell_vol_proxy": 0.0,
                    "odd_lot_trade_count": 0,
                    "odd_lot_volume": 0.0,
                    "block_trade_count": 0,
                    "block_trade_volume": 0.0,
                    "lit_volume": volume_by_symbol[symbol],
                    "off_exchange_volume": 0.0,
                    "lit_trade_count": 20,
                    "off_exchange_trade_count": 0,
                    "correction_count": 0,
                    "unique_exchange_count": 1,
                    "avg_report_lag_ns": 1_000.0,
                    "max_report_lag_ns": 1_000.0,
                }
            )
            previous = close

    bars_1m = pd.DataFrame(bars_rows)
    trade_flow_1m = pd.DataFrame(flow_rows)
    features_1m = build_intraday_features(bars_1m, trade_flow_1m)
    candidates = pd.DataFrame(
        [
            {
                "signal_timestamp": pd.Timestamp("2026-06-09T14:11:00Z"),
                "theme_path_id": "T001",
                "community_id": "C001",
                "members": "AAA,BBB,CCC",
                "member_count": 3,
                "confirmed_on_15m": True,
                "theme_score": 0.85,
            }
        ]
    )

    signals = generate_leadlag_signals(
        bars_1m,
        candidates,
        features_1m=features_1m,
        lookback_minutes=10,
        max_lag=3,
        top_followers=2,
    )

    assert "leadlag_flow_to_return" in set(signals["signal_type"])
    flow_signal = signals.loc[signals["signal_type"] == "leadlag_flow_to_return"].iloc[0]
    assert flow_signal["source_feature"] == "flow_impulse_score"
    assert flow_signal["leader_feature_value"] > 0
    assert "future_ret_1m" not in signals.columns
