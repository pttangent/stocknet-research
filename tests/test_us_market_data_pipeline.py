from __future__ import annotations

import gzip
import io
from pathlib import Path

import duckdb
import pandas as pd
import pyzipper

from stocknet_alpha.config import AlphaPaths
from stocknet_alpha.data.us_market_data import (
    build_daily_bars_only_artifacts,
    build_intraday_features,
    build_labels,
    build_split_adjustment_factors,
    discover_bars_only_zip_paths,
    finalize_trade_flow,
    generate_zip_password,
    initialize_market_database,
    load_vendor_1m_zip,
    normalize_vendor_1m_bars,
    read_encrypted_zip_csv,
    register_market_data_views,
    summarize_trade_chunk,
)


def test_generate_zip_password_and_read_encrypted_zip_csv(tmp_path: Path):
    zip_path = tmp_path / "20260202.zip"
    password = generate_zip_password(zip_path.name).encode("utf-8")

    with pyzipper.AESZipFile(zip_path, "w", encryption=pyzipper.WZ_AES) as archive:
        archive.setpassword(password)
        archive.writestr("A.csv", "exchange,symbol,open,high,low,close,amount,volume,bob,eob,type,sequence\nXNYS,A,1,2,0.5,1.5,150,100,2026-02-02 09:30:00-05:00,2026-02-02 09:31:00-05:00,21,1\n")

    frame = read_encrypted_zip_csv(zip_path, "A.csv")

    assert list(frame.columns) == [
        "exchange",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "amount",
        "volume",
        "bob",
        "eob",
        "type",
        "sequence",
    ]
    assert frame.loc[0, "symbol"] == "A"


def test_normalize_vendor_1m_bars_maps_fields_and_vwap():
    raw = pd.DataFrame(
        [
            {
                "exchange": "XNYS",
                "symbol": "A",
                "open": 132.63,
                "high": 133.45,
                "low": 132.63,
                "close": 133.45,
                "amount": 2362301.5464,
                "volume": 17796.0,
                "bob": "2026-02-02 09:30:00-05:00",
                "eob": "2026-02-02 09:31:00-05:00",
                "type": 21,
                "sequence": 1,
            }
        ]
    )

    normalized = normalize_vendor_1m_bars(raw, source="vendor_1m")

    assert list(normalized.columns) == [
        "symbol",
        "timestamp",
        "bar_end",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_volume",
        "vwap",
        "exchange",
        "bar_type",
        "sequence",
        "source",
    ]
    assert normalized.loc[0, "timestamp"] == pd.Timestamp("2026-02-02T14:30:00Z")
    assert normalized.loc[0, "bar_end"] == pd.Timestamp("2026-02-02T14:31:00Z")
    assert normalized.loc[0, "vwap"] == raw.loc[0, "amount"] / raw.loc[0, "volume"]


def test_normalize_vendor_1m_bars_allows_missing_sequence_column():
    raw = pd.DataFrame(
        [
            {
                "exchange": "XNYS",
                "symbol": "A",
                "open": 132.63,
                "high": 133.45,
                "low": 132.63,
                "close": 133.45,
                "amount": 2362301.5464,
                "volume": 17796.0,
                "bob": "2026-03-02 09:30:00-05:00",
                "eob": "2026-03-02 09:31:00-05:00",
                "type": 21,
            }
        ]
    )

    normalized = normalize_vendor_1m_bars(raw, source="vendor_1m")

    assert "sequence" in normalized.columns
    assert normalized.loc[0, "sequence"] == 0


def test_normalize_vendor_1m_bars_coerces_symbol_to_string():
    raw = pd.DataFrame(
        [
            {
                "exchange": True,
                "symbol": True,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "amount": 150.0,
                "volume": 100.0,
                "bob": "2026-01-02 09:30:00-05:00",
                "eob": "2026-01-02 09:31:00-05:00",
                "type": 21,
            }
        ]
    )

    normalized = normalize_vendor_1m_bars(raw, source="vendor_1m")

    assert normalized.loc[0, "symbol"] == "TRUE"
    assert normalized.loc[0, "exchange"] == "TRUE"


def test_summarize_trade_chunk_carries_tick_rule_state_across_chunks():
    chunk_one = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "participant_timestamp": 1_770_022_200_000_000_000,
                "price": 10.0,
                "size": 100,
                "exchange": 11,
                "sequence_number": 1,
                "conditions": "12,37",
                "correction": 0,
                "sip_timestamp": 1_770_022_200_100_000_000,
                "tape": 1,
                "trf_id": 0,
                "trf_timestamp": 0,
            },
            {
                "ticker": "AAA",
                "participant_timestamp": 1_770_022_230_000_000_000,
                "price": 11.0,
                "size": 200,
                "exchange": 11,
                "sequence_number": 2,
                "conditions": "12,37",
                "correction": 0,
                "sip_timestamp": 1_770_022_230_100_000_000,
                "tape": 1,
                "trf_id": 0,
                "trf_timestamp": 0,
            },
        ]
    )
    chunk_two = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "participant_timestamp": 1_770_022_250_000_000_000,
                "price": 11.0,
                "size": 50,
                "exchange": 201,
                "sequence_number": 3,
                "conditions": "12,37",
                "correction": 0,
                "sip_timestamp": 1_770_022_250_300_000_000,
                "tape": 1,
                "trf_id": 201,
                "trf_timestamp": 1_770_022_250_200_000_000,
            },
            {
                "ticker": "AAA",
                "participant_timestamp": 1_770_022_261_000_000_000,
                "price": 10.0,
                "size": 400,
                "exchange": 201,
                "sequence_number": 4,
                "conditions": "12,37",
                "correction": 1,
                "sip_timestamp": 1_770_022_261_600_000_000,
                "tape": 1,
                "trf_id": 201,
                "trf_timestamp": 1_770_022_261_400_000_000,
            },
        ]
    )

    partial_one, state = summarize_trade_chunk(chunk_one, state={})
    partial_two, state = summarize_trade_chunk(chunk_two, state=state)
    flow = finalize_trade_flow([partial_one, partial_two]).sort_values(["ticker", "minute"]).reset_index(drop=True)

    first_minute = flow.loc[0]
    second_minute = flow.loc[1]

    assert first_minute["buy_vol_proxy"] == 250
    assert first_minute["sell_vol_proxy"] == 0
    assert first_minute["off_exchange_volume"] == 50
    assert first_minute["lit_volume"] == 300
    assert second_minute["sell_vol_proxy"] == 400
    assert second_minute["correction_count"] == 1
    assert second_minute["max_report_lag_ns"] == 600_000_000
    assert state["AAA"]["last_price"] == 10.0
    assert state["AAA"]["last_side"] == -1


def test_build_split_adjustment_factors_computes_backward_price_factor():
    splits = pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2026-02-03", "from": 1, "to": 2},
            {"symbol": "AAA", "date": "2026-02-05", "from": 1, "to": 3},
        ]
    )
    trading_dates = ["2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06"]

    factors = build_split_adjustment_factors(splits, trading_dates)

    prices = factors.set_index("trade_date")["price_adjustment_factor"].to_dict()
    multipliers = factors.set_index("trade_date")["cumulative_split_multiplier"].to_dict()

    assert prices["2026-02-02"] == 1 / 6
    assert prices["2026-02-03"] == 1 / 3
    assert prices["2026-02-05"] == 1.0
    assert multipliers["2026-02-02"] == 1.0
    assert multipliers["2026-02-03"] == 2.0
    assert multipliers["2026-02-05"] == 6.0


def test_register_market_data_views_exposes_features_and_labels(tmp_path: Path):
    paths = AlphaPaths(repo_root=tmp_path)

    bars = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-02-02T14:30:00Z"), "bar_end": pd.Timestamp("2026-02-02T14:31:00Z"), "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100.0, "dollar_volume": 1_000.0, "vwap": 10.0, "exchange": "XNYS", "bar_type": 21, "sequence": 1, "source": "vendor_1m"},
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-02-02T14:31:00Z"), "bar_end": pd.Timestamp("2026-02-02T14:32:00Z"), "open": 10.0, "high": 11.0, "low": 10.0, "close": 11.0, "volume": 200.0, "dollar_volume": 2_200.0, "vwap": 11.0, "exchange": "XNYS", "bar_type": 21, "sequence": 2, "source": "vendor_1m"},
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-02-02T14:32:00Z"), "bar_end": pd.Timestamp("2026-02-02T14:33:00Z"), "open": 11.0, "high": 12.0, "low": 11.0, "close": 12.0, "volume": 300.0, "dollar_volume": 3_600.0, "vwap": 12.0, "exchange": "XNYS", "bar_type": 21, "sequence": 3, "source": "vendor_1m"},
        ]
    )
    flow = pd.DataFrame(
        [
            {"ticker": "AAA", "minute": pd.Timestamp("2026-02-02T14:30:00Z"), "trade_count": 2, "volume": 100.0, "dollar_volume": 1_000.0, "vwap": 10.0, "avg_trade_size": 50.0, "median_trade_size": 50.0, "max_trade_size": 80.0, "buy_vol_proxy": 70.0, "sell_vol_proxy": 30.0, "buy_trade_count_proxy": 1, "sell_trade_count_proxy": 1, "uptick_count": 1, "downtick_count": 1, "zero_tick_count": 0, "imbalance_proxy": 0.4, "trade_count_imbalance_proxy": 0.0, "large_trade_count": 0, "large_trade_volume": 0.0, "large_trade_dollar_volume": 0.0, "large_trade_buy_vol_proxy": 0.0, "large_trade_sell_vol_proxy": 0.0, "odd_lot_trade_count": 1, "odd_lot_volume": 20.0, "block_trade_count": 0, "block_trade_volume": 0.0, "lit_volume": 100.0, "off_exchange_volume": 0.0, "lit_trade_count": 2, "off_exchange_trade_count": 0, "correction_count": 0, "unique_exchange_count": 1, "avg_report_lag_ns": 1_000.0, "max_report_lag_ns": 1_000.0},
            {"ticker": "AAA", "minute": pd.Timestamp("2026-02-02T14:31:00Z"), "trade_count": 3, "volume": 200.0, "dollar_volume": 2_200.0, "vwap": 11.0, "avg_trade_size": 66.67, "median_trade_size": 60.0, "max_trade_size": 100.0, "buy_vol_proxy": 150.0, "sell_vol_proxy": 50.0, "buy_trade_count_proxy": 2, "sell_trade_count_proxy": 1, "uptick_count": 2, "downtick_count": 1, "zero_tick_count": 0, "imbalance_proxy": 0.5, "trade_count_imbalance_proxy": 1 / 3, "large_trade_count": 0, "large_trade_volume": 0.0, "large_trade_dollar_volume": 0.0, "large_trade_buy_vol_proxy": 0.0, "large_trade_sell_vol_proxy": 0.0, "odd_lot_trade_count": 0, "odd_lot_volume": 0.0, "block_trade_count": 0, "block_trade_volume": 0.0, "lit_volume": 200.0, "off_exchange_volume": 0.0, "lit_trade_count": 3, "off_exchange_trade_count": 0, "correction_count": 0, "unique_exchange_count": 1, "avg_report_lag_ns": 1_000.0, "max_report_lag_ns": 1_000.0},
            {"ticker": "AAA", "minute": pd.Timestamp("2026-02-02T14:32:00Z"), "trade_count": 4, "volume": 300.0, "dollar_volume": 3_600.0, "vwap": 12.0, "avg_trade_size": 75.0, "median_trade_size": 75.0, "max_trade_size": 120.0, "buy_vol_proxy": 220.0, "sell_vol_proxy": 80.0, "buy_trade_count_proxy": 3, "sell_trade_count_proxy": 1, "uptick_count": 3, "downtick_count": 1, "zero_tick_count": 0, "imbalance_proxy": 0.4666666667, "trade_count_imbalance_proxy": 0.5, "large_trade_count": 1, "large_trade_volume": 120.0, "large_trade_dollar_volume": 1_440.0, "large_trade_buy_vol_proxy": 120.0, "large_trade_sell_vol_proxy": 0.0, "odd_lot_trade_count": 0, "odd_lot_volume": 0.0, "block_trade_count": 0, "block_trade_volume": 0.0, "lit_volume": 300.0, "off_exchange_volume": 0.0, "lit_trade_count": 4, "off_exchange_trade_count": 0, "correction_count": 0, "unique_exchange_count": 1, "avg_report_lag_ns": 1_000.0, "max_report_lag_ns": 1_000.0},
        ]
    )

    bars_path = paths.ensure_parent(paths.raw_1m_path("2026-02-02"))
    flow_path = paths.ensure_parent(paths.trade_flow_1m_path("2026-02-02"))
    bars.to_parquet(bars_path, index=False)
    flow.to_parquet(flow_path, index=False)
    labels = build_labels(bars)
    labels.to_parquet(paths.ensure_parent(paths.labels_1m_path("2026-02-02")), index=False)
    db_path = initialize_market_database(paths)
    con = duckdb.connect(str(db_path))

    row = con.execute(
        """
        SELECT
            b.symbol,
            f.ret_1m_past,
            f.vwap_distance,
            l.future_ret_1m
        FROM raw.bars_1m b
        JOIN research.features_1m f
          ON b.symbol = f.symbol
         AND b.timestamp = f.timestamp
        JOIN research.labels_1m l
          ON b.symbol = l.symbol
         AND b.timestamp = l.timestamp
        WHERE b.symbol = 'AAA'
          AND b.timestamp = TIMESTAMP '2026-02-02 14:31:00+00'
        """
    ).fetchone()

    assert row[0] == "AAA"
    assert round(row[1], 6) == 0.1
    assert row[2] == 0.0
    assert round(row[3], 6) == round((12.0 / 11.0) - 1, 6)


def test_discover_bars_only_zip_paths_recurses_year_month_dirs(tmp_path: Path):
    year_root = tmp_path / "2021"
    (year_root / "202101").mkdir(parents=True)
    (year_root / "202102").mkdir(parents=True)
    (year_root / "202101" / "20210104.zip").write_bytes(b"")
    (year_root / "202102" / "20210205.zip").write_bytes(b"")

    discovered = discover_bars_only_zip_paths([year_root])

    assert [(trade_date, path.name) for trade_date, path in discovered] == [
        ("2021-01-04", "20210104.zip"),
        ("2021-02-05", "20210205.zip"),
    ]

    selected = discover_bars_only_zip_paths([year_root], selected_dates=["2021-02-05"])
    assert [(trade_date, path.name) for trade_date, path in selected] == [("2021-02-05", "20210205.zip")]


def test_build_intraday_features_preserves_trade_flow_schema_without_trades():
    bars = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-02-02T14:30:00Z"), "bar_end": pd.Timestamp("2026-02-02T14:31:00Z"), "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100.0, "dollar_volume": 1_000.0, "vwap": 10.0, "exchange": "XNYS", "bar_type": 21, "sequence": 1, "source": "vendor_1m"},
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-02-02T14:31:00Z"), "bar_end": pd.Timestamp("2026-02-02T14:32:00Z"), "open": 10.0, "high": 11.0, "low": 10.0, "close": 11.0, "volume": 120.0, "dollar_volume": 1_320.0, "vwap": 11.0, "exchange": "XNYS", "bar_type": 21, "sequence": 2, "source": "vendor_1m"},
        ]
    )

    features = build_intraday_features(bars, pd.DataFrame())

    assert "trade_count" in features.columns
    assert "large_trade_dollar_volume" in features.columns
    assert features["trade_count"].isna().all()
    assert round(features.loc[1, "ret_1m_past"], 6) == 0.1
    assert features.loc[1, "large_trade_ratio"] == 0.0


def test_build_daily_bars_only_artifacts_writes_all_outputs(tmp_path: Path):
    paths = AlphaPaths(repo_root=tmp_path)
    trade_date = "2026-02-03"
    zip_path = tmp_path / "20260203.zip"
    password = generate_zip_password(zip_path.name).encode("utf-8")

    csv_body = (
        "exchange,symbol,open,high,low,close,amount,volume,bob,eob,type,sequence\n"
        "XNYS,AAA,10,11,10,11,1100,100,2026-02-03 09:30:00-05:00,2026-02-03 09:31:00-05:00,21,1\n"
        "XNYS,AAA,11,12,11,12,1200,100,2026-02-03 09:31:00-05:00,2026-02-03 09:32:00-05:00,21,2\n"
    )
    with pyzipper.AESZipFile(zip_path, "w", encryption=pyzipper.WZ_AES) as archive:
        archive.setpassword(password)
        archive.writestr("AAA.csv", csv_body)

    result = build_daily_bars_only_artifacts(trade_date, paths=paths, bars_zip_path=zip_path)

    assert result["status"] == "built"
    assert paths.raw_1m_path(trade_date).exists()
    assert paths.bars_path(trade_date, "5m").exists()
    assert paths.bars_path(trade_date, "15m").exists()
    assert not paths.features_1m_path(trade_date).exists()
    assert paths.labels_1m_path(trade_date).exists()
    assert not paths.trade_flow_1m_path(trade_date).exists()
    labels = pd.read_parquet(paths.labels_1m_path(trade_date))

    assert round(labels.loc[0, "future_ret_1m"], 6) == round((12.0 / 11.0) - 1, 6)

    db_path = initialize_market_database(paths)
    con = duckdb.connect(str(db_path))
    feature_row = con.execute(
        """
        SELECT symbol, trade_count, ret_1m_past
        FROM research.features_1m
        WHERE date = DATE '2026-02-03'
        ORDER BY timestamp, symbol
        LIMIT 2
        """
    ).fetchall()

    assert len(feature_row) == 2
    assert feature_row[0][1] is None
    assert round(feature_row[1][2], 6) == round((12.0 / 11.0) - 1, 6)


def test_load_vendor_1m_zip_supports_legacy_gzip_daily_bars(tmp_path: Path):
    gz_path = tmp_path / "20210104.gz"
    csv_body = (
        "ticker,volume,open,close,high,low,window_start,transactions\n"
        "AAA,100,10,11,11,10,1609767000000000000,5\n"
        "AAA,120,11,12,12,11,1609767060000000000,6\n"
    )
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write(csv_body)

    bars = load_vendor_1m_zip(gz_path)

    assert len(bars) == 2
    assert bars.loc[0, "symbol"] == "AAA"
    assert bars.loc[0, "timestamp"] == pd.Timestamp("2021-01-04T13:30:00Z")
    assert bars.loc[0, "bar_end"] == pd.Timestamp("2021-01-04T13:31:00Z")
    assert bars.loc[0, "exchange"] == "UNK"
    assert bars.loc[0, "sequence"] == 5
    assert round(bars.loc[0, "vwap"], 6) == 10.5


def test_initialize_market_database_builds_causal_backtest_views(tmp_path: Path):
    paths = AlphaPaths(repo_root=tmp_path)

    bars = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-02-02T14:30:00Z"), "bar_end": pd.Timestamp("2026-02-02T14:31:00Z"), "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100.0, "dollar_volume": 1_000.0, "vwap": 10.0, "exchange": "XNYS", "bar_type": 21, "sequence": 1, "source": "vendor_1m"},
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-02-02T14:31:00Z"), "bar_end": pd.Timestamp("2026-02-02T14:32:00Z"), "open": 10.0, "high": 11.0, "low": 10.0, "close": 11.0, "volume": 200.0, "dollar_volume": 2_200.0, "vwap": 11.0, "exchange": "XNYS", "bar_type": 21, "sequence": 2, "source": "vendor_1m"},
            {"symbol": "AAA", "timestamp": pd.Timestamp("2026-02-02T14:32:00Z"), "bar_end": pd.Timestamp("2026-02-02T14:33:00Z"), "open": 11.0, "high": 12.0, "low": 11.0, "close": 12.0, "volume": 300.0, "dollar_volume": 3_600.0, "vwap": 12.0, "exchange": "XNYS", "bar_type": 21, "sequence": 3, "source": "vendor_1m"},
        ]
    )
    flow = pd.DataFrame(
        [
            {"ticker": "AAA", "minute": pd.Timestamp("2026-02-02T14:30:00Z"), "trade_count": 2, "volume": 100.0, "dollar_volume": 1_000.0, "vwap": 10.0, "avg_trade_size": 50.0, "median_trade_size": 50.0, "max_trade_size": 80.0, "buy_vol_proxy": 70.0, "sell_vol_proxy": 30.0, "buy_trade_count_proxy": 1, "sell_trade_count_proxy": 1, "uptick_count": 1, "downtick_count": 1, "zero_tick_count": 0, "imbalance_proxy": 0.4, "trade_count_imbalance_proxy": 0.0, "large_trade_count": 0, "large_trade_volume": 0.0, "large_trade_dollar_volume": 0.0, "large_trade_buy_vol_proxy": 0.0, "large_trade_sell_vol_proxy": 0.0, "odd_lot_trade_count": 1, "odd_lot_volume": 20.0, "block_trade_count": 0, "block_trade_volume": 0.0, "lit_volume": 100.0, "off_exchange_volume": 0.0, "lit_trade_count": 2, "off_exchange_trade_count": 0, "correction_count": 0, "unique_exchange_count": 1, "avg_report_lag_ns": 1_000.0, "max_report_lag_ns": 1_000.0},
            {"ticker": "AAA", "minute": pd.Timestamp("2026-02-02T14:31:00Z"), "trade_count": 3, "volume": 200.0, "dollar_volume": 2_200.0, "vwap": 11.0, "avg_trade_size": 66.67, "median_trade_size": 60.0, "max_trade_size": 100.0, "buy_vol_proxy": 150.0, "sell_vol_proxy": 50.0, "buy_trade_count_proxy": 2, "sell_trade_count_proxy": 1, "uptick_count": 2, "downtick_count": 1, "zero_tick_count": 0, "imbalance_proxy": 0.5, "trade_count_imbalance_proxy": 1 / 3, "large_trade_count": 0, "large_trade_volume": 0.0, "large_trade_dollar_volume": 0.0, "large_trade_buy_vol_proxy": 0.0, "large_trade_sell_vol_proxy": 0.0, "odd_lot_trade_count": 0, "odd_lot_volume": 0.0, "block_trade_count": 0, "block_trade_volume": 0.0, "lit_volume": 200.0, "off_exchange_volume": 0.0, "lit_trade_count": 3, "off_exchange_trade_count": 0, "correction_count": 0, "unique_exchange_count": 1, "avg_report_lag_ns": 1_000.0, "max_report_lag_ns": 1_000.0},
        ]
    )

    bars.to_parquet(paths.ensure_parent(paths.raw_1m_path("2026-02-02")), index=False)
    flow.to_parquet(paths.ensure_parent(paths.trade_flow_1m_path("2026-02-02")), index=False)

    labels = build_labels(bars)
    labels.to_parquet(paths.ensure_parent(paths.labels_1m_path("2026-02-02")), index=False)

    db_path = initialize_market_database(paths)
    con = duckdb.connect(str(db_path))

    causal_bar = con.execute(
        """
        SELECT timestamp, close
        FROM backtest.bars_1m
        WHERE symbol = 'AAA'
        ORDER BY timestamp
        LIMIT 1
        """
    ).fetchone()
    causal_flow = con.execute(
        """
        SELECT minute, trade_count, imbalance_proxy
        FROM backtest.trade_flow_1m
        WHERE ticker = 'AAA'
        ORDER BY minute
        LIMIT 1
        """
    ).fetchone()
    causal_label = con.execute(
        """
        SELECT timestamp, future_ret_1m
        FROM backtest.labels_1m
        WHERE symbol = 'AAA'
        ORDER BY timestamp
        LIMIT 1
        """
    ).fetchone()
    causal_feature = con.execute(
        """
        SELECT timestamp, trade_count, ret_1m_past
        FROM backtest.features_1m
        WHERE symbol = 'AAA'
        ORDER BY timestamp
        LIMIT 2
        """
    ).fetchall()

    assert causal_bar[0] == pd.Timestamp("2026-02-02 14:31:00")
    assert causal_bar[1] == 10.0
    assert causal_flow[0] == pd.Timestamp("2026-02-02 14:31:00")
    assert causal_flow[1] == 2
    assert causal_flow[2] == 0.4
    assert causal_label[0] == pd.Timestamp("2026-02-02 14:31:00")
    assert round(causal_label[1], 6) == 0.1
    assert causal_feature[0][0] == pd.Timestamp("2026-02-02 14:31:00")
    assert causal_feature[0][1] == 2
    assert causal_feature[0][2] is None
