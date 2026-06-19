from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from stocknetv2.application.services import graph_evaluation_pack_service as graph_pack_service
from stocknetv2.application.services.graph_evaluation_pack_service import (
    GraphEvaluationPackConfig,
    build_graph_evaluation_pack,
)
from stocknetv2.infrastructure.db.schema_manager import SchemaManager


def _create_graph_database(path: Path) -> None:
    connection = duckdb.connect(str(path))
    SchemaManager(connection).initialize()
    connection.execute(
        """
        INSERT INTO theme_discovery_run (
            run_id,
            run_name,
            date_start,
            date_end,
            frame_minutes,
            config_id,
            config_json,
            code_commit,
            data_version,
            status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "run_eval_2025-01-02",
            "Eval run",
            "2025-01-02",
            "2025-01-02",
            5,
            "config_eval",
            '{"config_id":"config_eval"}',
            "abc123",
            "bars_5m:2025-01-02|trade_flow_1m:2025-01-02|features_1m:2025-01-02",
            "completed",
        ],
    )
    connection.execute(
        """
        INSERT INTO graph_snapshot (
            snapshot_id,
            run_id,
            trade_date,
            timestamp,
            frame_minutes,
            market_session,
            graph_status,
            available_minutes_since_open
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "run_eval_2025-01-02_2025-01-02_1435",
            "run_eval_2025-01-02",
            "2025-01-02",
            "2025-01-02 22:35:00",
            5,
            "regular",
            "complete",
            5,
        ],
    )
    connection.execute(
        """
        INSERT INTO graph_edge_summary (
            run_id,
            snapshot_id,
            trade_date,
            graph_layer,
            edge_count,
            node_count,
            avg_weight,
            median_weight,
            p90_weight,
            threshold,
            top_k_per_symbol,
            effective_lookback_minutes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "run_eval_2025-01-02",
            "run_eval_2025-01-02_2025-01-02_1435",
            "2025-01-02",
            "return_corr_graph",
            1,
            2,
            0.82,
            0.82,
            0.82,
            0.65,
            3,
            60,
        ],
    )
    connection.execute(
        """
        INSERT INTO graph_edges_thresholded (
            run_id,
            snapshot_id,
            trade_date,
            timestamp,
            graph_layer,
            source_symbol,
            target_symbol,
            edge_type,
            weight,
            raw_score,
            edge_confidence,
            effective_lookback_minutes,
            window_start,
            window_end,
            support_points,
            config_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "run_eval_2025-01-02",
            "run_eval_2025-01-02_2025-01-02_1435",
            "2025-01-02",
            "2025-01-02 22:35:00",
            "return_corr_graph",
            "AAA",
            "BBB",
            "correlation",
            0.82,
            0.82,
            0.9,
            60,
            "2025-01-02 21:35:00",
            "2025-01-02 22:35:00",
            8,
            "config_eval",
        ],
    )
    connection.execute(
        """
        INSERT INTO layer_community (
            layer_community_id,
            run_id,
            snapshot_id,
            trade_date,
            graph_layer,
            community_local_id,
            members_json,
            member_count,
            edge_count,
            edge_density,
            avg_weight,
            min_weight,
            max_weight,
            community_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "lc_001",
            "run_eval_2025-01-02",
            "run_eval_2025-01-02_2025-01-02_1435",
            "2025-01-02",
            "return_corr_graph",
            "C001",
            '["AAA","BBB"]',
            2,
            1,
            1.0,
            0.82,
            0.82,
            0.82,
            "connected_components",
        ],
    )
    connection.execute(
        """
        INSERT INTO layer_community_membership (
            layer_community_id,
            run_id,
            snapshot_id,
            trade_date,
            graph_layer,
            community_local_id,
            symbol,
            member_rank,
            member_weight
        ) VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "lc_001",
            "run_eval_2025-01-02",
            "run_eval_2025-01-02_2025-01-02_1435",
            "2025-01-02",
            "return_corr_graph",
            "C001",
            "AAA",
            1,
            0.9,
            "lc_001",
            "run_eval_2025-01-02",
            "run_eval_2025-01-02_2025-01-02_1435",
            "2025-01-02",
            "return_corr_graph",
            "C001",
            "BBB",
            2,
            0.8,
        ],
    )
    connection.close()


def _create_market_database(path: Path) -> None:
    root = path.parent
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE features_1m (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            date DATE,
            close DOUBLE,
            volume DOUBLE,
            dollar_volume DOUBLE,
            trade_count DOUBLE,
            imbalance_proxy DOUBLE,
            large_trade_count DOUBLE,
            large_trade_dollar_volume DOUBLE,
            ret_1m_past DOUBLE,
            ret_3m_past DOUBLE,
            ret_5m_past DOUBLE,
            ret_15m_past DOUBLE,
            large_trade_ratio DOUBLE,
            volume_z_proxy DOUBLE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE labels_1m (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            future_ret_1m DOUBLE,
            future_ret_5m DOUBLE,
            future_ret_15m DOUBLE,
            future_ret_30m DOUBLE,
            date DATE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE trade_flow_1m (
            ticker VARCHAR,
            minute TIMESTAMP,
            trade_count DOUBLE,
            volume DOUBLE,
            dollar_volume DOUBLE,
            imbalance_proxy DOUBLE,
            large_trade_count DOUBLE,
            large_trade_dollar_volume DOUBLE,
            off_exchange_volume DOUBLE,
            date DATE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE bars_5m (
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            symbol VARCHAR,
            vwap DOUBLE,
            source VARCHAR,
            date DATE
        )
        """
    )
    connection.execute(
        """
        INSERT INTO features_1m VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "AAA", "2025-01-02 22:35:00", "2025-01-02", 10.1, 1000.0, 10100.0, 15.0, 0.4, 2.0, 1000.0, 0.01, 0.02, 0.03, 0.04, 0.1, 1.2,
            "BBB", "2025-01-02 22:35:00", "2025-01-02", 20.2, 2000.0, 40400.0, 18.0, 0.5, 3.0, 1500.0, 0.011, 0.021, 0.031, 0.041, 0.2, 1.5,
            "SPY", "2025-01-02 22:35:00", "2025-01-02", 500.0, 5000.0, 2500000.0, 40.0, 0.1, 1.0, 5000.0, 0.005, 0.006, 0.007, 0.008, 0.05, 0.8,
        ],
    )
    connection.execute(
        """
        INSERT INTO labels_1m VALUES
        (?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "AAA", "2025-01-02 22:35:00", 0.002, 0.006, 0.010, 0.015, "2025-01-02",
            "BBB", "2025-01-02 22:35:00", 0.001, 0.005, 0.009, 0.014, "2025-01-02",
            "SPY", "2025-01-02 22:35:00", 0.0005, 0.0025, 0.0040, 0.0060, "2025-01-02",
        ],
    )
    connection.execute(
        """
        INSERT INTO trade_flow_1m VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "AAA", "2025-01-02 22:35:00", 15.0, 1000.0, 10100.0, 0.4, 2.0, 1000.0, 100.0, "2025-01-02",
            "BBB", "2025-01-02 22:35:00", 18.0, 2000.0, 40400.0, 0.5, 3.0, 1500.0, 200.0, "2025-01-02",
            "SPY", "2025-01-02 22:35:00", 40.0, 5000.0, 2500000.0, 0.1, 1.0, 5000.0, 250.0, "2025-01-02",
        ],
    )
    connection.execute(
        """
        INSERT INTO bars_5m VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "2025-01-02 22:35:00", 10.0, 10.2, 9.9, 10.1, 1000.0, "AAA", 10.05, "test", "2025-01-02",
            "2025-01-02 22:35:00", 20.0, 20.3, 19.9, 20.2, 2000.0, "BBB", 20.1, "test", "2025-01-02",
            "2025-01-02 22:35:00", 499.0, 501.0, 498.0, 500.0, 5000.0, "SPY", 499.8, "test", "2025-01-02",
        ],
    )
    connection.close()

    features_partition = root / "features_1m" / "date=2025-01-02"
    labels_partition = root / "labels_1m" / "date=2025-01-02"
    trade_flow_partition = root / "trade_flow_1m" / "date=2025-01-02"
    bars_partition = root / "bars_5m" / "date=2025-01-02"
    for partition in (features_partition, labels_partition, trade_flow_partition, bars_partition):
        partition.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "timestamp": "2025-01-02 22:35:00",
                "date": "2025-01-02",
                "close": 10.1,
                "volume": 1000.0,
                "dollar_volume": 10100.0,
                "trade_count": 15.0,
                "imbalance_proxy": 0.4,
                "large_trade_count": 2.0,
                "large_trade_dollar_volume": 1000.0,
                "ret_1m_past": 0.01,
                "ret_3m_past": 0.02,
                "ret_5m_past": 0.03,
                "ret_15m_past": 0.04,
                "large_trade_ratio": 0.1,
                "volume_z_proxy": 1.2,
            },
            {
                "symbol": "BBB",
                "timestamp": "2025-01-02 22:35:00",
                "date": "2025-01-02",
                "close": 20.2,
                "volume": 2000.0,
                "dollar_volume": 40400.0,
                "trade_count": 18.0,
                "imbalance_proxy": 0.5,
                "large_trade_count": 3.0,
                "large_trade_dollar_volume": 1500.0,
                "ret_1m_past": 0.011,
                "ret_3m_past": 0.021,
                "ret_5m_past": 0.031,
                "ret_15m_past": 0.041,
                "large_trade_ratio": 0.2,
                "volume_z_proxy": 1.5,
            },
        ]
    ).to_parquet(features_partition / "features_1m.parquet", index=False)
    pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2025-01-02 22:35:00", "future_ret_1m": 0.002, "future_ret_5m": 0.006, "future_ret_15m": 0.010, "future_ret_30m": 0.015, "date": "2025-01-02"},
            {"symbol": "BBB", "timestamp": "2025-01-02 22:35:00", "future_ret_1m": 0.001, "future_ret_5m": 0.005, "future_ret_15m": 0.009, "future_ret_30m": 0.014, "date": "2025-01-02"},
            {"symbol": "SPY", "timestamp": "2025-01-02 22:35:00", "future_ret_1m": 0.0005, "future_ret_5m": 0.0025, "future_ret_15m": 0.0040, "future_ret_30m": 0.0060, "date": "2025-01-02"},
        ]
    ).to_parquet(labels_partition / "labels_1m.parquet", index=False)
    pd.DataFrame(
        [
            {"ticker": "AAA", "minute": "2025-01-02 22:35:00", "trade_count": 15.0, "volume": 1000.0, "dollar_volume": 10100.0, "imbalance_proxy": 0.4, "large_trade_count": 2.0, "large_trade_dollar_volume": 1000.0, "off_exchange_volume": 100.0, "date": "2025-01-02"},
            {"ticker": "BBB", "minute": "2025-01-02 22:35:00", "trade_count": 18.0, "volume": 2000.0, "dollar_volume": 40400.0, "imbalance_proxy": 0.5, "large_trade_count": 3.0, "large_trade_dollar_volume": 1500.0, "off_exchange_volume": 200.0, "date": "2025-01-02"},
            {"ticker": "SPY", "minute": "2025-01-02 22:35:00", "trade_count": 40.0, "volume": 5000.0, "dollar_volume": 2500000.0, "imbalance_proxy": 0.1, "large_trade_count": 1.0, "large_trade_dollar_volume": 5000.0, "off_exchange_volume": 250.0, "date": "2025-01-02"},
        ]
    ).to_parquet(trade_flow_partition / "trade_flow_1m.parquet", index=False)
    pd.DataFrame(
        [
            {"timestamp": "2025-01-02 22:30:00", "open": 9.9, "high": 10.0, "low": 9.8, "close": 10.0, "volume": 900.0, "symbol": "AAA", "vwap": 9.95, "source": "test", "date": "2025-01-02"},
            {"timestamp": "2025-01-02 22:35:00", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1000.0, "symbol": "AAA", "vwap": 10.05, "source": "test", "date": "2025-01-02"},
            {"timestamp": "2025-01-02 22:30:00", "open": 19.9, "high": 20.0, "low": 19.8, "close": 20.0, "volume": 1900.0, "symbol": "BBB", "vwap": 19.95, "source": "test", "date": "2025-01-02"},
            {"timestamp": "2025-01-02 22:35:00", "open": 20.0, "high": 20.3, "low": 19.9, "close": 20.2, "volume": 2000.0, "symbol": "BBB", "vwap": 20.10, "source": "test", "date": "2025-01-02"},
            {"timestamp": "2025-01-02 22:35:00", "open": 499.0, "high": 501.0, "low": 498.0, "close": 500.0, "volume": 5000.0, "symbol": "SPY", "vwap": 499.8, "source": "test", "date": "2025-01-02"},
        ]
    ).to_parquet(bars_partition / "bars_5m.parquet", index=False)


def test_build_graph_evaluation_pack_exports_review_artifacts(tmp_path):
    graph_database_path = tmp_path / "graph.duckdb"
    market_database_path = tmp_path / "market.duckdb"
    metadata_csv_path = tmp_path / "input_symbols.csv"
    output_dir = tmp_path / "evaluation_pack"

    _create_graph_database(graph_database_path)
    _create_market_database(market_database_path)
    metadata_csv_path.write_text(
        "\n".join(
            [
                "Ticker,Name,SectorCode,IndCode,Last,Rank,MktCap",
                "AAA,Alpha,TECH,SOFT,10.1,1,100000000",
                "BBB,Beta,TECH,SEMI,20.2,2,200000000",
                "SPY,SPY,ETF,INDEX,500,0,0",
            ]
        ),
        encoding="utf-8",
    )

    summary = build_graph_evaluation_pack(
        GraphEvaluationPackConfig(
            graph_database_path=graph_database_path,
            market_database_path=market_database_path,
            metadata_csv_path=metadata_csv_path,
            output_dir=output_dir,
            date_start="2025-01-02",
            date_end="2025-01-02",
            benchmark_symbols=("SPY",),
            generator_metadata={
                "git_head": "pack123",
                "git_branch": "test-branch",
                "repo_worktree_dirty": False,
                "relevant_worktree_dirty": False,
                "dirty_paths": [],
                "relevant_dirty_paths": [],
                "generated_at_utc": "2026-06-19T00:00:00Z",
            },
        )
    )

    assert summary.output_dir == output_dir.resolve()

    expected_files = [
        output_dir / "run_manifest.json",
        output_dir / "README.md",
        output_dir / "graph" / "all_edges",
        output_dir / "graph" / "snapshot_layer_diagnostics.csv",
        output_dir / "graph" / "node_layer_metrics",
        output_dir / "graph" / "community_metrics.parquet",
        output_dir / "graph" / "community_membership.parquet",
        output_dir / "graph" / "layer_review_candidates.csv",
        output_dir / "market" / "symbol_snapshot_features",
        output_dir / "market" / "symbol_forward_labels",
        output_dir / "market" / "symbol_master.csv",
        output_dir / "market" / "benchmark_series",
    ]
    for path in expected_files:
        assert path.exists(), path

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["date_start"] == "2025-01-02"
    assert manifest["date_end"] == "2025-01-02"
    assert manifest["counts"]["edge_rows"] == 1
    assert manifest["counts"]["community_rows"] == 1
    assert manifest["generator"]["git_head"] == "pack123"
    assert manifest["generator"]["git_branch"] == "test-branch"
    assert manifest["generator"]["relevant_worktree_dirty"] is False
    assert manifest["provenance"]["graph_build_commits"] == ["abc123"]
    assert manifest["provenance"]["evaluation_pack_generator"]["git_head"] == "pack123"
    assert manifest["provenance"]["config"]["sha256"]
    assert manifest["provenance"]["inputs"]["graph_database"]["sha256"]
    assert manifest["provenance"]["dependency_versions"]["duckdb"] == duckdb.__version__
    assert manifest["artifacts"]["run_manifest"]["size_bytes"] > 0

    connection = duckdb.connect()
    assert connection.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(output_dir / "graph" / "community_membership.parquet")],
    ).fetchone()[0] == 2
    assert connection.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(output_dir / "market" / "symbol_snapshot_features" / "*.parquet")],
    ).fetchone()[0] == 2
    assert connection.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(output_dir / "market" / "symbol_forward_labels" / "*.parquet")],
    ).fetchone()[0] == 2
    connection.close()


def test_resolve_generator_metadata_tolerates_output_dir_outside_repo(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    output_dir = tmp_path / "outside-pack"
    repo_root.mkdir()
    output_dir.mkdir()

    def fake_git_output(cwd: Path, args: list[str]) -> str | None:
        command = tuple(args)
        if command == ("rev-parse", "--show-toplevel"):
            return str(repo_root)
        if command == ("rev-parse", "HEAD"):
            return "head123"
        if command == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "branch-x"
        if command == ("status", "--porcelain=v1", "--untracked-files=all"):
            return " M data/generated.parquet\n M src/real_code.py\n"
        raise AssertionError(f"Unexpected git args: {args}")

    monkeypatch.setattr(graph_pack_service, "_git_output", fake_git_output)

    metadata = graph_pack_service._resolve_generator_metadata(
        provided_metadata=None,
        output_dir=output_dir,
    )

    assert metadata["git_head"] == "head123"
    assert metadata["git_branch"] == "branch-x"
    assert metadata["repo_worktree_dirty"] is True
    assert metadata["relevant_worktree_dirty"] is True
    assert metadata["dirty_paths"] == ["data/generated.parquet", "src/real_code.py"]
    assert metadata["relevant_dirty_paths"] == ["src/real_code.py"]
