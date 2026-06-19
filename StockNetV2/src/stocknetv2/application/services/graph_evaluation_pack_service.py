from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import duckdb
import pandas as pd


@dataclass(frozen=True)
class GraphEvaluationPackConfig:
    graph_database_path: Path | str
    market_database_path: Path | str
    output_dir: Path | str
    metadata_csv_path: Path | str | None = None
    date_start: str | None = None
    date_end: str | None = None
    benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA")
    compare_graph_database_path: Path | str | None = None
    generator_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class GraphEvaluationPackSummary:
    output_dir: Path
    artifact_paths: dict[str, Path]
    counts: dict[str, int]


def build_graph_evaluation_pack(
    config: GraphEvaluationPackConfig,
    *,
    log: Callable[[str], None] | None = None,
) -> GraphEvaluationPackSummary:
    logger = log or (lambda message: None)
    graph_database_path = Path(config.graph_database_path).expanduser().resolve()
    market_database_path = Path(config.market_database_path).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    metadata_csv_path = (
        Path(config.metadata_csv_path).expanduser().resolve()
        if config.metadata_csv_path is not None
        else None
    )
    compare_graph_database_path = (
        Path(config.compare_graph_database_path).expanduser().resolve()
        if config.compare_graph_database_path is not None
        else None
    )

    if not graph_database_path.exists():
        raise FileNotFoundError(f"Graph database not found: {graph_database_path}")
    if not market_database_path.exists():
        raise FileNotFoundError(f"Market database not found: {market_database_path}")
    if metadata_csv_path is not None and not metadata_csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv_path}")
    if compare_graph_database_path is not None and not compare_graph_database_path.exists():
        raise FileNotFoundError(f"Comparison graph database not found: {compare_graph_database_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    graph_output_dir = output_dir / "graph"
    market_output_dir = output_dir / "market"
    compare_output_dir = output_dir / "compare_old_vs_new"
    temp_output_dir = output_dir / ".duckdb_tmp"
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    market_output_dir.mkdir(parents=True, exist_ok=True)
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    if compare_graph_database_path is not None:
        compare_output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, Path] = {}

    connection: duckdb.DuckDBPyConnection | None = duckdb.connect()
    try:
        connection.execute("PRAGMA threads=8")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET memory_limit='110GB'")
        connection.execute(f"SET temp_directory='{_escape_sql_literal(str(temp_output_dir))}'")
        connection.execute("SET max_temp_directory_size='280GiB'")
        _attach_database(connection, "graph_db", graph_database_path)
        _attach_database(connection, "market_db", market_database_path)
        if compare_graph_database_path is not None:
            _attach_database(connection, "compare_db", compare_graph_database_path)

        resolved_date_start, resolved_date_end = _resolve_date_range(
            connection=connection,
            configured_start=config.date_start,
            configured_end=config.date_end,
        )
        benchmark_symbols = tuple(symbol.upper() for symbol in config.benchmark_symbols if symbol)
        benchmark_list_sql = _sql_string_list(benchmark_symbols)
        primary_benchmark = benchmark_symbols[0] if benchmark_symbols else "SPY"

        logger(f"[1/8] Preparing evaluation views for {resolved_date_start} to {resolved_date_end}.")
        _create_pack_views(
            connection=connection,
            date_start=resolved_date_start,
            date_end=resolved_date_end,
            benchmark_list_sql=benchmark_list_sql,
            primary_benchmark=primary_benchmark,
            metadata_csv_path=metadata_csv_path,
        )
        trade_dates = _trade_dates(connection)
        symbol_master_frame = connection.execute(
            """
            SELECT
                symbol,
                source_symbol,
                company_name,
                sector_code,
                industry_code,
                last_price,
                rank,
                market_cap
            FROM pack_symbol_master
            """
        ).fetchdf()

        logger("[2/8] Exporting graph edge and diagnostic artifacts.")
        artifact_paths["all_edges"] = graph_output_dir / "all_edges"
        _copy_query_to_partitioned_parquet(
            connection,
            """
            SELECT
                e.run_id,
                e.snapshot_id,
                e.trade_date,
                e.snapshot_timestamp,
                e.snapshot_clock_code,
                e.available_minutes_since_open,
                e.graph_layer,
                e.source_symbol,
                e.target_symbol,
                e.edge_type,
                e.weight,
                e.raw_score,
                e.edge_confidence,
                e.effective_lookback_minutes,
                e.window_start,
                e.window_end,
                e.support_points,
                e.config_id
            FROM pack_edges e
            """,
            artifact_paths["all_edges"],
            trade_date_column="trade_date",
        )
        artifact_paths["snapshot_layer_diagnostics"] = graph_output_dir / "snapshot_layer_diagnostics.csv"
        _copy_query_to_csv(
            connection,
            """
            WITH node_rollup AS (
                SELECT
                    snapshot_id,
                    graph_layer,
                    COUNT(*) AS active_node_count,
                    AVG(degree) AS average_degree,
                    quantile_cont(degree, 0.50) AS degree_p50,
                    quantile_cont(degree, 0.95) AS degree_p95,
                    MAX(degree) AS max_degree
                FROM pack_node_metrics_base
                GROUP BY 1, 2
            ),
            edge_rollup AS (
                SELECT
                    snapshot_id,
                    graph_layer,
                    COUNT(*) AS edge_count,
                    AVG(weight) AS average_edge_weight,
                    quantile_cont(weight, 0.50) AS edge_weight_p50,
                    quantile_cont(weight, 0.90) AS edge_weight_p90,
                    quantile_cont(COALESCE(support_points, 0), 0.50) AS support_points_p50,
                    quantile_cont(COALESCE(support_points, 0), 0.90) AS support_points_p90
                FROM pack_edges
                GROUP BY 1, 2
            ),
            community_rollup AS (
                SELECT
                    c.snapshot_id,
                    c.graph_layer,
                    COUNT(*) AS community_count,
                    quantile_cont(c.member_count, 0.50) AS community_size_p50,
                    quantile_cont(c.member_count, 0.95) AS community_size_p95,
                    MAX(c.member_count) AS community_size_max,
                    SUM(CASE WHEN c.member_count = 1 THEN 1 ELSE 0 END) AS singleton_community_count
                FROM pack_communities c
                GROUP BY 1, 2
            )
            SELECT
                s.run_id,
                s.snapshot_id,
                ctx.trade_date,
                ctx.snapshot_timestamp,
                ctx.snapshot_clock_code,
                ctx.available_minutes_since_open,
                s.graph_layer,
                s.edge_count AS summary_edge_count,
                s.node_count AS summary_node_count,
                s.avg_weight AS summary_avg_weight,
                s.median_weight AS summary_median_weight,
                s.p90_weight AS summary_p90_weight,
                s.threshold,
                s.top_k_per_symbol,
                s.effective_lookback_minutes,
                COALESCE(n.active_node_count, 0) AS active_node_count,
                COALESCE(n.average_degree, 0) AS average_degree,
                COALESCE(n.degree_p50, 0) AS degree_p50,
                COALESCE(n.degree_p95, 0) AS degree_p95,
                COALESCE(n.max_degree, 0) AS max_degree,
                COALESCE(e.average_edge_weight, 0) AS average_edge_weight,
                COALESCE(e.edge_weight_p50, 0) AS edge_weight_p50,
                COALESCE(e.edge_weight_p90, 0) AS edge_weight_p90,
                COALESCE(e.support_points_p50, 0) AS support_points_p50,
                COALESCE(e.support_points_p90, 0) AS support_points_p90,
                COALESCE(c.community_count, 0) AS community_count,
                COALESCE(c.community_size_p50, 0) AS community_size_p50,
                COALESCE(c.community_size_p95, 0) AS community_size_p95,
                COALESCE(c.community_size_max, 0) AS community_size_max,
                COALESCE(c.singleton_community_count, 0) AS singleton_community_count,
                CASE
                    WHEN COALESCE(n.active_node_count, 0) = 0 THEN 0
                    ELSE COALESCE(c.community_size_max, 0) * 1.0 / n.active_node_count
                END AS largest_community_ratio,
                CASE
                    WHEN COALESCE(n.active_node_count, 0) = 0 THEN FALSE
                    ELSE COALESCE(c.community_size_max, 0) * 1.0 / n.active_node_count >= 0.15
                END AS has_market_mode_cluster
            FROM graph_db.graph_edge_summary s
            JOIN pack_snapshot_context ctx
                ON ctx.snapshot_id = s.snapshot_id
            LEFT JOIN node_rollup n
                ON n.snapshot_id = s.snapshot_id
               AND n.graph_layer = s.graph_layer
            LEFT JOIN edge_rollup e
                ON e.snapshot_id = s.snapshot_id
               AND e.graph_layer = s.graph_layer
            LEFT JOIN community_rollup c
                ON c.snapshot_id = s.snapshot_id
               AND c.graph_layer = s.graph_layer
            ORDER BY ctx.trade_date, s.snapshot_id, s.graph_layer
            """,
            artifact_paths["snapshot_layer_diagnostics"],
        )

        logger("[3/8] Exporting node and community artifacts.")
        artifact_paths["node_layer_metrics"] = graph_output_dir / "node_layer_metrics"
        _copy_query_to_partitioned_parquet(
            connection,
            """
            SELECT
                n.run_id,
                n.snapshot_id,
                n.trade_date,
                n.snapshot_timestamp,
                n.snapshot_clock_code,
                n.available_minutes_since_open,
                n.graph_layer,
                n.symbol,
                n.degree,
                n.weighted_degree,
                n.avg_incident_weight,
                n.max_incident_weight,
                n.support_points_total,
                n.support_points_avg,
                n.raw_score_avg,
                n.edge_confidence_avg,
                COALESCE(m.layer_community_id, '') AS layer_community_id,
                COALESCE(m.community_local_id, '') AS community_local_id,
                COALESCE(m.community_assignment_count, 0) AS community_assignment_count,
                COALESCE(m.member_rank, 0) AS member_rank,
                COALESCE(m.member_weight, 0) AS member_weight,
                COALESCE(m.community_member_count, 0) AS community_member_count,
                sm.company_name,
                sm.sector_code,
                sm.industry_code,
                sm.market_cap
            FROM pack_node_metrics_base n
            LEFT JOIN pack_membership_lookup m
                ON m.snapshot_id = n.snapshot_id
               AND m.graph_layer = n.graph_layer
               AND m.symbol = n.symbol
            LEFT JOIN pack_symbol_master sm
                ON sm.symbol = n.symbol
            """,
            artifact_paths["node_layer_metrics"],
            trade_date_column="trade_date",
        )
        artifact_paths["community_metrics"] = graph_output_dir / "community_metrics.parquet"
        _copy_query_to_parquet(
            connection,
            """
            WITH layer_active_nodes AS (
                SELECT
                    snapshot_id,
                    graph_layer,
                    COUNT(*) AS active_node_count
                FROM pack_node_metrics_base
                GROUP BY 1, 2
            ),
            member_market_cap AS (
                SELECT
                    m.layer_community_id,
                    AVG(sm.market_cap) AS avg_market_cap,
                    quantile_cont(sm.market_cap, 0.50) AS market_cap_p50
                FROM pack_memberships m
                LEFT JOIN pack_symbol_master sm
                    ON sm.symbol = m.symbol
                GROUP BY 1
            ),
            member_degree AS (
                SELECT
                    m.layer_community_id,
                    AVG(n.degree) AS avg_member_degree,
                    quantile_cont(n.degree, 0.50) AS degree_p50,
                    quantile_cont(n.degree, 0.95) AS degree_p95,
                    SUM(n.weighted_degree) AS community_weighted_degree
                FROM pack_memberships m
                LEFT JOIN pack_node_metrics_base n
                    ON n.snapshot_id = m.snapshot_id
                   AND n.graph_layer = m.graph_layer
                   AND n.symbol = m.symbol
                GROUP BY 1
            ),
            sector_rank AS (
                SELECT
                    m.layer_community_id,
                    COALESCE(sm.sector_code, 'UNKNOWN') AS sector_code,
                    COUNT(*) AS sector_member_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.layer_community_id
                        ORDER BY COUNT(*) DESC, COALESCE(sm.sector_code, 'UNKNOWN')
                    ) AS sector_rank
                FROM pack_memberships m
                LEFT JOIN pack_symbol_master sm
                    ON sm.symbol = m.symbol
                GROUP BY 1, 2
            ),
            industry_rank AS (
                SELECT
                    m.layer_community_id,
                    COALESCE(sm.industry_code, 'UNKNOWN') AS industry_code,
                    COUNT(*) AS industry_member_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.layer_community_id
                        ORDER BY COUNT(*) DESC, COALESCE(sm.industry_code, 'UNKNOWN')
                    ) AS industry_rank
                FROM pack_memberships m
                LEFT JOIN pack_symbol_master sm
                    ON sm.symbol = m.symbol
                GROUP BY 1, 2
            )
            SELECT
                c.layer_community_id,
                c.run_id,
                c.snapshot_id,
                c.trade_date,
                ctx.snapshot_timestamp,
                ctx.snapshot_clock_code,
                ctx.available_minutes_since_open,
                c.graph_layer,
                c.community_local_id,
                c.member_count,
                c.edge_count,
                c.edge_density,
                c.avg_weight,
                c.min_weight,
                c.max_weight,
                c.community_method,
                COALESCE(n.active_node_count, 0) AS active_node_count,
                CASE
                    WHEN COALESCE(n.active_node_count, 0) = 0 THEN 0
                    ELSE c.member_count * 1.0 / n.active_node_count
                END AS layer_member_ratio,
                COALESCE(d.avg_member_degree, 0) AS avg_member_degree,
                COALESCE(d.degree_p50, 0) AS degree_p50,
                COALESCE(d.degree_p95, 0) AS degree_p95,
                COALESCE(d.community_weighted_degree, 0) AS community_weighted_degree,
                COALESCE(mc.avg_market_cap, 0) AS avg_market_cap,
                COALESCE(mc.market_cap_p50, 0) AS market_cap_p50,
                COALESCE(sr.sector_code, 'UNKNOWN') AS top_sector,
                CASE
                    WHEN c.member_count = 0 THEN 0
                    ELSE COALESCE(sr.sector_member_count, 0) * 1.0 / c.member_count
                END AS top_sector_ratio,
                COALESCE(ir.industry_code, 'UNKNOWN') AS top_industry,
                CASE
                    WHEN c.member_count = 0 THEN 0
                    ELSE COALESCE(ir.industry_member_count, 0) * 1.0 / c.member_count
                END AS top_industry_ratio,
                ROW_NUMBER() OVER (
                    PARTITION BY c.snapshot_id, c.graph_layer
                    ORDER BY c.member_count DESC, COALESCE(c.avg_weight, 0) DESC, c.layer_community_id
                ) AS size_rank_in_layer,
                CASE
                    WHEN COALESCE(n.active_node_count, 0) = 0 THEN FALSE
                    ELSE c.member_count * 1.0 / n.active_node_count >= 0.15
                END AS is_market_mode_community
            FROM pack_communities c
            JOIN pack_snapshot_context ctx
                ON ctx.snapshot_id = c.snapshot_id
            LEFT JOIN layer_active_nodes n
                ON n.snapshot_id = c.snapshot_id
               AND n.graph_layer = c.graph_layer
            LEFT JOIN member_market_cap mc
                ON mc.layer_community_id = c.layer_community_id
            LEFT JOIN member_degree d
                ON d.layer_community_id = c.layer_community_id
            LEFT JOIN sector_rank sr
                ON sr.layer_community_id = c.layer_community_id
               AND sr.sector_rank = 1
            LEFT JOIN industry_rank ir
                ON ir.layer_community_id = c.layer_community_id
               AND ir.industry_rank = 1
            """,
            artifact_paths["community_metrics"],
        )
        artifact_paths["community_membership"] = graph_output_dir / "community_membership.parquet"
        _copy_query_to_parquet(
            connection,
            """
            SELECT
                m.layer_community_id,
                m.run_id,
                m.snapshot_id,
                m.trade_date,
                ctx.snapshot_timestamp,
                ctx.snapshot_clock_code,
                ctx.available_minutes_since_open,
                m.graph_layer,
                m.community_local_id,
                m.symbol,
                m.member_rank,
                m.member_weight,
                c.member_count AS community_member_count,
                c.edge_count AS community_edge_count,
                c.edge_density,
                c.avg_weight AS community_avg_weight,
                sm.company_name,
                sm.sector_code,
                sm.industry_code,
                sm.market_cap
            FROM pack_memberships m
            JOIN pack_snapshot_context ctx
                ON ctx.snapshot_id = m.snapshot_id
            LEFT JOIN pack_communities c
                ON c.layer_community_id = m.layer_community_id
            LEFT JOIN pack_symbol_master sm
                ON sm.symbol = m.symbol
            """,
            artifact_paths["community_membership"],
        )
        artifact_paths["layer_review_candidates"] = graph_output_dir / "layer_review_candidates.csv"
        _copy_query_to_csv(
            connection,
            """
            WITH community_metrics AS (
                SELECT * FROM read_parquet($path)
            )
            SELECT
                layer_community_id,
                snapshot_id,
                trade_date,
                snapshot_timestamp,
                snapshot_clock_code,
                available_minutes_since_open,
                graph_layer,
                community_local_id,
                member_count,
                edge_count,
                edge_density,
                avg_weight,
                layer_member_ratio,
                avg_member_degree,
                top_sector,
                top_sector_ratio,
                top_industry,
                top_industry_ratio,
                is_market_mode_community,
                (
                    COALESCE(avg_weight, 0)
                    * LN(member_count + 1.0)
                    * CASE
                        WHEN layer_member_ratio >= 0.50 THEN 0.20
                        WHEN layer_member_ratio >= 0.15 THEN 0.50
                        ELSE 1.00
                    END
                ) AS review_priority_score,
                CASE
                    WHEN is_market_mode_community THEN 'market_mode'
                    WHEN member_count >= 50 THEN 'large_cluster'
                    WHEN top_sector_ratio >= 0.60 THEN 'sector_concentrated'
                    ELSE 'balanced_cluster'
                END AS review_reason
            FROM community_metrics
            ORDER BY review_priority_score DESC, member_count DESC, snapshot_id, graph_layer, community_local_id
            """,
            artifact_paths["layer_review_candidates"],
            parameters={"path": str(artifact_paths["community_metrics"])},
        )

        logger("[4/8] Exporting optional comparison artifacts.")
        if compare_graph_database_path is not None:
            artifact_paths["compare_layer_edge_summary"] = compare_output_dir / "layer_edge_summary_compare.csv"
            _copy_query_to_csv(
                connection,
                f"""
                WITH current_edges AS (
                    SELECT graph_layer, COUNT(*) AS edge_count
                    FROM pack_edges
                    GROUP BY 1
                ),
                baseline_edges AS (
                    SELECT graph_layer, COUNT(*) AS edge_count
                    FROM compare_db.graph_edges_thresholded
                    WHERE trade_date BETWEEN DATE '{resolved_date_start}' AND DATE '{resolved_date_end}'
                    GROUP BY 1
                )
                SELECT
                    COALESCE(b.graph_layer, c.graph_layer) AS graph_layer,
                    COALESCE(b.edge_count, 0) AS baseline_edge_count,
                    COALESCE(c.edge_count, 0) AS current_edge_count,
                    COALESCE(c.edge_count, 0) - COALESCE(b.edge_count, 0) AS edge_count_delta,
                    CASE
                        WHEN COALESCE(b.edge_count, 0) = 0 THEN NULL
                        ELSE COALESCE(c.edge_count, 0) * 1.0 / b.edge_count - 1.0
                    END AS edge_count_delta_ratio
                FROM baseline_edges b
                FULL OUTER JOIN current_edges c
                    ON c.graph_layer = b.graph_layer
                ORDER BY graph_layer
                """,
                artifact_paths["compare_layer_edge_summary"],
            )
            artifact_paths["compare_layer_community_distribution"] = compare_output_dir / "layer_community_distribution_compare.csv"
            _copy_query_to_csv(
                connection,
                f"""
                WITH baseline AS (
                    SELECT
                        graph_layer,
                        COUNT(*) AS community_count,
                        AVG(member_count) AS avg_member_count,
                        quantile_cont(member_count, 0.50) AS member_count_p50,
                        quantile_cont(member_count, 0.95) AS member_count_p95,
                        MAX(member_count) AS member_count_max
                    FROM compare_db.layer_community
                    WHERE trade_date BETWEEN DATE '{resolved_date_start}' AND DATE '{resolved_date_end}'
                    GROUP BY 1
                ),
                current AS (
                    SELECT
                        graph_layer,
                        COUNT(*) AS community_count,
                        AVG(member_count) AS avg_member_count,
                        quantile_cont(member_count, 0.50) AS member_count_p50,
                        quantile_cont(member_count, 0.95) AS member_count_p95,
                        MAX(member_count) AS member_count_max
                    FROM pack_communities
                    GROUP BY 1
                )
                SELECT
                    COALESCE(b.graph_layer, c.graph_layer) AS graph_layer,
                    COALESCE(b.community_count, 0) AS baseline_community_count,
                    COALESCE(c.community_count, 0) AS current_community_count,
                    b.avg_member_count AS baseline_avg_member_count,
                    c.avg_member_count AS current_avg_member_count,
                    b.member_count_p50 AS baseline_member_count_p50,
                    c.member_count_p50 AS current_member_count_p50,
                    b.member_count_p95 AS baseline_member_count_p95,
                    c.member_count_p95 AS current_member_count_p95,
                    b.member_count_max AS baseline_member_count_max,
                    c.member_count_max AS current_member_count_max
                FROM baseline b
                FULL OUTER JOIN current c
                    ON c.graph_layer = b.graph_layer
                ORDER BY graph_layer
                """,
                artifact_paths["compare_layer_community_distribution"],
            )

        active_snapshot_key_dir = temp_output_dir / "active_symbol_snapshots"
        _copy_query_to_partitioned_parquet(
            connection,
            """
            SELECT
                snapshot_id,
                trade_date,
                snapshot_timestamp,
                snapshot_clock_code,
                available_minutes_since_open,
                symbol
            FROM pack_active_symbol_snapshots
            """,
            active_snapshot_key_dir,
            trade_date_column="trade_date",
        )
        active_symbols = {
            row[0]
            for row in connection.execute(
                "SELECT symbol FROM pack_active_symbols"
            ).fetchall()
        }
        counts = {
            "run_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_runs"),
            "snapshot_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_snapshot_context"),
            "edge_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_edges"),
            "community_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_communities"),
            "community_membership_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_memberships"),
            "active_symbol_snapshot_rows": _scalar_int(connection, "SELECT COUNT(*) FROM pack_active_symbol_snapshots"),
            "active_symbol_count": _scalar_int(connection, "SELECT COUNT(*) FROM pack_active_symbols"),
        }
        code_commits = [
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT code_commit
                FROM pack_runs
                WHERE code_commit IS NOT NULL
                ORDER BY code_commit
                """
            ).fetchall()
        ]
        layers = [
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT graph_layer
                FROM pack_edges
                ORDER BY graph_layer
                """
            ).fetchall()
        ]

        connection.close()
        connection = None

        logger("[5/8] Exporting symbol features, labels, and benchmark context.")
        market_data_root = market_database_path.parent
        artifact_paths["symbol_snapshot_features"] = market_output_dir / "symbol_snapshot_features"
        _export_symbol_snapshot_feature_shards(
            trade_dates,
            active_snapshot_key_dir,
            artifact_paths["symbol_snapshot_features"],
            symbol_master_frame,
            market_data_root,
        )
        artifact_paths["symbol_forward_labels"] = market_output_dir / "symbol_forward_labels"
        _export_symbol_forward_label_shards(
            trade_dates,
            active_snapshot_key_dir,
            artifact_paths["symbol_forward_labels"],
            primary_benchmark,
            market_data_root,
        )
        artifact_paths["benchmark_series"] = market_output_dir / "benchmark_series"
        _export_benchmark_series_shards(
            trade_dates,
            artifact_paths["benchmark_series"],
            benchmark_symbols,
            market_data_root,
        )

        artifact_paths["symbol_master"] = market_output_dir / "symbol_master.csv"
        active_symbol_master = symbol_master_frame[symbol_master_frame["symbol"].isin(active_symbols)].copy()
        active_symbol_master.sort_values("symbol").to_csv(artifact_paths["symbol_master"], index=False)

        logger("[6/8] Writing README and manifest.")
        artifact_paths["README"] = output_dir / "README.md"
        _write_readme(
            artifact_paths["README"],
            date_start=resolved_date_start,
            date_end=resolved_date_end,
            primary_benchmark=primary_benchmark,
            compare_included=compare_graph_database_path is not None,
        )
        artifact_paths["run_manifest"] = output_dir / "run_manifest.json"
        generator_metadata = _resolve_generator_metadata(
            provided_metadata=config.generator_metadata,
            output_dir=output_dir,
        )
        _write_manifest(
            artifact_paths["run_manifest"],
            graph_database_path=graph_database_path,
            market_database_path=market_database_path,
            metadata_csv_path=metadata_csv_path,
            compare_graph_database_path=compare_graph_database_path,
            output_dir=output_dir,
            date_start=resolved_date_start,
            date_end=resolved_date_end,
            primary_benchmark=primary_benchmark,
            benchmark_symbols=benchmark_symbols,
            counts=counts,
            artifact_paths=artifact_paths,
            code_commits=code_commits,
            layers=layers,
            generator_metadata=generator_metadata,
        )
        logger("[7/8] Verifying artifact files.")
        for path in artifact_paths.values():
            if not path.exists():
                raise RuntimeError(f"Expected artifact missing after export: {path}")

        logger("[8/8] Evaluation pack is ready.")
        return GraphEvaluationPackSummary(
            output_dir=output_dir,
            artifact_paths=artifact_paths,
            counts=counts,
        )
    finally:
        if connection is not None:
            connection.close()


def _resolve_date_range(
    *,
    connection: duckdb.DuckDBPyConnection,
    configured_start: str | None,
    configured_end: str | None,
) -> tuple[str, str]:
    default_start, default_end = connection.execute(
        """
        SELECT
            CAST(MIN(trade_date) AS VARCHAR),
            CAST(MAX(trade_date) AS VARCHAR)
        FROM graph_db.graph_snapshot
        """
    ).fetchone()
    if default_start is None or default_end is None:
        raise RuntimeError("Graph database does not contain any snapshots.")
    return configured_start or default_start, configured_end or default_end


def _create_pack_views(
    *,
    connection: duckdb.DuckDBPyConnection,
    date_start: str,
    date_end: str,
    benchmark_list_sql: str,
    primary_benchmark: str,
    metadata_csv_path: Path | None,
) -> None:
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_runs AS
        SELECT *
        FROM graph_db.theme_discovery_run
        WHERE date_start <= DATE '{date_end}'
          AND date_end >= DATE '{date_start}'
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_snapshot_context AS
        SELECT
            snapshot_id,
            run_id,
            trade_date,
            timestamp AS snapshot_timestamp,
            frame_minutes,
            market_session,
            graph_status,
            available_minutes_since_open,
            RIGHT(snapshot_id, 4) AS snapshot_clock_code
        FROM graph_db.graph_snapshot
        WHERE trade_date BETWEEN DATE '{date_start}' AND DATE '{date_end}'
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW pack_edges AS
        SELECT
            e.run_id,
            e.snapshot_id,
            ctx.trade_date,
            ctx.snapshot_timestamp,
            ctx.snapshot_clock_code,
            ctx.available_minutes_since_open,
            e.graph_layer,
            e.source_symbol,
            e.target_symbol,
            e.edge_type,
            e.weight,
            e.raw_score,
            e.edge_confidence,
            e.effective_lookback_minutes,
            e.window_start,
            e.window_end,
            e.support_points,
            e.config_id
        FROM graph_db.graph_edges_thresholded e
        JOIN pack_snapshot_context ctx
            ON ctx.snapshot_id = e.snapshot_id
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW pack_communities AS
        SELECT c.*
        FROM graph_db.layer_community c
        JOIN pack_snapshot_context ctx
            ON ctx.snapshot_id = c.snapshot_id
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW pack_memberships AS
        SELECT m.*
        FROM graph_db.layer_community_membership m
        JOIN pack_snapshot_context ctx
            ON ctx.snapshot_id = m.snapshot_id
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE pack_node_metrics_base AS
        WITH incident_edges AS (
            SELECT
                run_id,
                snapshot_id,
                trade_date,
                snapshot_timestamp,
                snapshot_clock_code,
                available_minutes_since_open,
                graph_layer,
                source_symbol AS symbol,
                weight,
                raw_score,
                edge_confidence,
                support_points
            FROM pack_edges
            UNION ALL
            SELECT
                run_id,
                snapshot_id,
                trade_date,
                snapshot_timestamp,
                snapshot_clock_code,
                available_minutes_since_open,
                graph_layer,
                target_symbol AS symbol,
                weight,
                raw_score,
                edge_confidence,
                support_points
            FROM pack_edges
        )
        SELECT
            run_id,
            snapshot_id,
            trade_date,
            snapshot_timestamp,
            snapshot_clock_code,
            available_minutes_since_open,
            graph_layer,
            symbol,
            COUNT(*) AS degree,
            SUM(weight) AS weighted_degree,
            AVG(weight) AS avg_incident_weight,
            MAX(weight) AS max_incident_weight,
            SUM(COALESCE(support_points, 0)) AS support_points_total,
            AVG(COALESCE(support_points, 0)) AS support_points_avg,
            AVG(COALESCE(raw_score, weight)) AS raw_score_avg,
            AVG(COALESCE(edge_confidence, 1.0)) AS edge_confidence_avg
        FROM incident_edges
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE pack_membership_lookup AS
        SELECT
            m.snapshot_id,
            m.graph_layer,
            m.symbol,
            ANY_VALUE(m.layer_community_id) AS layer_community_id,
            ANY_VALUE(m.community_local_id) AS community_local_id,
            COUNT(*) AS community_assignment_count,
            MIN(COALESCE(m.member_rank, 0)) AS member_rank,
            AVG(COALESCE(m.member_weight, 0)) AS member_weight,
            MAX(COALESCE(c.member_count, 0)) AS community_member_count
        FROM pack_memberships m
        LEFT JOIN pack_communities c
            ON c.layer_community_id = m.layer_community_id
        GROUP BY 1, 2, 3
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE pack_active_symbol_snapshots AS
        SELECT DISTINCT
            snapshot_id,
            trade_date,
            snapshot_timestamp,
            snapshot_clock_code,
            available_minutes_since_open,
            symbol
        FROM pack_node_metrics_base
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE pack_active_symbols AS
        SELECT DISTINCT symbol
        FROM pack_active_symbol_snapshots
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_market_features AS
        SELECT *
        FROM market_db.features_1m
        WHERE date BETWEEN DATE '{date_start}' AND DATE '{date_end}'
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_market_labels AS
        SELECT *
        FROM market_db.labels_1m
        WHERE date BETWEEN DATE '{date_start}' AND DATE '{date_end}'
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_market_trade_flow AS
        SELECT *
        FROM market_db.trade_flow_1m
        WHERE date BETWEEN DATE '{date_start}' AND DATE '{date_end}'
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_market_bars_5m AS
        SELECT *
        FROM market_db.bars_5m
        WHERE date BETWEEN DATE '{date_start}' AND DATE '{date_end}'
        """
    )

    if metadata_csv_path is None:
        connection.execute(
            """
            CREATE OR REPLACE TEMP TABLE pack_symbol_master (
                symbol VARCHAR,
                source_symbol VARCHAR,
                company_name VARCHAR,
                sector_code VARCHAR,
                industry_code VARCHAR,
                last_price DOUBLE,
                rank INTEGER,
                market_cap DOUBLE
            )
            """
        )
    else:
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW pack_symbol_master AS
            SELECT DISTINCT
                UPPER(TRIM("Ticker")) AS symbol,
                TRIM("Ticker") AS source_symbol,
                TRIM(COALESCE("Name", '')) AS company_name,
                NULLIF(TRIM(COALESCE("SectorCode", '')), '') AS sector_code,
                NULLIF(TRIM(COALESCE("IndCode", '')), '') AS industry_code,
                TRY_CAST("Last" AS DOUBLE) AS last_price,
                TRY_CAST("Rank" AS INTEGER) AS rank,
                TRY_CAST("MktCap" AS DOUBLE) AS market_cap
            FROM read_csv_auto('{_escape_sql_literal(str(metadata_csv_path))}', HEADER=TRUE)
            WHERE NULLIF(TRIM(COALESCE("Ticker", '')), '') IS NOT NULL
            """
        )

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pack_benchmarks_in_scope AS
        SELECT DISTINCT symbol
        FROM pack_market_bars_5m
        WHERE symbol IN ({benchmark_list_sql})
        """
    )


def _write_readme(
    path: Path,
    *,
    date_start: str,
    date_end: str,
    primary_benchmark: str,
    compare_included: bool,
) -> None:
    compare_note = (
        "- `compare_old_vs_new/`: baseline-vs-current structural comparison for the same month.\n"
        if compare_included
        else ""
    )
    content = (
        "# Graph Evaluation Pack\n\n"
        f"Date range: `{date_start}` to `{date_end}`\n\n"
        "This pack is designed for manual graph-quality and financial-meaning review of the first month.\n"
        "It does not depend on rerunning the full T1 theme pipeline; instead it reconstructs evaluation context from the monthly graph-build database plus the market database.\n\n"
        "## Start Here\n\n"
        "1. Open `graph/layer_review_candidates.csv`.\n"
        "2. Use `graph/community_metrics.parquet` and `graph/community_membership.parquet` to inspect whether large communities are real themes, sector baskets, or market-mode clusters.\n"
        "3. Use `market/symbol_snapshot_features.parquet` to inspect the state of each member at the snapshot.\n"
        f"4. Use `market/symbol_forward_labels.parquet` to check whether members outperformed `{primary_benchmark}` over the next 1m/5m/15m/30m windows.\n"
        "5. Use `graph/snapshot_layer_diagnostics.csv` to find pathological layers, giant clusters, or snapshots where one layer dominates the universe.\n\n"
        "## Time Notes\n\n"
        "- `snapshot_clock_code` is the canonical market-clock label from the snapshot id suffix.\n"
        "- `snapshot_timestamp` is the stored timestamp value from the graph database.\n"
        "- `available_minutes_since_open` is the safest field for intraday sequencing if timezone display looks inconsistent.\n\n"
        "## Files\n\n"
        "- `graph/all_edges/`: thresholded graph edges, sharded by trade date as parquet.\n"
        "- `graph/snapshot_layer_diagnostics.csv`: per-snapshot, per-layer structure diagnostics.\n"
        "- `graph/node_layer_metrics/`: per-symbol, per-layer node metrics, sharded by trade date as parquet.\n"
        "- `graph/community_metrics.parquet`: community-level structure and concentration metrics.\n"
        "- `graph/community_membership.parquet`: member roster for each community.\n"
        "- `graph/layer_review_candidates.csv`: ranked shortlist for manual review.\n"
        "- `market/symbol_snapshot_features/`: snapshot-aligned symbol state features, derived from `bars_5m + trade_flow_1m` and sharded by trade date as parquet.\n"
        "- `market/symbol_forward_labels/`: forward returns and benchmark-relative labels, sharded by trade date as parquet.\n"
        "- `market/symbol_master.csv`: symbol metadata used for joins.\n"
        "- `market/benchmark_series/`: benchmark bar series for context, sharded by trade date as parquet.\n"
        f"{compare_note}"
        "\n## Suggested Evaluation Questions\n\n"
        "- Do top-ranked communities have reasonable member counts, or are they still market-mode clusters?\n"
        "- Are the members concentrated in one sector or industry for an interpretable reason?\n"
        "- Do symbols inside a community share similar flow, volume, and short-horizon forward return behavior?\n"
        "- Which layers create the most false giant clusters, and at what time of day?\n"
        "- Are review-worthy communities associated with positive benchmark-relative forward returns, or only with generic market beta?\n"
    )
    path.write_text(content, encoding="utf-8")


def _write_manifest(
    path: Path,
    *,
    graph_database_path: Path,
    market_database_path: Path,
    metadata_csv_path: Path | None,
    compare_graph_database_path: Path | None,
    output_dir: Path,
    date_start: str,
    date_end: str,
    primary_benchmark: str,
    benchmark_symbols: tuple[str, ...],
    counts: dict[str, int],
    artifact_paths: dict[str, Path],
    code_commits: list[str],
    layers: list[str],
    generator_metadata: dict[str, Any],
) -> None:
    manifest = _build_manifest_payload(
        graph_database_path=graph_database_path,
        market_database_path=market_database_path,
        metadata_csv_path=metadata_csv_path,
        compare_graph_database_path=compare_graph_database_path,
        output_dir=output_dir,
        date_start=date_start,
        date_end=date_end,
        primary_benchmark=primary_benchmark,
        benchmark_symbols=benchmark_symbols,
        counts=counts,
        artifact_paths=artifact_paths,
        code_commits=code_commits,
        layers=layers,
        generator_metadata=generator_metadata,
    )
    manifest["artifacts"]["run_manifest"]["size_bytes"] = 0
    for _ in range(3):
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        actual_size = path.stat().st_size
        if manifest["artifacts"]["run_manifest"]["size_bytes"] == actual_size:
            break
        manifest["artifacts"]["run_manifest"]["size_bytes"] = actual_size
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _attach_database(connection: duckdb.DuckDBPyConnection, alias: str, database_path: Path) -> None:
    connection.execute(f"ATTACH '{_escape_sql_literal(str(database_path))}' AS {alias} (READ_ONLY)")


def _build_manifest_payload(
    *,
    graph_database_path: Path,
    market_database_path: Path,
    metadata_csv_path: Path | None,
    compare_graph_database_path: Path | None,
    output_dir: Path,
    date_start: str,
    date_end: str,
    primary_benchmark: str,
    benchmark_symbols: tuple[str, ...],
    counts: dict[str, int],
    artifact_paths: dict[str, Path],
    code_commits: list[str],
    layers: list[str],
    generator_metadata: dict[str, Any],
) -> dict[str, Any]:
    config_payload = {
        "date_start": date_start,
        "date_end": date_end,
        "primary_benchmark": primary_benchmark,
        "benchmark_symbols": list(benchmark_symbols),
        "layers": layers,
        "compare_graph_database_path": str(compare_graph_database_path) if compare_graph_database_path is not None else None,
    }
    return {
        "date_start": date_start,
        "date_end": date_end,
        "primary_benchmark": primary_benchmark,
        "benchmark_symbols": list(benchmark_symbols),
        "counts": counts,
        "code_commits": code_commits,
        "layers": layers,
        "generator": generator_metadata,
        "provenance": {
            "graph_build_commits": code_commits,
            "evaluation_pack_generator": generator_metadata,
            "config": {
                **config_payload,
                "sha256": _sha256_json(config_payload),
            },
            "inputs": {
                "graph_database": _file_provenance(graph_database_path),
                "market_database": _file_provenance(market_database_path),
                "metadata_csv": _file_provenance(metadata_csv_path),
                "compare_graph_database": _file_provenance(compare_graph_database_path),
            },
            "dependency_versions": _dependency_versions(),
        },
        "sources": {
            "graph_database_path": str(graph_database_path),
            "market_database_path": str(market_database_path),
            "metadata_csv_path": str(metadata_csv_path) if metadata_csv_path is not None else None,
            "compare_graph_database_path": str(compare_graph_database_path) if compare_graph_database_path is not None else None,
        },
        "artifacts": {
            name: {
                "path": str(path_obj.relative_to(output_dir)),
                "size_bytes": _artifact_size_bytes(path_obj),
            }
            for name, path_obj in sorted(artifact_paths.items())
        },
    }


def _resolve_generator_metadata(
    *,
    provided_metadata: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    if provided_metadata is not None:
        return dict(provided_metadata)

    repo_root = _git_output(output_dir, ["rev-parse", "--show-toplevel"])
    generated_at_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if repo_root is None:
        return {
            "git_head": None,
            "git_branch": None,
            "repo_root": None,
            "repo_worktree_dirty": None,
            "relevant_worktree_dirty": None,
            "dirty_paths": [],
            "relevant_dirty_paths": [],
            "generated_at_utc": generated_at_utc,
        }

    repo_root_path = Path(repo_root)
    git_head = _git_output(repo_root_path, ["rev-parse", "HEAD"])
    git_branch = _git_output(repo_root_path, ["rev-parse", "--abbrev-ref", "HEAD"])
    status_output = _git_output(repo_root_path, ["status", "--porcelain=v1", "--untracked-files=all"]) or ""
    dirty_paths = _parse_git_status_paths(status_output)
    output_prefix: str | None = None
    try:
        relative_output_dir = output_dir.resolve().relative_to(repo_root_path.resolve())
    except ValueError:
        relative_output_dir = None
    if relative_output_dir is not None:
        output_prefix = relative_output_dir.as_posix().rstrip("/") + "/"
    excluded_prefixes = [
        "data/",
        "docs/superpowers/plans/",
    ]
    if output_prefix is not None:
        excluded_prefixes.append(output_prefix)
    relevant_dirty_paths = [
        path
        for path in dirty_paths
        if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in excluded_prefixes)
    ]
    return {
        "git_head": git_head,
        "git_branch": git_branch,
        "repo_root": str(repo_root_path),
        "repo_worktree_dirty": bool(dirty_paths),
        "relevant_worktree_dirty": bool(relevant_dirty_paths),
        "dirty_paths": dirty_paths,
        "relevant_dirty_paths": relevant_dirty_paths,
        "generated_at_utc": generated_at_utc,
    }


def _git_output(cwd: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _parse_git_status_paths(status_output: str) -> list[str]:
    paths: list[str] = []
    for line in status_output.splitlines():
        if not line:
            continue
        path_text = line[3:].strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", maxsplit=1)[1]
        paths.append(path_text.replace("\\", "/"))
    return paths


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_provenance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256_file(path) if path.exists() else None,
    }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _export_symbol_snapshot_feature_shards(
    trade_dates: list[str],
    active_snapshot_key_dir: Path,
    output_dir: Path,
    symbol_master_frame: pd.DataFrame,
    market_data_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for trade_date in trade_dates:
        active_day = pd.read_parquet(active_snapshot_key_dir / f"{trade_date}.parquet")
        if active_day.empty:
            _write_parquet_dataframe(active_day, output_dir / f"{trade_date}.parquet")
            continue
        bars_full_day = _read_partition_parquet(
            market_data_root,
            "bars_5m",
            trade_date,
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "date"],
        )
        if not bars_full_day.empty:
            bars_full_day = bars_full_day.sort_values(["symbol", "timestamp"]).copy()
            bars_full_day["bar_ret_5m_past"] = bars_full_day.groupby("symbol")["close"].pct_change(1)
            bars_full_day["bar_ret_15m_past"] = bars_full_day.groupby("symbol")["close"].pct_change(3)
            bars_full_day["bar_dollar_volume"] = bars_full_day["close"] * bars_full_day["volume"]
            bars_full_day["bar_volume_cs_z"] = bars_full_day.groupby("timestamp")["volume"].transform(_safe_zscore)
            bars_full_day["bar_dollar_volume_cs_z"] = bars_full_day.groupby("timestamp")["bar_dollar_volume"].transform(_safe_zscore)
            bars_day = bars_full_day.rename(
                columns={
                    "open": "bar_open",
                    "high": "bar_high",
                    "low": "bar_low",
                    "close": "bar_close",
                    "volume": "bar_volume",
                    "vwap": "bar_vwap",
                }
            )
        else:
            bars_day = bars_full_day

        trade_flow_day = _read_partition_parquet(
            market_data_root,
            "trade_flow_1m",
            trade_date,
            columns=[
                "ticker",
                "minute",
                "trade_count",
                "volume",
                "dollar_volume",
                "imbalance_proxy",
                "large_trade_count",
                "large_trade_dollar_volume",
                "off_exchange_volume",
            ],
        ).rename(
            columns={
                "trade_count": "flow_trade_count",
                "volume": "flow_volume",
                "dollar_volume": "flow_dollar_volume",
                "imbalance_proxy": "flow_imbalance_proxy",
                "large_trade_count": "flow_large_trade_count",
                "large_trade_dollar_volume": "flow_large_trade_dollar_volume",
            }
        )

        merged = active_day.merge(
            bars_day,
            how="left",
            left_on=["symbol", "snapshot_timestamp"],
            right_on=["symbol", "timestamp"],
        )
        if "timestamp" in merged.columns:
            merged = merged.drop(columns=["timestamp"])
        merged = merged.merge(trade_flow_day, how="left", left_on=["symbol", "snapshot_timestamp"], right_on=["ticker", "minute"])
        merged = merged.drop(columns=[column for column in ("ticker", "minute") if column in merged.columns])
        merged = merged.merge(symbol_master_frame, how="left", on="symbol")
        merged["feature_source"] = "bars_5m+trade_flow_1m_derived"
        _write_parquet_dataframe(merged, output_dir / f"{trade_date}.parquet")


def _export_symbol_forward_label_shards(
    trade_dates: list[str],
    active_snapshot_key_dir: Path,
    output_dir: Path,
    primary_benchmark: str,
    market_data_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for trade_date in trade_dates:
        active_day = pd.read_parquet(active_snapshot_key_dir / f"{trade_date}.parquet")
        if active_day.empty:
            _write_parquet_dataframe(active_day, output_dir / f"{trade_date}.parquet")
            continue
        labels_day = _read_partition_parquet(
            market_data_root,
            "labels_1m",
            trade_date,
            columns=["symbol", "timestamp", "future_ret_1m", "future_ret_5m", "future_ret_15m", "future_ret_30m"],
        )
        benchmark_day = labels_day.loc[labels_day["symbol"] == primary_benchmark, [
            "timestamp",
            "future_ret_1m",
            "future_ret_5m",
            "future_ret_15m",
            "future_ret_30m",
        ]].rename(
            columns={
                "future_ret_1m": "benchmark_future_ret_1m",
                "future_ret_5m": "benchmark_future_ret_5m",
                "future_ret_15m": "benchmark_future_ret_15m",
                "future_ret_30m": "benchmark_future_ret_30m",
            }
        )

        merged = active_day.merge(
            labels_day,
            how="left",
            left_on=["symbol", "snapshot_timestamp"],
            right_on=["symbol", "timestamp"],
        )
        if "timestamp" in merged.columns:
            merged = merged.drop(columns=["timestamp"])
        merged = merged.merge(
            benchmark_day,
            how="left",
            left_on="snapshot_timestamp",
            right_on="timestamp",
        )
        if "timestamp" in merged.columns:
            merged = merged.drop(columns=["timestamp"])
        merged.insert(5, "benchmark_symbol", primary_benchmark)
        merged["excess_future_ret_1m"] = merged["future_ret_1m"] - merged["benchmark_future_ret_1m"]
        merged["excess_future_ret_5m"] = merged["future_ret_5m"] - merged["benchmark_future_ret_5m"]
        merged["excess_future_ret_15m"] = merged["future_ret_15m"] - merged["benchmark_future_ret_15m"]
        merged["excess_future_ret_30m"] = merged["future_ret_30m"] - merged["benchmark_future_ret_30m"]
        _write_parquet_dataframe(merged, output_dir / f"{trade_date}.parquet")


def _export_benchmark_series_shards(
    trade_dates: list[str],
    output_dir: Path,
    benchmark_symbols: tuple[str, ...],
    market_data_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for trade_date in trade_dates:
        frame = _read_partition_parquet(
            market_data_root,
            "bars_5m",
            trade_date,
            columns=["timestamp", "open", "high", "low", "close", "volume", "symbol", "vwap", "source", "date"],
        )
        frame = frame.loc[frame["symbol"].isin(benchmark_symbols)].copy()
        if not frame.empty:
            frame = frame.sort_values(["symbol", "timestamp"])
            frame["ret_5m"] = frame.groupby("symbol")["close"].pct_change(1)
        _write_parquet_dataframe(frame, output_dir / f"{trade_date}.parquet")


def _copy_query_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> None:
    connection.execute(
        f"COPY ({query}) TO '{_escape_sql_literal(str(output_path))}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def _copy_query_to_partitioned_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_dir: Path,
    *,
    trade_date_column: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trade_dates = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT CAST(trade_date AS VARCHAR) FROM pack_snapshot_context ORDER BY 1"
        ).fetchall()
    ]
    for trade_date in trade_dates:
        output_path = output_dir / f"{trade_date}.parquet"
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM ({query}) AS shard_source
                WHERE {trade_date_column} = DATE '{trade_date}'
            ) TO '{_escape_sql_literal(str(output_path))}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )


def _copy_trade_date_queries_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    trade_dates: list[str],
    output_dir: Path,
    query_builder: Callable[[str], str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for trade_date in trade_dates:
        output_path = output_dir / f"{trade_date}.parquet"
        _copy_query_to_parquet(connection, query_builder(trade_date), output_path)


def _copy_query_to_csv(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
    *,
    parameters: dict[str, str] | None = None,
) -> None:
    if parameters:
        relation = connection.sql(query, params=parameters)
        relation.write_csv(str(output_path))
        return
    connection.execute(
        f"COPY ({query}) TO '{_escape_sql_literal(str(output_path))}' (HEADER, DELIMITER ',')"
    )


def _scalar_int(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    value = connection.execute(query).fetchone()[0]
    return int(value or 0)


def _trade_dates(connection: duckdb.DuckDBPyConnection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT CAST(trade_date AS VARCHAR) FROM pack_snapshot_context ORDER BY 1"
        ).fetchall()
    ]


def _artifact_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def _read_partition_parquet(
    market_data_root: Path,
    dataset_name: str,
    trade_date: str,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    partition_path = market_data_root / dataset_name / f"date={trade_date}" / f"{dataset_name}.parquet"
    if not partition_path.exists():
        return pd.DataFrame(columns=columns or [])
    try:
        frame = pd.read_parquet(partition_path, columns=columns)
    except Exception:
        frame = pd.read_parquet(partition_path)
        if columns is not None:
            existing_columns = [column for column in columns if column in frame.columns]
            frame = frame.loc[:, existing_columns]
    for column in ("timestamp", "minute"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
            if getattr(frame[column].dt, "tz", None) is not None:
                frame[column] = frame[column].dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _safe_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    mean = series.mean()
    return (series - mean) / std


def _write_parquet_dataframe(frame: pd.DataFrame, output_path: Path) -> None:
    try:
        frame.to_parquet(output_path, index=False, compression="zstd")
    except Exception:
        connection = duckdb.connect()
        try:
            connection.register("frame_df", frame)
            connection.execute(
                f"COPY frame_df TO '{_escape_sql_literal(str(output_path))}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            connection.close()


def _escape_sql_literal(value: str) -> str:
    return value.replace("\\", "/").replace("'", "''")


def _sql_string_list(values: tuple[str, ...]) -> str:
    if not values:
        return "'SPY'"
    return ", ".join(f"'{_escape_sql_literal(value)}'" for value in values)
