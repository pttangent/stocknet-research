from __future__ import annotations

from pathlib import Path

import duckdb

from stocknetv2.application.services.symbol_metadata_service import (
    empty_symbol_metadata_frame,
    read_symbol_metadata_csv,
)


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
        symbol_master_frame = empty_symbol_metadata_frame()
    else:
        symbol_master_frame = read_symbol_metadata_csv(metadata_csv_path)
    connection.register("pack_symbol_master_frame", symbol_master_frame)
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW pack_symbol_master AS
        SELECT *
        FROM pack_symbol_master_frame
        WHERE symbol IS NOT NULL
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
