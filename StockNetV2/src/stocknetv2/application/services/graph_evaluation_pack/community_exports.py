from __future__ import annotations

from pathlib import Path

import duckdb

from .alpha_reports import _augment_community_snapshot_features
from .io_utils import (
    _copy_query_to_parquet,
    _escape_sql_literal,
    _parquet_dataset_columns,
    _select_parquet_column_expr,
    _write_parquet_dataframe,
)


def _export_community_snapshot_features(
    community_membership_path: Path,
    symbol_snapshot_feature_dir: Path,
    output_path: Path,
) -> None:
    available_columns = _parquet_dataset_columns(symbol_snapshot_feature_dir)
    feature_timestamp_expr = _select_parquet_column_expr(
        available_columns,
        "sf",
        ["graph_input_feature_timestamp", "flow_feature_timestamp", "bar_timestamp", "timestamp"],
        "graph_input_feature_timestamp",
    )
    feature_available_expr = _select_parquet_column_expr(
        available_columns,
        "sf",
        ["graph_input_available_time", "flow_available_time", "bar_available_time"],
        "graph_input_available_time",
    )
    ret_expr = _select_parquet_column_expr(available_columns, "sf", ["ret_1m", "bar_ret_5m_past"], "ret_1m")
    volume_expr = _select_parquet_column_expr(
        available_columns,
        "sf",
        ["volume_z_12", "bar_volume_cs_z"],
        "volume_z_12",
    )
    imbalance_expr = _select_parquet_column_expr(
        available_columns,
        "sf",
        ["imbalance_z", "flow_imbalance_proxy"],
        "imbalance_z",
    )
    large_trade_expr = _select_parquet_column_expr(available_columns, "sf", ["large_trade_ratio_z"], "large_trade_ratio_z")
    flow_impulse_expr = _select_parquet_column_expr(available_columns, "sf", ["flow_impulse_score"], "flow_impulse_score")
    connection = duckdb.connect()
    try:
        query = f"""
        WITH member_features AS (
            SELECT
                m.trade_date,
                m.snapshot_id,
                m.snapshot_timestamp,
                m.snapshot_clock_code,
                m.graph_layer,
                m.layer_community_id,
                m.community_local_id,
                m.community_member_count,
                m.community_edge_count,
                m.edge_density,
                m.community_avg_weight,
                m.symbol,
                m.member_rank,
                m.member_weight,
                {feature_timestamp_expr},
                {feature_available_expr},
                {ret_expr},
                {volume_expr},
                {imbalance_expr},
                {large_trade_expr},
                {flow_impulse_expr},
                sf.bar_ret_5m_past,
                sf.bar_ret_15m_past,
                sf.market_cap,
                sf.sector_code,
                sf.industry_code
            FROM read_parquet('{_escape_sql_literal(str(community_membership_path))}') m
            LEFT JOIN read_parquet('{_escape_sql_literal(str(symbol_snapshot_feature_dir / "*.parquet"))}') sf
                ON sf.snapshot_id = m.snapshot_id
               AND sf.symbol = m.symbol
        ),
        layer_active_counts AS (
            SELECT
                snapshot_id,
                graph_layer,
                COUNT(DISTINCT symbol) AS layer_active_node_count
            FROM member_features
            GROUP BY 1, 2
        ),
        snapshot_active_counts AS (
            SELECT
                snapshot_id,
                COUNT(DISTINCT symbol) AS snapshot_active_symbol_count
            FROM member_features
            GROUP BY 1
        )
        SELECT
            mf.trade_date,
            mf.snapshot_id,
            mf.snapshot_timestamp,
            mf.snapshot_clock_code,
            mf.graph_layer,
            mf.layer_community_id,
            mf.community_local_id,
            MAX(mf.community_member_count) AS community_member_count,
            MAX(mf.community_edge_count) AS community_edge_count,
            MAX(mf.edge_density) AS edge_density,
            MAX(mf.community_avg_weight) AS community_avg_weight,
            COUNT(*) AS membership_rows,
            AVG(CASE WHEN ret_1m IS NOT NULL THEN 1.0 ELSE 0.0 END) AS feature_coverage_ratio,
            AVG(ret_1m) AS community_mean_ret_1m,
            AVG(volume_z_12) AS community_mean_volume_z_12,
            AVG(imbalance_z) AS community_mean_imbalance_z,
            AVG(large_trade_ratio_z) AS community_mean_large_trade_ratio_z,
            AVG(flow_impulse_score) AS community_mean_flow_impulse_score,
            AVG(bar_ret_5m_past) AS community_mean_bar_ret_5m_past,
            AVG(bar_ret_15m_past) AS community_mean_bar_ret_15m_past,
            AVG(CASE WHEN ret_1m > 0 THEN 1.0 ELSE 0.0 END) AS positive_ret_1m_breadth,
            AVG(CASE WHEN flow_impulse_score > 0 THEN 1.0 ELSE 0.0 END) AS positive_flow_breadth,
            AVG(CASE WHEN large_trade_ratio_z > 0 THEN 1.0 ELSE 0.0 END) AS positive_large_trade_breadth,
            AVG(CASE WHEN market_cap IS NOT NULL AND market_cap > 0 THEN 1.0 ELSE 0.0 END) AS market_cap_coverage_ratio,
            AVG(CASE WHEN sector_code IS NOT NULL AND UPPER(TRIM(sector_code)) <> 'UNKNOWN' THEN 1.0 ELSE 0.0 END) AS sector_coverage_ratio,
            AVG(CASE WHEN industry_code IS NOT NULL AND UPPER(TRIM(industry_code)) <> 'UNKNOWN' THEN 1.0 ELSE 0.0 END) AS industry_coverage_ratio,
            MAX(lac.layer_active_node_count) AS layer_active_node_count,
            MAX(sac.snapshot_active_symbol_count) AS snapshot_active_symbol_count,
            MIN(graph_input_feature_timestamp) AS earliest_graph_input_feature_timestamp,
            MAX(graph_input_feature_timestamp) AS latest_graph_input_feature_timestamp,
            MAX(graph_input_available_time) AS latest_graph_input_available_time
        FROM member_features mf
        LEFT JOIN layer_active_counts lac
            ON lac.snapshot_id = mf.snapshot_id
           AND lac.graph_layer = mf.graph_layer
        LEFT JOIN snapshot_active_counts sac
            ON sac.snapshot_id = mf.snapshot_id
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        """
        frame = connection.execute(query).fetchdf()
        _write_parquet_dataframe(_augment_community_snapshot_features(frame), output_path)
    finally:
        connection.close()


def _export_community_forward_labels(
    community_membership_path: Path,
    symbol_forward_label_dir: Path,
    output_path: Path,
) -> None:
    connection = duckdb.connect()
    try:
        query = f"""
        WITH member_labels AS (
            SELECT
                m.trade_date,
                m.snapshot_id,
                m.snapshot_timestamp,
                m.snapshot_clock_code,
                m.graph_layer,
                m.layer_community_id,
                m.community_local_id,
                m.community_member_count,
                m.community_edge_count,
                m.edge_density,
                m.community_avg_weight,
                m.symbol,
                m.member_rank,
                m.member_weight,
                m.member_core_score,
                sf.label_source_timestamp,
                sf.label_available_time,
                sf.future_ret_1m,
                sf.future_ret_5m,
                sf.future_ret_15m,
                sf.future_ret_30m,
                sf.excess_future_ret_1m,
                sf.excess_future_ret_5m,
                sf.excess_future_ret_15m,
                sf.excess_future_ret_30m,
                CAST(sf.benchmark_label_source AS VARCHAR) AS benchmark_label_source,
                CAST(sf.benchmark_proxy_price_method AS VARCHAR) AS benchmark_proxy_price_method
            FROM read_parquet('{_escape_sql_literal(str(community_membership_path))}') m
            LEFT JOIN read_parquet('{_escape_sql_literal(str(symbol_forward_label_dir / "*.parquet"))}') sf
                ON sf.snapshot_id = m.snapshot_id
               AND sf.symbol = m.symbol
        )
        SELECT
            trade_date,
            snapshot_id,
            snapshot_timestamp,
            snapshot_clock_code,
            graph_layer,
            layer_community_id,
            community_local_id,
            MAX(community_member_count) AS community_member_count,
            MAX(community_edge_count) AS community_edge_count,
            MAX(edge_density) AS edge_density,
            MAX(community_avg_weight) AS community_avg_weight,
            COUNT(*) AS membership_rows,
            AVG(CASE WHEN future_ret_1m IS NOT NULL THEN 1.0 ELSE 0.0 END) AS label_coverage_ratio,
            AVG(future_ret_1m) AS community_mean_future_ret_1m,
            AVG(future_ret_5m) AS community_mean_future_ret_5m,
            AVG(future_ret_15m) AS community_mean_future_ret_15m,
            AVG(future_ret_30m) AS community_mean_future_ret_30m,
            AVG(excess_future_ret_1m) AS community_equal_weight_excess_future_ret_1m,
            AVG(excess_future_ret_5m) AS community_equal_weight_excess_future_ret_5m,
            AVG(excess_future_ret_15m) AS community_equal_weight_excess_future_ret_15m,
            AVG(excess_future_ret_30m) AS community_equal_weight_excess_future_ret_30m,
            AVG(excess_future_ret_1m) AS community_mean_excess_future_ret_1m,
            AVG(excess_future_ret_5m) AS community_mean_excess_future_ret_5m,
            AVG(excess_future_ret_15m) AS community_mean_excess_future_ret_15m,
            AVG(excess_future_ret_30m) AS community_mean_excess_future_ret_30m,
            SUM(CASE WHEN excess_future_ret_1m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN excess_future_ret_1m * member_weight ELSE 0 END)
                / NULLIF(SUM(CASE WHEN excess_future_ret_1m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN member_weight ELSE 0 END), 0)
                AS community_member_weight_excess_future_ret_1m,
            SUM(CASE WHEN excess_future_ret_5m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN excess_future_ret_5m * member_weight ELSE 0 END)
                / NULLIF(SUM(CASE WHEN excess_future_ret_5m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN member_weight ELSE 0 END), 0)
                AS community_member_weight_excess_future_ret_5m,
            SUM(CASE WHEN excess_future_ret_15m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN excess_future_ret_15m * member_weight ELSE 0 END)
                / NULLIF(SUM(CASE WHEN excess_future_ret_15m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN member_weight ELSE 0 END), 0)
                AS community_member_weight_excess_future_ret_15m,
            SUM(CASE WHEN excess_future_ret_30m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN excess_future_ret_30m * member_weight ELSE 0 END)
                / NULLIF(SUM(CASE WHEN excess_future_ret_30m IS NOT NULL AND member_weight IS NOT NULL AND member_weight > 0 THEN member_weight ELSE 0 END), 0)
                AS community_member_weight_excess_future_ret_30m,
            SUM(CASE WHEN excess_future_ret_1m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN excess_future_ret_1m * member_core_score ELSE 0 END)
                / NULLIF(SUM(CASE WHEN excess_future_ret_1m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN member_core_score ELSE 0 END), 0)
                AS community_core_weighted_excess_future_ret_1m,
            SUM(CASE WHEN excess_future_ret_5m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN excess_future_ret_5m * member_core_score ELSE 0 END)
                / NULLIF(SUM(CASE WHEN excess_future_ret_5m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN member_core_score ELSE 0 END), 0)
                AS community_core_weighted_excess_future_ret_5m,
            SUM(CASE WHEN excess_future_ret_15m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN excess_future_ret_15m * member_core_score ELSE 0 END)
                / NULLIF(SUM(CASE WHEN excess_future_ret_15m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN member_core_score ELSE 0 END), 0)
                AS community_core_weighted_excess_future_ret_15m,
            SUM(CASE WHEN excess_future_ret_30m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN excess_future_ret_30m * member_core_score ELSE 0 END)
                / NULLIF(SUM(CASE WHEN excess_future_ret_30m IS NOT NULL AND member_core_score IS NOT NULL AND member_core_score > 0 THEN member_core_score ELSE 0 END), 0)
                AS community_core_weighted_excess_future_ret_30m,
            AVG(CASE WHEN member_rank <= 5 THEN excess_future_ret_1m END) AS community_top5_member_excess_future_ret_1m,
            AVG(CASE WHEN member_rank <= 5 THEN excess_future_ret_5m END) AS community_top5_member_excess_future_ret_5m,
            AVG(CASE WHEN member_rank <= 5 THEN excess_future_ret_15m END) AS community_top5_member_excess_future_ret_15m,
            AVG(CASE WHEN member_rank <= 5 THEN excess_future_ret_30m END) AS community_top5_member_excess_future_ret_30m,
            AVG(CASE WHEN member_rank <= 10 THEN excess_future_ret_1m END) AS community_top10_member_excess_future_ret_1m,
            AVG(CASE WHEN member_rank <= 10 THEN excess_future_ret_5m END) AS community_top10_member_excess_future_ret_5m,
            AVG(CASE WHEN member_rank <= 10 THEN excess_future_ret_15m END) AS community_top10_member_excess_future_ret_15m,
            AVG(CASE WHEN member_rank <= 10 THEN excess_future_ret_30m END) AS community_top10_member_excess_future_ret_30m,
            AVG(CASE WHEN future_ret_1m > 0 THEN 1.0 ELSE 0.0 END) AS positive_future_ret_1m_breadth,
            AVG(CASE WHEN future_ret_5m > 0 THEN 1.0 ELSE 0.0 END) AS positive_future_ret_5m_breadth,
            AVG(CASE WHEN future_ret_15m > 0 THEN 1.0 ELSE 0.0 END) AS positive_future_ret_15m_breadth,
            AVG(CASE WHEN future_ret_30m > 0 THEN 1.0 ELSE 0.0 END) AS positive_future_ret_30m_breadth,
            CASE
                WHEN COUNT(DISTINCT CASE WHEN benchmark_label_source IS NOT NULL THEN benchmark_label_source END) = 0 THEN 'missing'
                WHEN COUNT(DISTINCT CASE WHEN benchmark_label_source IS NOT NULL THEN benchmark_label_source END) = 1 THEN MAX(benchmark_label_source)
                ELSE 'mixed'
            END AS benchmark_label_source,
            CASE
                WHEN COUNT(DISTINCT CASE WHEN benchmark_proxy_price_method IS NOT NULL THEN benchmark_proxy_price_method END) = 0 THEN CAST(NULL AS VARCHAR)
                WHEN COUNT(DISTINCT CASE WHEN benchmark_proxy_price_method IS NOT NULL THEN benchmark_proxy_price_method END) = 1 THEN MAX(benchmark_proxy_price_method)
                ELSE 'mixed'
            END AS benchmark_proxy_price_method,
            MIN(label_source_timestamp) AS earliest_label_source_timestamp,
            MAX(label_source_timestamp) AS latest_label_source_timestamp,
            MAX(label_available_time) AS latest_label_available_time
        FROM member_labels
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        """
        _copy_query_to_parquet(connection, query, output_path)
    finally:
        connection.close()
