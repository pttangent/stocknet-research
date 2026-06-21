from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd


def _write_metadata_trust_policy(output_path: Path) -> None:
    policy_path = Path(__file__).resolve().parents[4] / "metadata_trust_policy.json"
    if policy_path.exists():
        output_path.write_text(policy_path.read_text(encoding="utf-8"), encoding="utf-8")
        return
    output_path.write_text(
        json.dumps(_default_metadata_trust_policy(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _default_metadata_trust_policy() -> dict[str, object]:
    return {
        "policy_version": "2026-06-20",
        "purpose": "Metadata is exported for post-hoc validation and review. It must not introduce future bias into model training or graph construction.",
        "safe_model_features": ["sector", "industry", "exchange", "country"],
        "time_dependent_features": ["market_cap_bucket_at_t", "price_at_t", "shares_outstanding_at_t"],
        "unsafe_interpretation_only": ["supplier", "customer", "theme", "narrative", "moat"],
        "notes": [
            "Current metadata exports are intended for ex-post evaluation only.",
            "Do not backfill modern narrative labels into historical model features.",
            "If market cap is required in modeling, reconstruct it at time t from contemporaneous price and historically valid shares outstanding.",
        ],
    }


def _prepare_active_snapshot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["snapshot_timestamp"] = pd.to_datetime(prepared["snapshot_timestamp"])
    return prepared.sort_values(["symbol", "snapshot_timestamp"]).reset_index(drop=True)


def _prepare_feature_review_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "graph_input_feature_timestamp",
                "graph_input_available_time",
                "ret_1m",
                "volume_z_12",
                "imbalance_z",
                "large_trade_ratio_z",
                "flow_impulse_score",
            ]
        )
    prepared = frame.copy()
    if "ret_1m" not in prepared.columns and "ret_1m_past" in prepared.columns:
        prepared["ret_1m"] = prepared["ret_1m_past"]
    if "volume_z_12" not in prepared.columns and "volume_z_proxy" in prepared.columns:
        prepared["volume_z_12"] = prepared["volume_z_proxy"]
    if "large_trade_ratio_z" not in prepared.columns and "large_trade_ratio" in prepared.columns:
        prepared["large_trade_ratio_z"] = prepared["large_trade_ratio"]
    if "imbalance_z" not in prepared.columns and "imbalance_proxy" in prepared.columns:
        prepared["imbalance_z"] = prepared["imbalance_proxy"]
    if "flow_impulse_score" not in prepared.columns:
        if "imbalance_z" in prepared.columns:
            prepared["flow_impulse_score"] = pd.to_numeric(prepared["imbalance_z"], errors="coerce").fillna(0.0)
        else:
            prepared["flow_impulse_score"] = 0.0
    if "graph_input_available_time" not in prepared.columns:
        if "available_time" in prepared.columns:
            prepared["graph_input_available_time"] = pd.to_datetime(prepared["available_time"])
        elif "bar_end" in prepared.columns:
            prepared["graph_input_available_time"] = pd.to_datetime(prepared["bar_end"])
        else:
            prepared["graph_input_available_time"] = pd.to_datetime(prepared["timestamp"]) + pd.Timedelta(minutes=1)
    prepared["graph_input_feature_timestamp"] = pd.to_datetime(prepared["timestamp"])
    return prepared.sort_values(["symbol", "graph_input_available_time"]).reset_index(drop=True)


def _prepare_trade_flow_review_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "flow_feature_timestamp",
                "flow_available_time",
                "flow_trade_count",
                "flow_volume",
                "flow_dollar_volume",
                "flow_imbalance_proxy",
                "flow_large_trade_count",
                "flow_large_trade_dollar_volume",
            ]
        )
    prepared = frame.copy()
    if "ticker" in prepared.columns:
        prepared = prepared.rename(columns={"ticker": "symbol"})
    if "minute" in prepared.columns:
        prepared["flow_feature_timestamp"] = pd.to_datetime(prepared["minute"])
    else:
        prepared["flow_feature_timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared["flow_available_time"] = prepared["flow_feature_timestamp"] + pd.Timedelta(minutes=1)
    return prepared.sort_values(["symbol", "flow_available_time"]).reset_index(drop=True)


def _prepare_label_review_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "label_source_timestamp",
                "label_available_time",
                "future_ret_1m",
                "future_ret_5m",
                "future_ret_15m",
                "future_ret_30m",
            ]
        )
    prepared = frame.copy()
    prepared["label_source_timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared["label_available_time"] = prepared["label_source_timestamp"] + pd.Timedelta(minutes=1)
    return prepared.sort_values(["symbol", "label_available_time"]).reset_index(drop=True)


def _merge_latest_available(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    right_time_column: str,
    by_column: str = "symbol",
) -> pd.DataFrame:
    if left.empty or right.empty:
        return left.copy()
    left_working = left.copy()
    left_working["_merge_row_order"] = range(len(left_working))
    right_working = right.copy()
    if by_column in left_working.columns:
        left_working[by_column] = pd.Series(left_working[by_column], dtype="string[python]")
    if by_column in right_working.columns:
        right_working[by_column] = pd.Series(right_working[by_column], dtype="string[python]")
    left_sorted = left_working.sort_values(["snapshot_timestamp", by_column]).reset_index(drop=True)
    right_sorted = right_working.sort_values([right_time_column, by_column]).reset_index(drop=True)
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        by=by_column,
        left_on="snapshot_timestamp",
        right_on=right_time_column,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("_merge_row_order").drop(columns=["_merge_row_order"]).reset_index(drop=True)


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA


def _parquet_dataset_columns(path: Path) -> set[str]:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    existing_files = [file_path for file_path in files if file_path.exists()]
    if not existing_files:
        return set()
    sample = pd.read_parquet(existing_files[0])
    return set(str(column) for column in sample.columns)


def _select_parquet_column_expr(
    available_columns: set[str],
    table_alias: str,
    candidates: list[str],
    output_alias: str,
) -> str:
    for candidate in candidates:
        if candidate in available_columns:
            return f"{table_alias}.{candidate} AS {output_alias}"
    return f"NULL AS {output_alias}"


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
    for column in ("timestamp", "minute", "bar_end", "available_time"):
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
