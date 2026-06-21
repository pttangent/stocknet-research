from __future__ import annotations

import os
import shutil
import tempfile
import time
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Protocol

import duckdb

from stocknetv2.infrastructure.db.schema_manager import SchemaManager
from stocknetv2.interfaces.cli.run_theme_discovery_t1 import run_theme_discovery

for _thread_env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env_var, "1")


class MarketCalendarProtocol(Protocol):
    def list_available_trade_dates(self, dataset_name: str) -> list[str]: ...


@dataclass(frozen=True)
class GraphBuildRangeConfig:
    data_root: Path | str
    output_database_path: Path | str
    date_start: str
    date_end: str
    run_prefix: str
    config_id: str
    config_name: str
    config_version: str
    code_commit: str
    symbol_limit: int | None = None
    continue_on_error: bool = False
    shard_directory: Path | str | None = None
    keep_shards: bool = False
    layer_workers_per_process: int = 1


@dataclass(frozen=True)
class GraphBuildShardTask:
    trade_date: str
    run_id: str
    run_name: str
    database_path: Path
    data_root: Path
    symbol_limit: int | None
    config_id: str
    config_name: str
    config_version: str
    code_commit: str
    layer_workers: int


@dataclass(frozen=True)
class GraphBuildShardResult:
    trade_date: str
    run_id: str
    database_path: Path
    snapshot_count: int
    data_version: str
    elapsed_seconds: float


@dataclass(frozen=True)
class GraphBuildShardFailure:
    trade_date: str
    run_id: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class GraphBuildRangeSummary:
    processed_dates: list[str]
    shard_results: list[GraphBuildShardResult]
    failures: list[GraphBuildShardFailure]
    failure_count: int
    elapsed_seconds: float


class GraphBuildRangeService:
    def __init__(
        self,
        *,
        market_calendar: MarketCalendarProtocol,
        max_workers: int = 1,
        shard_runner: Callable[[GraphBuildShardTask], GraphBuildShardResult] | None = None,
        executor_factory: Callable[[int], Executor] | None = None,
    ) -> None:
        self._market_calendar = market_calendar
        self._max_workers = max(1, max_workers)
        self._shard_runner = shard_runner or _run_graph_build_shard
        self._executor_factory = executor_factory or _build_graph_range_executor

    def run(
        self,
        config: GraphBuildRangeConfig,
        *,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> GraphBuildRangeSummary:
        started_at = time.perf_counter()
        trade_dates = [
            trade_date
            for trade_date in self._market_calendar.list_available_trade_dates("bars_5m")
            if config.date_start <= trade_date <= config.date_end
        ]
        if not trade_dates:
            raise RuntimeError("No available trade dates for the configured range.")

        output_database_path = Path(config.output_database_path).expanduser().resolve()
        output_database_path.parent.mkdir(parents=True, exist_ok=True)
        shard_directory, should_cleanup_shards = self._resolve_shard_directory(config, output_database_path)
        tasks = [
            GraphBuildShardTask(
                trade_date=trade_date,
                run_id=f"{config.run_prefix}_{trade_date}",
                run_name=f"{config.run_prefix} {trade_date}",
                database_path=shard_directory / f"{trade_date}.duckdb",
                data_root=Path(config.data_root).expanduser().resolve(),
                symbol_limit=config.symbol_limit,
                config_id=config.config_id,
                config_name=config.config_name,
                config_version=config.config_version,
                code_commit=config.code_commit,
                layer_workers=max(1, config.layer_workers_per_process),
            )
            for trade_date in trade_dates
        ]
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "range_started",
                    "date_start": config.date_start,
                    "date_end": config.date_end,
                    "total_dates": len(tasks),
                    "max_workers": self._max_workers,
                }
            )

        shard_results: list[GraphBuildShardResult] = []
        failures: list[GraphBuildShardFailure] = []

        try:
            if self._max_workers <= 1 or len(tasks) <= 1:
                for task in tasks:
                    try:
                        result = self._shard_runner(task)
                        shard_results.append(result)
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "status": "shard_completed",
                                    "trade_date": result.trade_date,
                                    "run_id": result.run_id,
                                    "snapshot_count": result.snapshot_count,
                                    "elapsed_seconds": result.elapsed_seconds,
                                    "completed_dates": len(shard_results),
                                    "total_dates": len(tasks),
                                }
                            )
                    except Exception as exc:
                        failure = GraphBuildShardFailure(
                            trade_date=task.trade_date,
                            run_id=task.run_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        failures.append(failure)
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "status": "shard_failed",
                                    "trade_date": failure.trade_date,
                                    "run_id": failure.run_id,
                                    "error_type": failure.error_type,
                                    "error_message": failure.error_message,
                                    "completed_dates": len(shard_results),
                                    "total_dates": len(tasks),
                                }
                            )
                        if not config.continue_on_error:
                            raise
            else:
                executor = self._executor_factory(self._max_workers)
                futures: dict[Future, GraphBuildShardTask] = {
                    executor.submit(self._shard_runner, task): task
                    for task in tasks
                }
                for future, task in futures.items():
                    try:
                        result = future.result()
                        shard_results.append(result)
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "status": "shard_completed",
                                    "trade_date": result.trade_date,
                                    "run_id": result.run_id,
                                    "snapshot_count": result.snapshot_count,
                                    "elapsed_seconds": result.elapsed_seconds,
                                    "completed_dates": len(shard_results),
                                    "total_dates": len(tasks),
                                }
                            )
                    except Exception as exc:
                        failure = GraphBuildShardFailure(
                            trade_date=task.trade_date,
                            run_id=task.run_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        failures.append(failure)
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "status": "shard_failed",
                                    "trade_date": failure.trade_date,
                                    "run_id": failure.run_id,
                                    "error_type": failure.error_type,
                                    "error_message": failure.error_message,
                                    "completed_dates": len(shard_results),
                                    "total_dates": len(tasks),
                                }
                            )
                        if not config.continue_on_error:
                            raise
                if hasattr(executor, "shutdown"):
                    executor.shutdown(wait=True, cancel_futures=False)

            if shard_results:
                _merge_shard_databases(output_database_path, [result.database_path for result in shard_results])

            summary = GraphBuildRangeSummary(
                processed_dates=[result.trade_date for result in shard_results],
                shard_results=sorted(shard_results, key=lambda result: result.trade_date),
                failures=sorted(failures, key=lambda failure: failure.trade_date),
                failure_count=len(failures),
                elapsed_seconds=round(time.perf_counter() - started_at, 2),
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "status": "range_completed",
                        "processed_dates": list(summary.processed_dates),
                        "failure_count": summary.failure_count,
                        "elapsed_seconds": summary.elapsed_seconds,
                    }
                )
            return summary
        finally:
            if should_cleanup_shards and shard_directory.exists():
                shutil.rmtree(shard_directory, ignore_errors=True)

    @staticmethod
    def _resolve_shard_directory(
        config: GraphBuildRangeConfig,
        output_database_path: Path,
    ) -> tuple[Path, bool]:
        if config.shard_directory is not None:
            shard_directory = Path(config.shard_directory).expanduser().resolve()
            shard_directory.mkdir(parents=True, exist_ok=True)
            return shard_directory, False
        shard_directory = Path(
            tempfile.mkdtemp(
                prefix=f"{output_database_path.stem}_shards_",
                dir=str(output_database_path.parent),
            )
        )
        return shard_directory, not config.keep_shards


def _run_graph_build_shard(task: GraphBuildShardTask) -> GraphBuildShardResult:
    task.database_path.parent.mkdir(parents=True, exist_ok=True)
    if task.database_path.exists():
        task.database_path.unlink()
    started_at = time.perf_counter()
    summary = run_theme_discovery(
        database_path=task.database_path,
        legacy_data_root=task.data_root,
        symbol_limit=task.symbol_limit,
        graph_build_only=True,
        run_id=task.run_id,
        run_name=task.run_name,
        date_start=task.trade_date,
        date_end=task.trade_date,
        config_id=task.config_id,
        config_name=task.config_name,
        config_scope="t1",
        config_version=task.config_version,
        code_commit=task.code_commit,
        layer_workers=task.layer_workers,
    )
    return GraphBuildShardResult(
        trade_date=task.trade_date,
        run_id=task.run_id,
        database_path=task.database_path,
        snapshot_count=summary.snapshot_count,
        data_version=summary.data_version,
        elapsed_seconds=round(time.perf_counter() - started_at, 2),
    )


def _merge_shard_databases(output_database_path: Path, shard_paths: list[Path]) -> None:
    if output_database_path.exists():
        output_database_path.unlink()
    wal_path = output_database_path.with_suffix(output_database_path.suffix + ".wal")
    if wal_path.exists():
        wal_path.unlink()

    connection = duckdb.connect(str(output_database_path))
    try:
        SchemaManager(connection).initialize()
        for index, shard_path in enumerate(sorted(shard_paths), start=1):
            alias = f"shard_{index}"
            escaped_path = str(shard_path).replace("\\", "/").replace("'", "''")
            connection.execute(f"ATTACH '{escaped_path}' AS {alias}")
            try:
                for table_name in (
                    "config_registry",
                    "theme_discovery_run",
                    "input_lineage",
                    "graph_snapshot",
                    "graph_edge_summary",
                    "graph_layer_diagnostic",
                    "graph_edges_thresholded",
                    "layer_community",
                    "layer_community_membership",
                    "consensus_theme_candidate",
                    "theme_membership",
                    "theme_semantic_label",
                    "theme_path_lifecycle",
                    "theme_level_flow_series",
                    "frontend_snapshot_cache",
                ):
                    insert_prefix = "INSERT OR REPLACE" if table_name == "config_registry" else "INSERT"
                    connection.execute(
                        f"{insert_prefix} INTO {table_name} SELECT * FROM {alias}.{table_name}"
                    )
            finally:
                connection.execute(f"DETACH {alias}")
    finally:
        connection.close()


def _build_graph_range_executor(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
    )
