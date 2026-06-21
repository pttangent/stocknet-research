from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import duckdb

from stocknetv2.application.services.theme_discovery_orchestrator import (
    ThemeDiscoveryOrchestrator,
    ThemeDiscoveryRunConfig,
)
from stocknetv2.application.services.consensus_service import ConsensusService
from stocknetv2.application.services.layer_execution_service import LayerExecutionService
from stocknetv2.application.services.lifecycle_service import LifecycleService
from stocknetv2.application.services.read_model_service import ReadModelService
from stocknetv2.application.services.semantic_service import SemanticService
from stocknetv2.application.services.theme_flow_service import ThemeFlowService
from stocknetv2.application.services.theme_quality_service import ThemeQualityService
from stocknetv2.domain.snapshot.snapshot_clock import SnapshotClock
from stocknetv2.domain.graph.layer_config import ThemeDiscoverySettings
from stocknetv2.infrastructure.db.schema_manager import SchemaManager
from stocknetv2.infrastructure.repositories.audit_repository import AuditRepository
from stocknetv2.infrastructure.repositories.graph_write_repository import GraphWriteRepository
from stocknetv2.infrastructure.repositories.market_read_repository import (
    LegacyDuckDBSource,
    LegacySourceLayout,
    MarketReadRepository,
)
from stocknetv2.infrastructure.repositories.read_model_repository import ReadModelRepository
from stocknetv2.infrastructure.repositories.theme_write_repository import ThemeWriteRepository


def run_theme_discovery(
    *,
    database_path: Path | str,
    legacy_data_root: Path | str | None = None,
    legacy_database_path: Path | str | None = None,
    symbol_limit: int | None = None,
    graph_build_only: bool = False,
    run_id: str,
    run_name: str,
    date_start: str,
    date_end: str,
    config_id: str,
    config_name: str,
    config_scope: str,
    config_version: str,
    code_commit: str,
    layer_workers: int = 1,
    dtw_backend: str = "cpu_python",
    dtw_torch_device: str = "auto",
    dtw_torch_batch_pair_threshold: int = 1024,
):
    resolved_database_path = Path(database_path).expanduser().resolve()
    resolved_database_path.parent.mkdir(parents=True, exist_ok=True)
    discovery_settings = ThemeDiscoverySettings(
        dtw_return=replace(
            ThemeDiscoverySettings().dtw_return,
            backend=dtw_backend,
            torch_device=dtw_torch_device,
            torch_batch_pair_threshold=max(1, dtw_torch_batch_pair_threshold),
        ),
        dtw_trade_flow=replace(
            ThemeDiscoverySettings().dtw_trade_flow,
            backend=dtw_backend,
            torch_device=dtw_torch_device,
            torch_batch_pair_threshold=max(1, dtw_torch_batch_pair_threshold),
        ),
    )

    market_source = _build_market_source(
        legacy_data_root=legacy_data_root,
        legacy_database_path=legacy_database_path,
    )

    connection = duckdb.connect(str(resolved_database_path))
    try:
        SchemaManager(connection).initialize()
        orchestrator = ThemeDiscoveryOrchestrator(
            market_repository=MarketReadRepository(
                market_source,
                symbol_limit=symbol_limit,
            ),
            audit_repository=AuditRepository(connection),
            snapshot_clock=SnapshotClock(),
            layer_execution_service=LayerExecutionService(
                parallel_workers=max(1, layer_workers),
                settings=discovery_settings,
            ),
            graph_write_repository=GraphWriteRepository(connection),
            consensus_service=ConsensusService(config=discovery_settings.consensus),
            theme_write_repository=ThemeWriteRepository(connection),
            semantic_service=SemanticService(),
            lifecycle_service=LifecycleService(),
            theme_quality_service=ThemeQualityService(),
            theme_flow_service=ThemeFlowService(),
            read_model_service=ReadModelService(),
            read_model_repository=ReadModelRepository(connection),
        )
        config = ThemeDiscoveryRunConfig(
            run_id=run_id,
            run_name=run_name,
            date_start=date_start,
            date_end=date_end,
            config_id=config_id,
            config_name=config_name,
            config_scope=config_scope,
            config_version=config_version,
            code_commit=code_commit,
            graph_build_only=graph_build_only,
            discovery_settings=discovery_settings.to_dict(),
        )
        return orchestrator.run(config)
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the StockNetV2 T1 theme discovery pipeline.")
    parser.add_argument("--database", required=True, help="Target StockNetV2 DuckDB path.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--legacy-data-root", help="Legacy StockNet data root.")
    source_group.add_argument("--legacy-database", help="Legacy StockNet DuckDB path.")
    parser.add_argument("--symbol-limit", type=int, help="Optional pilot limit on number of symbols.")
    parser.add_argument("--graph-build-only", action="store_true", help="Build graph layers only and skip theme downstream.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--date-start", required=True)
    parser.add_argument("--date-end", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--config-scope", default="t1")
    parser.add_argument("--config-version", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--layer-workers", type=int, help="Process workers for per-snapshot layer builds.")
    parser.add_argument(
        "--dtw-backend",
        default="cpu_python",
        choices=("cpu_python", "torch_cpu", "torch_cuda", "torch_auto"),
        help="DTW execution backend for dtw_return_similarity_graph and dtw_trade_flow_similarity_graph.",
    )
    parser.add_argument(
        "--dtw-torch-device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Preferred torch device when a torch DTW backend is used.",
    )
    parser.add_argument(
        "--dtw-torch-batch-pair-threshold",
        type=int,
        default=1024,
        help="Minimum pair count before switching DTW work to the torch backend.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_theme_discovery(
        database_path=args.database,
        legacy_data_root=args.legacy_data_root,
        legacy_database_path=args.legacy_database,
        symbol_limit=args.symbol_limit,
        graph_build_only=args.graph_build_only,
        run_id=args.run_id,
        run_name=args.run_name,
        date_start=args.date_start,
        date_end=args.date_end,
        config_id=args.config_id,
        config_name=args.config_name,
        config_scope=args.config_scope,
        config_version=args.config_version,
        code_commit=args.code_commit,
        layer_workers=args.layer_workers or _default_layer_workers(graph_build_only=args.graph_build_only),
        dtw_backend=args.dtw_backend,
        dtw_torch_device=args.dtw_torch_device,
        dtw_torch_batch_pair_threshold=args.dtw_torch_batch_pair_threshold,
    )
    print(
        f"Completed run {summary.run_id} for {len(summary.trade_dates_processed)} trade date(s) "
        f"with {summary.snapshot_count} snapshots."
    )
    return 0

def _build_market_source(
    *,
    legacy_data_root: Path | str | None,
    legacy_database_path: Path | str | None,
):
    if legacy_database_path:
        return LegacyDuckDBSource(database_path=legacy_database_path)
    if legacy_data_root:
        return LegacySourceLayout(data_root=legacy_data_root)
    raise ValueError("Either legacy_data_root or legacy_database_path must be provided.")


def _default_layer_workers(*, graph_build_only: bool) -> int:
    if graph_build_only:
        return 1
    available_cpus = os.cpu_count() or 1
    return max(1, min(6, available_cpus))


if __name__ == "__main__":
    raise SystemExit(main())
