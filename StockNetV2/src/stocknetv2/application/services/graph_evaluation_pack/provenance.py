from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .io_utils import _artifact_size_bytes, _escape_sql_literal


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
        "Status: `Graph evaluation artifact ready for manual review`\n"
        "Quality gate: `Not sufficient by itself to approve theme discovery, lifecycle analysis, or backtesting`\n\n"
        "See `ASSESSMENT.md` if a month-specific research conclusion has been written.\n\n"
        "This pack is designed for manual graph-quality and financial-meaning review of the first month.\n"
        "It does not depend on rerunning the full T1 theme pipeline; instead it reconstructs evaluation context from the monthly graph-build database plus the market database.\n\n"
        "## Start Here\n\n"
        "1. Open `graph/layer_review_candidates.csv`.\n"
        "2. Use `graph/community_member_symbols.csv` for a fast CSV roster of each community, then `graph/community_metrics.parquet` and `graph/community_membership.parquet` to inspect whether large communities are real themes, sector baskets, or market-mode clusters.\n"
        "3. Use `market/symbol_snapshot_features/` to inspect the causality-safe state of each member at the snapshot.\n"
        f"4. Use `market/symbol_forward_labels/` to check whether members outperformed `{primary_benchmark}` over the next 1m/5m/15m/30m windows.\n"
        "5. Use `market/community_snapshot_features.parquet`, `market/community_forward_labels.parquet`, `market/alpha_sanity_report.csv`, and `market/alpha_feature_ranking_by_layer.csv` for the first community-level alpha sanity pass.\n"
        "6. Use `graph/snapshot_layer_diagnostics.csv` to find pathological layers, giant clusters, or snapshots where one layer dominates the universe.\n\n"
        "## Time Notes\n\n"
        "- `snapshot_clock_code` is the canonical market-clock label from the snapshot id suffix.\n"
        "- `snapshot_timestamp` is the stored timestamp value from the graph database.\n"
        "- `available_minutes_since_open` is the safest field for intraday sequencing if timezone display looks inconsistent.\n"
        "- Symbol features now carry `graph_input_available_time` and trade-flow `flow_available_time`; they are only joined into a snapshot when `available_time <= snapshot_timestamp`.\n"
        "- Forward labels now carry `label_available_time` and are aligned from the last completed 1m bucket available at the snapshot, not from unfinished 1m bars.\n"
        "- Benchmark-relative labels now carry provenance: `benchmark_label_source` tells you whether they came from `labels_1m` or a `trade_flow_1m` proxy fallback.\n"
        "- Metadata is exported for post-hoc validation only; see `market/metadata_trust_policy.json` before using any field in modeling.\n\n"
        "## Files\n\n"
        "- `graph/all_edges/`: thresholded graph edges, sharded by trade date as parquet.\n"
        "- `graph/snapshot_layer_diagnostics.csv`: per-snapshot, per-layer structure diagnostics.\n"
        "- `graph/node_layer_metrics/`: per-symbol, per-layer node metrics, sharded by trade date as parquet.\n"
        "- `graph/community_metrics.parquet`: community-level structure and concentration metrics.\n"
        "- `graph/community_membership.parquet`: member roster for each community.\n"
        "- `graph/community_member_symbols.csv`: one CSV row per community with the ordered member-symbol list for quick theme review.\n"
        "- `graph/layer_review_candidates.csv`: ranked shortlist for manual review.\n"
        "- `market/symbol_snapshot_features/`: snapshot-aligned symbol state features and actual graph inputs, sharded by trade date as parquet.\n"
        "- `market/symbol_forward_labels/`: causality-safe forward returns and benchmark-relative labels, sharded by trade date as parquet.\n"
        "- `market/community_snapshot_features.parquet`: community-level feature aggregates built only from snapshot-time-available symbol inputs.\n"
        "- `market/community_forward_labels.parquet`: community-level forward labels kept physically separate from features.\n"
        "- `market/alpha_sanity_report.csv`: first-pass RankIC / decile / hit-rate summary for community-level evaluation.\n"
        "- `market/alpha_feature_ranking_by_layer.csv`: per-layer factor ranking with sample-size-aware confidence buckets and research actions.\n"
        "- `market/metadata_trust_policy.json`: allowed post-hoc validation use vs modeling restrictions for metadata fields.\n"
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
    output = completed.stdout.rstrip("\r\n")
    return output or None


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
