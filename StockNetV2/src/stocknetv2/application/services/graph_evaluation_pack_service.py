from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .graph_evaluation_pack.alpha_reports import (
    _alpha_factors_for_layer,
    _export_alpha_feature_ranking_report,
    _export_cross_window_alpha_comparison_report,
)
from .graph_evaluation_pack.builder import (
    GraphEvaluationPackConfig,
    GraphEvaluationPackSummary,
    build_graph_evaluation_pack,
)


def _resolve_generator_metadata(
    *,
    provided_metadata: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    from .graph_evaluation_pack import provenance

    original_git_output = provenance._git_output
    original_parse = provenance._parse_git_status_paths
    try:
        provenance._git_output = _git_output
        provenance._parse_git_status_paths = _parse_git_status_paths
        return provenance._resolve_generator_metadata(
            provided_metadata=provided_metadata,
            output_dir=output_dir,
        )
    finally:
        provenance._git_output = original_git_output
        provenance._parse_git_status_paths = original_parse


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


__all__ = [
    "GraphEvaluationPackConfig",
    "GraphEvaluationPackSummary",
    "build_graph_evaluation_pack",
    "_alpha_factors_for_layer",
    "_export_alpha_feature_ranking_report",
    "_export_cross_window_alpha_comparison_report",
    "_resolve_generator_metadata",
    "_git_output",
    "_parse_git_status_paths",
]
