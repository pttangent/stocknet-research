from __future__ import annotations

import json
from argparse import Namespace

from stocknetwork.run_metadata import create_run_context


def test_run_context_writes_stage_specific_metadata_files(tmp_path):
    output_dir = tmp_path / "shared_output"
    context = create_run_context(
        stage="build_graph_snapshots",
        output_dir=output_dir,
        args=Namespace(output=str(output_dir)),
        run_id="build_graph_snapshots_20260607_140000",
    )

    context.write_initial_metadata()
    context.write_validation({"compute_backend": "torch"})
    context.write_summary({"snapshot_count": 12})

    stage_run = json.loads((output_dir / "_run.build_graph_snapshots.json").read_text(encoding="utf-8"))
    stage_validation = json.loads((output_dir / "_validation.build_graph_snapshots.json").read_text(encoding="utf-8"))
    stage_summary = json.loads((output_dir / "_summary.build_graph_snapshots.json").read_text(encoding="utf-8"))

    assert stage_run["stage"] == "build_graph_snapshots"
    assert stage_validation["compute_backend"] == "torch"
    assert stage_summary["snapshot_count"] == 12
