from __future__ import annotations

import json
from argparse import Namespace
from datetime import UTC, datetime

from stocknetwork.run_metadata import create_run_context


def test_create_run_context_writes_expected_metadata_files(tmp_path):
    output_dir = tmp_path / "artifacts" / "parquet_15m"
    args = Namespace(
        input="input.csv",
        output=str(output_dir),
        interval="15m",
        workers=8,
    )

    context = create_run_context(
        stage="build_parquet",
        output_dir=output_dir,
        args=args,
        inputs={"input_csv": tmp_path / "input.csv"},
        run_id="build_parquet_20260607_130000",
        now=datetime(2026, 6, 7, 13, 0, 0, tzinfo=UTC),
    )

    context.write_initial_metadata()
    context.write_validation({"input_exists": True, "selected_symbols": 123})
    context.write_artifacts({"manifest_csv": output_dir / "_manifest.csv"})
    context.write_summary({"success_count": 120, "failure_count": 3})

    run_payload = json.loads((output_dir / "_run.json").read_text(encoding="utf-8"))
    inputs_payload = json.loads((output_dir / "_inputs.json").read_text(encoding="utf-8"))
    validation_payload = json.loads((output_dir / "_validation.json").read_text(encoding="utf-8"))
    artifacts_payload = json.loads((output_dir / "_artifacts.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((output_dir / "_summary.json").read_text(encoding="utf-8"))

    assert run_payload["stage"] == "build_parquet"
    assert run_payload["run_id"] == "build_parquet_20260607_130000"
    assert run_payload["parameters"]["interval"] == "15m"
    assert run_payload["parameters"]["workers"] == 8
    assert run_payload["started_at"] == "2026-06-07T13:00:00Z"

    assert inputs_payload["input_csv"].endswith("input.csv")
    assert validation_payload["input_exists"] is True
    assert validation_payload["selected_symbols"] == 123
    assert artifacts_payload["manifest_csv"].endswith("_manifest.csv")
    assert summary_payload["success_count"] == 120
    assert summary_payload["failure_count"] == 3


def test_create_run_context_generates_run_id_from_stage_and_time(tmp_path):
    output_dir = tmp_path / "artifacts" / "research_rotation"

    context = create_run_context(
        stage="analyze_rotation",
        output_dir=output_dir,
        args=Namespace(output=str(output_dir), benchmark="SPY"),
        now=datetime(2026, 6, 7, 13, 5, 9, tzinfo=UTC),
    )

    assert context.run_id == "analyze_rotation_20260607_130509"
