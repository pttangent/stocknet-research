from __future__ import annotations

import json

import pandas as pd

from stocknetwork.reporting import build_experiment_report


def test_build_experiment_report_writes_markdown_summary(tmp_path):
    snapshot_dir = tmp_path / "snapshots"
    baseline_dir = tmp_path / "baselines"
    tgnn_dir = tmp_path / "tgnn"
    output_dir = tmp_path / "report"
    snapshot_dir.mkdir()
    baseline_dir.mkdir()
    tgnn_dir.mkdir()
    output_dir.mkdir()

    (snapshot_dir / "_summary.json").write_text(json.dumps({"snapshot_count": 1552, "symbol_count": 20, "status": "completed"}), encoding="utf-8")
    (snapshot_dir / "_validation.json").write_text(json.dumps({"compute_backend": "torch"}), encoding="utf-8")
    (baseline_dir / "baseline_metrics.csv").write_text("baseline,metric,value\nstatic_graph_logistic,auc,0.73\n", encoding="utf-8")
    (tgnn_dir / "tgnn_metrics.csv").write_text("metric,value\nauc,0.74\naverage_precision,0.91\n", encoding="utf-8")
    (tgnn_dir / "_summary.json").write_text(json.dumps({"device": "cpu", "gpu_enabled": False}), encoding="utf-8")
    (output_dir / "model_comparison.csv").write_text("model,metric,value\nstatic_graph_logistic,auc,0.73\ntgnn_snapshot,auc,0.74\n", encoding="utf-8")

    result = build_experiment_report(
        output_dir=output_dir,
        snapshot_dir=snapshot_dir,
        baseline_dir=baseline_dir,
        tgnn_dir=tgnn_dir,
    )

    report_text = (output_dir / "experiment_report.md").read_text(encoding="utf-8")

    assert result["report_path"].name == "experiment_report.md"
    assert "Snapshot Dataset" in report_text
    assert "compute backend: `torch`" in report_text
    assert "static_graph_logistic" in report_text
    assert "tgnn_snapshot" in report_text
