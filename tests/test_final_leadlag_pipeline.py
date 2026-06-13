from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from stocknet_alpha.backtest.final_pipeline import (
    build_final_strategy_self_audit,
    build_strategy_robustness_scan,
    run_final_leadlag_pipeline,
)


def _sample_evaluated() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade_date, timestamp, theme_score, leadlag_score, horizon, net_return in [
        ("2025-09-10", "2025-09-10 15:35:00+00:00", 0.65, 0.50, 10, 0.0010),
        ("2025-10-10", "2025-10-10 15:35:00+00:00", 0.66, 0.50, 10, 0.0011),
        ("2025-11-10", "2025-11-10 15:35:00+00:00", 0.67, 0.50, 10, 0.0012),
        ("2025-12-10", "2025-12-10 15:35:00+00:00", 0.68, 0.50, 10, 0.0014),
        ("2026-01-10", "2026-01-10 15:35:00+00:00", 0.69, 0.50, 10, 0.0015),
        ("2025-09-11", "2025-09-11 15:35:00+00:00", 0.72, 0.60, 1, 0.0008),
        ("2025-10-11", "2025-10-11 15:35:00+00:00", 0.73, 0.60, 1, 0.0009),
        ("2025-11-11", "2025-11-11 15:35:00+00:00", 0.74, 0.60, 1, 0.0010),
        ("2025-12-11", "2025-12-11 15:35:00+00:00", 0.75, 0.60, 1, 0.0007),
        ("2026-01-11", "2026-01-11 15:35:00+00:00", 0.76, 0.60, 1, 0.0012),
    ]:
        rows.append(
            {
                "trade_date": trade_date,
                "signal_timestamp": pd.Timestamp(timestamp) - pd.Timedelta(minutes=1),
                "decision_timestamp": pd.Timestamp(timestamp),
                "execution_timestamp": pd.Timestamp(timestamp) + pd.Timedelta(minutes=1),
                "entry_time": pd.Timestamp(timestamp) + pd.Timedelta(minutes=1),
                "horizon_minutes": horizon,
                "theme_score": theme_score,
                "leadlag_score": leadlag_score,
                "confirmed_on_15m": True,
                "commission_bps": 0.5,
                "fees_bps": 0.5,
                "slippage_bps": 1.5,
                "gross_return": net_return + 0.0005,
                "net_return": net_return,
                "community_id": "C001",
            }
        )
    return pd.DataFrame(rows)


def test_build_strategy_robustness_scan_summarizes_positive_variants():
    evaluated = _sample_evaluated()

    scan, summary = build_strategy_robustness_scan(
        evaluated,
        min_train_counts=(1, 2),
        leadlag_thresholds=(0.50, 0.55),
    )

    assert not scan.empty
    assert summary["same_sign_rate"] == 1.0
    assert summary["min_weighted_test_avg_net_return"] > 0


def test_build_final_strategy_self_audit_mentions_five_checks_and_metrics():
    split_summary = pd.DataFrame([{"split_id": "wf_0001", "avg_net_return": 0.0010}])
    strategy_trades = pd.DataFrame(
        [
            {
                "signal_timestamp": pd.Timestamp("2025-09-10 15:34:00+00:00"),
                "decision_timestamp": pd.Timestamp("2025-09-10 15:35:00+00:00"),
                "execution_timestamp": pd.Timestamp("2025-09-10 15:36:00+00:00"),
                "entry_time": pd.Timestamp("2025-09-10 15:36:00+00:00"),
                "confirmed_on_15m": True,
                "commission_bps": 0.5,
                "fees_bps": 0.5,
                "slippage_bps": 1.5,
            },
            {
                "signal_timestamp": pd.Timestamp("2025-09-10 15:39:00+00:00"),
                "decision_timestamp": pd.Timestamp("2025-09-10 15:40:00+00:00"),
                "execution_timestamp": pd.Timestamp("2025-09-10 15:41:00+00:00"),
                "entry_time": pd.Timestamp("2025-09-10 15:41:00+00:00"),
                "confirmed_on_15m": False,
                "commission_bps": 0.5,
                "fees_bps": 0.5,
                "slippage_bps": 1.5,
            },
        ]
    )
    selection_summary = {
        "split_count": 1,
        "positive_test_splits": 1,
        "weighted_test_avg_net_return": 0.0012,
    }
    robustness_summary = {
        "same_sign_rate": 1.0,
        "min_weighted_test_avg_net_return": 0.0010,
        "max_weighted_test_avg_net_return": 0.0014,
    }
    audit_checks = {
        "lookahead_guard": {"status": "PASS", "evidence": "all rows respect signal <= decision < execution <= entry"},
        "survivorship_bias": {"status": "PASS", "evidence": "source audit confirms full raw-partition universe"},
        "parameter_robustness": {"status": "PASS", "evidence": "all tested variants stayed positive"},
        "logic_explainability": {"status": "PASS", "evidence": "selected rules belong to interpretable rule set"},
        "cost_realism": {"status": "PASS", "evidence": "positive cost columns present on evaluated trades"},
    }

    report = build_final_strategy_self_audit(
        selection_summary,
        robustness_summary,
        split_summary,
        strategy_trades,
        audit_checks=audit_checks,
    )

    assert "Lookahead guard" in report
    assert "Parameter robustness" in report
    assert "weighted_test_avg_net_return=0.001200" in report
    assert "Confirmed OOS trades" in report


def test_build_final_strategy_self_audit_uses_computed_statuses_instead_of_literal_pass():
    split_summary = pd.DataFrame([{"split_id": "wf_0001", "avg_net_return": 0.0010}])
    strategy_trades = pd.DataFrame([{"confirmed_on_15m": True}, {"confirmed_on_15m": False}])
    selection_summary = {
        "split_count": 1,
        "positive_test_splits": 1,
        "weighted_test_avg_net_return": 0.0012,
    }
    robustness_summary = {
        "same_sign_rate": 1.0,
        "min_weighted_test_avg_net_return": 0.0010,
        "max_weighted_test_avg_net_return": 0.0014,
    }
    audit_checks = {
        "lookahead_guard": {"status": "PASS", "evidence": "decision_timestamp < execution_timestamp"},
        "survivorship_bias": {"status": "FAIL", "evidence": "missing source-universe provenance"},
        "parameter_robustness": {"status": "PASS", "evidence": "all tested variants stayed positive"},
        "logic_explainability": {"status": "PASS", "evidence": "selected rules belong to interpretable rule set"},
        "cost_realism": {"status": "PASS", "evidence": "positive cost columns present on evaluated trades"},
    }

    report = build_final_strategy_self_audit(
        selection_summary,
        robustness_summary,
        split_summary,
        strategy_trades,
        audit_checks=audit_checks,
    )

    assert "- Survivorship bias guard: `FAIL`" in report
    assert "missing source-universe provenance" in report


def test_run_final_leadlag_pipeline_returns_complete_artifact_bundle():
    evaluated = _sample_evaluated()
    audit_evidence = {
        "survivorship_bias": {
            "status": "PASS",
            "evidence": "historical runner scanned all symbols present in raw daily partitions",
        }
    }

    bundle = run_final_leadlag_pipeline(
        evaluated,
        train_months=3,
        valid_months=1,
        test_months=1,
        min_train_count=1,
        leadlag_threshold=0.50,
        robustness_min_train_counts=(1, 2),
        robustness_leadlag_thresholds=(0.50, 0.55),
        audit_evidence=audit_evidence,
    )

    assert set(bundle) >= {
        "selections",
        "strategy_trades",
        "split_summary",
        "selection_summary",
        "strategy_report",
        "robustness_scan",
        "robustness_summary",
        "self_audit_report",
        "audit_checks",
    }
    assert not bundle["selections"].empty
    assert not bundle["strategy_trades"].empty
    assert bundle["selection_summary"]["weighted_test_avg_net_return"] > 0
    assert bundle["audit_checks"]["survivorship_bias"]["status"] == "PASS"


def test_run_final_leadlag_pipeline_rejects_missing_audit_evidence():
    evaluated = _sample_evaluated()

    with pytest.raises(ValueError, match="audit"):
        run_final_leadlag_pipeline(
            evaluated,
            train_months=3,
            valid_months=1,
            test_months=1,
            min_train_count=1,
            leadlag_threshold=0.50,
            robustness_min_train_counts=(1, 2),
            robustness_leadlag_thresholds=(0.50, 0.55),
            audit_evidence={},
        )


def test_build_strategy_robustness_scan_tolerates_missing_baseline_variant_result():
    evaluated = _sample_evaluated().iloc[:4].copy()

    scan, summary = build_strategy_robustness_scan(
        evaluated,
        train_months=1,
        valid_months=1,
        test_months=1,
        min_train_counts=(1, 2, 3),
        leadlag_thresholds=(0.45, 0.50, 0.55),
    )

    assert not scan.empty
    assert "baseline_variant" in summary


def test_build_strategy_robustness_scan_handles_none_baseline_weighted_return(monkeypatch):
    calls: list[tuple[int, float]] = []

    def fake_run_walk_forward_strategy(
        evaluated_trades,
        *,
        train_months,
        valid_months,
        test_months,
        min_train_count,
        min_valid_count,
        leadlag_threshold,
    ):
        calls.append((min_train_count, leadlag_threshold))
        weighted = None if (min_train_count == 2 and round(leadlag_threshold, 2) == 0.50) else 0.001
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "split_count": 1,
                "positive_test_splits": 1,
                "mean_test_avg_net_return": weighted,
                "median_test_avg_net_return": weighted,
                "weighted_test_avg_net_return": weighted,
            },
            "",
        )

    monkeypatch.setattr(
        "stocknet_alpha.backtest.final_pipeline.run_walk_forward_strategy",
        fake_run_walk_forward_strategy,
    )

    scan, summary = build_strategy_robustness_scan(
        pd.DataFrame([{"x": 1}]),
        min_train_counts=(1, 2, 3),
        leadlag_thresholds=(0.45, 0.50, 0.55),
    )

    assert len(scan) == 9
    assert summary["baseline_variant"] == "min_train_2_leadlag_0.50"
    assert summary["baseline_weighted_test_avg_net_return"] is None


def test_final_pipeline_cli_manifest_includes_run_and_input_provenance(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    evaluated_path = input_dir / "evaluated_trades.parquet"
    pd.concat([_sample_evaluated()] * 25, ignore_index=True).to_parquet(evaluated_path, index=False)
    (input_dir / "audit_summary.json").write_text(
        json.dumps(
            {
                "audit_checks": {
                    "survivorship_bias": {
                        "status": "PASS",
                        "evidence": "historical runner scanned all symbols present in raw daily partitions",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_final_leadlag_pipeline.py",
            "--evaluated-trades",
            str(evaluated_path),
            "--output-dir",
            str(output_dir),
            "--min-train-count",
            "1",
            "--robustness-min-train-counts",
            "1,2",
            "--robustness-leadlag-thresholds",
            "0.45,0.50",
            "--run-id",
            "final_cli_test_run",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "pipeline_manifest.json").read_text(encoding="utf-8"))

    assert manifest["run_id"] == "final_cli_test_run"
    assert manifest["artifact_schema_version"] == "leadlag_oos_v2"
    assert manifest["audit_status"] == "PASS"
    assert manifest["input_files"][0]["path"] == str(evaluated_path.resolve())
    assert manifest["input_files"][0]["sha256"]
    assert manifest["input_files"][0]["rows"] == len(_sample_evaluated()) * 25
    assert "git_commit" in manifest["git"]
    assert "code_dirty" in manifest["git"]
