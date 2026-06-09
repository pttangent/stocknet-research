from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_full_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_full_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_multi_resolution_plan_includes_5m_first_and_multires_steps():
    module = _load_module()
    plan = module.build_execution_plan(mode="multi-resolution", interval="15m")
    step_names = [step["name"] for step in plan]

    assert step_names[0] == "build_multi_resolution_panels"
    assert "analyze_rotation_5m" in step_names
    assert "analyze_rotation_15m" in step_names
    assert "analyze_rotation_30m" in step_names
    assert "build_multi_resolution_report" in step_names


def test_single_resolution_plan_keeps_existing_mainline():
    module = _load_module()
    plan = module.build_execution_plan(mode="single-resolution", interval="30m")
    step_names = [step["name"] for step in plan]

    assert step_names[0] == "build_parquet"
    assert "analyze_rotation" in step_names
    assert "build_snapshots" in step_names
    assert "build_multi_resolution_panels" not in step_names
