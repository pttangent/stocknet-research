from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_build_script_help_runs_without_optional_yfinance_dependency():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/build_15m_parquet.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--run-id" in result.stdout
