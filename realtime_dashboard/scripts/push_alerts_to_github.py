"""Publish scanner runtime artifacts to the headless runtime branch safely.

This script is designed to be called from the long-running scanner on `main`.
It never switches the live checkout. Instead it uses a dedicated git worktree
checked out to `realtime-scanner-headless`, copies runtime artifacts into that
worktree, then commits and pushes there.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNTIME_BRANCH = "realtime-scanner-headless"
DEFAULT_WORKTREE_PATH = REPO_ROOT / ".runtime_publish" / RUNTIME_BRANCH
RUNTIME_ARTIFACT_PATHS = [
    "data/bars_5m",
    "data/bars_15m",
    "data/theme_candidates",
    "data/leadlag_signals",
    "data/alpha_backtests",
    "realtime_dashboard/artifacts/scanner_state",
    "realtime_dashboard/artifacts/theme_state",
    "logs",
]


def run_git(args: list[str], cwd: str | Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command."""
    return subprocess.run(
        ["git"] + args,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=check,
    )


def _remote_branch_exists(branch: str) -> bool:
    result = run_git(["ls-remote", "--heads", "origin", branch], check=False)
    return bool(result.stdout.strip())


def _worktree_matches_branch(worktree_path: Path, branch: str) -> bool:
    result = run_git(["worktree", "list", "--porcelain"], check=False)
    current_path: str | None = None
    current_branch: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            current_path = line.split(" ", 1)[1].strip()
            current_branch = None
        elif line.startswith("branch "):
            current_branch = line.split(" ", 1)[1].strip()
            normalized_branch = current_branch.removeprefix("refs/heads/")
            if Path(current_path or "").resolve() == worktree_path.resolve() and normalized_branch == branch:
                return True
    return False


def prepare_runtime_publish_worktree(
    worktree_path: str | Path | None = None,
    branch: str = RUNTIME_BRANCH,
) -> Path:
    """Create or reuse a dedicated worktree for runtime artifact publishing."""
    target = Path(worktree_path or DEFAULT_WORKTREE_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)

    if _worktree_matches_branch(target, branch):
        logger.info("Using existing runtime publish worktree at %s", target)
        return target

    if _remote_branch_exists(branch):
        run_git(["fetch", "origin", branch], check=False)
        run_git(
            ["worktree", "add", "--force", "-B", branch, str(target), f"origin/{branch}"],
            check=False,
        )
    else:
        run_git(["worktree", "add", "--force", "-b", branch, str(target)], check=False)

    logger.info("Prepared runtime publish worktree at %s for branch %s", target, branch)
    return target


def _copy_path_into_worktree(source_root: Path, destination_root: Path, relative_path: str) -> None:
    src = source_root / relative_path
    dest = destination_root / relative_path
    if not src.exists():
        logger.info("Skipping missing runtime path: %s", src)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)


def publish_runtime_artifacts(
    artifact_paths: Iterable[str] | None = None,
    worktree_path: str | Path | None = None,
    branch: str = RUNTIME_BRANCH,
    commit_message: str | None = None,
) -> bool:
    """Copy runtime artifacts into the headless branch worktree and push them."""
    paths = list(artifact_paths or RUNTIME_ARTIFACT_PATHS)
    source_root = Path(REPO_ROOT)
    target = prepare_runtime_publish_worktree(worktree_path=worktree_path, branch=branch)

    for relative_path in paths:
        _copy_path_into_worktree(source_root, target, relative_path)

    run_git(["add", "--all"], cwd=target)

    diff = run_git(["diff", "--cached", "--quiet"], cwd=target, check=False)
    if diff.returncode == 0:
        logger.info("No runtime artifact changes to publish.")
        return True

    message = commit_message or (
        f"runtime: publish scanner artifacts {datetime.now(timezone.utc).isoformat()}"
    )
    run_git(["commit", "-m", message, "--no-verify"], cwd=target)
    run_git(["push", "origin", branch], cwd=target)
    logger.info("Published runtime artifacts to origin/%s", branch)
    return True


def main() -> None:
    try:
        publish_runtime_artifacts()
    except subprocess.CalledProcessError as exc:
        logger.error("Git command failed: %s\nstdout: %s\nstderr: %s", exc, exc.stdout, exc.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.error("Runtime artifact publish failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
