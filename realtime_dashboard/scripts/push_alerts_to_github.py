"""Push realtime alert logs to a dedicated GitHub branch.

This script reads the latest scanner state and theme events,
generates a markdown log file, and commits/pushes it to the
`realtime-logs` branch on GitHub.

Usage:
    python push_alerts_to_github.py

Environment:
    Must be run from within a git repository with write access.
    The script will create the `realtime-logs` branch if it doesn't exist.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Repository root (assumes script is under realtime_dashboard/scripts/)
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
REALTIME_LOGS_BRANCH = "realtime-logs"
LOGS_DIR = os.path.join(REPO_ROOT, "logs")


def run_git(args: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command."""
    cmd = ["git"] + args
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def ensure_branch_exists() -> bool:
    """Ensure the realtime-logs branch exists locally and remotely."""
    try:
        # Check if branch exists locally
        result = run_git(["branch", "--list", REALTIME_LOGS_BRANCH], check=False)
        if REALTIME_LOGS_BRANCH in result.stdout:
            return True

        # Check if branch exists remotely
        result = run_git(["ls-remote", "--heads", "origin", REALTIME_LOGS_BRANCH], check=False)
        if result.stdout.strip():
            # Fetch and checkout
            run_git(["fetch", "origin", REALTIME_LOGS_BRANCH], check=False)
            run_git(["checkout", "-b", REALTIME_LOGS_BRANCH, f"origin/{REALTIME_LOGS_BRANCH}"], check=False)
            return True

        # Create orphan branch
        run_git(["checkout", "--orphan", REALTIME_LOGS_BRANCH], check=False)
        # Remove all files from staging
        run_git(["rm", "-rf", "."], check=False)
        # Create initial README
        readme_path = os.path.join(REPO_ROOT, "README_LOGS.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Realtime Alert Logs\n\n")
            f.write("This branch contains automated logs from the realtime scanner.\n")
            f.write("Do not manually edit files in this branch.\n")
        run_git(["add", "README_LOGS.md"])
        run_git([
            "commit", "-m",
            "Initialize realtime-logs branch",
            "--no-verify",
        ], check=False)
        run_git(["push", "-u", "origin", REALTIME_LOGS_BRANCH], check=False)
        return True

    except Exception as exc:
        logger.error("Failed to ensure branch exists: %s", exc)
        return False


def switch_to_logs_branch() -> bool:
    """Switch to the realtime-logs branch."""
    try:
        result = run_git(["rev-parse", "--abbrev-ref", "HEAD"], check=False)
        current_branch = result.stdout.strip()

        if current_branch == REALTIME_LOGS_BRANCH:
            # Pull latest
            run_git(["pull", "origin", REALTIME_LOGS_BRANCH], check=False)
            return True

        run_git(["stash", "push", "-m", "auto-stash-before-log-push"], check=False)
        run_git(["checkout", REALTIME_LOGS_BRANCH], check=False)
        run_git(["pull", "origin", REALTIME_LOGS_BRANCH], check=False)
        return True

    except Exception as exc:
        logger.error("Failed to switch to logs branch: %s", exc)
        return False


def read_current_state() -> dict:
    """Read the latest scanner state."""
    state_path = os.path.join(
        REPO_ROOT, "realtime_dashboard", "artifacts", "scanner_state", "current_state.json"
    )
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read current_state.json: %s", exc)
        return {}


def read_theme_events() -> pd.DataFrame:
    """Read theme events parquet."""
    events_path = os.path.join(
        REPO_ROOT, "realtime_dashboard", "artifacts", "theme_state", "theme_events.parquet"
    )
    if not os.path.exists(events_path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(events_path)
    except Exception as exc:
        logger.warning("Failed to read theme_events.parquet: %s", exc)
        return pd.DataFrame()


def read_active_themes() -> list[dict]:
    """Read active theme paths."""
    active_path = os.path.join(
        REPO_ROOT, "realtime_dashboard", "artifacts", "theme_state", "active_theme_paths.json"
    )
    if not os.path.exists(active_path):
        return []
    try:
        with open(active_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("active_paths", [])
    except Exception as exc:
        logger.warning("Failed to read active_theme_paths.json: %s", exc)
        return []


def generate_markdown_log(
    state: dict,
    events_df: pd.DataFrame,
    active_themes: list[dict],
) -> str:
    """Generate a markdown log entry from scanner state."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S UTC")

    lines = [
        f"# Realtime Alert Log — {date_str} {time_str}",
        "",
        "## Scan Summary",
        "",
    ]

    scan_number = state.get("scan_number", 0)
    lines.append(f"- **Scan Number**: {scan_number}")
    lines.append(f"- **Universe**: {state.get('universe', 'unknown')}")
    lines.append(f"- **Symbols**: {state.get('symbols', 0)}")

    # 1m stats
    one_min = state.get("one_minute", {})
    lines.append(f"- **1m**: {one_min.get('community_count', 0)} communities, {one_min.get('alert_count', 0)} alerts")

    # 5m stats
    five_min = state.get("five_minute", {})
    lines.append(f"- **5m**: {five_min.get('community_count', 0)} communities, {five_min.get('alert_count', 0)} alerts")

    # 15m stats
    fifteen_min = state.get("fifteen_minute", {})
    lines.append(f"- **15m**: {fifteen_min.get('community_count', 0)} communities, {fifteen_min.get('alert_count', 0)} alerts")

    # Theme state
    theme_state = state.get("theme_state", {})
    if theme_state:
        lines.append("")
        lines.append("## Theme State")
        lines.append("")
        lines.append(f"- **Active Paths**: {theme_state.get('active_paths', 0)}")
        lines.append(f"- **Total Paths**: {theme_state.get('total_paths', 0)}")
        event_summary = theme_state.get("event_summary", {})
        if event_summary:
            lines.append(f"- **Total Events**: {event_summary.get('total_events', 0)}")
            lines.append(f"  - Births: {event_summary.get('birth_count', 0)}")
            lines.append(f"  - Continuations: {event_summary.get('continuation_count', 0)}")
            lines.append(f"  - Revivals: {event_summary.get('revival_count', 0)}")
            lines.append(f"  - Weak Continuations: {event_summary.get('weak_continuation_count', 0)}")

    # Recent events
    if not events_df.empty and "event_type" in events_df.columns:
        lines.append("")
        lines.append("## Recent Theme Events")
        lines.append("")

        # Get latest events (last 20)
        latest = events_df.tail(20)
        lines.append("| Time | Freq | Type | Theme Path | Match | Members | Radar |")
        lines.append("|------|------|------|------------|-------|---------|-------|")

        for _, row in latest.iterrows():
            ts = str(row.get("timestamp", "")).split("+")[0]  # Remove timezone for brevity
            freq = row.get("frequency", "")
            etype = row.get("event_type", "")
            tpid = row.get("theme_path_id", "")
            match = row.get("match_score", 0.0)
            members = row.get("member_count", 0)
            radar = row.get("radar_score", 0.0)
            lines.append(
                f"| {ts} | {freq} | {etype} | `{tpid}` | {match:.2f} | {members} | {radar:.3f} |"
            )

    # Active themes detail
    if active_themes:
        lines.append("")
        lines.append("## Active Themes")
        lines.append("")

        for theme in active_themes[:10]:
            tpid = theme.get("theme_path_id", "")
            state_label = theme.get("state", "")
            freq = theme.get("last_frequency", "")
            comm_id = theme.get("last_community_id", "")
            members = theme.get("member_count", 0)
            core = ", ".join(theme.get("core_members", [])[:8])
            radar = theme.get("last_radar_score", 0.0)
            peak = theme.get("peak_radar_score", 0.0)
            age = theme.get("age_events", 0)

            lines.append(f"### `{tpid}`")
            lines.append(f"- **State**: {state_label} | **Freq**: {freq} | **Comm**: {comm_id}")
            lines.append(f"- **Members**: {members} | **Age**: {age} events")
            lines.append(f"- **Radar**: {radar:.3f} (peak: {peak:.3f})")
            lines.append(f"- **Core**: {core}")
            lines.append("")

    # Top communities from current scan
    top_communities = one_min.get("top_communities", [])
    if top_communities:
        lines.append("## Top 1m Communities")
        lines.append("")
        lines.append("| Rank | Theme Path | Event | Members | Radar | Top Members |")
        lines.append("|------|------------|-------|---------|-------|-------------|")
        for i, comm in enumerate(top_communities[:10], 1):
            tpid = comm.get("theme_path_id", comm.get("community_id", ""))
            etype = comm.get("event_type", "")
            members = comm.get("member_count", 0)
            radar = comm.get("radar_score", 0.0)
            top = comm.get("top_members", "")
            lines.append(f"| {i} | `{tpid}` | {etype} | {members} | {radar:.3f} | {top} |")

    lines.append("")
    lines.append("---")
    lines.append(f"*Auto-generated at {time_str} by StockNet Realtime Scanner*")

    return "\n".join(lines)


def write_and_push_log(content: str) -> bool:
    """Write log file and push to GitHub."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")

    # Create directory structure: logs/YYYY-MM-DD/
    day_dir = os.path.join(LOGS_DIR, date_str)
    os.makedirs(day_dir, exist_ok=True)

    log_file = os.path.join(day_dir, f"realtime_alerts_{time_str}.md")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Log written to %s", log_file)

    # Git operations
    try:
        run_git(["add", log_file])

        # Also add any other new log files from today
        run_git(["add", day_dir])

        # Check if there are changes to commit
        diff = run_git(["diff", "--cached", "--quiet"], check=False)
        if diff.returncode == 0:
            logger.info("No changes to commit.")
            return True

        commit_msg = (
            f"[{date_str} {time_str}] Realtime alerts | "
            f"scan={datetime.now(timezone.utc).isoformat()}"
        )
        run_git(["commit", "-m", commit_msg, "--no-verify"])
        run_git(["push", "origin", REALTIME_LOGS_BRANCH])

        logger.info("Pushed to origin/%s", REALTIME_LOGS_BRANCH)
        return True

    except subprocess.CalledProcessError as exc:
        logger.error("Git operation failed: %s\nstdout: %s\nstderr: %s",
                     exc, exc.stdout, exc.stderr)
        return False
    except Exception as exc:
        logger.error("Unexpected error during git push: %s", exc)
        return False


def main() -> None:
    logger.info("=" * 60)
    logger.info("GitHub Alert Log Pusher")
    logger.info("=" * 60)

    # Step 1: Ensure branch exists
    if not ensure_branch_exists():
        logger.error("Cannot ensure branch exists. Aborting.")
        sys.exit(1)

    # Step 2: Switch to logs branch
    if not switch_to_logs_branch():
        logger.error("Cannot switch to logs branch. Aborting.")
        sys.exit(1)

    # Step 3: Read scanner state
    state = read_current_state()
    events_df = read_theme_events()
    active_themes = read_active_themes()

    if not state:
        logger.warning("No scanner state found. Creating minimal log.")

    # Step 4: Generate markdown
    content = generate_markdown_log(state, events_df, active_themes)

    # Step 5: Write and push
    success = write_and_push_log(content)

    # Step 6: Switch back to original branch if needed
    try:
        stash_list = run_git(["stash", "list"], check=False)
        if "auto-stash-before-log-push" in stash_list.stdout:
            run_git(["checkout", "-"], check=False)
            run_git(["stash", "pop"], check=False)
    except Exception:
        pass

    if success:
        logger.info("Done.")
        sys.exit(0)
    else:
        logger.error("Push failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
