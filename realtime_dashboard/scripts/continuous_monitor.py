"""Continuous market monitor that runs during US trading hours.

This script wraps the realtime scanner to provide:
- Automatic market hours detection (9:30-16:00 ET, Mon-Fri)
- Pre-market warm-up and initialization
- Graceful handling of market close
- Auto-restart on exceptions
- Periodic summary output
- GitHub log pushing (optional)

Usage:
    python continuous_monitor.py --universe core_500 --enable-15m

    # With GitHub logging
    python continuous_monitor.py --universe core_500 --enable-15m --github-push

    # With custom scan interval
    python continuous_monitor.py --universe core_500 --scan-interval-seconds 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo

# Fix import path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import RadarConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# US Eastern timezone
ET = ZoneInfo("America/New_York")

# Market hours
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)

# Pre-market warm-up window
PREMARKET_START = dt_time(9, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous market monitor for theme lifecycle tracking."
    )
    parser.add_argument(
        "--universe",
        choices=["watchlist", "core_500", "full_market"],
        default="core_500",
    )
    parser.add_argument(
        "--scan-interval-seconds",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--scan-mode",
        choices=["chunked", "full_parallel"],
        default="full_parallel",
    )
    parser.add_argument(
        "--one-minute-days",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--enable-15m",
        action="store_true",
    )
    parser.add_argument(
        "--github-push",
        action="store_true",
        help="Push alert logs to GitHub realtime-logs branch.",
    )
    parser.add_argument(
        "--premarket-warmup",
        action="store_true",
        default=True,
        help="Run initialization during pre-market (9:00-9:30 ET).",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=10,
        help="Max consecutive errors before giving up.",
    )
    parser.add_argument(
        "--error-cooldown-seconds",
        type=int,
        default=30,
        help="Cooldown between error retries.",
    )
    return parser.parse_args()


def now_et() -> datetime:
    """Current time in US Eastern."""
    return datetime.now(ET)


def is_trading_day(dt: datetime | None = None) -> bool:
    """Check if today is a US trading day (Mon-Fri)."""
    if dt is None:
        dt = now_et()
    return dt.weekday() < 5  # 0=Mon, 4=Fri


def is_premarket(dt: datetime | None = None) -> bool:
    """Check if currently in pre-market warm-up window."""
    if dt is None:
        dt = now_et()
    if not is_trading_day(dt):
        return False
    t = dt.time()
    return PREMARKET_START <= t < MARKET_OPEN


def is_market_open(dt: datetime | None = None) -> bool:
    """Check if US equity market is currently open."""
    if dt is None:
        dt = now_et()
    if not is_trading_day(dt):
        return False
    t = dt.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def is_after_hours(dt: datetime | None = None) -> bool:
    """Check if market has closed for the day."""
    if dt is None:
        dt = now_et()
    if not is_trading_day(dt):
        return False
    t = dt.time()
    return t > MARKET_CLOSE


def seconds_until_market_open(dt: datetime | None = None) -> float:
    """Calculate seconds until next market open."""
    if dt is None:
        dt = now_et()

    if is_market_open(dt):
        return 0.0

    # If after hours or weekend, compute next trading day 9:30
    next_day = dt.date()
    while True:
        next_day += timedelta(days=1)
        check = datetime.combine(next_day, MARKET_OPEN, tzinfo=ET)
        if is_trading_day(check):
            break

    next_open = datetime.combine(next_day, MARKET_OPEN, tzinfo=ET)
    return max(0.0, (next_open - dt).total_seconds())


def run_scanner_once(args: argparse.Namespace) -> dict:
    """Run a single scan cycle by calling run_realtime_scanner.py."""
    scanner_script = os.path.join(
        os.path.dirname(__file__), "run_realtime_scanner.py"
    )

    cmd = [
        sys.executable,
        scanner_script,
        "--universe", args.universe,
        "--scan-interval-seconds", str(args.scan_interval_seconds),
        "--workers", str(args.workers),
        "--scan-mode", args.scan_mode,
        "--one-minute-days", str(args.one_minute_days),
        "--max-scans", "1",
    ]
    if args.enable_15m:
        cmd.append("--enable-15m")

    logger.debug("Running: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=args.scan_interval_seconds * 10 + 60,
    )

    output = {}
    # Try to parse JSON from last line of stdout
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                output = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if result.returncode != 0:
        logger.error("Scanner failed (exit %d):\nstderr: %s", result.returncode, result.stderr)
        raise RuntimeError(f"Scanner exited with code {result.returncode}")

    return output


def push_logs_to_github() -> bool:
    """Push any new alert logs to the GitHub realtime-logs branch."""
    try:
        push_script = os.path.join(
            os.path.dirname(__file__), "push_alerts_to_github.py"
        )
        if not os.path.exists(push_script):
            logger.warning("GitHub push script not found: %s", push_script)
            return False

        result = subprocess.run(
            [sys.executable, push_script],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info("GitHub push successful")
            return True
        else:
            logger.warning("GitHub push failed: %s", result.stderr)
            return False

    except Exception as exc:
        logger.warning("GitHub push error: %s", exc)
        return False


def print_summary(scan_output: dict, scan_number: int, elapsed_ms: float) -> None:
    """Print a concise scan summary."""
    ts = now_et().strftime("%H:%M:%S ET")

    communities_1m = scan_output.get("one_minute_communities", 0)
    alerts_1m = scan_output.get("one_minute_alerts", 0)
    communities_5m = scan_output.get("five_minute_communities", 0)
    alerts_5m = scan_output.get("five_minute_alerts", 0)
    communities_15m = scan_output.get("fifteen_minute_communities", 0)
    alerts_15m = scan_output.get("fifteen_minute_alerts", 0)

    theme_state = scan_output.get("theme_state", {})
    active_paths = theme_state.get("active_paths", 0)
    total_paths = theme_state.get("total_paths", 0)

    summary = (
        f"[{ts}] Scan #{scan_number} | "
        f"1m:{communities_1m}c/{alerts_1m}a "
        f"5m:{communities_5m}c/{alerts_5m}a "
        f"15m:{communities_15m}c/{alerts_15m}a "
        f"| Themes: {active_paths}/{total_paths} active "
        f"| {elapsed_ms:.0f}ms"
    )
    logger.info(summary)


def run_premarket_warmup(args: argparse.Namespace) -> None:
    """Run scanner warm-up during pre-market hours."""
    logger.info("Pre-market warm-up starting...")

    scanner_script = os.path.join(
        os.path.dirname(__file__), "run_realtime_scanner.py"
    )

    cmd = [
        sys.executable,
        scanner_script,
        "--universe", args.universe,
        "--workers", str(args.workers),
        "--scan-mode", args.scan_mode,
        "--one-minute-days", str(args.one_minute_days),
        "--max-scans", "1",
        "--warmup-only",
    ]
    if args.enable_15m:
        cmd.append("--enable-15m")

    logger.info("Warm-up command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode == 0:
        logger.info("Pre-market warm-up complete.")
    else:
        logger.warning("Warm-up had issues (exit %d): %s", result.returncode, result.stderr)


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Continuous Market Monitor Starting")
    logger.info("  Universe: %s", args.universe)
    logger.info("  Scan interval: %ds", args.scan_interval_seconds)
    logger.info("  15m enabled: %s", args.enable_15m)
    logger.info("  GitHub push: %s", args.github_push)
    logger.info("  Pre-market warm-up: %s", args.premarket_warmup)
    logger.info("=" * 60)

    # Wait for market open (or pre-market if enabled)
    while True:
        current = now_et()

        if is_market_open(current):
            logger.info("Market is OPEN. Starting scans.")
            break

        if args.premarket_warmup and is_premarket(current):
            logger.info("Pre-market detected (%s). Running warm-up...", current.strftime("%H:%M ET"))
            try:
                run_premarket_warmup(args)
            except Exception as exc:
                logger.error("Pre-market warm-up failed: %s", exc)
            # Continue to wait for market open
            wait_seconds = max(30, args.scan_interval_seconds)
            logger.info("Waiting %ds for market open...", wait_seconds)
            time.sleep(wait_seconds)
            continue

        if is_after_hours(current):
            wait_sec = seconds_until_market_open(current)
            logger.info("Market closed. Next open in %.0f minutes (%.1f hours).",
                        wait_sec / 60, wait_sec / 3600)
            # Sleep in chunks to allow interruption
            sleep_chunk = min(300, wait_sec)
            time.sleep(sleep_chunk)
            continue

        if not is_trading_day(current):
            wait_sec = seconds_until_market_open(current)
            logger.info("Non-trading day. Next market open in %.1f hours.", wait_sec / 3600)
            sleep_chunk = min(600, wait_sec)
            time.sleep(sleep_chunk)
            continue

        # Before pre-market
        wait_sec = seconds_until_market_open(current)
        logger.info("Waiting for market open in %.0f minutes...", wait_sec / 60)
        time.sleep(min(60, wait_sec))

    # Main scanning loop
    scan_number = 0
    consecutive_errors = 0
    last_github_push = datetime.min.replace(tzinfo=timezone.utc)

    while True:
        current = now_et()

        if not is_market_open(current):
            if is_after_hours(current):
                logger.info("Market has closed. Shutting down until next session.")
                break
            # Shouldn't normally get here, but handle gracefully
            logger.warning("Market not open unexpectedly. Waiting...")
            time.sleep(60)
            continue

        scan_number += 1
        start_time = time.time()

        try:
            scan_output = run_scanner_once(args)
            elapsed_ms = (time.time() - start_time) * 1000
            consecutive_errors = 0

            print_summary(scan_output, scan_number, elapsed_ms)

            # GitHub push every 5 minutes or if there are alerts
            if args.github_push:
                now_utc = datetime.now(timezone.utc)
                has_alerts = (
                    scan_output.get("one_minute_alerts", 0) > 0
                    or scan_output.get("five_minute_alerts", 0) > 0
                    or scan_output.get("fifteen_minute_alerts", 0) > 0
                )
                time_since_push = (now_utc - last_github_push).total_seconds()
                if has_alerts or time_since_push >= 300:
                    push_logs_to_github()
                    last_github_push = now_utc

        except Exception as exc:
            consecutive_errors += 1
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(
                "Scan #%d failed after %.0fms (error %d/%d): %s",
                scan_number,
                elapsed_ms,
                consecutive_errors,
                args.max_consecutive_errors,
                exc,
            )
            if consecutive_errors >= args.max_consecutive_errors:
                logger.critical(
                    "Too many consecutive errors (%d). Shutting down.",
                    consecutive_errors,
                )
                break
            logger.info("Cooling down for %ds before retry...", args.error_cooldown_seconds)
            time.sleep(args.error_cooldown_seconds)
            continue

        # Calculate sleep time to maintain scan interval
        elapsed = time.time() - start_time
        sleep_time = max(1, args.scan_interval_seconds - elapsed)

        # Check if we'd sleep past market close
        seconds_to_close = (
            datetime.combine(now_et().date(), MARKET_CLOSE, tzinfo=ET) - now_et()
        ).total_seconds()
        if seconds_to_close <= 0:
            logger.info("Market close reached. Final scan complete.")
            break
        if sleep_time > seconds_to_close:
            # Do one more scan before close if there's time
            sleep_time = max(0, seconds_to_close - args.scan_interval_seconds)

        if sleep_time > 0:
            time.sleep(sleep_time)

    # Final push after market close
    if args.github_push:
        logger.info("Final GitHub push after market close...")
        push_logs_to_github()

    logger.info("=" * 60)
    logger.info("Continuous monitor complete. Total scans: %d", scan_number)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
