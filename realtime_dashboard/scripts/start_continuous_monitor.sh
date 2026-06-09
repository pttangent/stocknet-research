#!/bin/bash
# Continuous Market Monitor — Integration Startup Script
#
# Usage:
#   ./start_continuous_monitor.sh [universe] [options]
#
# Examples:
#   ./start_continuous_monitor.sh core_500
#   ./start_continuous_monitor.sh core_500 --github-push
#   ./start_continuous_monitor.sh watchlist --enable-15m

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

UNIVERSE="${1:-core_500}"
shift || true

# Optional: build historical theme state first
ARCHIVE_DIR="../data/archive_15m_from_1m"
if [ -d "$ARCHIVE_DIR" ]; then
    echo "[$(date)] Building historical theme state from $ARCHIVE_DIR ..."
    python build_historical_theme_state.py \
        --input-dir "$ARCHIVE_DIR" \
        --lookback-days 5 \
        --frequency 15m \
        || echo "Warning: historical theme state build failed, continuing with empty state"
fi

echo "[$(date)] Starting continuous monitor | universe=$UNIVERSE ..."

# Run the continuous monitor
python continuous_monitor.py \
    --universe "$UNIVERSE" \
    --scan-interval-seconds 60 \
    --workers 64 \
    --scan-mode full_parallel \
    --one-minute-days 7 \
    --enable-15m \
    --premarket-warmup \
    "$@"

echo "[$(date)] Continuous monitor exited."
