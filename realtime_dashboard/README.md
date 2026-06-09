# Realtime Scanner

Headless realtime scanner for intraday stock-community detection.

The frontend has been removed. This branch is intentionally scanner-only and is
meant to run as a continuous process, not as a dashboard app.

## Branch Strategy

This branch is the long-lived code branch for realtime scanning:

- branch: `realtime-scanner-headless`

Runtime scans do **not** create a new Git branch each time. That would be noisy,
slow, and hard to manage because scans are frequent and data-heavy. Instead:

- code changes live on the branch
- each scanner session writes a `run_id`
- operational outputs are stored under local `data/` and `artifacts/`
- Git is used for code lineage, not minute-by-minute runtime data

## What This Scanner Does

- pulls recent Yahoo-supported `1m` bars
- archives raw minute bars locally
- scans `1m` communities as early radar
- optionally aggregates the cached `1m` bars into `15m`
- scans `15m` communities as the higher-confidence confirmation layer
- writes machine-readable current state snapshots and alerts

## Why This Design

`1m` is useful as an early radar, but `15m` is currently the more reliable
structural scale. The scanner therefore supports:

- `1m` for early detection
- `15m` from aggregated `1m` for confirmation

This reduces dependency on separate live `15m` pulls and keeps both scales
aligned to the same raw minute tape.

## Important 1m Data Constraint

Yahoo / `yfinance` style minute data is not a true all-history source. Full
research continuity comes from:

`live polling + local archive`

not from unlimited historical backfill.

## Quick Start

### 1. Initialize warmup history

```bash
cd D:\DEV\stocknetwork\StockNet
python realtime_dashboard/scripts/initialize_live_radar.py --universe core_500
```

This prepares:

- `warmup_1m`
- `warmup_15m`
- initialization verification reports

### 2. Run the continuous scanner

```bash
cd D:\DEV\stocknetwork\StockNet
python realtime_dashboard/scripts/run_realtime_scanner.py --universe core_500 --scan-mode full_parallel --workers 64 --enable-15m
```

Useful options:

```bash
python realtime_dashboard/scripts/run_realtime_scanner.py --universe watchlist --max-scans 1 --enable-15m
python realtime_dashboard/scripts/run_realtime_scanner.py --universe core_500 --scan-interval-seconds 60 --workers 64 --enable-15m
python realtime_dashboard/scripts/run_realtime_scanner.py --universe full_market --scan-mode chunked --scan-interval-seconds 60 --enable-15m
```

## Universe Handling

The scanner supports:

- `watchlist`
- `core_500`
- `full_market`

and uses ETF / CEF exclusion by default from:

`D:\DEV\stocknetwork\P123_ETFCEF.csv`

## Outputs

### Daily logs

Written under:

`realtime_dashboard/data/YYYY-MM-DD/`

Key files:

- `1m_bars.parquet`
- `community_snapshots.csv`
- `community_members.csv`
- `live_alerts.csv`
- `community_edges.csv`

### Long-run archive

Written under:

`realtime_dashboard/data/archive_1m/`

### Scanner state

Written under:

`realtime_dashboard/artifacts/scanner_state/`

Key files:

- `current_state.json`
- `latest_alerts.json`

### Initialization reports

Written under:

`realtime_dashboard/artifacts/initialization/`

Key files:

- `initialization_report.json`
- `warmup_verification_report.json`

## Operational Guidance

- Use `1m` as the early radar.
- Use `15m` as the main confidence layer.
- Prefer `full_parallel` for strong hardware and moderate universes.
- Prefer `chunked` for very large universes where Yahoo throughput becomes the bottleneck.

## Removed

The following are no longer part of this branch:

- Streamlit dashboard
- dashboard components
- dashboard launcher

## License

Part of the StockNet project.
