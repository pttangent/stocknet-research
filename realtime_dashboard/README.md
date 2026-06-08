# Realtime Community Monitoring Radar

Realtime community and sector-rotation dashboard for scanning US equities from
intraday bars and turning them into live co-movement communities.

## What Changed

This dashboard is now aligned with a stricter intraday workflow:

- `ETF / CEF` symbols can be excluded from the monitored universe using
  `D:\DEV\stocknetwork\P123_ETFCEF.csv`
- a dedicated `1m` archive pipeline can continuously save Yahoo minute bars into
  a traceable local store
- the `Market Map` is upgraded into a realtime community bubble radar driven by
  `theme_path_id`, current metrics, and one-step structural drift

## Important 1m Data Constraint

Yahoo / `yfinance` is useful for live minute polling, but it is not a true
"pull any date in the past forever" source for `1m` bars. The practical way to
build a full historical `1m` research base is:

1. use Yahoo / `yfinance`-style minute polling during market hours
2. append the fetched bars into a local parquet archive every minute
3. reuse that archive for replay, model training, and post-session review

In other words:

`1m full history = live polling + local archival`, not `on-demand backfill`.

## Architecture

```text
Yahoo intraday polling
  -> 1m archive writer
  -> rolling feature engine
  -> graph builder
  -> community detector
  -> community scorer
  -> alert engine
  -> state tracker
  -> Streamlit realtime bubble radar
```

## Quick Start

### 1. Install dependencies

```bash
pip install streamlit plotly networkx pandas numpy python-louvain pyarrow
```

### 2. Launch the dashboard

```bash
cd D:\DEV\stocknetwork\StockNet
python -m streamlit run realtime_dashboard/dashboard/app.py
```

Frontend entry file:

`D:\DEV\stocknetwork\StockNet\realtime_dashboard\dashboard\app.py`

### 3. Build a local 1m archive

```bash
cd D:\DEV\stocknetwork\StockNet
python realtime_dashboard/scripts/archive_yfinance_1m.py --universe full_market --scans 390 --sleep-seconds 60
```

Useful options:

```bash
python realtime_dashboard/scripts/archive_yfinance_1m.py --universe core_500 --chunk-size 150 --scans 0
python realtime_dashboard/scripts/archive_yfinance_1m.py --universe watchlist --include-benchmarks
python realtime_dashboard/scripts/archive_yfinance_1m.py --exclude-csv D:\DEV\stocknetwork\P123_ETFCEF.csv
```

`--scans 0` means run continuously until you stop the process.

## Scan Modes

The live dashboard now supports two scan modes:

- `chunked`: scan one chunk of the universe per cycle
- `full_parallel`: scan the full loaded universe every cycle using the configured worker pool

`full_parallel` is the right choice when you have a strong workstation and want
the freshest possible whole-universe community map. It is still limited by
Yahoo response quality and rate limits, not by your CPU or GPU alone.

### 4. Run a simulated demo

```bash
python realtime_dashboard/scripts/live_demo.py --mode quick --quick-scans 30
```

## Universe Handling

Universe selection supports:

- `watchlist`
- `core_500`
- `full_market`

The dashboard and the archive script both use the same exclusion logic:

- default blacklist: `D:\DEV\stocknetwork\P123_ETFCEF.csv`
- default behavior: remove `ETF / CEF` symbols from community scanning
- optional behavior: keep benchmark ETFs only when `keep_benchmark_symbols=True`

This is meant to keep the radar focused on stock communities instead of index
wrappers and fund products.

## Output Files

Daily realtime session logs are written under:

```text
realtime_dashboard/data/YYYY-MM-DD/
```

Key files:

- `1m_bars.parquet`
- `community_snapshots.csv`
- `community_members.csv`
- `live_alerts.csv`
- `community_edges.csv`

Long-run minute archive is written under:

```text
realtime_dashboard/data/archive_1m/
```

Key archive files:

- `date=YYYY-MM-DD/1m_bars.parquet`
- `archive_manifest.csv`

The archive parquet keeps raw fetched bars plus traceability fields such as:

- `archive_fetched_at`
- `archive_provider`
- `archive_interval`

## Dashboard Pages

| Page | Description |
|------|-------------|
| `Live Radar` | live community table, alert stream, and community detail |
| `Timeline` | alert evolution and persistence review |
| `Market Map` | realtime bubble radar using `theme_path_id`, structural drift, and current community state |
| `Review` | post-session summary and markdown review |

## Realtime Bubble Radar

The upgraded market map is designed for intraday community motion, not just a
static scatter plot.

Each bubble represents the latest state of a persistent `theme_path_id`:

- `x` = relative return
- `y` = volume expansion
- `size` = member count
- `color` = current structural status such as `expansion`, `confirmed`, or `decay`
- line segment = previous location to latest location

This makes it easier to scan:

- which communities are accelerating
- which ones are losing coherence
- which ones are moving from noise to confirmation

## Recommended Workflow

For a production-like research loop:

1. run the `1m` archive script during market hours
2. point the dashboard at the same filtered universe
3. monitor live bubbles and alerts during the session
4. use the archived parquet for replay and community review after the close

## Roadmap

- causal community rotation alert layer
- stricter liquidity and ETF exclusion controls
- theme interpreter and news validation
- backend service for continuous market-hour archival

## License

Part of the StockNet project.
