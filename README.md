# StockNet

StockNet is an intraday market-network research system for discovering non-preset co-moving equity communities from U.S. `5m / 15m / 30m` bars and studying their persistence with graph-based models.

The current project state is best described as a research-system prototype:

- strongest today: pipeline skeleton, 15m mainline, dashboard shell, edge-persistence TGNN prototype
- in progress: honest multi-resolution integration, 5m-first resampling flow, shared feature wiring
- not yet a final research conclusion engine: consensus/null rigor, lifecycle label reliability, emergence modeling

## Status

Use these labels literally:

- `Implemented`: code exists and is part of a stable mainline
- `Prototype`: code exists but still needs validation or deeper integration
- `In Progress`: active architectural work is underway
- `Not Yet Integrated`: an idea or side path exists, but not in the main workflow

| Capability | Status | Notes |
|---|---|---|
| 5m parquet fetch | Implemented | Primary raw intraday source |
| 15m / 30m parquet fetch | Prototype | Supported, but moving toward 5m-first derivation |
| 5m -> 15m / 30m resampling flow | Implemented | Used by multi-resolution panel builder |
| Single-resolution research pipeline | Implemented | Mainline still centered on one interval at a time |
| Multi-resolution orchestration mode | In Progress | Explicit mode exists; validation depth still catching up |
| Shared feature module | Implemented | Residual returns, time-of-day volume z-score, co-jump, lead-lag |
| Graph snapshots using shared feature logic | Implemented | Snapshot features now route through `features.py` |
| GPU-first graph construction | Prototype | CPU fallback is still common; not all steps are GPU-saturated |
| Bootstrap consensus clustering | Prototype | Workflow exists, but probability accounting still needs tightening |
| Null-model validation | Prototype | Framework exists; scoring still needs stronger research semantics |
| Lifecycle / migration labels | Prototype | Current labels still need lifecycle-id-grade matching |
| TGNN edge persistence | Implemented | Usable proof-of-concept and metrics path |
| Edge emergence prediction | Not Yet Integrated | Separate task from persistence |
| Community detail dashboard | In Progress | Global dashboard works; deeper community drilldown still missing |

## Quick Start

### Dashboard

```bash
npm start
```

Open [http://localhost:3000](http://localhost:3000).

### Single-Resolution Full Pipeline

This is the current most reliable end-to-end path.

```bash
npm run research:full-pipeline -- \
  --mode single-resolution \
  --interval 15m \
  --input "D:/path/to/P123_Screen.csv" \
  --output-root artifacts/full_run_15m
```

### Multi-Resolution Orchestration

This mode now treats `5m` as the raw source and derives `15m` / `30m` from it for time-aligned analysis.

```bash
npm run research:full-pipeline -- \
  --mode multi-resolution \
  --input "D:/path/to/P123_Screen.csv" \
  --output-root artifacts/full_run_multi
```

### Realtime Scanner + Alpha Validation

The headless scanner now runs from `main`, publishes runtime artifacts to `realtime-scanner-headless`, and feeds a minimal `1m -> 5m / 15m -> lead-lag` alpha pipeline.

Default universe files:

- stock universe: `D:\DEV\stocknetwork\P123_Screen_0_20260606.csv`
- ETF / CEF exclusions: `D:\DEV\stocknetwork\P123_ETFCEF.csv`

Background monitor on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File realtime_dashboard/scripts/register_continuous_monitor_task.ps1
Start-ScheduledTask -TaskName StockNetContinuousMonitor
```

Alpha validation flow:

```powershell
python stocknet_alpha/data/resample_bars.py --date 2026-06-09 --from 1m --to 5m
python stocknet_alpha/data/resample_bars.py --date 2026-06-09 --from 1m --to 15m
python stocknet_alpha/theme/build_theme_candidates.py --date 2026-06-09
python stocknet_alpha/leadlag/generate_signals.py --date 2026-06-09
python stocknet_alpha/backtest/backtest_signals.py --date 2026-06-09
```

## Pipeline Modes

### `single-resolution`

Best for a stable per-interval research run.

Flow:

1. `build_15m_parquet.py --interval <interval>`
2. `analyze_rotation.py`
3. `curate_rotation_outputs.py`
4. `build_graph_snapshots.py`
5. `build_temporal_labels.py`
6. `run_edge_baselines.py`
7. `train_xgboost_baseline.py`
8. `train_tgnn_snapshot.py`
9. `build_model_comparison.py`
10. `build_experiment_report.py`

### `multi-resolution`

Best for aligned `5m / 15m / 30m` research artifacts.

Flow:

1. `build_multi_resolution_panels.py`
   - fetch `5m`
   - derive `15m` and `30m` from the 5m panel
2. Per resolution (`5m`, `15m`, `30m`):
   - `analyze_rotation.py`
   - `curate_rotation_outputs.py`
   - `build_consensus_clusters.py`
   - `build_graph_snapshots.py`
   - `build_temporal_labels.py`
   - `run_edge_baselines.py`
   - `train_tgnn_snapshot.py`
3. `build_multi_resolution_report.py`

What this mode does not yet claim:

- it is not yet a fully validated emergence/survival/migration research stack
- it does not yet make consensus/null metrics publication-grade by itself
- it does not yet make lifecycle matching fully reliable

## Key Files

### Data

- `scripts/build_15m_parquet.py`
- `scripts/build_multi_resolution_panels.py`

### Features

- `src/stocknetwork/features.py`

### Network / Snapshots

- `src/stocknetwork/gpu_graph.py`
- `src/stocknetwork/graph_snapshots.py`
- `src/stocknetwork/multi_resolution.py`

### Research / Validation

- `scripts/analyze_rotation.py`
- `scripts/build_consensus_clusters.py`
- `src/stocknetwork/consensus_clustering.py`
- `src/stocknetwork/null_models.py`
- `src/stocknetwork/temporal_labels.py`

### Models

- `src/stocknetwork/baselines.py`
- `src/stocknetwork/xgboost_baseline.py`
- `src/stocknetwork/tgnn_snapshot.py`
- `src/stocknetwork/tgnn_pyg.py`

### App / Dashboard

- `server.js`
- `public/app.js`
- `public/index.html`

## Current Research Caveats

These are intentionally explicit:

- `15m` remains the most mature mainline.
- Consensus clustering still needs pairwise observation-count normalization.
- Null-model scoring still needs better community-quality semantics than a simple persistence proxy.
- Lifecycle/migration labels still need lifecycle-id-grade matching instead of relying on local community numbering.
- TGNN currently proves edge persistence better than it proves edge emergence.

## Practical Direction

If you are extending this project, the highest-value next steps are:

1. keep `5m` as the only raw intraday source
2. derive `15m / 30m` from the same 5m panel
3. make all graph construction consume the shared feature module
4. tighten consensus probability accounting and null scoring
5. build lifecycle-id-based community matching
6. expose per-community drilldown in the dashboard
