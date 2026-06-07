# Stocknetwork Goal and Execution Roadmap

## Goal

Build `stocknetwork` into a reproducible intraday market research system that:

1. ingests and validates multi-symbol intraday bars,
2. discovers dynamic market communities from price/volume co-movement,
3. backtests community-rotation strategies without lookahead,
4. tracks every research run with comparable artifacts and metadata,
5. adds a GPU path for graph construction and training,
6. extends the current snapshot-based research flow toward TGNN-based prediction of market community evolution.

## Current Baseline

The repo already contains:

- `scripts/build_15m_parquet.py`: 15-minute parquet builder from a symbol list
- `scripts/analyze_rotation.py`: rolling graph/community discovery with Louvain
- `scripts/backtest_rotation_strategy.py`: no-lookahead rolling backtest
- `server.js` + `public/`: dashboard over current artifact folders

The repo does **not** yet have:

- stable run metadata across stages
- test coverage beyond ad hoc research use
- a graph snapshot dataset for model training
- GPU instrumentation or accelerated graph/community code paths
- TGNN baselines, training, evaluation, or experiment comparison structure

## Working Principles

- Keep the current research outputs usable by the dashboard.
- Harden the existing pipeline before adding model complexity.
- Add GPU acceleration where it materially reduces bottlenecks:
  - similarity / adjacency construction
  - community detection
  - TGNN training
- Start with snapshot-based temporal graph models; defer event-based TGN until labels, baselines, and evaluation are stable.

## Execution Plan

### Phase 1: Research Pipeline Hardening

Purpose: make build / analysis / backtest reproducible and comparable.

Deliverables:

- Shared run metadata files in every stage output:
  - `_run.json`
  - `_inputs.json`
  - `_validation.json`
  - `_artifacts.json`
  - `_summary.json`
- Standard CLI metadata flags:
  - `--run-id`
  - `--run-label`
  - `--run-notes`
- Script entrypoints that load even when optional dependencies are absent unless the dependency is actually needed
- Initial tests for metadata and CLI behavior

Acceptance criteria:

- Every stage writes a machine-readable run record.
- Output folders can be compared without reading console logs.
- CLI help works without forcing optional fallback dependencies.

### Phase 2: Snapshot Dataset Builder

Purpose: convert the current research pipeline into a model-ready temporal graph dataset.

New components:

- `scripts/build_graph_snapshots.py`
- `src/stocknetwork/graph_snapshots/`
- `data/processed/graphs/<interval>/<window>/`
- `snapshot_manifest.csv`

Snapshot contract:

- `G_t = (V_t, E_t, X_t, A_t, C_t)`
- node features:
  - return / residual return
  - volume z-score
  - intraday range
  - rolling volatility
  - rolling beta vs `SPY` / `QQQ`
  - liquidity score
  - centrality / clustering / community confidence
- edge features:
  - return correlation
  - residual-return correlation
  - volume-burst similarity
  - lead-lag score
  - co-jump score
  - persistence score

Acceptance criteria:

- Each snapshot is serialized in a stable schema.
- Time ordering is explicit and no feature leaks future information.
- A manifest reports snapshot count, symbol universe size, edge count, and feature completeness.

### Phase 3: Consensus Labels and Research Targets

Purpose: create weak labels for supervised temporal graph learning.

New outputs:

- `community_labels.csv`
- `edge_labels.csv`
- `node_migration_labels.csv`
- `lifecycle_labels.csv`

Prediction targets:

- edge persistence
- community survival
- node migration
- lifecycle stage

Labeling rules:

- labels at `t+h` must be built strictly from future snapshots
- features for `t` must use only data available at `t`
- splits must be chronological, not random

Acceptance criteria:

- Every target has a documented definition.
- Label confidence is tracked.
- A null / persistence baseline can be computed from the same label set.

### Phase 4: Baselines Before TGNN

Purpose: establish whether temporal graph learning adds value.

Baseline families:

- persistence baseline
- edge-score threshold baseline
- logistic regression / tree baseline on node-edge-community features
- static GNN baseline

Metrics:

- edge: AUC, AP, F1, Precision@K
- migration / lifecycle: macro-F1, calibration, confusion matrix
- community survival: AUC, F1

Acceptance criteria:

- Every TGNN task has at least one non-neural baseline.
- Comparison tables are written to artifacts for each run.

### Phase 5: GPU Acceleration Path

Purpose: make graph construction and training scale beyond the current CPU-only research loop.

GPU path scope:

- similarity matrix computation with PyTorch or CuPy
- accelerated graph analytics / Leiden with cuGraph where available
- PyTorch-based snapshot TGNN training

Run tracking additions:

- `gpu_enabled`
- device name
- peak GPU memory
- training minutes
- snapshots processed
- edges processed

Acceptance criteria:

- GPU execution is optional and discoverable from run metadata.
- CPU and GPU runs produce comparable artifact schemas.
- Performance deltas are measurable per run.

### Phase 6: Snapshot-Based TGNN v1

Purpose: predict the temporal evolution of dynamic market communities from snapshot sequences.

Candidate models:

- `GConvGRU`
- `GConvLSTM`
- `A3T-GCN`
- `EvolveGCN`

Initial scope:

- start with one task: edge persistence or community survival
- train on chronological rolling splits
- save checkpoints, metrics, and ablations

Acceptance criteria:

- TGNN beats at least the persistence baseline on one target with documented evidence.
- Training/evaluation artifacts are reproducible from a run configuration.

### Phase 7: Event-Based TGN v2

Purpose: move from discrete snapshots to timed graph events once the snapshot path is stable.

Event types:

- edge appear / disappear
- edge strengthen / weaken
- node join / leave community
- community split / merge

This phase is blocked until:

- snapshot labels are stable
- baselines are reliable
- run comparison is mature

## Dashboard Impact

Near term:

- keep current dashboard compatible with existing artifact directories
- optionally add a `latest run` resolver in server-side code rather than changing the front-end contract first

Later:

- add run selector
- add baseline vs TGNN comparison views
- add community survival / migration forecast panels
- add GPU run summaries

## Immediate Next Tasks

1. Extend Phase 1 verification beyond metadata to include artifact schema checks.
2. Define the snapshot manifest schema and implement `build_graph_snapshots.py`.
3. Add chronological split utilities and label builders.
4. Stand up the first baseline before touching TGNN training code.

## Work Started in This Session

- Declared the project goal explicitly.
- Read the article and extracted the missing GPU/TGNN requirements.
- Added shared run metadata support under `src/stocknetwork/run_metadata.py`.
- Wired run metadata into:
  - `scripts/build_15m_parquet.py`
  - `scripts/analyze_rotation.py`
  - `scripts/backtest_rotation_strategy.py`
- Added tests for metadata generation and CLI behavior.
- Fixed `build_15m_parquet.py` so `--help` does not fail when `yfinance` is absent.
