# Causal Theme And Lead-Lag MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first end-to-end causal `5m` theme persistence and `1m/5m` lead-lag backtest slice so StockNet can generate realistic, no-lookahead research artifacts and alpha evaluation tables from the new U.S. market database.

**Architecture:** Reuse the existing `src/stocknetwork` snapshot and lifecycle pipeline for offline research artifacts, reuse the existing `stocknet_alpha` theme/signals/backtest flow for tradeable outputs, and add a thin causal bridge between them rather than inventing a third identity system. `lifecycle_id` remains the per-snapshot-chain research identity, `theme_path_id` remains the cross-window/cross-day tradable identity, and every signal/output must be derived with explicit `decision_time < execution_time` semantics.

**Tech Stack:** Python 3.11, pandas, DuckDB-backed parquet lake, pytest, existing `stocknetwork` and `stocknet_alpha` modules.

---

### Task 1: Freeze The Causal Contract

**Files:**
- Modify: `D:\DEV\stocknetwork\StockNet\docs\research_protocol.md`
- Modify: `D:\DEV\stocknetwork\StockNet\docs\community_definition.md`
- Create: `D:\DEV\stocknetwork\StockNet\docs\metric_dictionary.md` (only if new fields need formal definitions)
- Test: no code test; verify by re-reading edited sections

- [ ] **Step 1: Document the four-time model**

Add explicit definitions for:
- `event_time`
- `available_time`
- `decision_time`
- `execution_time`

State that:
- graph construction uses only rows with `available_time <= decision_time`
- rolling statistics must use `.shift(1)` before any rolling mean/std/quantile
- labels never appear in feature tables or selection logic

- [ ] **Step 2: Document the new first-class artifact tables**

Add or update sections for:
- `theme_snapshots_5m`
- `community_membership_5m`
- `leadlag_edges_1m`
- `leadlag_edges_5m`
- `leadlag_edge_metrics`
- `walk_forward_splits`

- [ ] **Step 3: Re-read the edited docs**

Verify the docs explicitly forbid:
- full-sample normalization
- future-window community matching
- same-bar close execution for close-derived signals


### Task 2: Extract Reusable Theme Persistence Logic

**Files:**
- Create: `D:\DEV\stocknetwork\StockNet\src\stocknetwork\theme_persistence.py`
- Modify: `D:\DEV\stocknetwork\StockNet\src\stocknetwork\temporal_labels.py`
- Modify: `D:\DEV\stocknetwork\StockNet\src\stocknetwork\confirmation_backtest.py`
- Test: `D:\DEV\stocknetwork\StockNet\tests\test_theme_persistence.py`
- Test: `D:\DEV\stocknetwork\StockNet\tests\test_temporal_labels.py`
- Test: `D:\DEV\stocknetwork\StockNet\tests\test_confirmation_backtest_causal.py`

- [ ] **Step 1: Write the failing theme persistence tests**

Cover:
- stable `theme_path_id` continuation when member Jaccard stays above threshold
- new `theme_path_id` creation when overlap falls below threshold
- no future information required to assign today’s path
- day-to-day carryover works even when local `community_id` changes

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```powershell
.venv311\Scripts\python.exe -m pytest tests/test_theme_persistence.py tests/test_temporal_labels.py tests/test_confirmation_backtest_causal.py -q
```

Expected: at least the new persistence tests fail because the module and outputs do not exist yet.

- [ ] **Step 3: Implement `theme_persistence.py`**

Create a focused module that:
- accepts chronological `5m` community rows
- matches rows to prior active paths using member-overlap scoring
- emits `theme_path_id`, `matched_previous_theme_path_id`, `match_score`, `event_type`, `birth_time`, `age_bars`, and `status`

Prefer a small public surface such as:
```python
def assign_theme_paths(
    communities: pd.DataFrame,
    *,
    member_col: str = "members",
    timestamp_col: str = "timestamp",
    min_overlap: float = 0.40,
) -> pd.DataFrame:
    ...
```

- [ ] **Step 4: Refactor existing callers to reuse the new persistence module**

Update:
- `temporal_labels.py` to emit lifecycle/theme continuity tables without duplicate matching code
- `confirmation_backtest.py` to replace the current ad-hoc `_assign_theme_paths()` logic with the shared implementation

- [ ] **Step 5: Add concrete artifact outputs**

Expose or write DataFrames for:
- `theme_snapshots_5m`
- `community_membership_5m`

The rows must carry causal fields such as:
- `event_time`
- `available_time`
- `decision_time`
- `theme_path_id`
- `lifecycle_id`
- `stage`
- `member_count`
- `coherence`
- `breadth`

- [ ] **Step 6: Re-run the focused tests**

Run:
```powershell
.venv311\Scripts\python.exe -m pytest tests/test_theme_persistence.py tests/test_temporal_labels.py tests/test_confirmation_backtest_causal.py -q
```

Expected: PASS.


### Task 3: Build Causal Trade-Flow-Aware Lead-Lag Features

**Files:**
- Create: `D:\DEV\stocknetwork\StockNet\stocknet_alpha\leadlag\causal_features.py`
- Modify: `D:\DEV\stocknetwork\StockNet\stocknet_alpha\leadlag\generate_signals.py`
- Modify: `D:\DEV\stocknetwork\StockNet\stocknet_alpha\data\resample_bars.py` (only if helper reuse is needed)
- Test: `D:\DEV\stocknetwork\StockNet\tests\test_stocknet_alpha_pipeline.py`
- Test: `D:\DEV\stocknetwork\StockNet\tests\test_leadlag_causal_features.py`

- [ ] **Step 1: Write failing causal feature tests**

Cover:
- rolling z-scores use only past values
- trade-flow aggregation from `1m` to `5m` aligns to the just-closed bar
- signal-time features never include future bars
- `decision_time=bar_close`, `execution_time=next_bar_open`

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```powershell
.venv311\Scripts\python.exe -m pytest tests/test_stocknet_alpha_pipeline.py tests/test_leadlag_causal_features.py -q
```

Expected: FAIL on the newly added causal feature cases.

- [ ] **Step 3: Implement causal feature builders**

Add helpers for:
- `ret_1m_past`, `ret_3m_past`, `ret_5m_past`
- `volume_z_*` with `.shift(1)`
- trade-flow aggregates such as `imbalance_proxy`, `large_trade_ratio`, `off_exchange_ratio`
- optional `5m` aggregation helpers for later graph-node reuse

- [ ] **Step 4: Wire `generate_signals.py` to consume causal features**

Replace the current return-only leader/follower scan with a feature frame that stores:
- feature timestamp
- decision timestamp
- execution timestamp
- lead source feature type
- lag target symbol
- lag minutes

Do not compute forward returns inside the feature-building function. Forward returns belong to evaluation.

- [ ] **Step 5: Re-run the focused tests**

Run:
```powershell
.venv311\Scripts\python.exe -m pytest tests/test_stocknet_alpha_pipeline.py tests/test_leadlag_causal_features.py -q
```

Expected: PASS.


### Task 4: Separate Signals From Evaluation

**Files:**
- Create: `D:\DEV\stocknetwork\StockNet\stocknet_alpha\leadlag\evaluate_edges.py`
- Modify: `D:\DEV\stocknetwork\StockNet\stocknet_alpha\backtest\backtest_signals.py`
- Modify: `D:\DEV\stocknetwork\StockNet\stocknet_alpha\config.py`
- Test: `D:\DEV\stocknetwork\StockNet\tests\test_leadlag_edge_evaluation.py`
- Test: `D:\DEV\stocknetwork\StockNet\tests\test_stocknet_alpha_pipeline.py`

- [ ] **Step 1: Write failing evaluation tests**

Cover:
- signal generation produces no forward-return columns
- evaluation attaches realized returns only after entry/exit rules are specified
- fills use next bar open or explicitly configured execution rule
- edge metrics include `mean_ret`, `hit_rate`, `n`, `sharpe`, and cost-aware return

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```powershell
.venv311\Scripts\python.exe -m pytest tests/test_leadlag_edge_evaluation.py tests/test_stocknet_alpha_pipeline.py -q
```

Expected: FAIL because the new evaluator and output contract do not exist yet.

- [ ] **Step 3: Implement edge evaluation tables**

Create outputs for:
- `leadlag_edges_1m`
- `leadlag_edges_5m`
- `leadlag_edge_metrics`

Suggested columns:
- `trade_date`
- `theme_path_id`
- `lead_symbol`
- `lag_symbol`
- `signal_type`
- `lag_minutes`
- `decision_time`
- `execution_time`
- `entry_price`
- `exit_time`
- `exit_price`
- `gross_return`
- `net_return`
- `hit`

- [ ] **Step 4: Update `backtest_signals.py` to summarize evaluated trades, not raw signals**

Keep the current markdown summary path, but base it on realized trade rows and cost-aware returns.

- [ ] **Step 5: Re-run the focused tests**

Run:
```powershell
.venv311\Scripts\python.exe -m pytest tests/test_leadlag_edge_evaluation.py tests/test_stocknet_alpha_pipeline.py -q
```

Expected: PASS.


### Task 5: Add Walk-Forward Split Plumbing

**Files:**
- Create: `D:\DEV\stocknetwork\StockNet\stocknet_alpha\backtest\walk_forward.py`
- Create: `D:\DEV\stocknetwork\StockNet\scripts\run_leadlag_walk_forward.py`
- Modify: `D:\DEV\stocknetwork\StockNet\stocknet_alpha\config.py`
- Test: `D:\DEV\stocknetwork\StockNet\tests\test_walk_forward.py`

- [ ] **Step 1: Write failing split tests**

Cover:
- train/valid/test windows are chronological and non-overlapping
- fit state is derived from train only
- validation chooses parameters before test starts
- test rows are never reused for fitting or threshold selection

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```powershell
.venv311\Scripts\python.exe -m pytest tests/test_walk_forward.py -q
```

Expected: FAIL because the split builder does not exist yet.

- [ ] **Step 3: Implement walk-forward split generation**

Create a small API like:
```python
def build_walk_forward_splits(
    start_month: str,
    end_month: str,
    *,
    train_months: int,
    valid_months: int,
    test_months: int,
) -> pd.DataFrame:
    ...
```

- [ ] **Step 4: Persist split metadata**

Write `walk_forward_splits` with:
- `split_id`
- `train_start`
- `train_end`
- `valid_start`
- `valid_end`
- `test_start`
- `test_end`
- `feature_version`
- `model_version`

- [ ] **Step 5: Re-run the tests**

Run:
```powershell
.venv311\Scripts\python.exe -m pytest tests/test_walk_forward.py -q
```

Expected: PASS.


### Task 6: Run The First Real Historical Slice

**Files:**
- Modify: `D:\DEV\stocknetwork\StockNet\docs\pipeline-commands.md`
- Use: `D:\DEV\stocknetwork\StockNet\data\stocknet_us.duckdb`
- Use: `D:\DEV\stocknetwork\StockNet\data\bars_5m`
- Use: `D:\DEV\stocknetwork\StockNet\data\trade_flow_1m`

- [ ] **Step 1: Add a reproducible command block for the historical run**

Document commands for:
- building `theme_snapshots_5m`
- generating causal lead-lag signals
- evaluating `leadlag_edges_*`
- running the first walk-forward slice

- [ ] **Step 2: Run the historical slice on the available window**

Start with:
- train: `2025-09` to `2026-01`
- valid: `2026-02`
- test: `2026-03`

Use the ingested U.S. database rather than legacy sample CSVs.

- [ ] **Step 3: Save realized artifacts**

At minimum produce:
- one split record
- one lead-lag metrics file
- one markdown report

- [ ] **Step 4: Record whether the slice is positive after costs**

Do not claim success unless the output explicitly shows:
- signal count
- average net return
- hit rate
- turnover or trade count
- whether PnL remains positive after configured transaction costs


### Task 7: Full Verification Before Any Performance Claim

**Files:**
- Verify: `D:\DEV\stocknetwork\StockNet\tests\test_theme_persistence.py`
- Verify: `D:\DEV\stocknetwork\StockNet\tests\test_leadlag_causal_features.py`
- Verify: `D:\DEV\stocknetwork\StockNet\tests\test_leadlag_edge_evaluation.py`
- Verify: `D:\DEV\stocknetwork\StockNet\tests\test_walk_forward.py`
- Verify: existing regression tests touched by the new work

- [ ] **Step 1: Run the focused causal test suite**

Run:
```powershell
.venv311\Scripts\python.exe -m pytest ^
  tests/test_theme_persistence.py ^
  tests/test_temporal_labels.py ^
  tests/test_confirmation_backtest_causal.py ^
  tests/test_leadlag_causal_features.py ^
  tests/test_leadlag_edge_evaluation.py ^
  tests/test_walk_forward.py ^
  tests/test_stocknet_alpha_pipeline.py -q
```

Expected: all pass.

- [ ] **Step 2: Run any required script smoke tests**

At minimum run the new historical command path once and confirm the generated files exist.

- [ ] **Step 3: Only then report outcome**

Report one of:
- causal pipeline is correct but PnL not yet positive
- causal pipeline is correct and the tested slice is positive after costs

Do not blur these into one claim.
