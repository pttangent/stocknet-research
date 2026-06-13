# U.S. Bars-Only Historical Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing U.S. market data pipeline so 2019-2025 encrypted 1m bars can be ingested into the independent `data` database layout before trade files are available.

**Architecture:** Keep the existing partitioned parquet layout and DuckDB views, then add a bars-only daily build path that writes `raw_1m`, `bars_5m`, `bars_15m`, `features_1m`, and `labels_1m` with the same schema expected by downstream readers. Historical year ingestion will recurse through year/month bar folders, stream one day per worker without unpacking to disk, and append results into the existing independent market database roots.

**Tech Stack:** Python, pandas, DuckDB, pyzipper, pytest

---

### Task 1: Add failing bars-only pipeline tests

**Files:**
- Modify: `D:\DEV\stocknetwork\StockNet\tests\test_us_market_data_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- bars-only zip discovery across year/month folders
- bars-only daily artifact generation without trades
- feature generation retaining trade-flow columns as null/empty when no trades exist

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_us_market_data_pipeline.py -q`
Expected: FAIL because the new bars-only helpers do not exist yet.

### Task 2: Implement bars-only ingestion support

**Files:**
- Modify: `D:\DEV\stocknetwork\StockNet\stocknet_alpha\data\us_market_data.py`

- [ ] **Step 1: Write minimal implementation**

Implement:
- a canonical empty trade-flow schema helper
- bars-only daily artifact builder
- recursive encrypted bar zip discovery for year/month roots
- feature generation that tolerates missing trade flow while preserving schema

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_us_market_data_pipeline.py -q`
Expected: PASS

### Task 3: Add a historical bars-only batch script

**Files:**
- Create: `D:\DEV\stocknetwork\StockNet\scripts\build_us_bars_only_database.py`

- [ ] **Step 1: Write the script**

The script should:
- accept one or more historical `1m` roots
- recurse to daily encrypted zips
- build reference tables once
- process daily bars in parallel
- append/update the ingest summary parquet
- rebuild the independent market database views

- [ ] **Step 2: Run a smoke test on a tiny date subset**

Run: `python scripts/build_us_bars_only_database.py --bars-root <one-year-root> --date <one-date>`
Expected: PASS and write the five bars-derived outputs for that date.

### Task 4: Bulk historical ingest and verify

**Files:**
- Modify: `D:\DEV\stocknetwork\StockNet\data\...` (generated partitions)
- Modify: `D:\DEV\stocknetwork\StockNet\data\stocknet_us.duckdb`
- Modify: `D:\DEV\stocknetwork\StockNet\data\stocknet_us_backtest.duckdb`

- [ ] **Step 1: Run the historical bars-only ingest**

Run the new script across:
- `C:\Users\A001\Downloads\US\1m\2019`
- `C:\Users\A001\Downloads\US\1m\2020`
- `C:\Users\A001\Downloads\US\1m\2021`
- `C:\Users\A001\Downloads\US\1m\2022`
- `C:\Users\A001\Downloads\US\1m\2023`
- `C:\Users\A001\Downloads\US\1m\2024`
- `C:\Users\A001\Downloads\US\1m\2025`

- [ ] **Step 2: Rebuild the independent backtest database**

Run the backtest database initialization so the causal views cover the new historical bar partitions too.

- [ ] **Step 3: Verify outputs**

Run:
- `pytest tests/test_us_market_data_pipeline.py tests/test_us_market_validation.py -q`
- targeted DuckDB checks for historical partition counts and sample feature rows

Expected: PASS, with historical bars-derived partitions available and no trade-flow requirement for pre-2026 dates.
