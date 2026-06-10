# Main + Headless Runtime Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `main` the primary scanner development branch, keep `realtime-scanner-headless` as the runtime artifacts branch, and add the runtime foundation needed for a background scanner that can push artifacts safely without switching the live checkout.

**Architecture:** The live scanner will run from `main` and write artifacts locally. A dedicated publish worktree for `realtime-scanner-headless` will receive copied runtime artifacts and commit/push them independently, so the running scanner never changes its own checked-out branch. The alpha/backtest work will start only after this foundation is in place.

**Tech Stack:** Python, pytest, Git worktrees, Windows Task Scheduler helper scripts, parquet/json runtime artifacts

---

### Task 1: Add regression tests for background publishing and runtime branch isolation

**Files:**
- Modify: `tests/test_realtime_scanner_multiscale.py`
- Test: `tests/test_realtime_scanner_multiscale.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_prepare_runtime_publish_worktree_creates_branch_checkout(...):
    ...

def test_publish_runtime_artifacts_copies_scanner_outputs_to_worktree(...):
    ...

def test_continuous_monitor_only_publishes_when_alerts_exist(...):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_realtime_scanner_multiscale.py -q`
Expected: FAIL because runtime publish helpers do not exist or monitor still uses old branch-switching flow.

- [ ] **Step 3: Write minimal implementation**

```python
def prepare_runtime_publish_worktree(...):
    ...

def publish_runtime_artifacts(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_realtime_scanner_multiscale.py -q`
Expected: PASS for the new regression tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_realtime_scanner_multiscale.py realtime_dashboard/scripts/push_alerts_to_github.py realtime_dashboard/scripts/continuous_monitor.py
git commit -m "feat: isolate runtime artifact publishing from live scanner checkout"
```

### Task 2: Make runtime publishing target the headless branch from main safely

**Files:**
- Modify: `realtime_dashboard/scripts/push_alerts_to_github.py`
- Modify: `realtime_dashboard/scripts/continuous_monitor.py`
- Modify: `realtime_dashboard/README.md`
- Test: `tests/test_realtime_scanner_multiscale.py`

- [ ] **Step 1: Write the failing test**

```python
def test_publish_runtime_artifacts_stages_scanner_state_and_theme_state(...):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_realtime_scanner_multiscale.py -q`
Expected: FAIL because the publisher still writes markdown logs on `realtime-logs`.

- [ ] **Step 3: Write minimal implementation**

```python
RUNTIME_BRANCH = "realtime-scanner-headless"
RUNTIME_ARTIFACT_PATHS = [...]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_realtime_scanner_multiscale.py -q`
Expected: PASS and runtime publishing is branch-safe.

- [ ] **Step 5: Commit**

```bash
git add realtime_dashboard/scripts/push_alerts_to_github.py realtime_dashboard/scripts/continuous_monitor.py realtime_dashboard/README.md tests/test_realtime_scanner_multiscale.py
git commit -m "feat: publish scanner runtime artifacts to headless branch"
```

### Task 3: Add Windows background-runner entrypoints and docs

**Files:**
- Create: `realtime_dashboard/scripts/start_continuous_monitor.ps1`
- Create: `realtime_dashboard/scripts/register_continuous_monitor_task.ps1`
- Modify: `realtime_dashboard/README.md`
- Test: `tests/test_realtime_scanner_multiscale.py`

- [ ] **Step 1: Write the failing test**

```python
def test_background_runner_scripts_exist_and_reference_continuous_monitor():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_realtime_scanner_multiscale.py -q`
Expected: FAIL because the PowerShell helpers do not exist.

- [ ] **Step 3: Write minimal implementation**

```powershell
python realtime_dashboard/scripts/continuous_monitor.py ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_realtime_scanner_multiscale.py -q`
Expected: PASS and docs reference the background flow.

- [ ] **Step 5: Commit**

```bash
git add realtime_dashboard/scripts/start_continuous_monitor.ps1 realtime_dashboard/scripts/register_continuous_monitor_task.ps1 realtime_dashboard/README.md tests/test_realtime_scanner_multiscale.py
git commit -m "feat: add windows background monitor entrypoints"
```

### Task 4: Integrate headless scanner code into main and verify the merge baseline

**Files:**
- Modify: branch state in git
- Test: repo status and scanner verification commands

- [ ] **Step 1: Review merge delta**

Run: `git diff --stat main..realtime-scanner-headless`
Expected: Understand which scanner files will land on `main`.

- [ ] **Step 2: Merge into main**

Run: `git checkout main && git merge --no-ff realtime-scanner-headless`
Expected: `main` contains scanner runtime foundation and current headless code.

- [ ] **Step 3: Run verification**

Run: `pytest tests/test_realtime_scanner_multiscale.py -q`
Expected: PASS on `main`.

- [ ] **Step 4: Push main**

Run: `git push origin main`
Expected: Remote `main` updated with headless scanner foundation.

- [ ] **Step 5: Commit / merge result**

```bash
git push origin main
```

### Task 5: Prepare alpha module scaffold after main is updated

**Files:**
- Create: `stocknet_alpha/__init__.py`
- Create: `stocknet_alpha/data/resample_bars.py`
- Create: `stocknet_alpha/leadlag/generate_signals.py`
- Create: `stocknet_alpha/backtest/backtest_signals.py`
- Create: `stocknet_alpha/README.md`

- [ ] **Step 1: Stop after merge baseline is verified**

Run: `git status --short --branch`
Expected: Clean `main` before alpha module implementation starts.

- [ ] **Step 2: Begin a fresh TDD cycle for alpha work**

Run: `pytest tests/test_realtime_scanner_multiscale.py -q`
Expected: Current scanner foundation still passes before alpha changes begin.
