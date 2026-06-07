# Pipeline Commands

## Existing research stages

```bash
npm run research:build-parquet -- --input <portfolio123.csv> --output artifacts/parquet_15m
npm run research:analyze-rotation -- --input <portfolio123.csv> --parquet-root artifacts/parquet_15m --output artifacts/research_rotation --benchmark SPY
npm run research:backtest-rotation -- --parquet-root artifacts/parquet_15m --rotation-dir artifacts/research_rotation_tuned --output artifacts/backtest_rotation --benchmark SPY --use-curated-only
```

## Graph dataset and labels

```bash
npm run research:build-snapshots -- --parquet-root artifacts/parquet_15m_test --output artifacts/graph_snapshots_test --benchmark SPY --window-bars 10 --min-history-bars 10 --top-k 5 --edge-threshold 0.05
npm run research:build-labels -- --dataset-dir artifacts/graph_snapshots_test --horizon 1
npm run research:edge-baselines -- --dataset-dir artifacts/graph_snapshots_test --train-fraction 0.6 --validation-fraction 0.2
```

Optional Torch-backed snapshot compute in the Python 3.11 environment:

```powershell
.\.venv311\Scripts\python.exe scripts/build_graph_snapshots.py --parquet-root artifacts/parquet_15m_test --output artifacts/graph_snapshots_torch_test --benchmark SPY --window-bars 10 --min-history-bars 10 --top-k 5 --edge-threshold 0.05 --compute-backend torch
```

## TGNN runtime

The snapshot TGNN training path uses a local Python 3.11 virtual environment because the default Python 3.14 environment is not a stable base for the Torch stack in this repo.

Setup:

```powershell
npm run research:setup-tgnn-env
```

Train:

```powershell
npm run research:train-tgnn -- --dataset-dir artifacts/graph_snapshots_test --output artifacts/tgnn_snapshot_test --sequence-length 8 --hidden-dim 32 --epochs 2 --device cpu
```

## Comparison report

```bash
npm run research:model-comparison -- --baseline-dir artifacts/graph_snapshots_test --tgnn-dir artifacts/tgnn_snapshot_test --output-dir artifacts/model_comparison_test
```

## Verified sample artifact chain

The repo now has a verified sample run with these outputs:

- graph snapshots: `artifacts/graph_snapshots_test`
- graph snapshots (Torch backend): `artifacts/graph_snapshots_torch_test`
- temporal labels: `artifacts/graph_snapshots_test/*.csv`
- baseline metrics: `artifacts/graph_snapshots_test/baseline_metrics.csv`
- TGNN metrics: `artifacts/tgnn_snapshot_test/tgnn_metrics.csv`
- combined comparison: `artifacts/model_comparison_test/model_comparison.csv`

Verified sample comparison highlights:

- `static_graph_logistic` AUC `0.731427`
- `tgnn_snapshot` AUC `0.729531`
