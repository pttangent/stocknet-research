# StockNet: 非預設動態題材發現與配置系統

基於 5m/15m/30m intraday bars 的金融網絡動量分析平台，從市場共同運動中自動發現新興板塊，並通過 Temporal GNN 預測社群演變。

## 核心能力

| 能力 | 狀態 |
|------|------|
| 5m/15m/30m 多分辨率數據抓取 | ✅ |
| GPU 加速相關計算 (PyTorch) | ✅ |
| Bootstrap 共識社群偵測 | ✅ |
| Null Model 顯著性驗證 | ✅ |
| 市場中性 residual returns | ✅ |
| Co-jump / Lead-lag 邊特徵 | ✅ |
| Edge Persistence 基線 (Persistence/Logistic/XGBoost) | ✅ |
| Snapshot TGNN (純 PyTorch GRU) | ✅ |
| PyG Temporal GNN (GConvGRU/GConvLSTM) | ✅ |
| Community Survival 預測 | ✅ |
| Node Migration 預測 | ✅ |
| Ablation Study 框架 | ✅ |
| 回測強化 (現金閘門 + Volatility Targeting) | ✅ |
| Dashboard (網絡圖 + 共識面板 + TGNN 預測) | ✅ |
| 一鍵 Full Pipeline | ✅ |

## 快速開始

### 1. 啟動 Dashboard

```bash
npm start
```

訪問 http://localhost:3000

### 2. 一鍵 Full Pipeline

```bash
npm run research:full-pipeline -- \
  --input "/path/to/P123_Screen.csv" \
  --output-root artifacts/full_run_$(date +%Y%m%d)
```

### 3. 分步執行

```bash
# Phase 0: 多分辨率數據
npm run research:build-multi-res -- --input ... --output-root artifacts/

# Phase 1: 板塊分析 + curation
npm run research:analyze-rotation -- --input ... --parquet-root artifacts/parquet_15m --output artifacts/research_rotation
npm run research:curate-outputs -- --input-dir artifacts/research_rotation --output-dir artifacts/research_rotation_tuned

# Phase 2: 共識分群 + Null 驗證
npm run research:build-consensus -- --parquet-root artifacts/parquet_15m --output artifacts/consensus_clusters

# Phase 3: Graph Snapshots + 特徵工程
npm run research:build-snapshots -- --parquet-root artifacts/parquet_15m --output artifacts/graph_snapshots --compute-backend torch
npm run research:build-labels -- --dataset-dir artifacts/graph_snapshots

# Phase 4: 基線 + XGBoost + TGNN
npm run research:edge-baselines -- --dataset-dir artifacts/graph_snapshots
npm run research:train-xgboost -- --dataset-dir artifacts/graph_snapshots --output artifacts/graph_snapshots/xgboost
npm run research:train-tgnn -- --dataset-dir artifacts/graph_snapshots --output artifacts/tgnn_snapshot --device cuda --epochs 20

# Phase 5: 報告
npm run research:model-comparison -- --output-dir artifacts/
npm run research:build-report -- --output-dir artifacts/ --snapshot-dir artifacts/graph_snapshots --baseline-dir artifacts/graph_snapshots --tgnn-dir artifacts/tgnn_snapshot
```

## 技術架構

```
Data Layer
    ├── build_15m_parquet.py          (Yahoo 抓取, 支持 5m/15m/30m)
    └── build_multi_resolution_panels.py

Analysis Layer
    ├── analyze_rotation.py           (Louvain + lifecycle tracking)
    ├── curate_rotation_outputs.py    (Dashboard artifact 生成)
    └── backtest_rotation_strategy.py (Rolling Ridge 回測)

GPU / Graph Layer
    ├── gpu_graph.py                  (torch GPU corr + Leiden fallback)
    ├── consensus_clustering.py       (Bootstrap 200次 consensus)
    ├── null_models.py                (Time/Label/Sector null)
    └── multi_resolution.py           (跨解析度 NMI + emergence)

Feature Layer
    ├── features.py                   (Residual returns, co-jump, lead-lag)
    └── graph_snapshots.py            (Graph snapshot 生成)

Model Layer
    ├── baselines.py                  (Persistence/Logistic/Static GNN)
    ├── xgboost_baseline.py           (XGBoost edge predictor)
    ├── tgnn_snapshot.py              (純 PyTorch GRU TGNN)
    └── tgnn_pyg.py                   (PyTorch Geometric GConvGRU)

Portfolio Layer
    └── portfolio.py                  (現金閘門 + Vol Targeting + Sortino/Calmar)

Dashboard Layer
    ├── server.js                     (API server)
    └── app.js                        (前端可視化)
```

## GPU 需求

- **必須**: NVIDIA GPU + CUDA (已測試 CUDA 12.8)
- **PyTorch**: 2.12.0+ cu128
- **PyTorch Geometric**: 2.8.0+
- **leidenalg**: 0.11.0 (CPU Leiden fallback)
- **cuGraph**: 可選，Windows 上建議用 leidenalg fallback

## 關鍵修正 (v0.2+)

1. **Volume z-score lookahead bias 已修復**: 從 `groupby.transform(mean)` 改為 rolling time-of-day z-score
2. **Benchmark 自動注入**: SPY/QQQ/IWM/DIA 默認自動加入
3. **Dashboard artifact 自動生成**: `curate_rotation_outputs.py` 橋接分析輸出和前端展示

## 許可

個人研究使用。Yahoo Finance 數據非官方接口，不適合生產環境。
