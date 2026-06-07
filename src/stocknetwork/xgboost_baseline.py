"""XGBoost baseline for edge persistence and community survival prediction.

Provides a strong non-GNN baseline using gradient boosted trees on
engineered edge, node, and community features.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def build_xgboost_feature_table(
    dataset_dir: Path | str,
    label_type: str = "edge",  # "edge", "community", "node"
) -> pd.DataFrame:
    """Build feature table for XGBoost from graph snapshots and labels.

    label_type:
    - "edge": edge persistence prediction
    - "community": community survival prediction
    - "node": node migration prediction
    """
    dataset_dir = Path(dataset_dir).expanduser().resolve()

    if label_type == "edge":
        return _build_edge_feature_table(dataset_dir)
    elif label_type == "community":
        return _build_community_feature_table(dataset_dir)
    elif label_type == "node":
        return _build_node_feature_table(dataset_dir)
    else:
        raise ValueError(f"Unknown label_type: {label_type}")


def _build_edge_feature_table(dataset_dir: Path) -> pd.DataFrame:
    edge_labels = pd.read_csv(dataset_dir / "edge_labels.csv")
    manifest = pd.read_csv(dataset_dir / "snapshot_manifest.csv")

    # Load snapshot metadata
    snapshot_data: dict[str, dict[str, Any]] = {}
    for _, row in manifest.iterrows():
        import pickle
        with (dataset_dir / str(row["path"])).open("rb") as f:
            payload = pickle.load(f)
        snapshot_data[str(row["snapshot_id"])] = payload

    rows: list[dict[str, Any]] = []
    for _, label in edge_labels.iterrows():
        snap_id = str(label["snapshot_id"])
        payload = snapshot_data.get(snap_id, {})
        symbols = list(payload.get("symbols", []))
        edge_index = np.asarray(payload.get("edge_index", []))
        edge_attr = np.asarray(payload.get("edge_attr", []), dtype=np.float32)
        node_x = np.asarray(payload.get("x", []), dtype=np.float32)

        left_sym = str(label["symbol_left"])
        right_sym = str(label["symbol_right"])
        left_idx = symbols.index(left_sym) if left_sym in symbols else -1
        right_idx = symbols.index(right_sym) if right_sym in symbols else -1

        if left_idx < 0 or right_idx < 0:
            continue

        # Find edge attributes
        edge_features: list[float] = []
        for pos in range(edge_attr.shape[0]):
            if edge_index[0, pos] == left_idx and edge_index[1, pos] == right_idx:
                edge_features = edge_attr[pos].tolist()
                break

        if not edge_features:
            continue

        # Node features
        left_node = node_x[left_idx].tolist() if left_idx < node_x.shape[0] else []
        right_node = node_x[right_idx].tolist() if right_idx < node_x.shape[0] else []

        row = {
            "snapshot_id": snap_id,
            "timestamp": label["timestamp"],
            "symbol_left": left_sym,
            "symbol_right": right_sym,
            "persists": int(label["persists"]),
        }
        # Edge features
        edge_names = list(payload.get("edge_feature_names", []))
        for idx, name in enumerate(edge_names):
            if idx < len(edge_features):
                row[name] = edge_features[idx]
        # Node features
        node_names = list(payload.get("feature_names", []))
        for idx, name in enumerate(node_names):
            if idx < len(left_node):
                row[f"left_{name}"] = left_node[idx]
            if idx < len(right_node):
                row[f"right_{name}"] = right_node[idx]

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["timestamp", "symbol_left", "symbol_right"]).reset_index(drop=True)


def _build_community_feature_table(dataset_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(dataset_dir / "community_labels.csv")
    if labels.empty:
        return pd.DataFrame()
    # For now, use basic community features
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    return labels.sort_values("timestamp").reset_index(drop=True)


def _build_node_feature_table(dataset_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(dataset_dir / "node_migration_labels.csv")
    if labels.empty:
        return pd.DataFrame()
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    return labels.sort_values("timestamp").reset_index(drop=True)


def split_chronologically(
    frame: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamps = sorted(frame["timestamp"].drop_duplicates().tolist())
    total = len(timestamps)
    train_cut = max(1, int(total * train_fraction))
    validation_cut = max(train_cut + 1, int(total * (train_fraction + validation_fraction)))
    train_times = set(timestamps[:train_cut])
    validation_times = set(timestamps[train_cut:validation_cut])
    test_times = set(timestamps[validation_cut:]) or set(timestamps[-1:])
    return (
        frame[frame["timestamp"].isin(train_times)].copy(),
        frame[frame["timestamp"].isin(validation_times)].copy(),
        frame[frame["timestamp"].isin(test_times)].copy(),
    )


def train_xgboost_edge_persistence(
    dataset_dir: Path | str,
    output_dir: Path | str,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    """Train XGBoost on edge persistence prediction."""
    import xgboost as xgb

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_xgboost_feature_table(dataset_dir, label_type="edge")
    if df.empty:
        return {"error": "No edge feature data"}

    train_df, _, test_df = split_chronologically(df, train_fraction, validation_fraction)

    if train_df.empty or test_df.empty:
        return {"error": "Insufficient data for train/test split"}

    # Feature columns (exclude metadata and target)
    exclude = {"snapshot_id", "timestamp", "symbol_left", "symbol_right", "persists"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in "fiub"]

    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["persists"].values
    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["persists"].values

    if len(np.unique(y_train)) < 2:
        return {"error": "Training data has only one class"}

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "auc": _safe_metric(lambda: roc_auc_score(y_test, prob)),
        "average_precision": _safe_metric(lambda: average_precision_score(y_test, prob)),
        "f1": _safe_metric(lambda: f1_score(y_test, pred)),
        "positive_rate": float(np.mean(pred)),
        "rows": float(len(y_test)),
    }

    # Save
    metrics_df = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
    metrics_df.to_csv(output_dir / "xgboost_edge_metrics.csv", index=False)

    predictions = pd.DataFrame({
        "snapshot_id": test_df["snapshot_id"],
        "timestamp": test_df["timestamp"],
        "symbol_left": test_df["symbol_left"],
        "symbol_right": test_df["symbol_right"],
        "probability": prob,
        "prediction": pred,
        "actual": y_test,
    })
    predictions.to_csv(output_dir / "xgboost_edge_predictions.csv", index=False)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "xgboost_edge_importance.csv", index=False)

    model.save_model(str(output_dir / "xgboost_edge_model.json"))

    return {
        "metrics": metrics,
        "prediction_rows": len(predictions),
        "feature_count": len(feature_cols),
    }


def _safe_metric(func) -> float:
    try:
        return float(func())
    except Exception:
        return math.nan
