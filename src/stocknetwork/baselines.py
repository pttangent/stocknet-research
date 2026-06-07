from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def run_edge_persistence_baselines(
    dataset_dir: Path | str,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, int]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    edge_table = build_edge_feature_table(dataset_dir)
    train_df, _, test_df = split_chronologically(edge_table, train_fraction=train_fraction, validation_fraction=validation_fraction)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    persistence_probability = (test_df["edge_persistence"] > 0).astype(float).to_numpy()
    persistence_prediction = (persistence_probability >= 0.5).astype(int)
    metric_rows.extend(_metric_rows("persistence", test_df["persists"].to_numpy(), persistence_probability, persistence_prediction))
    prediction_rows.extend(_prediction_rows("persistence", test_df, persistence_probability, persistence_prediction))

    strength_probability = _normalize_scores(test_df["edge_strength"].to_numpy(dtype=float))
    strength_prediction = (strength_probability >= 0.5).astype(int)
    metric_rows.extend(_metric_rows("edge_strength", test_df["persists"].to_numpy(), strength_probability, strength_prediction))
    prediction_rows.extend(_prediction_rows("edge_strength", test_df, strength_probability, strength_prediction))

    logistic_result = _run_logistic(train_df, test_df)
    if logistic_result is not None:
        metric_rows.extend(_metric_rows("logistic_regression", test_df["persists"].to_numpy(), logistic_result["probability"], logistic_result["prediction"]))
        prediction_rows.extend(_prediction_rows("logistic_regression", test_df, logistic_result["probability"], logistic_result["prediction"]))

    static_graph_result = _run_static_graph_logistic(train_df, test_df)
    if static_graph_result is not None:
        metric_rows.extend(_metric_rows("static_graph_logistic", test_df["persists"].to_numpy(), static_graph_result["probability"], static_graph_result["prediction"]))
        prediction_rows.extend(_prediction_rows("static_graph_logistic", test_df, static_graph_result["probability"], static_graph_result["prediction"]))

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    metrics_df.to_csv(dataset_dir / "baseline_metrics.csv", index=False)
    predictions_df.to_csv(dataset_dir / "baseline_predictions.csv", index=False)
    return {"metric_rows": len(metrics_df), "prediction_rows": len(predictions_df)}


def build_edge_feature_table(dataset_dir: Path | str) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    edge_labels = pd.read_csv(dataset_dir / "edge_labels.csv")
    manifest = pd.read_csv(dataset_dir / "snapshot_manifest.csv")
    snapshot_data_by_id = {
        str(row["snapshot_id"]): _load_snapshot_payload(dataset_dir / str(row["path"])) for _, row in manifest.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for _, row in edge_labels.iterrows():
        snapshot_data = snapshot_data_by_id.get(str(row["snapshot_id"]), {})
        feature_map = snapshot_data.get("edge_features", {})
        edge_key = tuple(sorted((str(row["symbol_left"]), str(row["symbol_right"]))))
        attrs = feature_map.get(edge_key)
        if attrs is None:
            continue
        node_attrs = _edge_node_context(snapshot_data, str(row["symbol_left"]), str(row["symbol_right"]))
        rows.append(
            {
                "snapshot_id": row["snapshot_id"],
                "timestamp": row["timestamp"],
                "symbol_left": row["symbol_left"],
                "symbol_right": row["symbol_right"],
                "persists": int(row["persists"]),
                **attrs,
                **node_attrs,
            }
        )
    return pd.DataFrame(rows).sort_values(["timestamp", "symbol_left", "symbol_right"]).reset_index(drop=True)


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


def _load_snapshot_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    feature_names = list(payload["edge_feature_names"])
    symbols = list(payload["symbols"])
    node_feature_names = list(payload.get("feature_names", []))
    node_x = np.asarray(payload.get("x", []), dtype=np.float32)
    edge_index = payload["edge_index"]
    edge_attr = payload["edge_attr"]
    feature_map: dict[tuple[str, str], dict[str, float]] = {}
    for position in range(len(edge_attr)):
        left_idx = int(edge_index[0][position])
        right_idx = int(edge_index[1][position])
        if left_idx >= right_idx:
            continue
        edge_key = tuple(sorted((symbols[left_idx], symbols[right_idx])))
        feature_map[edge_key] = {
            feature_name: float(edge_attr[position][feature_idx])
            for feature_idx, feature_name in enumerate(feature_names)
        }
    node_map: dict[str, dict[str, float]] = {}
    for idx, symbol in enumerate(symbols):
        node_map[symbol] = {
            feature_name: float(node_x[idx][feature_idx])
            for feature_idx, feature_name in enumerate(node_feature_names)
        }
    return {
        "symbols": symbols,
        "node_features": node_map,
        "edge_features": feature_map,
    }


def _edge_node_context(snapshot_data: dict[str, Any], left_symbol: str, right_symbol: str) -> dict[str, float]:
    left_features = snapshot_data.get("node_features", {}).get(left_symbol, {})
    right_features = snapshot_data.get("node_features", {}).get(right_symbol, {})
    context: dict[str, float] = {}
    for prefix, feature_map in (("left", left_features), ("right", right_features)):
        for feature_name, value in feature_map.items():
            context[f"{prefix}_{feature_name}"] = value
    return context


def _run_logistic(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, np.ndarray] | None:
    if train_df.empty or test_df.empty:
        return None
    y_train = train_df["persists"].to_numpy()
    if len(np.unique(y_train)) < 2:
        return None
    feature_columns = ["return_corr", "residual_corr", "volume_corr", "edge_strength", "edge_persistence"]
    model = LogisticRegression(max_iter=1000)
    model.fit(train_df[feature_columns], y_train)
    probability = model.predict_proba(test_df[feature_columns])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {"probability": probability, "prediction": prediction}


def _run_static_graph_logistic(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, np.ndarray] | None:
    if train_df.empty or test_df.empty:
        return None
    y_train = train_df["persists"].to_numpy()
    if len(np.unique(y_train)) < 2:
        return None
    feature_columns = [
        "return_corr",
        "residual_corr",
        "volume_corr",
        "edge_strength",
        "edge_persistence",
        "left_degree_centrality",
        "right_degree_centrality",
        "left_pagerank",
        "right_pagerank",
        "left_community_confidence",
        "right_community_confidence",
        "left_log_return",
        "right_log_return",
        "left_residual_return",
        "right_residual_return",
    ]
    available_columns = [column for column in feature_columns if column in train_df.columns and column in test_df.columns]
    if not available_columns:
        return None
    model = LogisticRegression(max_iter=1000)
    model.fit(train_df[available_columns], y_train)
    probability = model.predict_proba(test_df[available_columns])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {"probability": probability, "prediction": prediction}


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if math.isclose(min_value, max_value):
        return np.full_like(values, 0.5, dtype=float)
    return (values - min_value) / (max_value - min_value)


def _metric_rows(baseline: str, y_true: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_values = {
        "auc": _safe_metric(lambda: roc_auc_score(y_true, probability)),
        "average_precision": _safe_metric(lambda: average_precision_score(y_true, probability)),
        "f1": _safe_metric(lambda: f1_score(y_true, prediction)),
        "positive_rate": float(np.mean(prediction)) if len(prediction) else math.nan,
        "rows": float(len(y_true)),
    }
    for metric, value in metric_values.items():
        rows.append({"baseline": baseline, "metric": metric, "value": value})
    return rows


def _prediction_rows(baseline: str, frame: pd.DataFrame, probability: np.ndarray, prediction: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (_, item), prob, pred in zip(frame.iterrows(), probability, prediction, strict=False):
        rows.append(
            {
                "baseline": baseline,
                "snapshot_id": item["snapshot_id"],
                "timestamp": item["timestamp"],
                "symbol_left": item["symbol_left"],
                "symbol_right": item["symbol_right"],
                "probability": float(prob),
                "prediction": int(pred),
                "actual": int(item["persists"]),
            }
        )
    return rows


def _safe_metric(func) -> float:
    try:
        return float(func())
    except Exception:  # noqa: BLE001
        return math.nan
