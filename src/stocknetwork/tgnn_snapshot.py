from __future__ import annotations

import copy
import math
import pickle
import sys
from itertools import chain
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def build_sequence_index(
    manifest: pd.DataFrame,
    edge_labels: pd.DataFrame,
    sequence_length: int,
) -> list[dict[str, Any]]:
    manifest = manifest.sort_values("timestamp").reset_index(drop=True)
    label_lookup = edge_labels.groupby("snapshot_id")["future_snapshot_id"].first().to_dict()
    snapshot_ids = manifest["snapshot_id"].tolist()
    index_rows: list[dict[str, Any]] = []
    for end_position in range(sequence_length - 1, len(snapshot_ids)):
        target_snapshot_id = snapshot_ids[end_position]
        future_snapshot_id = label_lookup.get(target_snapshot_id)
        if future_snapshot_id is None:
            continue
        start_position = end_position - sequence_length + 1
        input_snapshot_ids = snapshot_ids[start_position : end_position + 1]
        index_rows.append(
            {
                "input_snapshot_ids": input_snapshot_ids,
                "target_snapshot_id": target_snapshot_id,
                "future_snapshot_id": future_snapshot_id,
                "target_timestamp": manifest.iloc[end_position]["timestamp"],
            }
        )
    return index_rows


def split_sequence_index(
    index_rows: list[dict[str, Any]],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(index_rows)
    train_cut = max(1, int(total * train_fraction))
    validation_cut = max(train_cut + 1, int(total * (train_fraction + validation_fraction)))
    train_rows = index_rows[:train_cut]
    validation_rows = index_rows[train_cut:validation_cut]
    test_rows = index_rows[validation_cut:] or index_rows[-1:]
    return train_rows, validation_rows, test_rows


def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


@dataclass
class SnapshotSequenceDataset:
    dataset_dir: Path
    manifest: pd.DataFrame
    edge_labels: pd.DataFrame
    sequence_index: list[dict[str, Any]]
    snapshots: dict[str, dict[str, Any]]

    @classmethod
    def from_dataset_dir(cls, dataset_dir: Path | str, sequence_length: int) -> "SnapshotSequenceDataset":
        dataset_dir = Path(dataset_dir).expanduser().resolve()
        manifest = pd.read_csv(dataset_dir / "snapshot_manifest.csv")
        edge_labels = pd.read_csv(dataset_dir / "edge_labels.csv")
        sequence_index = build_sequence_index(manifest, edge_labels, sequence_length=sequence_length)
        snapshots = {}
        for _, row in manifest.iterrows():
            with (dataset_dir / str(row["path"])).open("rb") as handle:
                snapshots[str(row["snapshot_id"])] = pickle.load(handle)
        return cls(dataset_dir, manifest, edge_labels, sequence_index, snapshots)

    def materialize_example(self, row: dict[str, Any]) -> dict[str, Any]:
        current_snapshot = self.snapshots[row["target_snapshot_id"]]
        sequence_snapshots = [self.snapshots[snapshot_id] for snapshot_id in row["input_snapshot_ids"]]
        label_rows = self.edge_labels[self.edge_labels["snapshot_id"] == row["target_snapshot_id"]].copy()
        symbol_to_index = {symbol: index for index, symbol in enumerate(current_snapshot["symbols"])}

        edge_pairs: list[tuple[int, int]] = []
        edge_features: list[list[float]] = []
        y: list[float] = []
        feature_lookup = _current_edge_feature_lookup(current_snapshot)
        for _, label in label_rows.iterrows():
            left_symbol = str(label["symbol_left"])
            right_symbol = str(label["symbol_right"])
            edge_key = tuple(sorted((left_symbol, right_symbol)))
            attrs = feature_lookup.get(edge_key)
            if attrs is None:
                continue
            edge_pairs.append((symbol_to_index[left_symbol], symbol_to_index[right_symbol]))
            edge_features.append(attrs)
            y.append(float(label["persists"]))

        return {
            "sequence_x": [np.asarray(snapshot["x"], dtype=np.float32) for snapshot in sequence_snapshots],
            "sequence_edge_index": [np.asarray(snapshot["edge_index"], dtype=np.int64) for snapshot in sequence_snapshots],
            "target_edge_pairs": edge_pairs,
            "target_edge_features": np.asarray(edge_features, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
            "target_snapshot_id": row["target_snapshot_id"],
        }


def _current_edge_feature_lookup(snapshot_payload: dict[str, Any]) -> dict[tuple[str, str], list[float]]:
    symbols = list(snapshot_payload["symbols"])
    edge_index = np.asarray(snapshot_payload["edge_index"])
    edge_attr = np.asarray(snapshot_payload["edge_attr"], dtype=np.float32)
    feature_lookup: dict[tuple[str, str], list[float]] = {}
    for edge_position in range(edge_attr.shape[0]):
        left_idx = int(edge_index[0, edge_position])
        right_idx = int(edge_index[1, edge_position])
        if left_idx >= right_idx:
            continue
        edge_key = tuple(sorted((symbols[left_idx], symbols[right_idx])))
        feature_lookup[edge_key] = edge_attr[edge_position].tolist()
    return feature_lookup


def train_edge_tgnn(
    dataset_dir: Path | str,
    output_dir: Path | str,
    sequence_length: int = 8,
    hidden_dim: int = 32,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    device: str = "cpu",
) -> dict[str, Any]:
    torch, nn = _import_torch()
    gpu_enabled = device.startswith("cuda")
    if gpu_enabled and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device}, but CUDA is not available in this runtime.")
    if gpu_enabled:
        torch.cuda.reset_peak_memory_stats(device)
    dataset = SnapshotSequenceDataset.from_dataset_dir(dataset_dir, sequence_length=sequence_length)
    train_index, validation_index, test_index = split_sequence_index(
        dataset.sequence_index,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset.materialize_example(dataset.sequence_index[0])
    node_feature_dim = sample["sequence_x"][0].shape[1]
    edge_feature_dim = sample["target_edge_features"].shape[1]
    model = EdgePersistenceTGNN(node_feature_dim, hidden_dim, edge_feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    history_rows: list[dict[str, Any]] = []
    best_state = None
    best_val_ap = -math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for row in train_index:
            example = dataset.materialize_example(row)
            if example["y"].size == 0:
                continue
            optimizer.zero_grad()
            probability = model.forward_example(example, device=device)
            target = torch.tensor(example["y"], dtype=torch.float32, device=device)
            loss = criterion(probability, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_edge_tgnn(model, dataset, validation_index, device=device) if validation_index else {"average_precision": math.nan}
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else math.nan,
                "validation_average_precision": val_metrics.get("average_precision", math.nan),
                "validation_auc": val_metrics.get("auc", math.nan),
                "validation_f1": val_metrics.get("f1", math.nan),
            }
        )
        val_ap = val_metrics.get("average_precision", math.nan)
        if not math.isnan(val_ap) and val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, prediction_rows = evaluate_edge_tgnn(model, dataset, test_index, device=device, return_predictions=True)
    pd.DataFrame(history_rows).to_csv(output_dir / "tgnn_training_history.csv", index=False)
    pd.DataFrame(
        [{"metric": metric, "value": value} for metric, value in test_metrics.items()]
    ).to_csv(output_dir / "tgnn_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "tgnn_predictions.csv", index=False)
    torch.save(model.state_dict(), output_dir / "tgnn_snapshot_edge.pt")
    return {
        "history_rows": len(history_rows),
        "test_metrics": test_metrics,
        "prediction_rows": len(prediction_rows),
        "train_sequences": len(train_index),
        "validation_sequences": len(validation_index),
        "test_sequences": len(test_index),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_enabled": gpu_enabled,
        "gpu_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
            if gpu_enabled
            else 0.0
        ),
    }


def evaluate_edge_tgnn(
    model,
    dataset: SnapshotSequenceDataset,
    index_rows: list[dict[str, Any]],
    device: str = "cpu",
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[dict[str, Any]]]:
    torch, _ = _import_torch()
    model.eval()
    y_true: list[float] = []
    y_prob: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in index_rows:
            example = dataset.materialize_example(row)
            if example["y"].size == 0:
                continue
            probability = model.forward_example(example, device=device).detach().cpu().numpy()
            y_true.extend(example["y"].tolist())
            y_prob.extend(probability.tolist())
            if return_predictions:
                current_snapshot = dataset.snapshots[row["target_snapshot_id"]]
                for (left_idx, right_idx), prob, actual in zip(
                    example["target_edge_pairs"],
                    probability.tolist(),
                    example["y"].tolist(),
                    strict=False,
                ):
                    prediction_rows.append(
                        {
                            "snapshot_id": row["target_snapshot_id"],
                            "future_snapshot_id": row["future_snapshot_id"],
                            "timestamp": row["target_timestamp"],
                            "symbol_left": current_snapshot["symbols"][left_idx],
                            "symbol_right": current_snapshot["symbols"][right_idx],
                            "probability": float(prob),
                            "prediction": int(prob >= 0.5),
                            "actual": int(actual),
                        }
                    )
    if not y_true:
        empty_metrics = {"auc": math.nan, "average_precision": math.nan, "f1": math.nan, "rows": 0.0}
        return (empty_metrics, prediction_rows) if return_predictions else empty_metrics
    y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]
    metrics = {
        "auc": _safe_metric(lambda: roc_auc_score(y_true, y_prob)),
        "average_precision": _safe_metric(lambda: average_precision_score(y_true, y_prob)),
        "f1": _safe_metric(lambda: f1_score(y_true, y_pred)),
        "rows": float(len(y_true)),
    }
    return (metrics, prediction_rows) if return_predictions else metrics


class EdgePersistenceTGNN:
    def __init__(self, node_feature_dim: int, hidden_dim: int, edge_feature_dim: int):
        torch, nn = _import_torch()
        self.torch = torch
        self.hidden_dim = hidden_dim
        self.message_linear = nn.Linear(node_feature_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def to(self, device: str):
        self.message_linear.to(device)
        self.gru.to(device)
        self.edge_mlp.to(device)
        return self

    def parameters(self):
        return chain(self.message_linear.parameters(), self.gru.parameters(), self.edge_mlp.parameters())

    def train(self):
        self.message_linear.train()
        self.gru.train()
        self.edge_mlp.train()

    def eval(self):
        self.message_linear.eval()
        self.gru.eval()
        self.edge_mlp.eval()

    def state_dict(self):
        return {
            "message_linear": self.message_linear.state_dict(),
            "gru": self.gru.state_dict(),
            "edge_mlp": self.edge_mlp.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.message_linear.load_state_dict(state_dict["message_linear"])
        self.gru.load_state_dict(state_dict["gru"])
        self.edge_mlp.load_state_dict(state_dict["edge_mlp"])

    def forward_example(self, example: dict[str, Any], device: str = "cpu"):
        torch = self.torch
        hidden = None
        for x_np, edge_index_np in zip(example["sequence_x"], example["sequence_edge_index"], strict=False):
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
            hidden = self._graph_step(x, edge_index, hidden)

        edge_pairs = torch.tensor(example["target_edge_pairs"], dtype=torch.long, device=device)
        edge_features = torch.tensor(example["target_edge_features"], dtype=torch.float32, device=device)
        left_embeddings = hidden[edge_pairs[:, 0]]
        right_embeddings = hidden[edge_pairs[:, 1]]
        logits = self.edge_mlp(torch.cat([left_embeddings, right_embeddings, edge_features], dim=1)).squeeze(-1)
        return logits

    def _graph_step(self, x, edge_index, hidden):
        torch = self.torch
        num_nodes = x.shape[0]
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=x.device)
        if edge_index.numel() > 0:
            adjacency[edge_index[0], edge_index[1]] = 1.0
        adjacency = adjacency + torch.eye(num_nodes, dtype=torch.float32, device=x.device)
        degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        aggregated = adjacency @ x / degree
        message = self.message_linear(aggregated)
        if hidden is None:
            hidden = torch.zeros((num_nodes, self.hidden_dim), dtype=torch.float32, device=x.device)
        return self.gru(message, hidden)


def _safe_metric(func) -> float:
    try:
        return float(func())
    except Exception:  # noqa: BLE001
        return math.nan
