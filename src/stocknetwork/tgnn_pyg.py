"""PyTorch Geometric Temporal GNN for financial network evolution prediction.

Supports three prediction tasks:
1. Edge Persistence: will an edge still exist in the future?
2. Community Survival: will a community persist?
3. Node Migration: stay / migrate / isolated

Uses GConvGRU or GConvLSTM for temporal graph modeling.
"""
from __future__ import annotations

import copy
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv, GATConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


class TemporalGNN(nn.Module):
    """Multi-task Temporal GNN using PyTorch Geometric.

    Architecture:
    - Input: sequence of graph snapshots (node features + edge_index)
    - Spatial: GCNConv layers for graph message passing
    - Temporal: GRUCell for temporal state update
    - Task heads: EdgePersistenceHead, CommunitySurvivalHead, NodeMigrationHead
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        edge_feature_dim: int,
        num_gnn_layers: int = 2,
        temporal_model: str = "GCN_GRU",
        K: int = 2,
    ):
        if not _HAS_PYG:
            raise ImportError("PyTorch Geometric is required. Install: pip install torch_geometric")

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_gnn_layers

        # GCN layers for spatial message passing
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # GRU cells for temporal update
        self.gru_cells = nn.ModuleList()
        self.gru_cells.append(nn.GRUCell(hidden_dim, hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.gru_cells.append(nn.GRUCell(hidden_dim, hidden_dim))

        # Task heads
        self.edge_head = EdgePersistenceHead(hidden_dim, edge_feature_dim)
        self.community_head = CommunitySurvivalHead(hidden_dim)
        self.migration_head = NodeMigrationHead(hidden_dim)

    def forward_snapshot(self, x, edge_index, edge_weight, hidden_states):
        """Process a single snapshot.

        Args:
            x: [N, F] node features
            edge_index: [2, E] edge indices
            edge_weight: [E] edge weights
            hidden_states: list of hidden states from previous snapshots

        Returns:
            h: [N, H] node embeddings
            new_hidden_states: updated hidden states
        """
        h = x
        new_hidden_states = []
        for layer_idx, (gcn, gru) in enumerate(zip(self.gcn_layers, self.gru_cells)):
            # Spatial message passing
            spatial = gcn(h, edge_index, edge_weight)
            spatial = torch.relu(spatial)
            # Temporal update
            prev_h = hidden_states[layer_idx] if hidden_states and layer_idx < len(hidden_states) else None
            if prev_h is None:
                prev_h = torch.zeros_like(spatial)
            h = gru(spatial, prev_h)
            new_hidden_states.append(h)
        return h, new_hidden_states

    def forward_sequence(self, sequence_data: list[dict], device: str = "cpu"):
        """Process a sequence of snapshots.

        Args:
            sequence_data: list of dicts with keys:
                - x: node features
                - edge_index: edge indices
                - edge_weight: edge weights
            device: torch device

        Returns:
            final_h: [N, H] final node embeddings
        """
        hidden_states = None
        for snap in sequence_data:
            x = torch.tensor(snap["x"], dtype=torch.float32, device=device)
            edge_index = torch.tensor(snap["edge_index"], dtype=torch.long, device=device)
            edge_weight = torch.tensor(snap.get("edge_weight", np.ones(edge_index.shape[1])), dtype=torch.float32, device=device)
            h, hidden_states = self.forward_snapshot(x, edge_index, edge_weight, hidden_states)
        return h


class EdgePersistenceHead(nn.Module):
    """Predict whether an edge will persist into the future."""

    def __init__(self, hidden_dim: int, edge_feature_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_embeddings, edge_pairs, edge_features):
        """
        Args:
            node_embeddings: [N, H]
            edge_pairs: [E, 2] (source, target)
            edge_features: [E, F_edge]
        Returns:
            probs: [E] persistence probabilities
        """
        src_emb = node_embeddings[edge_pairs[:, 0]]
        tgt_emb = node_embeddings[edge_pairs[:, 1]]
        combined = torch.cat([src_emb, tgt_emb, edge_features], dim=-1)
        return self.mlp(combined).squeeze(-1)


class CommunitySurvivalHead(nn.Module):
    """Predict whether a community will survive into the future."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, community_embedding):
        """
        Args:
            community_embedding: [C, H] aggregated community embeddings
        Returns:
            probs: [C] survival probabilities
        """
        return self.mlp(community_embedding).squeeze(-1)


class NodeMigrationHead(nn.Module):
    """Predict node migration: stay / migrate / isolated (3-class)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3),  # 3 classes
        )

    def forward(self, node_embeddings):
        """
        Args:
            node_embeddings: [N, H]
        Returns:
            logits: [N, 3] class logits
        """
        return self.mlp(node_embeddings)


def build_sequence_index(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    sequence_length: int,
) -> list[dict[str, Any]]:
    """Build chronological sequence index for temporal training."""
    manifest = manifest.sort_values("timestamp").reset_index(drop=True)
    label_lookup = labels.groupby("snapshot_id")["future_snapshot_id"].first().to_dict()
    snapshot_ids = manifest["snapshot_id"].tolist()

    index_rows: list[dict[str, Any]] = []
    for end_pos in range(sequence_length - 1, len(snapshot_ids)):
        target_id = snapshot_ids[end_pos]
        future_id = label_lookup.get(target_id)
        if future_id is None:
            continue
        start_pos = end_pos - sequence_length + 1
        input_ids = snapshot_ids[start_pos:end_pos + 1]
        index_rows.append({
            "input_snapshot_ids": input_ids,
            "target_snapshot_id": target_id,
            "future_snapshot_id": future_id,
            "target_timestamp": manifest.iloc[end_pos]["timestamp"],
        })
    return index_rows


def split_sequence_index(
    index_rows: list[dict[str, Any]],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(index_rows)
    train_cut = max(1, int(total * train_fraction))
    validation_cut = max(train_cut + 1, int(total * (train_fraction + validation_fraction)))
    return (
        index_rows[:train_cut],
        index_rows[train_cut:validation_cut],
        index_rows[validation_cut:] or index_rows[-1:],
    )


class SnapshotSequenceDatasetPyG:
    """Dataset loader for PyG temporal GNN."""

    def __init__(self, dataset_dir: Path | str, sequence_length: int, task: str = "edge"):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.manifest = pd.read_csv(self.dataset_dir / "snapshot_manifest.csv")
        self.task = task

        # Load appropriate labels
        if task == "edge":
            labels = pd.read_csv(self.dataset_dir / "edge_labels.csv")
        elif task == "community":
            labels = pd.read_csv(self.dataset_dir / "community_labels.csv")
        elif task == "node":
            labels = pd.read_csv(self.dataset_dir / "node_migration_labels.csv")
        else:
            raise ValueError(f"Unknown task: {task}")

        self.sequence_index = build_sequence_index(self.manifest, labels, sequence_length)

        # Load all snapshots into memory
        self.snapshots: dict[str, dict[str, Any]] = {}
        for _, row in self.manifest.iterrows():
            with (self.dataset_dir / str(row["path"])).open("rb") as f:
                self.snapshots[str(row["snapshot_id"])] = pickle.load(f)

        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequence_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.sequence_index[idx]
        target_snap = self.snapshots[row["target_snapshot_id"]]

        # Build sequence
        sequence = []
        for snap_id in row["input_snapshot_ids"]:
            snap = self.snapshots[snap_id]
            sequence.append({
                "x": np.asarray(snap["x"], dtype=np.float32),
                "edge_index": np.asarray(snap["edge_index"], dtype=np.int64),
                "edge_weight": np.ones(snap["edge_index"].shape[1], dtype=np.float32) if snap["edge_index"].shape[1] > 0 else np.array([], dtype=np.float32),
            })

        result = {
            "sequence": sequence,
            "target_snapshot_id": row["target_snapshot_id"],
            "future_snapshot_id": row["future_snapshot_id"],
            "timestamp": row["target_timestamp"],
        }

        # Task-specific labels
        if self.task == "edge":
            label_rows = self.labels[self.labels["snapshot_id"] == row["target_snapshot_id"]]
            symbols = list(target_snap["symbols"])
            symbol_to_idx = {s: i for i, s in enumerate(symbols)}

            edge_pairs = []
            edge_features = []
            y = []

            # Build edge feature lookup
            edge_index = np.asarray(target_snap["edge_index"])
            edge_attr = np.asarray(target_snap["edge_attr"], dtype=np.float32)
            feature_lookup: dict[tuple[int, int], list[float]] = {}
            for pos in range(edge_attr.shape[0]):
                src = int(edge_index[0, pos])
                tgt = int(edge_index[1, pos])
                feature_lookup[(src, tgt)] = edge_attr[pos].tolist()

            for _, lbl in label_rows.iterrows():
                left = str(lbl["symbol_left"])
                right = str(lbl["symbol_right"])
                if left in symbol_to_idx and right in symbol_to_idx:
                    li = symbol_to_idx[left]
                    ri = symbol_to_idx[right]
                    edge_pairs.append([li, ri])
                    feat = feature_lookup.get((li, ri), [0.0] * edge_attr.shape[1])
                    edge_features.append(feat)
                    y.append(float(lbl["persists"]))

            result["edge_pairs"] = np.asarray(edge_pairs, dtype=np.int64)
            result["edge_features"] = np.asarray(edge_features, dtype=np.float32)
            result["y"] = np.asarray(y, dtype=np.float32)

        elif self.task == "community":
            label_rows = self.labels[self.labels["snapshot_id"] == row["target_snapshot_id"]]
            # target_snap["community_ids"] maps each node to its community
            node_community_ids = np.asarray(target_snap.get("community_ids", []), dtype=np.int64)
            # Build community_member_indices mapping
            community_members: dict[int, list[int]] = {}
            for node_idx, comm_id in enumerate(node_community_ids):
                if comm_id >= 0:
                    community_members.setdefault(int(comm_id), []).append(node_idx)

            # For each labeled community, store member indices and label
            member_indices: list[list[int]] = []
            y_values: list[float] = []
            for _, lbl in label_rows.iterrows():
                cid = int(lbl["community_id"])
                if cid in community_members:
                    member_indices.append(community_members[cid])
                    y_values.append(float(lbl["survives"]))

            result["community_member_indices"] = member_indices
            result["y"] = np.asarray(y_values, dtype=np.float32)

        elif self.task == "node":
            label_rows = self.labels[self.labels["snapshot_id"] == row["target_snapshot_id"]]
            symbols = list(target_snap["symbols"])
            symbol_to_idx = {s: i for i, s in enumerate(symbols)}

            # Map labels to symbol indices
            y = np.full(len(symbols), -1, dtype=np.int64)
            for _, lbl in label_rows.iterrows():
                sym = str(lbl["symbol"])
                if sym in symbol_to_idx:
                    label_map = {"stay": 0, "migrate": 1, "isolated": 2}
                    y[symbol_to_idx[sym]] = label_map.get(str(lbl["migration_label"]), -1)

            result["y"] = y

        return result


def train_tgnn_pyg(
    dataset_dir: Path | str,
    output_dir: Path | str,
    task: str = "edge",
    sequence_length: int = 8,
    hidden_dim: int = 64,
    num_gnn_layers: int = 2,
    temporal_model: str = "GConvGRU",
    epochs: int = 20,
    learning_rate: float = 1e-3,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    device: str = "cuda",
    batch_size: int = 1,
) -> dict[str, Any]:
    """Train PyG Temporal GNN."""
    if not _HAS_PYG:
        return {"error": "PyTorch Geometric Temporal not installed"}

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    dataset = SnapshotSequenceDatasetPyG(dataset_dir, sequence_length, task=task)
    train_idx, val_idx, test_idx = split_sequence_index(
        dataset.sequence_index, train_fraction, validation_fraction
    )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer dimensions from first sample
    sample = dataset[0]
    node_feature_dim = sample["sequence"][0]["x"].shape[1]
    edge_feature_dim = sample.get("edge_features", np.zeros((1, 1))).shape[1] if "edge_features" in sample else 1

    model = TemporalGNN(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        edge_feature_dim=edge_feature_dim,
        num_gnn_layers=num_gnn_layers,
        temporal_model=temporal_model,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_val_ap = -math.inf
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for i in range(0, len(train_idx), batch_size):
            batch_rows = train_idx[i:i + batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0
            count = 0
            for row in batch_rows:
                sample = dataset[dataset.sequence_index.index(row)]
                final_h = model.forward_sequence(sample["sequence"], device=device)

                if task == "edge" and sample["y"].size > 0:
                    edge_pairs = torch.tensor(sample["edge_pairs"], dtype=torch.long, device=device)
                    edge_features = torch.tensor(sample["edge_features"], dtype=torch.float32, device=device)
                    probs = model.edge_head(final_h, edge_pairs, edge_features)
                    target = torch.tensor(sample["y"], dtype=torch.float32, device=device)
                    loss = F.binary_cross_entropy(probs, target)
                    batch_loss += loss
                    count += 1
                elif task == "community" and sample["y"].size > 0:
                    # Aggregate community embeddings by mean pooling member nodes
                    member_indices = sample.get("community_member_indices", [])
                    if member_indices:
                        comm_embs = []
                        valid_targets = []
                        for idx, members in enumerate(member_indices):
                            if len(members) > 0:
                                member_tensor = torch.tensor(members, dtype=torch.long, device=device)
                                comm_embs.append(final_h[member_tensor].mean(dim=0))
                                valid_targets.append(sample["y"][idx])
                        if comm_embs:
                            comm_emb = torch.stack(comm_embs)
                            target = torch.tensor(valid_targets, dtype=torch.float32, device=device)
                            probs = model.community_head(comm_emb)
                            loss = F.binary_cross_entropy(probs, target)
                            batch_loss += loss
                            count += 1
                elif task == "node":
                    logits = model.migration_head(final_h)
                    target = torch.tensor(sample["y"], dtype=torch.long, device=device)
                    valid = target >= 0
                    if valid.any():
                        loss = F.cross_entropy(logits[valid], target[valid])
                        batch_loss += loss
                        count += 1

            if count > 0:
                batch_loss = batch_loss / count
                batch_loss.backward()
                optimizer.step()
                train_losses.append(float(batch_loss.detach().cpu()))

        # Validation
        val_metrics = evaluate_tgnn_pyg(model, dataset, val_idx, task=task, device=device)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else math.nan,
            "val_auc": val_metrics.get("auc", math.nan),
            "val_ap": val_metrics.get("average_precision", math.nan),
            "val_f1": val_metrics.get("f1", math.nan),
        })

        val_ap = val_metrics.get("average_precision", math.nan)
        if not math.isnan(val_ap) and val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_tgnn_pyg(model, dataset, test_idx, task=task, device=device)

    # Save
    pd.DataFrame(history).to_csv(output_dir / "tgnn_pyg_history.csv", index=False)
    pd.DataFrame([{"metric": k, "value": v} for k, v in test_metrics.items()]).to_csv(
        output_dir / "tgnn_pyg_metrics.csv", index=False
    )
    torch.save(model.state_dict(), output_dir / "tgnn_pyg_model.pt")

    return {
        "task": task,
        "test_metrics": test_metrics,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "device": device,
        "gpu_peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.startswith("cuda") else 0.0,
    }


def evaluate_tgnn_pyg(
    model: TemporalGNN,
    dataset: SnapshotSequenceDatasetPyG,
    index_rows: list[dict[str, Any]],
    task: str = "edge",
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate PyG Temporal GNN."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    model.eval()
    y_true = []
    y_prob = []
    y_pred = []

    with torch.no_grad():
        for row in index_rows:
            idx = dataset.sequence_index.index(row)
            sample = dataset[idx]
            final_h = model.forward_sequence(sample["sequence"], device=device)

            if task == "edge" and sample["y"].size > 0:
                edge_pairs = torch.tensor(sample["edge_pairs"], dtype=torch.long, device=device)
                edge_features = torch.tensor(sample["edge_features"], dtype=torch.float32, device=device)
                probs = model.edge_head(final_h, edge_pairs, edge_features)
                y_true.extend(sample["y"].tolist())
                y_prob.extend(probs.cpu().numpy().tolist())
                y_pred.extend((probs >= 0.5).cpu().numpy().tolist())
            elif task == "community" and sample["y"].size > 0:
                member_indices = sample.get("community_member_indices", [])
                if member_indices:
                    comm_embs = []
                    valid_targets = []
                    for idx, members in enumerate(member_indices):
                        if len(members) > 0:
                            member_tensor = torch.tensor(members, dtype=torch.long, device=device)
                            comm_embs.append(final_h[member_tensor].mean(dim=0))
                            valid_targets.append(sample["y"][idx])
                    if comm_embs:
                        comm_emb = torch.stack(comm_embs)
                        probs = model.community_head(comm_emb)
                        y_true.extend(valid_targets)
                        y_prob.extend(probs.cpu().numpy().tolist())
                        y_pred.extend((probs >= 0.5).cpu().numpy().tolist())
            elif task == "node":
                logits = model.migration_head(final_h)
                target = torch.tensor(sample["y"], dtype=torch.long, device=device)
                valid = target >= 0
                if valid.any():
                    probs = F.softmax(logits[valid], dim=1)
                    # For migration, use class 0 (stay) probability for AUC-like metrics
                    y_true.extend((target[valid] == 0).cpu().numpy().tolist())
                    y_prob.extend(probs[:, 0].cpu().numpy().tolist())
                    y_pred.extend(probs.argmax(dim=1).cpu().numpy().tolist())

    if not y_true:
        return {"auc": math.nan, "average_precision": math.nan, "f1": math.nan, "rows": 0.0}

    metrics = {
        "auc": _safe_metric(lambda: roc_auc_score(y_true, y_prob)),
        "average_precision": _safe_metric(lambda: average_precision_score(y_true, y_prob)),
        "f1": _safe_metric(lambda: f1_score(y_true, y_pred)),
        "rows": float(len(y_true)),
    }
    return metrics


def _safe_metric(func) -> float:
    try:
        return float(func())
    except Exception:
        return math.nan
