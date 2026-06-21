from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import (
    _ALPHA_FACTOR_COLUMNS_BY_LAYER,
    _ALPHA_LABEL_VARIANTS,
    _LAYER_RESEARCH_ROLES,
)
from .io_utils import _safe_zscore


def _export_alpha_sanity_report(
    community_snapshot_feature_path: Path,
    community_forward_label_path: Path,
    output_path: Path,
) -> None:
    features = pd.read_parquet(community_snapshot_feature_path)
    labels = pd.read_parquet(community_forward_label_path)
    labels = labels.drop(
        columns=[
            column
            for column in (
                "trade_date",
                "snapshot_timestamp",
                "snapshot_clock_code",
                "community_local_id",
                "community_member_count",
                "community_edge_count",
                "edge_density",
                "community_avg_weight",
            )
            if column in labels.columns
        ]
    )
    merged = features.merge(
        labels,
        how="inner",
        on=["snapshot_id", "graph_layer", "layer_community_id"],
        suffixes=("_feature", "_label"),
    )
    if "community_member_count_feature" in merged.columns:
        merged["community_member_count"] = merged["community_member_count_feature"]
    if "edge_density_feature" not in merged.columns and "edge_density" in merged.columns:
        merged["edge_density_feature"] = merged["edge_density"]
    target_variants = [
        (label_variant, horizon_name, f"{column_prefix}_{horizon_name}")
        for label_variant, column_prefix in _ALPHA_LABEL_VARIANTS
        for horizon_name in ("1m", "5m", "15m", "30m")
    ]
    rows: list[dict[str, Any]] = []
    for graph_layer, layer_frame in merged.groupby("graph_layer", dropna=False):
        for factor_name in _alpha_factors_for_layer(graph_layer):
            if factor_name not in layer_frame.columns:
                continue
            for label_variant, horizon_name, target_name in target_variants:
                if target_name not in layer_frame.columns:
                    continue
                sample = layer_frame[[factor_name, target_name]].dropna()
                if sample.empty:
                    rows.append(
                        {
                            "graph_layer": graph_layer,
                            "factor_name": factor_name,
                            "label_horizon": horizon_name,
                            "label_variant": label_variant,
                            "sample_size": 0,
                            "rank_ic": None,
                            "top_decile_mean": None,
                            "bottom_decile_mean": None,
                            "top_bottom_spread": None,
                            "top_decile_hit_rate": None,
                        }
                    )
                    continue
                rank_ic = sample[factor_name].rank().corr(sample[target_name].rank())
                decile_size = max(1, len(sample) // 10)
                sorted_sample = sample.sort_values(factor_name)
                bottom = sorted_sample.head(decile_size)[target_name]
                top = sorted_sample.tail(decile_size)[target_name]
                rows.append(
                    {
                        "graph_layer": graph_layer,
                        "factor_name": factor_name,
                        "label_horizon": horizon_name,
                        "label_variant": label_variant,
                        "sample_size": int(len(sample)),
                        "rank_ic": None if pd.isna(rank_ic) else float(rank_ic),
                        "top_decile_mean": float(top.mean()),
                        "bottom_decile_mean": float(bottom.mean()),
                        "top_bottom_spread": float(top.mean() - bottom.mean()),
                        "top_decile_hit_rate": float((top > 0).mean()),
                    }
                )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _export_alpha_feature_ranking_report(
    alpha_sanity_report_path: Path,
    output_path: Path,
) -> None:
    report = pd.read_csv(alpha_sanity_report_path)
    if report.empty:
        pd.DataFrame(
            columns=[
                "graph_layer",
                "layer_role",
                "factor_name",
                "label_horizon",
                "label_variant",
                "sample_size",
                "rank_ic",
                "top_bottom_spread",
                "top_decile_hit_rate",
                "score",
                "confidence_bucket",
                "research_action",
            ]
        ).to_csv(output_path, index=False)
        return
    ranking = report.copy()
    ranking["sample_size"] = pd.to_numeric(ranking["sample_size"], errors="coerce").fillna(0).astype(int)
    ranking["rank_ic"] = pd.to_numeric(ranking["rank_ic"], errors="coerce")
    ranking["top_bottom_spread"] = pd.to_numeric(ranking["top_bottom_spread"], errors="coerce")
    ranking["top_decile_hit_rate"] = pd.to_numeric(ranking["top_decile_hit_rate"], errors="coerce")
    ranking["layer_role"] = ranking["graph_layer"].map(_LAYER_RESEARCH_ROLES).fillna("unclassified_layer")
    ranking["score"] = ranking.apply(_alpha_ranking_score, axis=1)
    ranking["confidence_bucket"] = ranking["sample_size"].apply(_alpha_confidence_bucket)
    ranking["research_action"] = ranking.apply(_alpha_research_action, axis=1)
    ranking = ranking.sort_values(
        ["score", "sample_size", "graph_layer", "factor_name", "label_variant", "label_horizon"],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)
    ranking.to_csv(output_path, index=False)


def _export_cross_window_alpha_comparison_report(
    first_window_ranking_path: Path,
    second_window_ranking_path: Path,
    output_path: Path,
    *,
    first_window_id: str,
    second_window_id: str,
) -> None:
    key_columns = [
        "graph_layer",
        "layer_role",
        "factor_name",
        "label_horizon",
        "label_variant",
    ]
    metric_columns = [
        "sample_size",
        "rank_ic",
        "top_bottom_spread",
        "top_decile_hit_rate",
        "score",
        "confidence_bucket",
        "research_action",
    ]
    first = _prepare_cross_window_ranking_frame(first_window_ranking_path, key_columns, metric_columns, "first")
    second = _prepare_cross_window_ranking_frame(second_window_ranking_path, key_columns, metric_columns, "second")
    comparison = first.merge(second, how="outer", on=key_columns)
    if comparison.empty:
        pd.DataFrame(
            columns=key_columns
            + [
                "first_window_id",
                "second_window_id",
                "first_score_sign",
                "second_score_sign",
                "score_direction_consistent",
                "rank_ic_direction_consistent",
                "sample_qualified_both",
                "research_action_consistent",
                "score_delta",
                "sample_size_delta",
                "stability_bucket",
                "research_decision",
            ]
        ).to_csv(output_path, index=False)
        return
    comparison["first_window_id"] = first_window_id
    comparison["second_window_id"] = second_window_id
    for prefix in ("first", "second"):
        comparison[f"{prefix}_sample_size"] = pd.to_numeric(
            comparison[f"{prefix}_sample_size"],
            errors="coerce",
        ).fillna(0).astype(int)
        comparison[f"{prefix}_rank_ic"] = pd.to_numeric(comparison[f"{prefix}_rank_ic"], errors="coerce")
        comparison[f"{prefix}_top_bottom_spread"] = pd.to_numeric(
            comparison[f"{prefix}_top_bottom_spread"],
            errors="coerce",
        )
        comparison[f"{prefix}_top_decile_hit_rate"] = pd.to_numeric(
            comparison[f"{prefix}_top_decile_hit_rate"],
            errors="coerce",
        )
        comparison[f"{prefix}_score"] = pd.to_numeric(comparison[f"{prefix}_score"], errors="coerce")
        comparison[f"{prefix}_score_sign"] = comparison[f"{prefix}_score"].apply(_alpha_sign)
    comparison["score_direction_consistent"] = comparison.apply(
        lambda row: row["first_score_sign"] != 0
        and row["second_score_sign"] != 0
        and row["first_score_sign"] == row["second_score_sign"],
        axis=1,
    )
    comparison["rank_ic_direction_consistent"] = comparison.apply(
        lambda row: _alpha_sign(row["first_rank_ic"]) != 0
        and _alpha_sign(row["second_rank_ic"]) != 0
        and _alpha_sign(row["first_rank_ic"]) == _alpha_sign(row["second_rank_ic"]),
        axis=1,
    )
    comparison["sample_qualified_both"] = comparison.apply(
        lambda row: row["first_sample_size"] >= 3000 and row["second_sample_size"] >= 3000,
        axis=1,
    )
    comparison["research_action_consistent"] = (
        comparison["first_research_action"].fillna("") == comparison["second_research_action"].fillna("")
    )
    comparison["score_delta"] = comparison["second_score"] - comparison["first_score"]
    comparison["sample_size_delta"] = comparison["second_sample_size"] - comparison["first_sample_size"]
    comparison["stability_bucket"] = comparison.apply(_cross_window_stability_bucket, axis=1)
    comparison["research_decision"] = comparison["stability_bucket"].map(
        {
            "stable_positive": "confirm_layer_role",
            "stable_negative": "deprioritize",
            "insufficient_sample": "needs_more_sample",
            "missing_in_one_window": "rebuild_missing_window",
        }
    ).fillna("review_manually")
    comparison = comparison.sort_values(
        ["stability_bucket", "graph_layer", "factor_name", "label_variant", "label_horizon"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    comparison.to_csv(output_path, index=False)


def _prepare_cross_window_ranking_frame(
    ranking_path: Path,
    key_columns: list[str],
    metric_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    frame = pd.read_csv(ranking_path)
    available_key_columns = [column for column in key_columns if column in frame.columns]
    available_metric_columns = [column for column in metric_columns if column in frame.columns]
    prepared = frame.loc[:, available_key_columns + available_metric_columns].copy()
    for column in key_columns:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    for column in metric_columns:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    rename_map = {column: f"{prefix}_{column}" for column in metric_columns}
    return prepared.loc[:, key_columns + metric_columns].rename(columns=rename_map)


def _alpha_ranking_score(row: pd.Series) -> float:
    rank_ic = row.get("rank_ic")
    spread = row.get("top_bottom_spread")
    sample_size = int(row.get("sample_size", 0) or 0)
    if pd.isna(rank_ic) or pd.isna(spread) or sample_size <= 0:
        return 0.0
    spread_sign = 1.0 if spread > 0 else -1.0 if spread < 0 else 0.0
    return float(abs(rank_ic) * math.log10(sample_size + 1) * spread_sign)


def _alpha_confidence_bucket(sample_size: int) -> str:
    if sample_size < 500:
        return "ignore"
    if sample_size < 3000:
        return "watch"
    if sample_size <= 10000:
        return "usable"
    return "strong_sample"


def _alpha_research_action(row: pd.Series) -> str:
    sample_size = int(row.get("sample_size", 0) or 0)
    score = float(row.get("score", 0.0) or 0.0)
    layer_role = row.get("layer_role")
    if sample_size < 500:
        return "ignore_sparse"
    if pd.isna(row.get("rank_ic")) or pd.isna(row.get("top_bottom_spread")):
        return "insufficient_signal"
    if score > 0:
        if sample_size > 10000 and layer_role == "theme_candidate_layer":
            return "prioritize_for_next_round"
        if sample_size >= 3000:
            return "keep_for_next_round"
        return "watch"
    if sample_size >= 3000:
        return "downgrade"
    return "watch"


def _alpha_sign(value: Any) -> int:
    if pd.isna(value):
        return 0
    numeric_value = float(value)
    if numeric_value > 0:
        return 1
    if numeric_value < 0:
        return -1
    return 0


def _cross_window_stability_bucket(row: pd.Series) -> str:
    if row["first_sample_size"] == 0 or row["second_sample_size"] == 0:
        return "missing_in_one_window"
    if not row["sample_qualified_both"]:
        return "insufficient_sample"
    if row["score_direction_consistent"]:
        if row["first_score_sign"] > 0:
            return "stable_positive"
        if row["first_score_sign"] < 0:
            return "stable_negative"
    return "unstable_direction"


def _alpha_factors_for_layer(graph_layer: Any) -> list[str]:
    return list(_ALPHA_FACTOR_COLUMNS_BY_LAYER.get(str(graph_layer), ["community_quality_score"]))


def _augment_community_snapshot_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        for column in (
            "edge_density_feature",
            "community_avg_weight_feature",
            "layer_member_ratio",
            "flow_member_count_z",
            "flow_layer_participation_ratio",
            "flow_breadth_expansion",
            "community_quality_score",
        ):
            frame[column] = pd.Series(dtype="float64")
        return frame
    augmented = frame.copy()
    augmented["snapshot_timestamp"] = pd.to_datetime(augmented["snapshot_timestamp"])
    augmented["edge_density_feature"] = pd.to_numeric(augmented["edge_density"], errors="coerce")
    augmented["community_avg_weight_feature"] = pd.to_numeric(augmented["community_avg_weight"], errors="coerce")
    augmented["community_member_count"] = pd.to_numeric(augmented["community_member_count"], errors="coerce")
    augmented["feature_coverage_ratio"] = pd.to_numeric(augmented["feature_coverage_ratio"], errors="coerce")
    augmented["layer_active_node_count"] = pd.to_numeric(augmented["layer_active_node_count"], errors="coerce")
    augmented["snapshot_active_symbol_count"] = pd.to_numeric(augmented["snapshot_active_symbol_count"], errors="coerce")
    augmented["layer_member_ratio"] = (
        augmented["community_member_count"] / augmented["layer_active_node_count"].replace(0, pd.NA)
    )
    augmented["flow_layer_participation_ratio"] = (
        augmented["layer_active_node_count"] / augmented["snapshot_active_symbol_count"].replace(0, pd.NA)
    )
    augmented["flow_member_count_z"] = (
        augmented.groupby("graph_layer", dropna=False)["community_member_count"].transform(_safe_zscore)
    )
    augmented["quality_density_z"] = augmented.groupby("graph_layer", dropna=False)["edge_density_feature"].transform(_safe_zscore)
    augmented["quality_avg_weight_z"] = augmented.groupby("graph_layer", dropna=False)["community_avg_weight_feature"].transform(_safe_zscore)
    augmented["quality_coverage_z"] = augmented.groupby("graph_layer", dropna=False)["feature_coverage_ratio"].transform(_safe_zscore)
    augmented["quality_size_penalty_z"] = augmented.groupby("graph_layer", dropna=False)["layer_member_ratio"].transform(_safe_zscore)
    augmented["community_quality_score"] = (
        augmented["quality_density_z"]
        + augmented["quality_avg_weight_z"]
        + augmented["quality_coverage_z"]
        - augmented["quality_size_penalty_z"]
    )
    breadth_base = (
        augmented.loc[:, ["snapshot_id", "graph_layer", "snapshot_timestamp", "flow_layer_participation_ratio"]]
        .drop_duplicates()
        .sort_values(["graph_layer", "snapshot_timestamp", "snapshot_id"])
        .reset_index(drop=True)
    )
    breadth_base["flow_participation_level"] = breadth_base["flow_layer_participation_ratio"]
    breadth_base["flow_participation_delta_1"] = (
        breadth_base.groupby("graph_layer", dropna=False)["flow_layer_participation_ratio"].diff().fillna(0.0)
    )
    breadth_base["flow_participation_delta_3"] = (
        breadth_base.groupby("graph_layer", dropna=False)["flow_layer_participation_ratio"].diff(3).fillna(0.0)
    )
    breadth_base["flow_breadth_expansion"] = breadth_base["flow_participation_delta_1"]
    augmented = augmented.merge(
        breadth_base[
            [
                "snapshot_id",
                "graph_layer",
                "flow_breadth_expansion",
                "flow_participation_level",
                "flow_participation_delta_1",
                "flow_participation_delta_3",
            ]
        ],
        how="left",
        on=["snapshot_id", "graph_layer"],
    )
    return augmented
