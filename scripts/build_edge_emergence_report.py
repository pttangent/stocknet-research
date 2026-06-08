#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a markdown report for edge emergence results.")
    parser.add_argument("--input-dir", required=True, help="Directory containing edge emergence metrics and predictions.")
    parser.add_argument("--output", required=True, help="Markdown output path.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top case rows to include.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    metrics = pd.read_csv(input_dir / "xgboost_edge_emergence_metrics.csv")
    predictions = pd.read_csv(input_dir / "xgboost_edge_emergence_predictions.csv")
    importance = pd.read_csv(input_dir / "xgboost_edge_emergence_importance.csv")

    top_true_positive = predictions[(predictions["prediction"] == 1) & (predictions["actual"] == 1)].nlargest(args.top_k, "probability")
    top_false_positive = predictions[(predictions["prediction"] == 1) & (predictions["actual"] == 0)].nlargest(args.top_k, "probability")
    top_false_negative = predictions[(predictions["prediction"] == 0) & (predictions["actual"] == 1)].nsmallest(args.top_k, "probability")

    metric_lookup = {row["metric"]: row["value"] for _, row in metrics.iterrows()}
    lines = [
        "# Edge Emergence Report",
        "",
        "## Summary",
        "",
        f"- AUC: `{metric_lookup.get('auc', 'n/a')}`",
        f"- Average Precision: `{metric_lookup.get('average_precision', 'n/a')}`",
        f"- F1: `{metric_lookup.get('f1', 'n/a')}`",
        f"- Positive rate: `{metric_lookup.get('positive_rate', 'n/a')}`",
        f"- Test rows: `{metric_lookup.get('rows', 'n/a')}`",
        "",
        "## Feature Importance",
        "",
    ]
    lines.extend(_markdown_table(importance.head(15)))
    lines.extend(["", "## Top True Positives", ""])
    lines.extend(_markdown_table(top_true_positive[["snapshot_id", "symbol_left", "symbol_right", "probability", "actual"]]))
    lines.extend(["", "## Top False Positives", ""])
    lines.extend(_markdown_table(top_false_positive[["snapshot_id", "symbol_left", "symbol_right", "probability", "actual"]]))
    lines.extend(["", "## Top False Negatives", ""])
    lines.extend(_markdown_table(top_false_negative[["snapshot_id", "symbol_left", "symbol_right", "probability", "actual"]]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Edge emergence report written to {output_path}")
    return 0


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        parts = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                parts.append(f"{value:.6f}")
            else:
                parts.append(str(value))
        lines.append("| " + " | ".join(parts) + " |")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
