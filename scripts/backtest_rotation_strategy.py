#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.run_metadata import create_run_context


NY_TZ = "America/New_York"
DEFAULT_TRAIN_DAYS = 15
DEFAULT_TOP_THEMES = 3
DEFAULT_TOP_STOCKS_PER_THEME = 5
DEFAULT_COST_BPS = 10.0
DEFAULT_MAX_WEIGHT_PER_STOCK = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling no-lookahead backtest for the sector rotation strategy.")
    parser.add_argument("--parquet-root", required=True, help="Path to the symbol-partitioned parquet database.")
    parser.add_argument("--rotation-dir", required=True, help="Directory containing tuned rotation outputs.")
    parser.add_argument("--output", required=True, help="Output directory for backtest artifacts.")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark symbol. Default: SPY")
    parser.add_argument("--train-days", type=int, default=DEFAULT_TRAIN_DAYS, help="Rolling train window in signal dates.")
    parser.add_argument("--top-themes", type=int, default=DEFAULT_TOP_THEMES, help="How many themes to hold each rebalance.")
    parser.add_argument(
        "--top-stocks-per-theme",
        type=int,
        default=DEFAULT_TOP_STOCKS_PER_THEME,
        help="How many stocks to select inside each chosen theme.",
    )
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS, help="Round-turn trading cost in basis points.")
    parser.add_argument(
        "--max-weight-per-stock",
        type=float,
        default=DEFAULT_MAX_WEIGHT_PER_STOCK,
        help="Maximum portfolio weight per single stock.",
    )
    parser.add_argument(
        "--use-curated-only",
        action="store_true",
        help="Restrict the strategy universe to curated lifecycles if curated_sector_summary.csv exists.",
    )
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier for metadata tracking.")
    parser.add_argument("--run-label", default="", help="Optional short label for this run.")
    parser.add_argument("--run-notes", default="", help="Optional notes for this run.")
    return parser.parse_args()


def load_daily_close(parquet_root: Path, benchmark_symbol: str) -> pd.DataFrame:
    manifest = pd.read_csv(parquet_root / "_manifest.csv")
    success_symbols = manifest.loc[manifest["status"] == "success", "symbol"].astype(str).tolist()
    close_series: dict[str, pd.Series] = {}
    for symbol in success_symbols:
        file_path = parquet_root / f"symbol={symbol}" / "part-000.parquet"
        if not file_path.exists():
            continue
        frame = pd.read_parquet(file_path, columns=["timestamp", "close"])
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        close_series[symbol] = frame.set_index("timestamp")["close"].astype(float)

    close_df = pd.DataFrame(close_series).sort_index()
    if benchmark_symbol not in close_df.columns:
        raise RuntimeError(f"Benchmark symbol {benchmark_symbol} not found in parquet database.")

    local_index = close_df.index.tz_convert(NY_TZ)
    close_df = close_df.copy()
    close_df["trade_date"] = local_index.normalize()
    daily_close = close_df.groupby("trade_date").last()
    daily_close.index = pd.to_datetime(daily_close.index).tz_localize(None)
    return daily_close


def compute_stock_features(daily_close: pd.DataFrame, benchmark_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_returns = daily_close.pct_change()
    benchmark_returns = daily_returns[benchmark_symbol]
    rel_strength_10d = (
        daily_close / daily_close.shift(10) - 1.0
    ).sub((daily_close[benchmark_symbol] / daily_close[benchmark_symbol].shift(10) - 1.0), axis=0)
    volatility_10d = daily_returns.rolling(10).std()
    return rel_strength_10d, volatility_10d


def parse_snapshots(rotation_dir: Path) -> pd.DataFrame:
    snapshots = pd.read_pickle(rotation_dir / "cluster_snapshots.pkl").copy()
    snapshots["trade_date"] = pd.to_datetime(snapshots["trade_date"]).dt.tz_localize(None)
    snapshots["primary_members"] = snapshots["primary_members"].apply(list)
    snapshots["member_weights"] = snapshots["member_weights"].apply(dict)
    return snapshots


def select_lifecycles(rotation_dir: Path, use_curated_only: bool, snapshots: pd.DataFrame) -> set[str]:
    curated_path = rotation_dir / "curated_sector_summary.csv"
    if use_curated_only and curated_path.exists():
        curated = pd.read_csv(curated_path)
        return set(curated["lifecycle_id"].astype(str))
    return set(snapshots["lifecycle_id"].astype(str))


def attach_targets(snapshots: pd.DataFrame, daily_returns: pd.DataFrame) -> pd.DataFrame:
    trade_dates = list(daily_returns.index)
    date_to_loc = {trade_date: idx for idx, trade_date in enumerate(trade_dates)}

    def forward_mean_return(members: list[str], trade_date: pd.Timestamp, horizon: int) -> float:
        if trade_date not in date_to_loc:
            return math.nan
        start_loc = date_to_loc[trade_date] + 1
        end_loc = start_loc + horizon
        if start_loc >= len(trade_dates):
            return math.nan
        future_index = trade_dates[start_loc:end_loc]
        if len(future_index) == 0:
            return math.nan
        member_slice = daily_returns.reindex(index=future_index, columns=members)
        series = member_slice.mean(axis=1, skipna=True).dropna()
        if series.empty:
            return math.nan
        return float(np.expm1(np.log1p(series).sum()))

    target_1d: list[float] = []
    target_5d: list[float] = []
    next_dates: list[pd.Timestamp | pd.NaT] = []

    for _, row in snapshots.iterrows():
        trade_date = row["trade_date"]
        loc = date_to_loc.get(trade_date)
        if loc is None or loc + 1 >= len(trade_dates):
            next_dates.append(pd.NaT)
        else:
            next_dates.append(trade_dates[loc + 1])
        members = row["primary_members"]
        target_1d.append(forward_mean_return(members, trade_date, 1))
        target_5d.append(forward_mean_return(members, trade_date, 5))

    snapshots["next_trade_date"] = next_dates
    snapshots["target_1d"] = target_1d
    snapshots["target_5d"] = target_5d
    return snapshots


def build_model() -> Pipeline:
    numeric_features = [
        "momentum_score",
        "coherence",
        "breadth",
        "volume_expansion",
        "return_3d",
        "return_10d",
        "return_20d",
        "relative_strength_10d",
        "size",
        "overlap_size",
        "lifecycle_age",
        "size_change",
    ]
    categorical_features = ["stage"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def compute_stock_scores(
    snapshot_row: pd.Series,
    rel_strength_10d: pd.DataFrame,
    volatility_10d: pd.DataFrame,
    benchmark_symbol: str,
    top_stocks_per_theme: int,
) -> list[dict[str, Any]]:
    trade_date = snapshot_row["trade_date"]
    theme_score = float(snapshot_row["predicted_score"])
    stock_records: list[dict[str, Any]] = []
    stock_rel = rel_strength_10d.loc[trade_date] if trade_date in rel_strength_10d.index else pd.Series(dtype=float)
    stock_vol = volatility_10d.loc[trade_date] if trade_date in volatility_10d.index else pd.Series(dtype=float)

    for symbol, membership_weight in snapshot_row["member_weights"].items():
        if symbol == benchmark_symbol:
            continue
        rel_strength = float(stock_rel.get(symbol, 0.0) if not pd.isna(stock_rel.get(symbol, np.nan)) else 0.0)
        volatility = float(stock_vol.get(symbol, np.nan) if not pd.isna(stock_vol.get(symbol, np.nan)) else 0.04)
        score = theme_score * float(membership_weight) + 0.15 * rel_strength - 0.05 * volatility
        stock_records.append(
            {
                "symbol": symbol,
                "theme_lifecycle_id": snapshot_row["lifecycle_id"],
                "theme_name": snapshot_row["cluster_name"],
                "theme_score": theme_score,
                "membership_weight": float(membership_weight),
                "stock_rel_strength_10d": rel_strength,
                "stock_volatility_10d": volatility,
                "stock_score": score,
            }
        )

    stock_df = pd.DataFrame(stock_records)
    if stock_df.empty:
        return []
    stock_df = stock_df.sort_values(["stock_score", "membership_weight"], ascending=False).head(top_stocks_per_theme)
    return stock_df.to_dict("records")


def cap_and_normalize_weights(weight_series: pd.Series, max_weight_per_stock: float) -> pd.Series:
    if weight_series.empty:
        return weight_series

    weights = weight_series.clip(lower=0.0)
    if weights.sum() <= 0:
        return pd.Series(dtype=float)
    weights = weights / weights.sum()

    while True:
        capped = weights.clip(upper=max_weight_per_stock)
        excess = 1.0 - capped.sum()
        if excess >= -1e-9:
            remaining_mask = capped < max_weight_per_stock - 1e-12
            if remaining_mask.any() and excess > 1e-9:
                redistribute = capped[remaining_mask]
                redistribute = redistribute / redistribute.sum()
                capped.loc[remaining_mask] = capped.loc[remaining_mask] + redistribute * excess
            weights = capped / capped.sum()
            break
        weights = capped / capped.sum()

    return weights


def run_backtest(
    snapshots: pd.DataFrame,
    daily_close: pd.DataFrame,
    rel_strength_10d: pd.DataFrame,
    volatility_10d: pd.DataFrame,
    benchmark_symbol: str,
    train_days: int,
    top_themes: int,
    top_stocks_per_theme: int,
    cost_bps: float,
    max_weight_per_stock: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_returns = daily_close.pct_change()
    signal_dates = sorted(snapshots["trade_date"].drop_duplicates())
    model = build_model()

    prediction_rows: list[dict[str, Any]] = []
    positions_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    previous_weights = pd.Series(dtype=float)

    for signal_index in range(train_days, len(signal_dates) - 1):
        signal_date = signal_dates[signal_index]
        execution_date = signal_dates[signal_index + 1]
        train_dates = signal_dates[signal_index - train_days : signal_index]

        train_slice = snapshots[(snapshots["trade_date"].isin(train_dates)) & snapshots["target_1d"].notna()].copy()
        live_slice = snapshots[snapshots["trade_date"] == signal_date].copy()
        if train_slice.empty or live_slice.empty:
            continue

        feature_columns = [
            "momentum_score",
            "coherence",
            "breadth",
            "volume_expansion",
            "return_3d",
            "return_10d",
            "return_20d",
            "relative_strength_10d",
            "size",
            "overlap_size",
            "lifecycle_age",
            "size_change",
            "stage",
        ]
        model.fit(train_slice[feature_columns], train_slice["target_1d"])
        live_slice["predicted_score"] = model.predict(live_slice[feature_columns])
        live_slice["signal_date"] = signal_date
        live_slice["execution_date"] = execution_date

        selected_themes = live_slice[live_slice["predicted_score"] > 0].sort_values(
            ["predicted_score", "momentum_score"], ascending=False
        ).head(top_themes)
        if selected_themes.empty:
            selected_themes = live_slice.sort_values(["predicted_score", "momentum_score"], ascending=False).head(top_themes)

        stock_picks: list[dict[str, Any]] = []
        for _, theme_row in selected_themes.iterrows():
            stock_picks.extend(
                compute_stock_scores(
                    snapshot_row=theme_row,
                    rel_strength_10d=rel_strength_10d,
                    volatility_10d=volatility_10d,
                    benchmark_symbol=benchmark_symbol,
                    top_stocks_per_theme=top_stocks_per_theme,
                )
            )
            prediction_rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "lifecycle_id": theme_row["lifecycle_id"],
                    "cluster_name": theme_row["cluster_name"],
                    "predicted_score": float(theme_row["predicted_score"]),
                    "momentum_score": float(theme_row["momentum_score"]),
                    "coherence": float(theme_row["coherence"]),
                    "breadth": float(theme_row["breadth"]),
                    "target_1d_realized": float(theme_row["target_1d"]) if pd.notna(theme_row["target_1d"]) else math.nan,
                }
            )

        stock_pick_df = pd.DataFrame(stock_picks)
        if stock_pick_df.empty:
            target_weights = pd.Series(dtype=float)
        else:
            stock_weight_signal = stock_pick_df.groupby("symbol")["stock_score"].sum()
            target_weights = cap_and_normalize_weights(stock_weight_signal, max_weight_per_stock=max_weight_per_stock)

        if execution_date not in daily_returns.index:
            continue
        realized_stock_returns = daily_returns.loc[execution_date].reindex(target_weights.index).fillna(0.0)
        gross_return = float((target_weights * realized_stock_returns).sum()) if not target_weights.empty else 0.0

        aligned_prev = previous_weights.reindex(target_weights.index.union(previous_weights.index)).fillna(0.0)
        aligned_target = target_weights.reindex(aligned_prev.index).fillna(0.0)
        turnover = float((aligned_target - aligned_prev).abs().sum())
        cost = turnover * (cost_bps / 10000.0)
        net_return = gross_return - cost
        benchmark_return = float(daily_returns.loc[execution_date, benchmark_symbol]) if benchmark_symbol in daily_returns.columns else 0.0

        for symbol, weight in target_weights.items():
            positions_rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "symbol": symbol,
                    "target_weight": float(weight),
                    "realized_return": float(realized_stock_returns.get(symbol, 0.0)),
                }
            )

        portfolio_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "n_positions": int(len(target_weights)),
            }
        )
        previous_weights = target_weights

    portfolio_df = pd.DataFrame(portfolio_rows).sort_values("execution_date").reset_index(drop=True)
    prediction_df = pd.DataFrame(prediction_rows).sort_values(["signal_date", "predicted_score"], ascending=[True, False]).reset_index(drop=True)
    positions_df = pd.DataFrame(positions_rows).sort_values(["execution_date", "target_weight"], ascending=[True, False]).reset_index(drop=True)
    return portfolio_df, prediction_df, positions_df


def summarize_metrics(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    if portfolio_df.empty:
        return pd.DataFrame()

    net = portfolio_df["net_return"].fillna(0.0)
    bench = portfolio_df["benchmark_return"].fillna(0.0)
    excess = net - bench

    def max_drawdown(returns: pd.Series) -> float:
        curve = (1.0 + returns).cumprod()
        drawdown = curve / curve.cummax() - 1.0
        return float(drawdown.min())

    ann_factor = 252
    mean_daily = float(net.mean())
    std_daily = float(net.std(ddof=0))
    sharpe = (mean_daily / std_daily) * math.sqrt(ann_factor) if std_daily > 0 else math.nan

    metrics = [
        ("total_return", float((1.0 + net).prod() - 1.0)),
        ("benchmark_total_return", float((1.0 + bench).prod() - 1.0)),
        ("excess_total_return", float((1.0 + excess).prod() - 1.0)),
        ("annualized_return", float((1.0 + net).prod() ** (ann_factor / max(len(net), 1)) - 1.0)),
        ("benchmark_annualized_return", float((1.0 + bench).prod() ** (ann_factor / max(len(bench), 1)) - 1.0)),
        ("annualized_volatility", std_daily * math.sqrt(ann_factor)),
        ("sharpe", sharpe),
        ("max_drawdown", max_drawdown(net)),
        ("hit_rate", float((net > 0).mean())),
        ("avg_turnover", float(portfolio_df["turnover"].mean())),
        ("avg_positions", float(portfolio_df["n_positions"].mean())),
        ("days", int(len(portfolio_df))),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def write_report(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
) -> None:
    report_lines = [
        "# Rolling Sector Rotation Backtest",
        "",
        "This backtest uses a rolling training window and executes signals one trading day later.",
        "",
        "## Anti-Lookahead Rules",
        "",
        "- Theme features on signal date `t` use only information available by the close of `t`.",
        "- Model fitting for signal date `t` uses only rows whose realized next-day returns were already known before `t`.",
        "- Portfolio return is measured from `t+1` close over the next daily bar; the strategy never earns the same bar that generated the signal.",
        "- Transaction costs are applied on turnover at each rebalance.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for _, row in metrics_df.iterrows():
        value = row["value"]
        if row["metric"] in {"sharpe", "avg_turnover", "avg_positions", "days"}:
            display = f"{value:.4f}" if isinstance(value, float) else str(value)
        else:
            display = f"{value:.2%}" if pd.notna(value) else "nan"
        report_lines.append(f"| {row['metric']} | {display} |")

    report_lines.extend(
        [
            "",
            "## Recent Top Theme Predictions",
            "",
            "| Signal Date | Execution Date | Lifecycle | Theme | Predicted 1D | Realized 1D | Momentum | Coherence | Breadth |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in prediction_df.head(30).iterrows():
        report_lines.append(
            f"| {pd.Timestamp(row['signal_date']).date()} | {pd.Timestamp(row['execution_date']).date()} | {row['lifecycle_id']} | {row['cluster_name']} | {row['predicted_score']:.2%} | {row['target_1d_realized']:.2%} | {row['momentum_score']:.2f} | {row['coherence']:.2f} | {row['breadth']:.2f} |"
        )

    report_lines.extend(
        [
            "",
            "## Recent Portfolio Returns",
            "",
            "| Execution Date | Net Return | Benchmark | Turnover | Positions |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in portfolio_df.tail(20).iterrows():
        report_lines.append(
            f"| {pd.Timestamp(row['execution_date']).date()} | {row['net_return']:.2%} | {row['benchmark_return']:.2%} | {row['turnover']:.2f} | {int(row['n_positions'])} |"
        )

    (output_dir / "backtest_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    parquet_root = Path(args.parquet_root).expanduser().resolve()
    rotation_dir = Path(args.rotation_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="backtest_rotation",
        output_dir=output_dir,
        args=args,
        inputs={"parquet_root": parquet_root, "rotation_dir": rotation_dir},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    daily_close = load_daily_close(parquet_root, args.benchmark)
    rel_strength_10d, volatility_10d = compute_stock_features(daily_close, args.benchmark)
    snapshots = parse_snapshots(rotation_dir)
    keep_lifecycles = select_lifecycles(rotation_dir, args.use_curated_only, snapshots)
    snapshots = snapshots[snapshots["lifecycle_id"].isin(keep_lifecycles)].copy()
    snapshots = attach_targets(snapshots, daily_close.pct_change())

    portfolio_df, prediction_df, positions_df = run_backtest(
        snapshots=snapshots,
        daily_close=daily_close,
        rel_strength_10d=rel_strength_10d,
        volatility_10d=volatility_10d,
        benchmark_symbol=args.benchmark,
        train_days=args.train_days,
        top_themes=args.top_themes,
        top_stocks_per_theme=args.top_stocks_per_theme,
        cost_bps=args.cost_bps,
        max_weight_per_stock=args.max_weight_per_stock,
    )

    if portfolio_df.empty:
        raise RuntimeError("Backtest produced no portfolio rows.")

    portfolio_df["equity_curve"] = (1.0 + portfolio_df["net_return"]).cumprod()
    portfolio_df["benchmark_curve"] = (1.0 + portfolio_df["benchmark_return"]).cumprod()
    metrics_df = summarize_metrics(portfolio_df)

    portfolio_df.to_csv(output_dir / "portfolio_returns.csv", index=False)
    prediction_df.to_csv(output_dir / "theme_predictions.csv", index=False)
    positions_df.to_csv(output_dir / "positions.csv", index=False)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    write_report(output_dir, metrics_df, prediction_df, portfolio_df)
    run_context.write_validation(
        {
            "benchmark_present": args.benchmark in daily_close.columns,
            "daily_rows": len(daily_close),
            "snapshot_rows": len(snapshots),
            "portfolio_rows": len(portfolio_df),
            "prediction_rows": len(prediction_df),
            "positions_rows": len(positions_df),
        }
    )
    run_context.write_artifacts(
        {
            "portfolio_returns_csv": output_dir / "portfolio_returns.csv",
            "theme_predictions_csv": output_dir / "theme_predictions.csv",
            "positions_csv": output_dir / "positions.csv",
            "metrics_csv": output_dir / "metrics.csv",
            "backtest_report_md": output_dir / "backtest_report.md",
        }
    )
    metrics_map = dict(zip(metrics_df["metric"], metrics_df["value"], strict=False))
    run_context.write_summary(
        {
            "status": "completed",
            "days": int(metrics_map.get("days", 0)),
            "total_return": float(metrics_map.get("total_return", 0.0)),
            "benchmark_total_return": float(metrics_map.get("benchmark_total_return", 0.0)),
            "sharpe": float(metrics_map.get("sharpe", 0.0)),
            "max_drawdown": float(metrics_map.get("max_drawdown", 0.0)),
            "output_dir": output_dir,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
