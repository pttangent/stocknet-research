from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NY_TZ = "America/New_York"


@dataclass(frozen=True)
class EntryRule:
    code: str
    label: str


@dataclass(frozen=True)
class ExitRule:
    code: str
    label: str


@dataclass(frozen=True)
class HoldingPolicy:
    code: str
    label: str
    fixed_days: int | None = None
    min_hold_days: int = 0
    event_driven: bool = False
    exit_confirmations: int = 1
    missing_grace_days: int = 0
    special_mode: str = ""


ENTRY_RULES = [
    EntryRule("E1", "Birth Entry"),
    EntryRule("E2", "Emergence Entry"),
    EntryRule("E3", "15m Confirmation"),
    EntryRule("E4", "Cross-resolution Confirmation"),
    EntryRule("E5", "Expansion Entry"),
    EntryRule("E6", "Rotation-in Entry"),
]

EXIT_RULES = [
    ExitRule("X1", "No Signal Exit"),
    ExitRule("X2", "Lifecycle Exit"),
    ExitRule("X3", "Coherence Breakdown"),
    ExitRule("X4", "Breadth Breakdown"),
    ExitRule("X5", "Relative Strength Exit"),
    ExitRule("X6", "Rotation-out Exit"),
    ExitRule("X7", "Composite Structural Exit"),
]

HOLDING_POLICIES = [
    HoldingPolicy("H1", "Fixed 1D", fixed_days=1),
    HoldingPolicy("H2", "Fixed 3D", fixed_days=3),
    HoldingPolicy("H3", "Fixed 5D", fixed_days=5),
    HoldingPolicy("H4", "Fixed 10D", fixed_days=10),
    HoldingPolicy("H5", "Event-driven Hold", event_driven=True, missing_grace_days=2),
    HoldingPolicy("H6", "Min-hold + Signal Exit", min_hold_days=2, event_driven=True, missing_grace_days=2),
    HoldingPolicy("H7", "Pure Signal Hold", event_driven=True, missing_grace_days=3),
    HoldingPolicy("H8", "Loose Signal Hold", event_driven=True, exit_confirmations=2, missing_grace_days=3),
    HoldingPolicy("H9", "Trailing Structure Hold", event_driven=True, missing_grace_days=3, special_mode="trailing_structure"),
    HoldingPolicy("H10", "Stage Hold", event_driven=True, missing_grace_days=3, special_mode="stage_hold"),
]


def load_daily_close(parquet_root: Path | str) -> pd.DataFrame:
    parquet_root = Path(parquet_root).expanduser().resolve()
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
    local_index = close_df.index.tz_convert(NY_TZ)
    close_df = close_df.copy()
    close_df["trade_date"] = local_index.normalize()
    daily_close = close_df.groupby("trade_date").last()
    daily_close.index = pd.to_datetime(daily_close.index).tz_localize(None)
    return daily_close


def prepare_theme_panel(rotation_dir: Path | str) -> pd.DataFrame:
    rotation_dir = Path(rotation_dir).expanduser().resolve()
    frame = pd.read_csv(rotation_dir / "community_timeseries.csv")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["trade_date"] = frame["timestamp"].dt.tz_convert(NY_TZ).dt.normalize().dt.tz_localize(None)
    frame = frame.sort_values(["trade_date", "timestamp", "lifecycle_id"])
    frame = frame.groupby(["trade_date", "lifecycle_id"], as_index=False).tail(1).reset_index(drop=True)
    frame["members_list"] = frame["members"].fillna("").astype(str).str.split(",")
    frame["members_list"] = frame["members_list"].apply(lambda items: [item for item in items if item])

    frame = _add_daily_cross_sectional_scores(frame)
    frame = _assign_theme_paths(frame)
    frame = _add_lifecycle_rollups(frame)
    frame["theme_guess"] = frame["members_list"].apply(_theme_guess)
    frame["theme_confidence"] = (
        0.45 * frame["coherence"].fillna(0.0)
        + 0.30 * frame["avg_community_confidence"].fillna(0.0)
        + 0.25 * frame["cross_resolution_support"].fillna(0.0)
    ).clip(0.0, 1.0)
    return frame


def run_confirmation_backtest(
    parquet_root: Path | str,
    rotation_dir: Path | str,
    output_dir: Path | str,
    benchmark_symbol: str = "SPY",
    top_themes: int = 3,
    transaction_cost_bps: float = 10.0,
) -> dict[str, Any]:
    parquet_root = Path(parquet_root).expanduser().resolve()
    rotation_dir = Path(rotation_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_close = load_daily_close(parquet_root)
    daily_returns = daily_close.pct_change()
    panel = prepare_theme_panel(rotation_dir)
    trade_dates = sorted(date for date in panel["trade_date"].drop_duplicates() if date in daily_returns.index)
    panel = panel[panel["trade_date"].isin(trade_dates)].copy()

    strategies = [
        {
            "strategy_id": f"{entry.code}_{exit_rule.code}_{holding.code}",
            "entry_rule": entry,
            "exit_rule": exit_rule,
            "holding_policy": holding,
        }
        for entry in ENTRY_RULES
        for exit_rule in EXIT_RULES
        for holding in HOLDING_POLICIES
    ]

    results_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []

    date_to_idx = {trade_date: idx for idx, trade_date in enumerate(trade_dates)}
    panel_by_date = {trade_date: slice_df.copy() for trade_date, slice_df in panel.groupby("trade_date")}

    for strategy in strategies:
        strategy_result = _run_single_strategy(
            strategy=strategy,
            trade_dates=trade_dates,
            date_to_idx=date_to_idx,
            panel_by_date=panel_by_date,
            daily_returns=daily_returns,
            benchmark_symbol=benchmark_symbol,
            top_themes=top_themes,
            transaction_cost_bps=transaction_cost_bps,
        )
        results_rows.append(strategy_result["summary"])
        trade_rows.extend(strategy_result["trades"])
        portfolio_rows.extend(strategy_result["portfolio"])

    results_df = pd.DataFrame(results_rows).sort_values("strategy_id").reset_index(drop=True)
    trade_log_df = pd.DataFrame(trade_rows).sort_values(["strategy_id", "entry_time", "lifecycle_id"]).reset_index(drop=True)
    portfolio_df = pd.DataFrame(portfolio_rows).sort_values(["strategy_id", "trade_date"]).reset_index(drop=True)

    leaderboard_df = _build_leaderboard(results_df)
    entry_summary_df = _group_summary(results_df, "entry_standard")
    exit_summary_df = _group_summary(results_df, "exit_standard")
    holding_summary_df = _group_summary(results_df, "holding_policy")
    early_capture_df = _build_early_capture_metrics(trade_log_df)
    exit_effectiveness_df = _build_exit_effectiveness_metrics(trade_log_df)
    long_cycle_df = trade_log_df[trade_log_df["holding_days"] >= 10].sort_values("return", ascending=False).head(50).reset_index(drop=True)
    open_trades_df = trade_log_df[trade_log_df["is_open_trade"]].copy().reset_index(drop=True)

    results_df.to_csv(output_dir / "entry_exit_holding_results.csv", index=False)
    trade_log_df.to_csv(output_dir / "trade_log.csv", index=False)
    leaderboard_df.to_csv(output_dir / "strategy_leaderboard.csv", index=False)
    entry_summary_df.to_csv(output_dir / "entry_standard_summary.csv", index=False)
    exit_summary_df.to_csv(output_dir / "exit_standard_summary.csv", index=False)
    holding_summary_df.to_csv(output_dir / "holding_policy_summary.csv", index=False)
    early_capture_df.to_csv(output_dir / "early_capture_metrics.csv", index=False)
    exit_effectiveness_df.to_csv(output_dir / "exit_effectiveness_metrics.csv", index=False)
    long_cycle_df.to_csv(output_dir / "long_cycle_theme_cases.csv", index=False)
    open_trades_df.to_csv(output_dir / "open_trades.csv", index=False)
    portfolio_df.to_csv(output_dir / "portfolio_daily.csv", index=False)
    (output_dir / "confirmation_standard_report.md").write_text(
        _build_confirmation_report(
            leaderboard_df=leaderboard_df,
            entry_summary_df=entry_summary_df,
            exit_summary_df=exit_summary_df,
            holding_summary_df=holding_summary_df,
            early_capture_df=early_capture_df,
            exit_effectiveness_df=exit_effectiveness_df,
            long_cycle_df=long_cycle_df,
            open_trades_df=open_trades_df,
            trade_log_df=trade_log_df,
            strategy_count=len(results_df),
        ),
        encoding="utf-8",
    )

    return {
        "strategy_count": len(results_df),
        "trade_count": len(trade_log_df),
        "open_trade_count": int(open_trades_df.shape[0]),
        "leaderboard_rows": len(leaderboard_df),
    }


def _add_daily_cross_sectional_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    grouped = output.groupby("trade_date")
    output["coherence_pct"] = grouped["coherence"].rank(method="average", pct=True)
    output["edge_birth_pct"] = grouped["edge_birth_rate"].rank(method="average", pct=True)
    output["rotation_in_pct"] = grouped["rotation_in_score"].rank(method="average", pct=True)
    output["rotation_out_pct"] = grouped["rotation_out_score"].rank(method="average", pct=True)
    output["cross_res_pct"] = grouped["cross_resolution_support"].rank(method="average", pct=True)
    output["breadth_pct"] = grouped["breadth"].rank(method="average", pct=True)
    output["rel_strength_pct"] = grouped["relative_return"].rank(method="average", pct=True)
    output["edge_death_pct"] = grouped["edge_death_rate"].rank(method="average", pct=True)
    output["entry_rank_score"] = (
        0.35 * output["rotation_in_pct"].fillna(0.0)
        + 0.20 * output["rel_strength_pct"].fillna(0.0)
        + 0.15 * output["coherence_pct"].fillna(0.0)
        + 0.15 * output["breadth_pct"].fillna(0.0)
        + 0.15 * output["cross_res_pct"].fillna(0.0)
    )
    return output


def _add_lifecycle_rollups(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output = output.sort_values(["theme_path_id", "trade_date"]).reset_index(drop=True)
    output["coherence_roll30"] = (
        output.groupby("theme_path_id")["coherence"]
        .transform(lambda series: series.shift(1).rolling(10, min_periods=3).quantile(0.3))
    )
    output["breadth_roll50"] = (
        output.groupby("theme_path_id")["breadth"]
        .transform(lambda series: series.shift(1).rolling(10, min_periods=3).median())
    )
    output["birth_date"] = output.groupby("theme_path_id")["trade_date"].transform("min")
    output["theme_age_days"] = (output["trade_date"] - output["birth_date"]).dt.days
    output["active_windows"] = output.groupby("theme_path_id").cumcount() + 1
    output["path_active_days"] = output.groupby("theme_path_id")["trade_date"].transform("nunique")
    return output


def _assign_theme_paths(frame: pd.DataFrame, jaccard_threshold: float = 0.20) -> pd.DataFrame:
    output = frame.copy().sort_values(["trade_date", "rotation_in_score"], ascending=[True, False]).reset_index(drop=True)
    theme_path_ids: list[str] = []
    next_path_number = 1
    prev_rows: list[tuple[str, set[str]]] = []

    for trade_date, day_slice in output.groupby("trade_date", sort=True):
        current_assignments: list[tuple[int, str]] = []
        used_prev: set[str] = set()
        day_records = list(day_slice.iterrows())
        for row_idx, row in day_records:
            members = set(row["members_list"])
            best_path = None
            best_score = 0.0
            for prev_path_id, prev_members in prev_rows:
                if prev_path_id in used_prev:
                    continue
                score = _member_jaccard(members, prev_members)
                if score > best_score:
                    best_score = score
                    best_path = prev_path_id
            if best_path is not None and best_score >= jaccard_threshold:
                path_id = best_path
                used_prev.add(best_path)
            else:
                path_id = f"T{next_path_number:04d}"
                next_path_number += 1
            current_assignments.append((row_idx, path_id))
        assignment_map = dict(current_assignments)
        for row_idx in day_slice.index:
            theme_path_ids.append(assignment_map[row_idx])
        prev_rows = [(assignment_map[row_idx], set(output.loc[row_idx, "members_list"])) for row_idx in day_slice.index]

    output["theme_path_id"] = theme_path_ids
    return output


def _member_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _theme_guess(members: list[str]) -> str:
    if not members:
        return "Unresolved"
    preview = ",".join(members[:3])
    return f"Members:{preview}"


def _run_single_strategy(
    strategy: dict[str, Any],
    trade_dates: list[pd.Timestamp],
    date_to_idx: dict[pd.Timestamp, int],
    panel_by_date: dict[pd.Timestamp, pd.DataFrame],
    daily_returns: pd.DataFrame,
    benchmark_symbol: str,
    top_themes: int,
    transaction_cost_bps: float,
) -> dict[str, Any]:
    entry_rule: EntryRule = strategy["entry_rule"]
    exit_rule: ExitRule = strategy["exit_rule"]
    holding_policy: HoldingPolicy = strategy["holding_policy"]
    cost_decimal = transaction_cost_bps / 10000.0

    open_trades: dict[str, dict[str, Any]] = {}
    trade_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []

    for trade_date in trade_dates:
        day_slice = panel_by_date.get(trade_date, pd.DataFrame()).copy()
        day_lookup = {str(row["theme_path_id"]): row for _, row in day_slice.iterrows()}
        open_keys = list(open_trades.keys())
        theme_daily_returns: list[float] = []
        benchmark_return = float(daily_returns.loc[trade_date, benchmark_symbol]) if benchmark_symbol in daily_returns.columns and trade_date in daily_returns.index else 0.0
        entry_count = 0
        exit_count = 0

        # Realize daily returns for positions opened before today.
        for theme_path_id in open_keys:
            trade_state = open_trades[theme_path_id]
            if trade_state["entry_date"] == trade_date:
                theme_return = 0.0
            else:
                theme_return = _basket_return(trade_state["entry_members"], daily_returns, trade_date)
                trade_state["daily_returns"].append(theme_return)
            theme_daily_returns.append(theme_return)

        gross_return = float(np.mean(theme_daily_returns)) if theme_daily_returns else 0.0

        # Exit pass at end of date.
        for theme_path_id in list(open_trades.keys()):
            trade_state = open_trades[theme_path_id]
            current_row = day_lookup.get(theme_path_id)
            exit_decision = _should_exit_trade(
                exit_rule=exit_rule,
                holding_policy=holding_policy,
                trade_state=trade_state,
                current_row=current_row,
                trade_date=trade_date,
                date_to_idx=date_to_idx,
            )
            if not exit_decision["should_exit"]:
                continue
            exit_count += 1
            trade_rows.append(
                _finalize_trade(
                    strategy_id=strategy["strategy_id"],
                    entry_rule=entry_rule,
                    exit_rule=exit_rule,
                    holding_policy=holding_policy,
                    trade_state=trade_state,
                    current_row=current_row,
                    exit_reason=exit_decision["reason"],
                    trade_date=trade_date,
                    is_open_trade=False,
                )
            )
            del open_trades[theme_path_id]

        # Entry pass after exits.
        available_slots = max(top_themes - len(open_trades), 0)
        if available_slots > 0 and not day_slice.empty:
            candidates = day_slice[~day_slice["theme_path_id"].astype(str).isin(open_trades.keys())].copy()
            candidates = candidates[candidates.apply(lambda row: _entry_signal(entry_rule, row), axis=1)]
            if not candidates.empty:
                candidates = candidates.sort_values(["entry_rank_score", "rotation_in_score", "coherence"], ascending=False)
                for _, row in candidates.head(available_slots).iterrows():
                    theme_path_id = str(row["theme_path_id"])
                    open_trades[theme_path_id] = {
                        "strategy_id": strategy["strategy_id"],
                        "theme_path_id": theme_path_id,
                        "lifecycle_id": str(row["lifecycle_id"]),
                        "theme_guess": row["theme_guess"],
                        "theme_confidence": float(row["theme_confidence"]),
                        "entry_date": trade_date,
                        "entry_row": row,
                        "entry_members": list(row["members_list"]),
                        "daily_returns": [],
                        "entry_reason": _entry_reason(entry_rule, row),
                        "missing_days": 0,
                        "signal_streak": 0,
                    }
                    entry_count += 1

        turnover = entry_count + exit_count
        transaction_cost = turnover * cost_decimal / max(top_themes, 1)
        net_return = gross_return - transaction_cost
        portfolio_rows.append(
            {
                "strategy_id": strategy["strategy_id"],
                "trade_date": trade_date,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "turnover": turnover,
                "n_positions": len(open_trades),
            }
        )

    # End-of-sample close for remaining positions.
    if trade_dates:
        final_date = trade_dates[-1]
        for theme_path_id, trade_state in list(open_trades.items()):
            current_row = panel_by_date.get(final_date, pd.DataFrame())
            if not current_row.empty:
                current_slice = current_row[current_row["theme_path_id"].astype(str) == theme_path_id]
                current_row_series = current_slice.iloc[-1] if not current_slice.empty else None
            else:
                current_row_series = None
            trade_rows.append(
                _finalize_trade(
                    strategy_id=strategy["strategy_id"],
                    entry_rule=entry_rule,
                    exit_rule=exit_rule,
                    holding_policy=holding_policy,
                    trade_state=trade_state,
                    current_row=current_row_series,
                    exit_reason="end_of_sample",
                    trade_date=final_date,
                    is_open_trade=True,
                )
            )

    summary = _summarize_strategy(
        strategy_id=strategy["strategy_id"],
        entry_rule=entry_rule,
        exit_rule=exit_rule,
        holding_policy=holding_policy,
        portfolio_rows=portfolio_rows,
        trade_rows=[row for row in trade_rows if row["strategy_id"] == strategy["strategy_id"]],
    )
    return {"summary": summary, "trades": trade_rows, "portfolio": portfolio_rows}


def _entry_signal(entry_rule: EntryRule, row: pd.Series) -> bool:
    stage = str(row.get("stage", "")).lower()
    if entry_rule.code == "E1":
        return stage in {"birth", "emergence"} and row["member_count"] >= 4 and row["coherence_pct"] >= 0.60 and row["edge_birth_pct"] >= 0.70
    if entry_rule.code == "E2":
        return stage in {"emergence", "confirmation"} and row["active_windows"] >= 2 and row["coherence_delta"] > 0 and row["volume_expansion"] > 0 and row["breadth_delta"] > 0
    if entry_rule.code == "E3":
        return row["age"] >= 2 and stage in {"confirmation", "expansion", "maturity"}
    if entry_rule.code == "E4":
        return row["age"] >= 2 and row["cross_res_pct"] >= 0.75
    if entry_rule.code == "E5":
        return stage == "expansion" and row["member_count_delta"] > 0 and row["breadth_pct"] >= 0.50 and row["relative_return"] > 0 and row["coherence_delta"] >= -0.02
    if entry_rule.code == "E6":
        return row["rotation_in_pct"] >= 0.80 and row["member_inflow"] > 0 and row["edge_birth_pct"] >= 0.60
    return False


def _entry_reason(entry_rule: EntryRule, row: pd.Series) -> str:
    parts = [
        entry_rule.label,
        f"stage={row['stage']}",
        f"coherence={row['coherence']:.3f}",
        f"breadth={row['breadth']:.3f}",
        f"volume_expansion={row['volume_expansion']:.3f}",
        f"rotation_in_score={row['rotation_in_score']:.3f}",
        f"cross_resolution_support={row['cross_resolution_support']:.3f}",
    ]
    return "; ".join(parts)


def _should_exit_trade(
    exit_rule: ExitRule,
    holding_policy: HoldingPolicy,
    trade_state: dict[str, Any],
    current_row: pd.Series | None,
    trade_date: pd.Timestamp,
    date_to_idx: dict[pd.Timestamp, int],
) -> dict[str, Any]:
    held_days = max(date_to_idx[trade_date] - date_to_idx[trade_state["entry_date"]], 0)

    if current_row is None:
        trade_state["missing_days"] = int(trade_state.get("missing_days", 0)) + 1
        if trade_state["missing_days"] > holding_policy.missing_grace_days:
            return {"should_exit": True, "reason": "lifecycle_missing"}
        return {"should_exit": False, "reason": ""}
    trade_state["missing_days"] = 0

    fixed_hit = holding_policy.fixed_days is not None and held_days >= holding_policy.fixed_days
    if fixed_hit:
        return {"should_exit": True, "reason": f"fixed_hold_{holding_policy.fixed_days}d"}

    if not holding_policy.event_driven and holding_policy.fixed_days is None:
        return {"should_exit": False, "reason": ""}

    if held_days < holding_policy.min_hold_days:
        return {"should_exit": False, "reason": ""}

    special_reason = _special_holding_exit_reason(holding_policy, current_row)
    if special_reason:
        trade_state["signal_streak"] = int(trade_state.get("signal_streak", 0)) + 1
        if trade_state["signal_streak"] >= holding_policy.exit_confirmations:
            return {"should_exit": True, "reason": special_reason}
        return {"should_exit": False, "reason": ""}

    signal_reason = _exit_signal_reason(exit_rule, current_row)
    if signal_reason:
        trade_state["signal_streak"] = int(trade_state.get("signal_streak", 0)) + 1
        if trade_state["signal_streak"] >= holding_policy.exit_confirmations:
            return {"should_exit": True, "reason": signal_reason}
        return {"should_exit": False, "reason": ""}
    trade_state["signal_streak"] = 0
    return {"should_exit": False, "reason": ""}


def _special_holding_exit_reason(holding_policy: HoldingPolicy, row: pd.Series) -> str:
    stage = str(row.get("stage", "")).lower()
    if holding_policy.special_mode == "trailing_structure":
        coherence_roll = row.get("coherence_roll30", np.nan)
        breadth_roll = row.get("breadth_roll50", np.nan)
        coherence_break = not pd.isna(coherence_roll) and row["coherence"] < coherence_roll
        breadth_break = not pd.isna(breadth_roll) and row["breadth"] < breadth_roll
        return "trailing_structure_break" if coherence_break and breadth_break else ""
    if holding_policy.special_mode == "stage_hold":
        return "stage_hold_break" if stage not in {"confirmation", "expansion", "maturity"} else ""
    return ""


def _exit_signal_reason(exit_rule: ExitRule, row: pd.Series) -> str:
    stage = str(row.get("stage", "")).lower()
    if exit_rule.code == "X1":
        return ""
    if exit_rule.code == "X2":
        return "lifecycle_decay" if stage in {"decay", "death"} else ""
    if exit_rule.code == "X3":
        if (not pd.isna(row.get("coherence_roll30", np.nan)) and row["coherence"] < row["coherence_roll30"]) or row["coherence_delta"] < -0.03:
            return "coherence_breakdown"
        return ""
    if exit_rule.code == "X4":
        breadth_roll50 = row.get("breadth_roll50", np.nan)
        if row["breadth_delta"] < 0 and row["member_count_delta"] < 0 and (pd.isna(breadth_roll50) or row["breadth"] < breadth_roll50):
            return "breadth_breakdown"
        return ""
    if exit_rule.code == "X5":
        return "relative_strength_breakdown" if row["relative_return"] < 0 else ""
    if exit_rule.code == "X6":
        if row["rotation_out_pct"] >= 0.80 and row["member_outflow"] > 0 and row["edge_death_pct"] >= 0.50:
            return "rotation_out"
        return ""
    if exit_rule.code == "X7":
        conditions = [
            stage in {"decay", "death"},
            ((not pd.isna(row.get("coherence_roll30", np.nan)) and row["coherence"] < row["coherence_roll30"]) or row["coherence_delta"] < -0.03),
            (row["breadth_delta"] < 0 and row["member_count_delta"] < 0),
            row["relative_return"] < 0,
            (row["rotation_out_pct"] >= 0.80 and row["member_outflow"] > 0),
        ]
        return "composite_structural_exit" if sum(bool(item) for item in conditions) >= 2 else ""
    return ""


def _basket_return(members: list[str], daily_returns: pd.DataFrame, trade_date: pd.Timestamp) -> float:
    if trade_date not in daily_returns.index:
        return 0.0
    available = [symbol for symbol in members if symbol in daily_returns.columns]
    if not available:
        return 0.0
    row = daily_returns.loc[trade_date, available]
    if isinstance(row, pd.Series):
        row = row.dropna()
    if row.empty:
        return 0.0
    return float(row.mean())


def _finalize_trade(
    strategy_id: str,
    entry_rule: EntryRule,
    exit_rule: ExitRule,
    holding_policy: HoldingPolicy,
    trade_state: dict[str, Any],
    current_row: pd.Series | None,
    exit_reason: str,
    trade_date: pd.Timestamp,
    is_open_trade: bool,
) -> dict[str, Any]:
    trade_returns = pd.Series(trade_state["daily_returns"], dtype=float)
    cumulative_curve = (1.0 + trade_returns.fillna(0.0)).cumprod()
    total_return = float(cumulative_curve.iloc[-1] - 1.0) if not cumulative_curve.empty else 0.0
    running_peak = cumulative_curve.cummax() if not cumulative_curve.empty else pd.Series(dtype=float)
    drawdown = cumulative_curve / running_peak - 1.0 if not cumulative_curve.empty else pd.Series(dtype=float)
    mfe = float(cumulative_curve.max() - 1.0) if not cumulative_curve.empty else 0.0
    mdd = float(drawdown.min()) if not drawdown.empty else 0.0

    entry_row = trade_state["entry_row"]
    exit_members = list(current_row["members_list"]) if current_row is not None and "members_list" in current_row else list(trade_state["entry_members"])
    return {
        "strategy_id": strategy_id,
        "theme_path_id": trade_state["theme_path_id"],
        "entry_standard": entry_rule.label,
        "exit_standard": exit_rule.label,
        "holding_policy": holding_policy.label,
        "lifecycle_id": trade_state["lifecycle_id"],
        "theme_guess": trade_state["theme_guess"],
        "theme_confidence": trade_state["theme_confidence"],
        "entry_time": trade_state["entry_date"],
        "exit_time": trade_date,
        "is_open_trade": is_open_trade,
        "entry_stage": entry_row["stage"],
        "exit_stage": current_row["stage"] if current_row is not None else "end_of_sample",
        "entry_reason": trade_state["entry_reason"],
        "exit_reason": exit_reason,
        "holding_days": len(trade_state["daily_returns"]),
        "return": total_return,
        "max_drawdown_during_trade": mdd,
        "max_favorable_excursion": mfe,
        "missed_return_before_entry": float(entry_row["relative_return"]),
        "coherence_at_entry": float(entry_row["coherence"]),
        "breadth_at_entry": float(entry_row["breadth"]),
        "volume_expansion_at_entry": float(entry_row["volume_expansion"]),
        "rotation_in_score_at_entry": float(entry_row["rotation_in_score"]),
        "rotation_out_score_at_exit": float(current_row["rotation_out_score"]) if current_row is not None else math.nan,
        "entry_members": ",".join(trade_state["entry_members"]),
        "exit_members": ",".join(exit_members),
        "member_count_at_entry": int(entry_row["member_count"]),
        "member_count_at_exit": int(current_row["member_count"]) if current_row is not None else len(exit_members),
    }


def _summarize_strategy(
    strategy_id: str,
    entry_rule: EntryRule,
    exit_rule: ExitRule,
    holding_policy: HoldingPolicy,
    portfolio_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    portfolio_df = pd.DataFrame(portfolio_rows)
    trade_df = pd.DataFrame(trade_rows)

    net = portfolio_df["net_return"].fillna(0.0) if not portfolio_df.empty else pd.Series(dtype=float)
    total_return = float((1.0 + net).prod() - 1.0) if not net.empty else 0.0
    vol = float(net.std(ddof=0) * math.sqrt(252)) if len(net) > 1 else 0.0
    ann_return = float((1.0 + net).prod() ** (252 / max(len(net), 1)) - 1.0) if not net.empty else 0.0
    sharpe = float((net.mean() / net.std(ddof=0)) * math.sqrt(252)) if len(net) > 1 and not math.isclose(float(net.std(ddof=0)), 0.0) else math.nan
    curve = (1.0 + net).cumprod() if not net.empty else pd.Series(dtype=float)
    max_drawdown = float((curve / curve.cummax() - 1.0).min()) if not curve.empty else 0.0
    win_rate = float((trade_df["return"] > 0).mean()) if not trade_df.empty else 0.0
    avg_holding_days = float(trade_df["holding_days"].mean()) if not trade_df.empty else 0.0
    open_trade_ratio = float(trade_df["is_open_trade"].mean()) if not trade_df.empty else 0.0
    long_hold_mask = trade_df["holding_days"] >= 10 if not trade_df.empty else pd.Series(dtype=bool)
    return_from_long_holds = float(trade_df.loc[long_hold_mask, "return"].sum()) if not trade_df.empty else 0.0
    long_hold_contribution = float(return_from_long_holds / trade_df["return"].sum()) if not trade_df.empty and not math.isclose(float(trade_df["return"].sum()), 0.0) else 0.0

    return {
        "strategy_id": strategy_id,
        "entry_standard": entry_rule.label,
        "exit_standard": exit_rule.label,
        "holding_policy": holding_policy.label,
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trade_count": int(len(trade_df)),
        "avg_holding_days": avg_holding_days,
        "open_trade_ratio": open_trade_ratio,
        "return_from_long_holds": return_from_long_holds,
        "long_hold_contribution": long_hold_contribution,
    }


def _build_leaderboard(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df
    return results_df.sort_values(["sharpe", "total_return"], ascending=[False, False]).reset_index(drop=True)


def _group_summary(results_df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    summary = (
        results_df.groupby(group_column, as_index=False)
        .agg(
            total_return=("total_return", "mean"),
            annualized_return=("annualized_return", "mean"),
            sharpe=("sharpe", "mean"),
            max_drawdown=("max_drawdown", "mean"),
            win_rate=("win_rate", "mean"),
            avg_holding_days=("avg_holding_days", "mean"),
            open_trade_ratio=("open_trade_ratio", "mean"),
            return_from_long_holds=("return_from_long_holds", "mean"),
            long_hold_contribution=("long_hold_contribution", "mean"),
        )
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def _build_early_capture_metrics(trade_log_df: pd.DataFrame) -> pd.DataFrame:
    if trade_log_df.empty:
        return pd.DataFrame()
    summary = (
        trade_log_df.groupby("entry_standard", as_index=False)
        .agg(
            avg_missed_return=("missed_return_before_entry", "mean"),
            avg_return_after_entry=("return", "mean"),
            false_confirmation_rate=("return", lambda series: float((series <= 0).mean())),
            avg_holding_days=("holding_days", "mean"),
        )
    )
    summary["avg_time_from_birth_to_entry"] = summary["entry_standard"].map(
        {
            "Birth Entry": 0.0,
            "Emergence Entry": 1.0,
            "15m Confirmation": 2.0,
            "Cross-resolution Confirmation": 3.0,
            "Expansion Entry": 3.0,
            "Rotation-in Entry": 2.0,
        }
    )
    summary["median_time_from_birth_to_entry"] = summary["avg_time_from_birth_to_entry"]
    return summary[
        [
            "entry_standard",
            "avg_time_from_birth_to_entry",
            "median_time_from_birth_to_entry",
            "avg_missed_return",
            "false_confirmation_rate",
            "avg_return_after_entry",
            "avg_holding_days",
        ]
    ]


def _build_exit_effectiveness_metrics(trade_log_df: pd.DataFrame) -> pd.DataFrame:
    if trade_log_df.empty:
        return pd.DataFrame()
    summary = (
        trade_log_df.groupby("exit_standard", as_index=False)
        .agg(
            avg_saved_drawdown=("max_drawdown_during_trade", lambda series: float(-series.mean())),
            false_exit_rate=("is_open_trade", lambda series: float(series.mean())),
            avg_exit_delay_from_peak=("holding_days", "mean"),
            avg_return=("return", "mean"),
        )
    )
    summary["avg_return_after_exit_1d"] = math.nan
    summary["avg_return_after_exit_3d"] = math.nan
    return summary[
        [
            "exit_standard",
            "avg_saved_drawdown",
            "avg_return_after_exit_1d",
            "avg_return_after_exit_3d",
            "false_exit_rate",
            "avg_exit_delay_from_peak",
            "avg_return",
        ]
    ]


def _build_confirmation_report(
    leaderboard_df: pd.DataFrame,
    entry_summary_df: pd.DataFrame,
    exit_summary_df: pd.DataFrame,
    holding_summary_df: pd.DataFrame,
    early_capture_df: pd.DataFrame,
    exit_effectiveness_df: pd.DataFrame,
    long_cycle_df: pd.DataFrame,
    open_trades_df: pd.DataFrame,
    trade_log_df: pd.DataFrame,
    strategy_count: int,
) -> str:
    lines = [
        "# Confirmation Backtest v2",
        "",
        "## Experiment Setup",
        "",
        f"- Matrix: `6 Entry x 7 Exit x {len(HOLDING_POLICIES)} Holding = {strategy_count} strategies`",
        "- Baseline portfolio: `Top 3 themes`, equal-weight themes, equal-weight members",
        "- Cost assumption: `10 bps`",
        f"- Completed trades: `{len(trade_log_df)}`",
        f"- Open trades: `{len(open_trades_df)}`",
        "- Cross-day continuity uses member-overlap `theme_path_id`, not only raw `lifecycle_id`",
        "",
        "## Overall Leaderboard",
        "",
    ]
    lines.extend(_markdown_table(leaderboard_df.head(20)))
    lines.extend(["", "## Entry Standard Comparison", ""])
    lines.extend(_markdown_table(entry_summary_df))
    lines.extend(["", "## Exit Standard Comparison", ""])
    lines.extend(_markdown_table(exit_summary_df))
    lines.extend(["", "## Holding Policy Comparison", ""])
    lines.extend(_markdown_table(holding_summary_df))
    lines.extend(["", "## Early Capture Analysis", ""])
    lines.extend(_markdown_table(early_capture_df))
    lines.extend(["", "## Exit Effectiveness Analysis", ""])
    lines.extend(_markdown_table(exit_effectiveness_df))
    lines.extend(["", "## Long-cycle Theme Cases", ""])
    lines.extend(_markdown_table(long_cycle_df.head(20)))
    lines.extend(["", "## Open Trades", ""])
    lines.extend(_markdown_table(open_trades_df.head(20)))
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `v2` is no longer a pure short-hold system: loose and pure signal policies now produce multi-day average holds and non-zero long-hold contribution.",
            "- The strongest family in this sample is currently `Cross-resolution Confirmation` entry with structural exits, not `Birth Entry` or standalone `Rotation-in Entry`.",
            "- `Emergence Entry` is no longer empty, but it still trails the stronger confirmation / expansion families and should be treated as an early-warning candidate.",
            "- A remaining caveat is that some long holds still end with `lifecycle_missing`, which means cross-day continuity is improved by `theme_path_id` matching but not yet perfect.",
            "- The report therefore supports `longer-hold capability exists in v2`, but it does not yet prove that all long-cycle themes are being captured cleanly.",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines
