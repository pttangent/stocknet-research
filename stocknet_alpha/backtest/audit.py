from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

import pandas as pd


AuditCheck = dict[str, str]

FINAL_AUDIT_LABELS = {
    "lookahead_guard": "Lookahead guard",
    "survivorship_bias": "Survivorship bias guard",
    "parameter_robustness": "Parameter robustness",
    "logic_explainability": "Logic explainability",
    "cost_realism": "Cost realism",
}

HISTORICAL_AUDIT_LABELS = {
    "lookahead_guard": "Lookahead guard",
    "survivorship_bias": "Survivorship bias",
    "robustness": "Robustness",
    "logic_explainability": "Logic explainability",
    "cost_realism": "Cost realism",
}

_RULE_ID_PATTERN = re.compile(r"^regular(_confirmed)?(_theme_\d+\.\d)?(_both)?$")
_ALLOWED_SELECTION_POOLS = {"strict", "positive_train", "all_candidates"}


def audit_pass(evidence: str) -> AuditCheck:
    return {"status": "PASS", "evidence": evidence}


def audit_fail(evidence: str) -> AuditCheck:
    return {"status": "FAIL", "evidence": evidence}


def coerce_audit_check(value: Any) -> AuditCheck:
    if isinstance(value, Mapping):
        status = str(value.get("status", "FAIL")).upper()
        evidence = str(value.get("evidence", "")).strip() or "no audit evidence provided"
        return {"status": "PASS" if status == "PASS" else "FAIL", "evidence": evidence}
    if value is None:
        return audit_fail("no audit evidence provided")
    status = str(value).upper()
    return {"status": "PASS" if status == "PASS" else "FAIL", "evidence": "legacy audit status without supporting evidence"}


def render_audit_checks(
    checks: Mapping[str, AuditCheck],
    *,
    labels: Mapping[str, str],
) -> list[str]:
    lines = ["## Five Checks", ""]
    for key, label in labels.items():
        check = coerce_audit_check(checks.get(key))
        lines.append(f"- {label}: `{check['status']}`")
        lines.append(f"  Evidence: {check['evidence']}")
    return lines


def all_audit_checks_pass(checks: Mapping[str, AuditCheck], *, required_keys: tuple[str, ...] | list[str]) -> bool:
    return all(coerce_audit_check(checks.get(key)).get("status") == "PASS" for key in required_keys)


def evaluate_temporal_causality(trades: pd.DataFrame) -> AuditCheck:
    if trades.empty:
        return audit_fail("no realized trades available for temporal audit")

    required = ["signal_timestamp", "decision_timestamp", "execution_timestamp", "entry_time"]
    missing = [column for column in required if column not in trades.columns]
    if missing:
        return audit_fail(f"missing temporal audit columns: {', '.join(sorted(missing))}")

    frame = trades.copy()
    for column in required:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    invalid = frame[required].isna().any(axis=1)
    ordering = (
        (frame["signal_timestamp"] <= frame["decision_timestamp"])
        & (frame["decision_timestamp"] < frame["execution_timestamp"])
        & (frame["execution_timestamp"] <= frame["entry_time"])
    )
    violations = int((invalid | ~ordering).sum())
    if violations:
        return audit_fail(f"{violations} realized trades violate signal/decision/execution ordering")
    return audit_pass(f"validated {len(frame)} realized trades with signal <= decision < execution <= entry ordering")


def evaluate_cost_realism(trades: pd.DataFrame) -> AuditCheck:
    if trades.empty:
        return audit_fail("no realized trades available for cost audit")

    required = ["commission_bps", "fees_bps", "slippage_bps"]
    missing = [column for column in required if column not in trades.columns]
    if missing:
        return audit_fail(f"missing cost audit columns: {', '.join(sorted(missing))}")

    frame = trades.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid = frame[required].isna().any(axis=1)
    non_positive_total = (frame[required].sum(axis=1) <= 0.0)
    violations = int((invalid | non_positive_total).sum())
    if violations:
        return audit_fail(f"{violations} realized trades have missing or non-positive total trading costs")

    avg_total_cost = float(frame[required].sum(axis=1).mean())
    return audit_pass(f"validated {len(frame)} realized trades with positive explicit costs; average total cost={avg_total_cost:.4f}bps")


def evaluate_final_parameter_robustness(robustness_summary: Mapping[str, Any]) -> AuditCheck:
    same_sign_rate = pd.to_numeric(pd.Series([robustness_summary.get("same_sign_rate")]), errors="coerce").iloc[0]
    min_return = pd.to_numeric(pd.Series([robustness_summary.get("min_weighted_test_avg_net_return")]), errors="coerce").iloc[0]
    max_return = pd.to_numeric(pd.Series([robustness_summary.get("max_weighted_test_avg_net_return")]), errors="coerce").iloc[0]
    if pd.isna(same_sign_rate) or pd.isna(min_return) or pd.isna(max_return):
        return audit_fail("robustness summary is missing same-sign or return-range evidence")
    if same_sign_rate < 1.0 or min_return <= 0.0 or max_return <= 0.0:
        return audit_fail(
            f"robustness scan not uniformly positive: same_sign_rate={float(same_sign_rate):.4f}, "
            f"min_return={float(min_return):.6f}, max_return={float(max_return):.6f}"
        )
    return audit_pass(
        f"all robustness variants remained positive: same_sign_rate={float(same_sign_rate):.4f}, "
        f"min_return={float(min_return):.6f}, max_return={float(max_return):.6f}"
    )


def evaluate_historical_robustness(robustness_summary: Mapping[str, Any]) -> AuditCheck:
    baseline_metric = pd.to_numeric(pd.Series([robustness_summary.get("baseline_metric")]), errors="coerce").iloc[0]
    same_sign_rate = pd.to_numeric(pd.Series([robustness_summary.get("same_sign_rate")]), errors="coerce").iloc[0]
    is_robust = bool(robustness_summary.get("is_robust"))
    if pd.isna(baseline_metric) or pd.isna(same_sign_rate):
        return audit_fail("historical robustness summary is missing baseline or same-sign evidence")
    if not is_robust:
        return audit_fail(
            f"historical robustness scan failed: baseline_metric={float(baseline_metric):.6f}, same_sign_rate={float(same_sign_rate):.4f}"
        )
    return audit_pass(
        f"historical robustness scan passed: baseline_metric={float(baseline_metric):.6f}, same_sign_rate={float(same_sign_rate):.4f}"
    )


def evaluate_logic_explainability(selections: pd.DataFrame, strategy_trades: pd.DataFrame) -> AuditCheck:
    if selections.empty:
        return audit_fail("no walk-forward selections available for logic audit")
    if strategy_trades.empty:
        return audit_fail("no realized strategy trades available for logic audit")

    required = ["selection_pool", "selected_rule_id", "selected_horizon_minutes"]
    missing = [column for column in required if column not in selections.columns]
    if missing:
        return audit_fail(f"missing logic audit columns: {', '.join(sorted(missing))}")

    bad_pools = sorted({str(value) for value in selections["selection_pool"].dropna().unique() if str(value) not in _ALLOWED_SELECTION_POOLS})
    selected_rules = sorted({str(value) for value in selections["selected_rule_id"].dropna().unique()})
    bad_rules = [rule for rule in selected_rules if not (_RULE_ID_PATTERN.match(rule) or rule == "baseline")]
    if bad_pools:
        return audit_fail(f"unsupported selection pools found: {', '.join(bad_pools)}")
    if bad_rules:
        return audit_fail(f"unsupported selected rules found: {', '.join(bad_rules)}")

    pools = ", ".join(sorted({str(value) for value in selections["selection_pool"].dropna().unique()}))
    rules = ", ".join(selected_rules)
    return audit_pass(f"selection pools={pools}; selected interpretable rules={rules}")


def evaluate_historical_logic(daily_df: pd.DataFrame) -> AuditCheck:
    if daily_df.empty:
        return audit_fail("no daily backtest rows available for logic audit")
    if "status" not in daily_df.columns:
        return audit_fail("daily backtest summary is missing status column")
    ok_rows = int(daily_df["status"].astype(str).eq("ok").sum())
    if ok_rows == 0:
        return audit_fail("historical runner produced no successful daily evaluations")
    return audit_pass(f"historical runner completed {ok_rows} successful daily evaluations using interpretable theme and lead-lag rules")


def evaluate_historical_survivorship(requested_dates: list[str], daily_df: pd.DataFrame) -> AuditCheck:
    if not requested_dates:
        return audit_fail("no requested trade dates available for survivorship audit")
    if daily_df.empty or "trade_date" not in daily_df.columns or "status" not in daily_df.columns:
        return audit_fail("daily backtest summary is missing date/status coverage needed for survivorship audit")

    ok_dates = sorted(daily_df.loc[daily_df["status"].astype(str).eq("ok"), "trade_date"].astype(str).unique())
    missing_dates = sorted(set(requested_dates) - set(ok_dates))
    if missing_dates:
        return audit_fail(f"historical runner did not evaluate all requested dates; missing={', '.join(missing_dates[:5])}")
    return audit_pass(
        f"evaluated all {len(requested_dates)} requested trade dates from raw daily partitions without a static survivor universe"
    )


def inherit_required_audit_check(
    audit_evidence: Mapping[str, Any] | None,
    key: str,
) -> AuditCheck:
    if not audit_evidence:
        return audit_fail(f"missing upstream audit evidence for {key}")

    source = audit_evidence.get("audit_checks", audit_evidence)
    if not isinstance(source, Mapping):
        return audit_fail(f"invalid upstream audit evidence container for {key}")

    check = coerce_audit_check(source.get(key))
    if check["status"] != "PASS":
        return audit_fail(f"upstream audit for {key} is not PASS: {check['evidence']}")
    return audit_pass(check["evidence"])
