from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from stocknetwork.theme_persistence import assign_theme_paths


THEME_CANDIDATE_COLUMNS = [
    "candidate_id",
    "trade_date",
    "signal_timestamp",
    "theme_path_id",
    "community_id",
    "members",
    "member_count",
    "confirmed_on_15m",
    "confirmed_by_age_3x5m",
    "confirmed_by_15m_graph",
    "confirmation_source",
    "confirmation_timestamp",
    "confirmation_match_score",
    "radar_score_5m",
    "confirmation_score_5m",
    "coherence_5m",
    "breadth_5m",
    "relative_return_5m",
    "volume_expansion_5m",
    "theme_score",
    "event_type",
    "age_bars",
]


def aggregate_trade_flow_to_5m(trade_flow_1m: pd.DataFrame) -> pd.DataFrame:
    if trade_flow_1m.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "imbalance_proxy",
                "dollar_volume",
                "buy_vol_proxy",
                "sell_vol_proxy",
                "large_trade_dollar_volume",
                "off_exchange_volume",
                "volume",
                "trade_count",
            ]
        )

    frame = trade_flow_1m.copy()
    frame["minute"] = pd.to_datetime(frame["minute"], utc=True)
    frame["symbol"] = frame["ticker"].astype(str).str.upper()
    frame["timestamp"] = frame["minute"].dt.floor("5min") + pd.Timedelta(minutes=5)
    for column in [
        "imbalance_proxy",
        "dollar_volume",
        "buy_vol_proxy",
        "sell_vol_proxy",
        "large_trade_dollar_volume",
        "off_exchange_volume",
        "volume",
        "trade_count",
    ]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    grouped = (
        frame.groupby(["symbol", "timestamp"], as_index=False)[
            [
                "imbalance_proxy",
                "dollar_volume",
                "buy_vol_proxy",
                "sell_vol_proxy",
                "large_trade_dollar_volume",
                "off_exchange_volume",
                "volume",
                "trade_count",
            ]
        ]
        .sum()
    )
    return grouped.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def build_theme_candidates_from_market_data(
    bars_5m: pd.DataFrame,
    trade_flow_1m: pd.DataFrame,
    *,
    trade_date: str,
    lookback_bars: int = 6,
    top_symbols: int = 200,
    min_members: int = 3,
    min_theme_score: float = 0.5,
    min_pair_corr: float = 0.6,
    max_members_per_candidate: int = 25,
    theme_path_score_method: str = "jaccard",
    theme_path_min_overlap: float = 0.40,
) -> pd.DataFrame:
    if bars_5m.empty:
        return pd.DataFrame(columns=THEME_CANDIDATE_COLUMNS)

    bars = bars_5m.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["symbol"] = bars["symbol"].astype(str).str.upper()
    bars["volume"] = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars["open"] = pd.to_numeric(bars["open"], errors="coerce")
    bars["vwap"] = pd.to_numeric(bars.get("vwap"), errors="coerce")
    bars["dollar_volume_5m"] = bars["close"].fillna(0.0) * bars["volume"].fillna(0.0)
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    flow_5m = aggregate_trade_flow_to_5m(trade_flow_1m)
    merged = bars.merge(flow_5m, on=["symbol", "timestamp"], how="left", suffixes=("", "_flow"))
    for column in [
        "imbalance_proxy",
        "dollar_volume",
        "buy_vol_proxy",
        "sell_vol_proxy",
        "large_trade_dollar_volume",
        "off_exchange_volume",
        "trade_count",
        "volume_flow",
    ]:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    merged["trade_date"] = merged["timestamp"].dt.date.astype(str)
    merged = merged[merged["trade_date"] == str(trade_date)].copy()
    if merged.empty:
        return pd.DataFrame(columns=THEME_CANDIDATE_COLUMNS)

    merged = _add_causal_features(merged)
    candidate_rows: list[dict[str, object]] = []

    for timestamp, current_slice in merged.groupby("timestamp", sort=True):
        eligible = current_slice.dropna(subset=["ret_5m_past"]).copy()
        if eligible.empty:
            continue
        liquid_pool_size = max(top_symbols * 5, top_symbols)
        eligible = (
            eligible.sort_values(["liquidity_score", "symbol"], ascending=[False, True])
            .head(liquid_pool_size)
            .sort_values(["seed_score", "liquidity_score", "symbol"], ascending=[False, False, True])
            .head(top_symbols)
            .copy()
        )
        if len(eligible) < min_members:
            continue

        history = merged[
            (merged["timestamp"] <= timestamp)
            & (merged["timestamp"] >= timestamp - pd.Timedelta(minutes=5 * max(lookback_bars - 1, 0)))
            & (merged["symbol"].isin(eligible["symbol"]))
        ].copy()
        if history.empty:
            continue
        pivot = history.pivot(index="timestamp", columns="symbol", values="ret_5m_past").sort_index()
        if len(pivot) < max(lookback_bars, 3):
            continue
        pivot = pivot.tail(lookback_bars)
        corr = pivot.corr(min_periods=max(3, lookback_bars // 2)).fillna(0.0)

        graph = _build_similarity_graph(eligible["symbol"].tolist(), corr, min_pair_corr=min_pair_corr)
        communities = [sorted(component) for component in _connected_components(graph) if len(component) >= min_members]
        if not communities:
            continue

        for community_id, members in enumerate(sorted(communities, key=len, reverse=True), start=1):
            if len(members) > max_members_per_candidate:
                ranked_members = (
                    eligible[eligible["symbol"].isin(members)]
                    .sort_values(["seed_score", "liquidity_score", "symbol"], ascending=[False, False, True])["symbol"]
                    .head(max_members_per_candidate)
                    .tolist()
                )
                members = sorted(ranked_members)
            member_slice = eligible[eligible["symbol"].isin(members)].copy()
            coherence = _average_pairwise_corr(corr, members)
            breadth = float((member_slice["ret_5m_past"] > 0).mean())
            relative_return = float(member_slice["ret_5m_past"].mean())
            volume_expansion = float(member_slice["volume_z_12"].mean())
            radar_score = float(member_slice["seed_score"].mean())
            confirmation_score = 0.6 * coherence + 0.4 * breadth
            theme_score = 0.4 * radar_score + 0.3 * confirmation_score + 0.3 * max(relative_return, 0.0)
            if theme_score < min_theme_score:
                continue
            candidate_rows.append(
                {
                    "trade_date": str(trade_date),
                    "signal_timestamp": timestamp,
                    "community_id": f"C{community_id:03d}",
                    "members": ",".join(members),
                    "member_count": len(members),
                    "radar_score_5m": radar_score,
                    "confirmation_score_5m": confirmation_score,
                    "coherence_5m": coherence,
                    "breadth_5m": breadth,
                    "relative_return_5m": relative_return,
                    "volume_expansion_5m": volume_expansion,
                    "theme_score": theme_score,
                }
            )

    if not candidate_rows:
        return pd.DataFrame(columns=THEME_CANDIDATE_COLUMNS)

    candidates = pd.DataFrame(candidate_rows)
    candidates = candidates.sort_values(["signal_timestamp", "theme_score"], ascending=[True, False]).reset_index(drop=True)
    candidates = assign_theme_paths(
        candidates,
        member_col="members",
        timestamp_col="signal_timestamp",
        min_overlap=theme_path_min_overlap,
        score_method=theme_path_score_method,
    )
    candidates["candidate_id"] = candidates.apply(_build_candidate_id, axis=1)
    candidates["confirmed_by_age_3x5m"] = candidates["age_bars"] >= 3
    candidates["confirmed_by_15m_graph"] = False
    candidates["confirmed_on_15m"] = candidates["confirmed_by_age_3x5m"] | candidates["confirmed_by_15m_graph"]
    candidates["confirmation_source"] = np.where(
        candidates["confirmed_by_15m_graph"],
        "15m_graph",
        np.where(candidates["confirmed_by_age_3x5m"], "age_3x5m", ""),
    )
    candidates["confirmation_timestamp"] = candidates["signal_timestamp"].where(candidates["confirmed_on_15m"], pd.NaT)
    candidates["confirmation_match_score"] = candidates["match_score"].where(candidates["confirmed_on_15m"], pd.NA)
    for column in THEME_CANDIDATE_COLUMNS:
        if column not in candidates.columns:
            candidates[column] = pd.NA
    return candidates[THEME_CANDIDATE_COLUMNS].sort_values(
        ["signal_timestamp", "theme_score", "member_count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def _add_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output = output.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    by_symbol = output.groupby("symbol", sort=False)
    output["ret_5m_past"] = by_symbol["close"].pct_change(1)
    output["ret_15m_past"] = by_symbol["close"].pct_change(3)
    output["volume_z_12"] = by_symbol["volume"].transform(lambda s: _rolling_zscore(s, 12))
    output["imbalance_z_12"] = by_symbol["imbalance_proxy"].transform(lambda s: _rolling_zscore(s, 12))
    output["liquidity_score"] = by_symbol["dollar_volume_5m"].transform(lambda s: s.shift(1).rolling(12, min_periods=3).median())
    output["large_trade_ratio"] = np.where(
        output["dollar_volume"].abs() > 0,
        output["large_trade_dollar_volume"] / output["dollar_volume"].replace(0, np.nan),
        0.0,
    )
    output["off_exchange_ratio"] = np.where(
        output["volume"].abs() > 0,
        output["off_exchange_volume"] / output["volume"].replace(0, np.nan),
        0.0,
    )
    for column in ["ret_5m_past", "ret_15m_past", "volume_z_12", "imbalance_z_12"]:
        output[f"{column}_xs"] = output.groupby("timestamp", sort=False)[column].transform(_cross_sectional_zscore)
    output["seed_score"] = (
        0.35 * output["ret_5m_past_xs"].fillna(0.0)
        + 0.20 * output["ret_15m_past_xs"].fillna(0.0)
        + 0.20 * output["volume_z_12_xs"].fillna(0.0)
        + 0.25 * output["imbalance_z_12_xs"].fillna(0.0)
    )
    return output


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.shift(1).rolling(window, min_periods=3).mean()
    std = series.shift(1).rolling(window, min_periods=3).std(ddof=0).replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if np.isclose(std, 0.0) or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def _build_similarity_graph(symbols: list[str], corr: pd.DataFrame, *, min_pair_corr: float) -> dict[str, set[str]]:
    graph = {symbol: set() for symbol in symbols}
    for idx, left in enumerate(symbols):
        for right in symbols[idx + 1 :]:
            pair_corr = float(corr.loc[left, right]) if left in corr.index and right in corr.columns else 0.0
            if pair_corr >= min_pair_corr:
                graph[left].add(right)
                graph[right].add(left)
    return graph


def _connected_components(graph: dict[str, set[str]]) -> Iterable[set[str]]:
    seen: set[str] = set()
    for node in graph:
        if node in seen:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(graph[current] - seen)
        yield component


def _average_pairwise_corr(corr: pd.DataFrame, members: list[str]) -> float:
    values: list[float] = []
    for idx, left in enumerate(members):
        for right in members[idx + 1 :]:
            if left in corr.index and right in corr.columns:
                value = float(corr.loc[left, right])
                if not np.isnan(value):
                    values.append(value)
    return float(np.mean(values)) if values else 0.0


def _build_candidate_id(row: pd.Series) -> str:
    trade_date = str(row.get("trade_date", "")).strip()
    signal_timestamp = pd.to_datetime(row.get("signal_timestamp"), utc=True)
    members = str(row.get("members", "")).strip()
    ts_text = signal_timestamp.isoformat().replace("+00:00", "Z") if not pd.isna(signal_timestamp) else ""
    return f"{trade_date}_{ts_text}_{members}"
