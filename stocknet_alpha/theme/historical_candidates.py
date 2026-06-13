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
    "price_theme_score",
    "flow_theme_score",
    "confirmed_by_flow",
    "flow_breadth",
    "large_trade_breadth",
    "imbalance_breadth",
    "coherence_5m",
    "breadth_5m",
    "relative_return_5m",
    "volume_expansion_5m",
    "theme_score",
    "event_type",
    "age_bars",
]


def aggregate_trade_flow_to_5m(trade_flow_1m: pd.DataFrame) -> pd.DataFrame:
    return aggregate_trade_flow_to_interval(trade_flow_1m, interval_minutes=5)


def aggregate_trade_flow_to_interval(
    trade_flow_1m: pd.DataFrame,
    interval_minutes: int,
) -> pd.DataFrame:
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
    frame["timestamp"] = frame["minute"].dt.floor(f"{int(interval_minutes)}min") + pd.Timedelta(minutes=int(interval_minutes))
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
    theme_path_max_gap: pd.Timedelta | str | None = None,
    theme_path_reset_on_trade_date_change: bool = False,
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

    flow_5m = aggregate_trade_flow_to_interval(trade_flow_1m, interval_minutes=5)
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

    candidates = _build_candidate_frame(
        merged,
        trade_date=trade_date,
        interval_minutes=5,
        lookback_bars=lookback_bars,
        top_symbols=top_symbols,
        min_members=min_members,
        min_theme_score=min_theme_score,
        min_pair_corr=min_pair_corr,
        max_members_per_candidate=max_members_per_candidate,
    )

    if candidates.empty:
        return pd.DataFrame(columns=THEME_CANDIDATE_COLUMNS)
    candidates = candidates.sort_values(["signal_timestamp", "theme_score"], ascending=[True, False]).reset_index(drop=True)
    candidates = assign_theme_paths(
        candidates,
        member_col="members",
        timestamp_col="signal_timestamp",
        min_overlap=theme_path_min_overlap,
        score_method=theme_path_score_method,
        max_gap=theme_path_max_gap,
        reset_on_trade_date_change=theme_path_reset_on_trade_date_change,
    )
    fifteen_minute_candidates = _build_15m_confirmation_candidates(
        bars_5m,
        trade_flow_1m,
        trade_date=trade_date,
        lookback_bars=lookback_bars,
        top_symbols=top_symbols,
        min_members=min_members,
        min_theme_score=min_theme_score,
        min_pair_corr=min_pair_corr,
        max_members_per_candidate=max_members_per_candidate,
    )
    candidates["candidate_id"] = candidates.apply(_build_candidate_id, axis=1)
    candidates["confirmed_by_age_3x5m"] = candidates["age_bars"] >= 3
    candidates = _apply_15m_graph_confirmation(
        candidates,
        fifteen_minute_candidates,
        min_overlap=theme_path_min_overlap,
        score_method=theme_path_score_method,
    )
    candidates["confirmed_on_15m"] = candidates["confirmed_by_age_3x5m"] | candidates["confirmed_by_15m_graph"]
    candidates["confirmation_source"] = np.where(
        candidates["confirmed_by_15m_graph"],
        "15m_graph",
        np.where(candidates["confirmed_by_age_3x5m"], "age_3x5m", ""),
    )
    candidates["confirmation_timestamp"] = np.where(
        candidates["confirmed_by_15m_graph"],
        candidates["confirmation_timestamp"],
        candidates["signal_timestamp"].where(candidates["confirmed_by_age_3x5m"], pd.NaT),
    )
    candidates["confirmation_timestamp"] = pd.to_datetime(candidates["confirmation_timestamp"], utc=True, errors="coerce")
    candidates["confirmation_match_score"] = np.where(
        candidates["confirmed_by_15m_graph"],
        candidates["confirmation_match_score"],
        candidates["match_score"].where(candidates["confirmed_by_age_3x5m"], pd.NA),
    )
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
    output["dollar_volume_z_12"] = by_symbol["dollar_volume"].transform(lambda s: _rolling_zscore(s, 12))
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
    output["large_trade_ratio_z_12"] = by_symbol["large_trade_ratio"].transform(lambda s: _rolling_zscore(s, 12))
    output["off_exchange_ratio_z_12"] = by_symbol["off_exchange_ratio"].transform(lambda s: _rolling_zscore(s, 12))
    for column in [
        "ret_5m_past",
        "ret_15m_past",
        "volume_z_12",
        "imbalance_z_12",
        "dollar_volume_z_12",
        "large_trade_ratio_z_12",
        "off_exchange_ratio_z_12",
    ]:
        output[f"{column}_xs"] = output.groupby("timestamp", sort=False)[column].transform(_cross_sectional_zscore)
    output["seed_score"] = (
        0.35 * output["ret_5m_past_xs"].fillna(0.0)
        + 0.20 * output["ret_15m_past_xs"].fillna(0.0)
        + 0.20 * output["volume_z_12_xs"].fillna(0.0)
        + 0.25 * output["imbalance_z_12_xs"].fillna(0.0)
    )
    output["flow_impulse_score"] = (
        0.35 * output["dollar_volume_z_12_xs"].fillna(0.0)
        + 0.30 * output["imbalance_z_12_xs"].fillna(0.0)
        + 0.20 * output["large_trade_ratio_z_12_xs"].fillna(0.0)
        + 0.15 * output["off_exchange_ratio_z_12_xs"].fillna(0.0)
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


def _split_members(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().upper() for item in value if str(item).strip()}
    return {item.strip().upper() for item in str(value).split(",") if item.strip()}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _overlap_small(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _match_score(left: set[str], right: set[str], *, score_method: str) -> float:
    method = str(score_method).strip().lower()
    if method == "jaccard":
        return _jaccard(left, right)
    if method == "overlap_small":
        return _overlap_small(left, right)
    raise ValueError(f"Unsupported score_method: {score_method}")


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


def _build_candidate_frame(
    merged: pd.DataFrame,
    *,
    trade_date: str,
    interval_minutes: int,
    lookback_bars: int,
    top_symbols: int,
    min_members: int,
    min_theme_score: float,
    min_pair_corr: float,
    max_members_per_candidate: int,
) -> pd.DataFrame:
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
            & (merged["timestamp"] >= timestamp - pd.Timedelta(minutes=interval_minutes * max(lookback_bars - 1, 0)))
            & (merged["symbol"].isin(eligible["symbol"]))
        ].copy()
        if history.empty:
            continue
        pivot = history.pivot(index="timestamp", columns="symbol", values="ret_5m_past").sort_index()
        min_required_bars = max(2, min(int(lookback_bars), 3))
        if len(pivot) < min_required_bars:
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
            price_theme_score = float(
                0.45 * member_slice["ret_5m_past_xs"].fillna(0.0).mean()
                + 0.25 * member_slice["ret_15m_past_xs"].fillna(0.0).mean()
                + 0.30 * member_slice["volume_z_12_xs"].fillna(0.0).mean()
            )
            flow_breadth = float((member_slice["flow_impulse_score"].fillna(0.0) > 0.0).mean())
            large_trade_breadth = float((member_slice["large_trade_ratio"].fillna(0.0) > 0.0).mean())
            imbalance_breadth = float((member_slice["imbalance_proxy"].fillna(0.0) > 0.0).mean())
            flow_theme_score = float(
                0.30 * member_slice["flow_impulse_score"].fillna(0.0).mean()
                + 0.25 * flow_breadth
                + 0.20 * imbalance_breadth
                + 0.15 * large_trade_breadth
                + 0.10 * member_slice["off_exchange_ratio"].fillna(0.0).mean()
            )
            lifecycle_score = confirmation_score
            theme_score = 0.55 * price_theme_score + 0.30 * flow_theme_score + 0.15 * lifecycle_score
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
                    "price_theme_score": price_theme_score,
                    "flow_theme_score": flow_theme_score,
                    "confirmed_by_flow": bool(flow_theme_score > 0.0),
                    "flow_breadth": flow_breadth,
                    "large_trade_breadth": large_trade_breadth,
                    "imbalance_breadth": imbalance_breadth,
                    "coherence_5m": coherence,
                    "breadth_5m": breadth,
                    "relative_return_5m": relative_return,
                    "volume_expansion_5m": volume_expansion,
                    "theme_score": theme_score,
                }
            )
    return pd.DataFrame(candidate_rows)


def _build_15m_confirmation_candidates(
    bars_5m: pd.DataFrame,
    trade_flow_1m: pd.DataFrame,
    *,
    trade_date: str,
    lookback_bars: int,
    top_symbols: int,
    min_members: int,
    min_theme_score: float,
    min_pair_corr: float,
    max_members_per_candidate: int,
) -> pd.DataFrame:
    bars_15m = _resample_bars_from_5m(bars_5m, target_interval_minutes=15)
    if bars_15m.empty:
        return pd.DataFrame(columns=["signal_timestamp", "members", "theme_score"])
    bars_15m["dollar_volume_5m"] = pd.to_numeric(bars_15m["close"], errors="coerce").fillna(0.0) * pd.to_numeric(
        bars_15m["volume"], errors="coerce"
    ).fillna(0.0)
    flow_15m = aggregate_trade_flow_to_interval(trade_flow_1m, interval_minutes=15)
    merged = bars_15m.merge(flow_15m, on=["symbol", "timestamp"], how="left", suffixes=("", "_flow"))
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
    merged["trade_date"] = pd.to_datetime(merged["timestamp"], utc=True).dt.date.astype(str)
    merged = merged[merged["trade_date"] == str(trade_date)].copy()
    if merged.empty:
        return pd.DataFrame(columns=["signal_timestamp", "members", "theme_score"])
    return _build_candidate_frame(
        merged,
        trade_date=trade_date,
        interval_minutes=15,
        lookback_bars=3,
        top_symbols=top_symbols,
        min_members=min_members,
        min_theme_score=min_theme_score,
        min_pair_corr=min_pair_corr,
        max_members_per_candidate=max_members_per_candidate,
    )


def _apply_15m_graph_confirmation(
    candidates: pd.DataFrame,
    confirmation_candidates: pd.DataFrame,
    *,
    min_overlap: float,
    score_method: str,
) -> pd.DataFrame:
    output = candidates.copy()
    output["confirmed_by_15m_graph"] = False
    output["confirmation_timestamp"] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
    output["confirmation_match_score"] = pd.NA
    if confirmation_candidates.empty:
        return output

    conf = confirmation_candidates.copy()
    conf["signal_timestamp"] = pd.to_datetime(conf["signal_timestamp"], utc=True)
    conf["_member_set"] = conf["members"].map(_split_members)
    output["signal_timestamp"] = pd.to_datetime(output["signal_timestamp"], utc=True)
    for idx, row in output.iterrows():
        member_set = _split_members(row.get("members"))
        eligible = conf[conf["signal_timestamp"] <= row["signal_timestamp"]]
        best_score = 0.0
        best_timestamp = pd.NaT
        for _, conf_row in eligible.iterrows():
            score = _match_score(member_set, conf_row["_member_set"], score_method=score_method)
            if score > best_score:
                best_score = score
                best_timestamp = conf_row["signal_timestamp"]
        if best_score >= min_overlap:
            output.at[idx, "confirmed_by_15m_graph"] = True
            output.at[idx, "confirmation_timestamp"] = best_timestamp
            output.at[idx, "confirmation_match_score"] = float(best_score)
    return output


def _resample_bars_from_5m(bars_5m: pd.DataFrame, *, target_interval_minutes: int) -> pd.DataFrame:
    if bars_5m.empty:
        return bars_5m.copy()
    frame = bars_5m.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["bucket_end"] = (frame["timestamp"] - pd.Timedelta(minutes=5)).dt.floor(f"{int(target_interval_minutes)}min") + pd.Timedelta(minutes=int(target_interval_minutes))
    rows: list[dict[str, object]] = []
    for (symbol, bucket_end), group in frame.groupby(["symbol", "bucket_end"], sort=True):
        group = group.sort_values("timestamp")
        row = {
            "timestamp": bucket_end,
            "symbol": symbol,
            "open": pd.to_numeric(group["open"], errors="coerce").iloc[0],
            "high": pd.to_numeric(group["high"], errors="coerce").max(),
            "low": pd.to_numeric(group["low"], errors="coerce").min(),
            "close": pd.to_numeric(group["close"], errors="coerce").iloc[-1],
            "volume": pd.to_numeric(group["volume"], errors="coerce").fillna(0.0).sum(),
            "source": group["source"].iloc[-1] if "source" in group.columns else "resampled_15m",
        }
        if "vwap" in group.columns:
            weights = pd.to_numeric(group["volume"], errors="coerce").fillna(0.0)
            vwap_values = pd.to_numeric(group["vwap"], errors="coerce")
            total_weight = float(weights.sum())
            row["vwap"] = float((vwap_values.fillna(0.0) * weights).sum() / total_weight) if total_weight > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
