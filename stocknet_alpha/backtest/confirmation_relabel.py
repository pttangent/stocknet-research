from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocknet_alpha.config import AlphaPaths
from stocknet_alpha.theme.build_historical_theme_candidates import load_daily_market_inputs
from stocknet_alpha.theme.historical_candidates import build_theme_candidates_from_market_data


CONFIRMATION_FLAG_COLUMNS = [
    "candidate_id",
    "trade_date",
    "decision_timestamp",
    "community_id",
    "reconfirmed_on_15m",
    "reconfirmed_by_age_3x5m",
    "reconfirmed_by_15m_graph",
    "reconfirmation_source",
    "reconfirmation_timestamp",
    "reconfirmation_match_score",
    "reconfirmation_age_bars",
    "reconfirmation_theme_path_id",
]


def build_confirmation_flag_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    """Project theme-candidate confirmation state into evaluated-trade merge keys."""

    if candidates.empty:
        return pd.DataFrame(columns=CONFIRMATION_FLAG_COLUMNS)

    frame = candidates.copy()
    frame["signal_timestamp"] = pd.to_datetime(frame["signal_timestamp"], utc=True)
    if "candidate_id" not in frame.columns:
        frame["candidate_id"] = pd.NA
    return (
        frame.rename(
            columns={
                "signal_timestamp": "decision_timestamp",
                "confirmed_on_15m": "reconfirmed_on_15m",
                "confirmed_by_age_3x5m": "reconfirmed_by_age_3x5m",
                "confirmed_by_15m_graph": "reconfirmed_by_15m_graph",
                "confirmation_source": "reconfirmation_source",
                "confirmation_timestamp": "reconfirmation_timestamp",
                "match_score": "reconfirmation_match_score",
                "age_bars": "reconfirmation_age_bars",
                "theme_path_id": "reconfirmation_theme_path_id",
            }
        )[CONFIRMATION_FLAG_COLUMNS]
        .drop_duplicates(subset=_flag_dedup_keys(frame), keep="last")
        .reset_index(drop=True)
    )


def relabel_evaluated_confirmations(
    evaluated_trades: pd.DataFrame,
    confirmation_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Overwrite evaluated confirmation state using rebuilt causal candidate flags."""

    if evaluated_trades.empty:
        return evaluated_trades.copy()

    frame = evaluated_trades.copy()
    frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True)
    if confirmation_flags.empty:
        frame["confirmed_on_15m"] = False
        frame["confirmed_by_age_3x5m"] = False
        frame["confirmed_by_15m_graph"] = False
        frame["confirmation_source"] = pd.NA
        frame["confirmation_timestamp"] = pd.NaT
        frame["confirmation_match_score"] = pd.NA
        frame["confirmation_age_bars"] = pd.NA
        frame["confirmation_theme_path_id"] = pd.NA
        return frame

    flags = confirmation_flags.copy()
    flags["decision_timestamp"] = pd.to_datetime(flags["decision_timestamp"], utc=True)
    merge_keys = _confirmation_merge_keys(frame, flags)
    merged = frame.merge(
        flags,
        on=merge_keys,
        how="left",
    )
    merged["confirmed_on_15m"] = merged["reconfirmed_on_15m"].fillna(False).astype(bool)
    merged["confirmed_by_age_3x5m"] = merged["reconfirmed_by_age_3x5m"].fillna(False).astype(bool)
    merged["confirmed_by_15m_graph"] = merged["reconfirmed_by_15m_graph"].fillna(False).astype(bool)
    merged["confirmation_source"] = merged["reconfirmation_source"]
    merged["confirmation_timestamp"] = pd.to_datetime(merged["reconfirmation_timestamp"], utc=True, errors="coerce")
    merged["confirmation_match_score"] = merged["reconfirmation_match_score"]
    merged["confirmation_age_bars"] = merged["reconfirmation_age_bars"]
    merged["confirmation_theme_path_id"] = merged["reconfirmation_theme_path_id"]
    return merged.drop(
        columns=[
            "reconfirmed_on_15m",
            "reconfirmed_by_age_3x5m",
            "reconfirmed_by_15m_graph",
            "reconfirmation_source",
            "reconfirmation_timestamp",
            "reconfirmation_match_score",
            "reconfirmation_age_bars",
            "reconfirmation_theme_path_id",
        ],
        errors="ignore",
    )


def discover_confirmation_trade_dates(
    paths: AlphaPaths,
    *,
    start_date: str,
    end_date: str,
) -> list[str]:
    discovered: set[str] = set()
    for root in [
        paths.raw_1m_root,
        paths.bars_5m_root,
        paths.bars_15m_root,
        paths.trade_flow_1m_root,
    ]:
        if not root.exists():
            continue
        for child in root.glob("date=*"):
            trade_date = child.name.replace("date=", "")
            if start_date <= trade_date <= end_date:
                discovered.add(trade_date)
    return sorted(discovered)


def rebuild_confirmation_flags(
    paths: AlphaPaths,
    *,
    start_date: str,
    end_date: str,
    lookback_bars: int,
    top_symbols: int,
    min_members: int,
    min_theme_score: float,
    min_pair_corr: float,
    theme_path_score_method: str,
    theme_path_min_overlap: float,
) -> pd.DataFrame:
    """Rebuild candidate confirmation flags for a date range."""

    dates = discover_confirmation_trade_dates(paths, start_date=start_date, end_date=end_date)
    flag_frames: list[pd.DataFrame] = []
    for trade_date in dates:
        bars_5m, flow_1m = load_daily_market_inputs(paths, trade_date)
        candidates = build_theme_candidates_from_market_data(
            bars_5m,
            flow_1m,
            trade_date=trade_date,
            lookback_bars=lookback_bars,
            top_symbols=top_symbols,
            min_members=min_members,
            min_theme_score=min_theme_score,
            min_pair_corr=min_pair_corr,
            theme_path_score_method=theme_path_score_method,
            theme_path_min_overlap=theme_path_min_overlap,
        )
        if not candidates.empty:
            flag_frames.append(build_confirmation_flag_frame(candidates))
    if not flag_frames:
        return build_confirmation_flag_frame(pd.DataFrame())
    return pd.concat(flag_frames, ignore_index=True)


def relabel_evaluated_trades_file(
    evaluated_path: Path | str,
    *,
    output_path: Path | str,
    paths: AlphaPaths,
    start_date: str,
    end_date: str,
    lookback_bars: int,
    top_symbols: int,
    min_members: int,
    min_theme_score: float,
    min_pair_corr: float,
    theme_path_score_method: str,
    theme_path_min_overlap: float,
) -> Path:
    """Rebuild confirmations and persist a relabeled evaluated-trades parquet."""

    source = Path(evaluated_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    evaluated = pd.read_parquet(source)
    flags = rebuild_confirmation_flags(
        paths,
        start_date=start_date,
        end_date=end_date,
        lookback_bars=lookback_bars,
        top_symbols=top_symbols,
        min_members=min_members,
        min_theme_score=min_theme_score,
        min_pair_corr=min_pair_corr,
        theme_path_score_method=theme_path_score_method,
        theme_path_min_overlap=theme_path_min_overlap,
    )
    relabeled = relabel_evaluated_confirmations(evaluated, flags)
    target.parent.mkdir(parents=True, exist_ok=True)
    relabeled.to_parquet(target, index=False)
    return target


def _confirmation_merge_keys(evaluated: pd.DataFrame, flags: pd.DataFrame) -> list[str]:
    if "candidate_id" in evaluated.columns and "candidate_id" in flags.columns:
        if evaluated["candidate_id"].notna().any() and flags["candidate_id"].notna().any():
            return ["candidate_id"]
    return ["trade_date", "decision_timestamp", "community_id"]


def _flag_dedup_keys(frame: pd.DataFrame) -> list[str]:
    if "candidate_id" in frame.columns and frame["candidate_id"].notna().any():
        return ["candidate_id"]
    return ["trade_date", "signal_timestamp", "community_id"]
