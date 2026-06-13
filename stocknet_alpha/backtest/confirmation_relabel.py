from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocknet_alpha.backtest.historical_leadlag import discover_trade_dates
from stocknet_alpha.config import AlphaPaths
from stocknet_alpha.theme.build_historical_theme_candidates import load_daily_market_inputs
from stocknet_alpha.theme.historical_candidates import build_theme_candidates_from_market_data


def build_confirmation_flag_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    """Project theme-candidate confirmation state into evaluated-trade merge keys."""

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "decision_timestamp",
                "community_id",
                "reconfirmed_on_15m",
                "reconfirmation_timestamp",
                "reconfirmation_age_bars",
                "reconfirmation_theme_path_id",
            ]
        )

    frame = candidates.copy()
    frame["signal_timestamp"] = pd.to_datetime(frame["signal_timestamp"], utc=True)
    return (
        frame.rename(
            columns={
                "signal_timestamp": "decision_timestamp",
                "confirmed_on_15m": "reconfirmed_on_15m",
                "confirmation_timestamp": "reconfirmation_timestamp",
                "age_bars": "reconfirmation_age_bars",
                "theme_path_id": "reconfirmation_theme_path_id",
            }
        )[
            [
                "trade_date",
                "decision_timestamp",
                "community_id",
                "reconfirmed_on_15m",
                "reconfirmation_timestamp",
                "reconfirmation_age_bars",
                "reconfirmation_theme_path_id",
            ]
        ]
        .drop_duplicates(subset=["trade_date", "decision_timestamp", "community_id"], keep="last")
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
        frame["confirmation_timestamp"] = pd.NaT
        frame["confirmation_age_bars"] = pd.NA
        frame["confirmation_theme_path_id"] = pd.NA
        return frame

    flags = confirmation_flags.copy()
    flags["decision_timestamp"] = pd.to_datetime(flags["decision_timestamp"], utc=True)
    merged = frame.merge(
        flags,
        on=["trade_date", "decision_timestamp", "community_id"],
        how="left",
    )
    merged["confirmed_on_15m"] = merged["reconfirmed_on_15m"].fillna(False).astype(bool)
    merged["confirmation_timestamp"] = pd.to_datetime(merged["reconfirmation_timestamp"], utc=True, errors="coerce")
    merged["confirmation_age_bars"] = merged["reconfirmation_age_bars"]
    merged["confirmation_theme_path_id"] = merged["reconfirmation_theme_path_id"]
    return merged.drop(
        columns=[
            "reconfirmed_on_15m",
            "reconfirmation_timestamp",
            "reconfirmation_age_bars",
            "reconfirmation_theme_path_id",
        ],
        errors="ignore",
    )


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

    dates = discover_trade_dates(paths.trade_flow_1m_root, start_date=start_date, end_date=end_date)
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
