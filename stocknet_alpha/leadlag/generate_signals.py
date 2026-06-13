from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.config import AlphaPaths
from stocknet_alpha.data.resample_bars import load_raw_1m_bars
from stocknetwork.features import compute_leadlag_scores


SIGNAL_COLUMNS = [
    "trade_date",
    "signal_timestamp",
    "feature_max_timestamp",
    "decision_timestamp",
    "execution_timestamp",
    "theme_path_id",
    "community_id",
    "leader_symbol",
    "follower_symbol",
    "signal_type",
    "lag_minutes",
    "leadlag_score",
    "best_lag_correlation",
    "confirmed_on_15m",
    "theme_score",
]


def generate_leadlag_signals(
    bars_1m: pd.DataFrame,
    candidates: pd.DataFrame,
    lookback_minutes: int = 60,
    max_lag: int = 5,
    top_followers: int = 3,
    max_members: int = 12,
) -> pd.DataFrame:
    """Turn scanner themes into causal leader-follower 1m signals."""

    if bars_1m.empty or candidates.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    bars = bars_1m.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    if "bar_end" in bars.columns:
        bars["bar_end"] = pd.to_datetime(bars["bar_end"], utc=True)
    else:
        bars["bar_end"] = bars["timestamp"] + pd.Timedelta(minutes=1)
    bars = bars.sort_values(["symbol", "timestamp"]).drop_duplicates(subset=["symbol", "timestamp"], keep="last").reset_index(drop=True)
    bars["close"] = bars["close"].astype(float)
    bars["volume"] = bars["volume"].astype(float)

    signal_rows: list[dict[str, object]] = []

    for _, candidate in candidates.iterrows():
        signal_timestamp = pd.Timestamp(candidate["signal_timestamp"])
        if signal_timestamp.tzinfo is None:
            signal_timestamp = signal_timestamp.tz_localize("UTC")
        else:
            signal_timestamp = signal_timestamp.tz_convert("UTC")
        members = _coerce_members(candidate.get("members", ""))
        history = bars[
            (bars["symbol"].isin(members))
            & (bars["bar_end"] <= signal_timestamp)
            & (bars["timestamp"] >= signal_timestamp - pd.Timedelta(minutes=lookback_minutes))
        ].copy()
        if history.empty:
            continue
        history = history.sort_values(["symbol", "timestamp"]).drop_duplicates(subset=["symbol", "timestamp"], keep="last")

        top_members = (
            history.assign(dollar_volume=history["close"] * history["volume"])
            .groupby("symbol", as_index=False)["dollar_volume"]
            .sum()
            .sort_values("dollar_volume", ascending=False)["symbol"]
            .head(max_members)
            .tolist()
        )
        history = history[history["symbol"].isin(top_members)]
        close_pivot = history.pivot(index="timestamp", columns="symbol", values="close").sort_index()
        close_pivot = close_pivot.ffill().dropna(axis=1, thresh=max(3, len(close_pivot) // 2))
        if close_pivot.shape[1] < 2:
            continue

        returns = close_pivot.pct_change().dropna(how="all").fillna(0.0)
        if returns.empty:
            continue
        leadlag_scores = compute_leadlag_scores(returns, max_lag=max_lag, window_bars=len(returns))
        positive_scores = leadlag_scores.clip(lower=0.0)
        outbound = positive_scores.sum(axis=1)
        if outbound.empty or float(outbound.max()) <= 0.0:
            continue
        leader = str(outbound.idxmax())
        followers = positive_scores.loc[leader].drop(labels=[leader], errors="ignore").sort_values(ascending=False)
        followers = followers[followers > 0.0].head(top_followers)
        if followers.empty:
            continue

        for follower, score in followers.items():
            lag_minutes, best_corr = _best_lag(returns[leader], returns[str(follower)], max_lag=max_lag)
            feature_max_timestamp = pd.Timestamp(history["bar_end"].max())
            row = {
                "trade_date": signal_timestamp.date().isoformat(),
                "signal_timestamp": signal_timestamp,
                "feature_max_timestamp": feature_max_timestamp,
                "decision_timestamp": signal_timestamp,
                "execution_timestamp": signal_timestamp + pd.Timedelta(minutes=1),
                "theme_path_id": candidate.get("theme_path_id", ""),
                "community_id": candidate.get("community_id", ""),
                "leader_symbol": leader,
                "follower_symbol": str(follower),
                "signal_type": "leadlag_return",
                "lag_minutes": lag_minutes,
                "leadlag_score": float(score),
                "best_lag_correlation": float(best_corr),
                "confirmed_on_15m": bool(candidate.get("confirmed_on_15m", False)),
                "theme_score": float(candidate.get("theme_score", 0.0) or 0.0),
            }
            signal_rows.append(row)

    if not signal_rows:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    frame = pd.DataFrame(signal_rows)
    for column in SIGNAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.sort_values(
        ["signal_timestamp", "theme_score", "leadlag_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def write_signals(frame: pd.DataFrame, paths: AlphaPaths, trade_date: str) -> Path:
    output_path = paths.ensure_parent(paths.signals_path(trade_date))
    frame.to_parquet(output_path, index=False)
    return output_path


def load_theme_candidates(paths: AlphaPaths, trade_date: str, input_path: Path | str | None = None) -> pd.DataFrame:
    source = Path(input_path).expanduser().resolve() if input_path else paths.theme_candidates_path(trade_date)
    frame = pd.read_parquet(source)
    if "signal_timestamp" in frame.columns:
        frame["signal_timestamp"] = pd.to_datetime(frame["signal_timestamp"], utc=True)
    return frame


def _coerce_members(raw_members: object) -> list[str]:
    return [member.strip().upper() for member in str(raw_members).split(",") if member.strip()]


def _best_lag(leader_returns: pd.Series, follower_returns: pd.Series, max_lag: int) -> tuple[int, float]:
    best_lag = 1
    best_corr = 0.0
    for lag in range(1, max_lag + 1):
        lead = leader_returns.iloc[:-lag]
        follow = follower_returns.iloc[lag:]
        if len(lead) < 3 or len(follow) < 3:
            continue
        if np.isclose(float(lead.std(ddof=0)), 0.0) or np.isclose(float(follow.std(ddof=0)), 0.0):
            continue
        corr = float(np.corrcoef(lead.values, follow.values)[0, 1])
        if not np.isnan(corr) and abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
    return best_lag, best_corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 1m lead-lag alpha signals from scanner theme candidates.")
    parser.add_argument("--date", required=True, help="Trade date partition in YYYY-MM-DD format.")
    parser.add_argument("--bars-1m", default="", help="Optional override path for the 1m source parquet.")
    parser.add_argument("--theme-candidates", default="", help="Optional override path for theme candidate parquet.")
    parser.add_argument("--lookback-minutes", type=int, default=60)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--top-followers", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AlphaPaths()
    bars_1m = load_raw_1m_bars(paths, args.date, input_path=args.bars_1m or None)
    candidates = load_theme_candidates(paths, args.date, input_path=args.theme_candidates or None)
    signals = generate_leadlag_signals(
        bars_1m,
        candidates,
        lookback_minutes=args.lookback_minutes,
        max_lag=args.max_lag,
        top_followers=args.top_followers,
    )
    output_path = write_signals(signals, paths, args.date)
    print(f"Wrote {len(signals)} lead-lag signals to {output_path}")


if __name__ == "__main__":
    main()
