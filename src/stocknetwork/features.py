"""Enhanced feature engineering for intraday financial networks.

Provides market-neutral residual returns, co-jump detection, lead-lag scores,
and corrected rolling volume z-scores without lookahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_residual_returns(
    close_df: pd.DataFrame,
    benchmark_symbols: list[str] = None,
    window_bars: int = 78,
) -> pd.DataFrame:
    """Compute market-neutral residual returns via multi-factor regression.

    For each symbol, regress log-returns on SPY/QQQ/IWM betas and return residuals.
    These residuals capture alpha movement independent of market beta.
    """
    if benchmark_symbols is None:
        benchmark_symbols = ["SPY", "QQQ", "IWM"]

    available_benchmarks = [b for b in benchmark_symbols if b in close_df.columns]
    if not available_benchmarks:
        # No benchmarks available; return raw returns as residuals
        returns = np.log(close_df / close_df.shift(1))
        return returns

    returns = np.log(close_df / close_df.shift(1))
    benchmark_returns = returns[available_benchmarks]
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for symbol in returns.columns:
        if symbol in available_benchmarks:
            residuals[symbol] = returns[symbol] - benchmark_returns.mean(axis=1)
            continue

        # Rolling regression for beta estimation
        y = returns[symbol]
        X = benchmark_returns

        # Use expanding window regression (no lookahead)
        def _rolling_residual(y_series: pd.Series, X_df: pd.DataFrame) -> pd.Series:
            resid = pd.Series(np.nan, index=y_series.index)
            for i in range(len(y_series)):
                if i < 20:
                    continue
                y_window = y_series.iloc[:i].dropna()
                X_window = X_df.iloc[:i].reindex(y_window.index).dropna()
                if len(X_window) < 20:
                    continue
                y_aligned = y_window.reindex(X_window.index).dropna()
                X_aligned = X_window.reindex(y_aligned.index).dropna()
                if len(X_aligned) < 20:
                    continue
                # OLS: beta = (X'X)^-1 X'y
                X_mat = np.column_stack([np.ones(len(X_aligned)), X_aligned.values])
                y_vec = y_aligned.values
                try:
                    beta = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]
                    pred = X_mat[-1] @ beta
                    resid.iloc[i] = y_series.iloc[i] - pred
                except Exception:
                    resid.iloc[i] = y_series.iloc[i]
            return resid

        residuals[symbol] = _rolling_residual(y, X)

    return residuals


def compute_rolling_volume_zscore(
    volume_df: pd.DataFrame,
    lookback_days: int = 20,
    min_periods: int = 5,
) -> pd.DataFrame:
    """Compute rolling time-of-day volume z-scores WITHOUT lookahead bias.

    For each time-of-day bucket, use only prior observations within lookback
    to estimate mean and std.
    """
    ny_tz = "America/New_York"
    local_index = volume_df.index.tz_convert(ny_tz)
    time_bucket = local_index.strftime("%H:%M")
    out = pd.DataFrame(index=volume_df.index, columns=volume_df.columns, dtype=float)

    for bucket in sorted(set(time_bucket)):
        mask = time_bucket == bucket
        bucket_volume = volume_df.loc[mask]
        rolling_mean = bucket_volume.shift(1).rolling(lookback_days, min_periods=min_periods).mean()
        rolling_std = bucket_volume.shift(1).rolling(lookback_days, min_periods=min_periods).std(ddof=0).replace(0, np.nan)
        out.loc[mask] = (bucket_volume - rolling_mean) / rolling_std

    return out.replace([np.inf, -np.inf], np.nan)


def compute_cojump_scores(
    returns_df: pd.DataFrame,
    z_threshold: float = 2.0,
    window_bars: int = 26,
) -> pd.DataFrame:
    """Compute co-jump scores: how often two stocks simultaneously exceed z-threshold.

    For each pair of stocks, count the number of bars within the window where
    both have |return_z| > threshold. Higher = stronger co-jump relationship.
    """
    symbols = returns_df.columns.tolist()
    n = len(symbols)

    # Compute z-scores per symbol (within window)
    z_returns = (returns_df - returns_df.rolling(window_bars, min_periods=10).mean()) / (
        returns_df.rolling(window_bars, min_periods=10).std(ddof=0).replace(0, np.nan)
    )
    z_returns = z_returns.replace([np.inf, -np.inf], np.nan)

    # Count co-jumps
    cojump_matrix = np.zeros((n, n), dtype=np.float32)
    jump_mask = z_returns.abs() > z_threshold

    for i, sym_i in enumerate(symbols):
        for j, sym_j in enumerate(symbols):
            if i >= j:
                continue
            both_jump = jump_mask[sym_i] & jump_mask[sym_j]
            count = both_jump.sum()
            total = len(both_jump.dropna())
            if total > 0:
                cojump_matrix[i, j] = float(count / total)
                cojump_matrix[j, i] = cojump_matrix[i, j]

    return pd.DataFrame(cojump_matrix, index=symbols, columns=symbols)


def compute_leadlag_scores(
    returns_df: pd.DataFrame,
    max_lag: int = 3,
    window_bars: int = 78,
) -> pd.DataFrame:
    """Compute lead-lag cross-correlation scores.

    For each pair (i, j), compute max cross-correlation across lags 1..max_lag.
    Higher score = stronger lead-lag relationship (one leads, the other follows).
    """
    symbols = returns_df.columns.tolist()
    n = len(symbols)

    # Use recent window
    recent = returns_df.tail(window_bars).fillna(0.0)
    leadlag_matrix = np.zeros((n, n), dtype=np.float32)

    for i, sym_i in enumerate(symbols):
        for j, sym_j in enumerate(symbols):
            if i == j:
                continue
            x = recent[sym_i].values
            y = recent[sym_j].values
            if len(x) < max_lag + 5 or len(y) < max_lag + 5:
                continue

            max_corr = 0.0
            for lag in range(1, max_lag + 1):
                x_lag = x[:-lag]
                y_lead = y[lag:]
                if x_lag.std() == 0 or y_lead.std() == 0:
                    continue
                corr = float(np.corrcoef(x_lag, y_lead)[0, 1])
                if not np.isnan(corr) and abs(corr) > abs(max_corr):
                    max_corr = corr

            leadlag_matrix[i, j] = max_corr

    return pd.DataFrame(leadlag_matrix, index=symbols, columns=symbols)


def compute_intraday_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compute intraday features from OHLCV panel.

    Returns DataFrame with columns:
    - log_return
    - residual_return (if benchmark available)
    - intraday_range: (high - low) / close
    - rolling_volatility: rolling std of log returns
    - liquidity_score: log(close * volume)
    - volume_zscore: rolling z-score (no lookahead)
    """
    frame = panel.copy()
    frame["log_return"] = frame.groupby("symbol")["close"].transform(lambda s: np.log(s / s.shift(1)))
    frame["intraday_range"] = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
    frame["rolling_volatility"] = frame.groupby("symbol")["log_return"].transform(
        lambda s: s.rolling(window=5, min_periods=3).std(ddof=0)
    )
    frame["liquidity_score"] = np.log1p(frame["close"] * frame["volume"])

    # Volume z-score per symbol (rolling, no lookahead)
    vol_z = (
        frame.groupby("symbol")["volume"]
        .transform(lambda s: _rolling_zscore(s, window=20))
    )
    frame["volume_zscore"] = vol_z

    return frame


def _rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.shift(1).rolling(window=window, min_periods=5).mean()
    std = series.shift(1).rolling(window=window, min_periods=5).std(ddof=0).replace(0, np.nan)
    zscore = (series - mean) / std
    return zscore.replace([np.inf, -np.inf], np.nan)
