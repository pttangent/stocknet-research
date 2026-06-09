"""Portfolio construction and risk management for sector rotation strategy.

Provides:
- Cash gate based on market regime (QQQ vs 20MA)
- Volatility targeting
- Walk-forward parameter validation
- Weight capping and normalization
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def compute_market_regime(
    daily_close: pd.DataFrame,
    benchmark_symbol: str = "QQQ",
    ma_window: int = 20,
) -> pd.Series:
    """Compute market regime: 1 if benchmark > MA, 0 otherwise.

    Used as a cash gate: if market is below MA, reduce exposure.
    """
    if benchmark_symbol not in daily_close.columns:
        return pd.Series(1.0, index=daily_close.index)

    benchmark = daily_close[benchmark_symbol]
    ma = benchmark.rolling(ma_window, min_periods=10).mean()
    regime = (benchmark >= ma).astype(float)
    return regime


def compute_volatility_target_weight(
    returns: pd.Series,
    target_volatility: float = 0.15,
    vol_window: int = 20,
) -> float:
    """Compute position size scaling based on realized volatility.

    target_volatility: annualized target volatility (e.g., 0.15 = 15%)
    Returns a scaling factor in [0, 2].
    """
    realized_vol = returns.rolling(vol_window, min_periods=10).std(ddof=0) * np.sqrt(252)
    if realized_vol.iloc[-1] <= 0:
        return 1.0
    scale = target_volatility / realized_vol.iloc[-1]
    return float(np.clip(scale, 0.0, 2.0))


def apply_cash_gate(
    target_weights: pd.Series,
    regime: float,
    regime_threshold: float = 0.5,
    max_exposure_bear: float = 0.5,
    max_exposure_bull: float = 1.0,
) -> pd.Series:
    """Apply cash gate based on market regime.

    If regime < threshold (bear market), cap exposure to max_exposure_bear.
    Otherwise, cap to max_exposure_bull.
    """
    if target_weights.empty:
        return target_weights

    max_exposure = max_exposure_bull if regime >= regime_threshold else max_exposure_bear
    current_exposure = target_weights.sum()
    if current_exposure > max_exposure:
        target_weights = target_weights * (max_exposure / current_exposure)

    return target_weights


def cap_and_normalize_weights(
    weight_series: pd.Series,
    max_weight_per_stock: float = 0.10,
) -> pd.Series:
    """Cap individual stock weights and normalize to sum to target.

    Iteratively caps weights above max and redistributes excess.
    """
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


def compute_drawdown(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from returns."""
    curve = (1.0 + returns).cumprod()
    peak = curve.cummax()
    drawdown = curve / peak - 1.0
    return drawdown


def compute_sortino(
    returns: pd.Series,
    target_return: float = 0.0,
    ann_factor: float = 252.0,
) -> float:
    """Compute Sortino ratio (downside deviation only)."""
    excess = returns - target_return
    downside = excess[excess < 0]
    if downside.empty or downside.std(ddof=0) == 0:
        return math.nan
    downside_std = downside.std(ddof=0) * np.sqrt(ann_factor)
    mean_excess = returns.mean() * ann_factor
    return mean_excess / downside_std


def compute_calmar(
    returns: pd.Series,
    ann_factor: float = 252.0,
) -> float:
    """Compute Calmar ratio (annualized return / max drawdown)."""
    ann_return = returns.mean() * ann_factor
    max_dd = abs(compute_drawdown(returns).min())
    if max_dd == 0:
        return math.nan
    return ann_return / max_dd


def enhanced_portfolio_metrics(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Compute enhanced portfolio metrics including Sortino and Calmar."""
    if portfolio_df.empty:
        return pd.DataFrame()

    net = portfolio_df["net_return"].fillna(0.0)
    bench = portfolio_df["benchmark_return"].fillna(0.0)
    excess = net - bench
    ann_factor = 252

    mean_daily = float(net.mean())
    std_daily = float(net.std(ddof=0))
    sharpe = (mean_daily / std_daily) * np.sqrt(ann_factor) if std_daily > 0 else math.nan
    sortino = compute_sortino(net)
    calmar = compute_calmar(net)
    max_dd = float(compute_drawdown(net).min())

    metrics = [
        ("total_return", float((1.0 + net).prod() - 1.0)),
        ("benchmark_total_return", float((1.0 + bench).prod() - 1.0)),
        ("excess_total_return", float((1.0 + excess).prod() - 1.0)),
        ("annualized_return", float((1.0 + net).prod() ** (ann_factor / max(len(net), 1)) - 1.0)),
        ("benchmark_annualized_return", float((1.0 + bench).prod() ** (ann_factor / max(len(bench), 1)) - 1.0)),
        ("annualized_volatility", std_daily * np.sqrt(ann_factor)),
        ("sharpe", sharpe),
        ("sortino", sortino),
        ("calmar", calmar),
        ("max_drawdown", max_dd),
        ("hit_rate", float((net > 0).mean())),
        ("avg_turnover", float(portfolio_df["turnover"].mean())),
        ("avg_positions", float(portfolio_df["n_positions"].mean())),
        ("days", int(len(portfolio_df))),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])
