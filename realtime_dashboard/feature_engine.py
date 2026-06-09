"""Rolling feature computation for intraday community detection."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from .config import FeatureConfig
except ImportError:
    from config import FeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class SymbolFeatures:
    """Features for a single symbol at a single point in time."""
    symbol: str
    timestamp: pd.Timestamp
    return_1m: float = 0.0
    return_5m: float = 0.0
    return_15m: float = 0.0
    relative_return_spy: float = 0.0
    relative_return_qqq: float = 0.0
    volume_zscore: float = 0.0
    dollar_volume: float = 0.0
    intraday_vwap_distance: float = 0.0
    range_percent: float = 0.0
    gap_from_prev_close: float = 0.0


class RollingFeatureEngine:
    """Compute rolling features from 1m bar data."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._bar_buffer: pd.DataFrame = pd.DataFrame()
        self._prev_close: Optional[Dict[str, float]] = None

    def ingest_bars(self, bars_df: pd.DataFrame) -> None:
        """Add new bars to the rolling buffer."""
        required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        if not required.issubset(bars_df.columns):
            missing = required - set(bars_df.columns)
            raise ValueError(f"Missing columns: {missing}")

        bars_df = bars_df.copy()
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"])

        # Append to buffer and keep last N periods for lookback
        max_lookback = max(self.config.windows) + self.config.volume_zscore_lookback + 5
        self._bar_buffer = pd.concat([self._bar_buffer, bars_df], ignore_index=True)

        if len(self._bar_buffer) > 0:
            # Keep enough history for all windows + volume z-score lookback
            # Group by symbol and keep per-symbol history
            self._bar_buffer = (
                self._bar_buffer
                .sort_values(["symbol", "timestamp"])
                .groupby("symbol")
                .tail(max_lookback)
                .reset_index(drop=True)
            )

    def _ensure_buffer(self):
        if self._bar_buffer.empty:
            raise ValueError("No bars in buffer. Call ingest_bars() first.")

    def compute_features(
        self,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Compute features for all symbols at the given timestamp."""
        self._ensure_buffer()

        df = self._bar_buffer.copy()
        df = df.sort_values(["symbol", "timestamp"])

        if timestamp is not None:
            df = df[df["timestamp"] <= timestamp]

        if df.empty:
            return pd.DataFrame()

        # Compute per-symbol features
        features_list = []
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values("timestamp")
            feats = self._compute_symbol_features(symbol, group)
            if feats is not None:
                features_list.append(feats)

        if not features_list:
            return pd.DataFrame()

        result = pd.DataFrame([f.__dict__ for f in features_list])
        return result

    def _compute_symbol_features(
        self,
        symbol: str,
        group: pd.DataFrame,
    ) -> Optional[SymbolFeatures]:
        """Compute features for a single symbol from its bar history."""
        if len(group) < 1:
            return None

        latest = group.iloc[-1]
        feats = SymbolFeatures(
            symbol=symbol,
            timestamp=latest["timestamp"],
        )

        # Returns over different windows
        for window in self.config.windows:
            if len(group) >= window:
                old_close = group.iloc[-window - 1]["close"] if len(group) > window else group.iloc[0]["open"]
                if old_close > 0:
                    ret = (latest["close"] - old_close) / old_close
                else:
                    ret = 0.0
            else:
                ret = 0.0
            setattr(feats, f"return_{window}m", ret)

        # Volume z-score (relative to recent history)
        if len(group) >= self.config.volume_zscore_lookback:
            vol_hist = group["volume"].iloc[-self.config.volume_zscore_lookback:]
            vol_mean = vol_hist.mean()
            vol_std = vol_hist.std() or 1.0
            feats.volume_zscore = (latest["volume"] - vol_mean) / vol_std
        else:
            feats.volume_zscore = 0.0

        # Dollar volume
        feats.dollar_volume = latest["close"] * latest["volume"]

        # VWAP distance
        if "vwap" in group.columns and not pd.isna(latest.get("vwap")):
            feats.intraday_vwap_distance = (latest["close"] - latest["vwap"]) / latest["vwap"] if latest["vwap"] != 0 else 0
        else:
            # Approximate intraday VWAP from today's bars
            today_bars = group  # Assume all bars are from same session for simplicity
            tpv = ((today_bars["high"] + today_bars["low"] + today_bars["close"]) / 3 * today_bars["volume"]).sum()
            tv = today_bars["volume"].sum()
            vwap = tpv / tv if tv > 0 else latest["close"]
            feats.intraday_vwap_distance = (latest["close"] - vwap) / vwap if vwap != 0 else 0

        # Range percent
        if latest["open"] > 0:
            feats.range_percent = (latest["high"] - latest["low"]) / latest["open"]
        else:
            feats.range_percent = 0.0

        # Gap from prev close (if we have it)
        if self._prev_close and symbol in self._prev_close:
            prev = self._prev_close[symbol]
            if prev > 0:
                feats.gap_from_prev_close = (latest["open"] - prev) / prev
        elif len(group) > 1:
            prev = group.iloc[-2]["close"]
            if prev > 0:
                feats.gap_from_prev_close = (latest["open"] - prev) / prev

        return feats

    def compute_relative_returns(
        self,
        features_df: pd.DataFrame,
        benchmark_returns: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Compute relative returns vs benchmarks."""
        df = features_df.copy()

        if benchmark_returns:
            if "SPY" in benchmark_returns:
                df["relative_return_spy"] = df["return_1m"] - benchmark_returns["SPY"]
            if "QQQ" in benchmark_returns:
                df["relative_return_qqq"] = df["return_1m"] - benchmark_returns["QQQ"]

        return df

    def compute_benchmark_returns(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        """Compute benchmark returns from benchmark bars."""
        benchmarks = {}
        for bench in self.config.relative_benchmarks:
            bench_bars = bars_df[bars_df["symbol"] == bench]
            if len(bench_bars) >= 2:
                ret = (bench_bars["close"].iloc[-1] - bench_bars["close"].iloc[-2]) / bench_bars["close"].iloc[-2]
                benchmarks[bench] = ret
        return benchmarks

    def compute_cross_sectional_zscores(
        self,
        features_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute cross-sectional z-scores for feature columns."""
        df = features_df.copy()

        if columns is None:
            columns = ["return_1m", "return_5m", "return_15m",
                      "volume_zscore", "range_percent"]

        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std() or 1.0
                df[f"{col}_z"] = (df[col] - mean) / std

        return df

    def set_prev_close(self, prev_close: Dict[str, float]):
        """Set previous day closing prices for gap calculation."""
        self._prev_close = prev_close
