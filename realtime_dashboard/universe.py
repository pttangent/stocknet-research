"""Universe loading and ETF/CEF exclusion helpers."""

from __future__ import annotations

import os
from io import StringIO
from typing import Iterable, Sequence

import pandas as pd

try:
    from .config import RadarConfig
except ImportError:
    from config import RadarConfig


_SYMBOL_COLUMNS = (
    "Ticker",
    "ticker",
    "Symbol",
    "symbol",
    "Code",
    "code",
)


def _normalize_symbol(symbol: str) -> str:
    value = str(symbol).strip().upper()
    return value.replace("/", "-")


def load_symbol_csv(csv_path: str) -> list[str]:
    """Load symbols from a CSV using the first matching symbol column."""
    if not csv_path or not os.path.exists(csv_path):
        return []

    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as handle:
        lines = handle.readlines()

    header_index = 0
    for index, line in enumerate(lines):
        parts = [part.strip() for part in line.split(",")]
        if any(name in parts for name in _SYMBOL_COLUMNS):
            header_index = index
            break

    csv_buffer = StringIO("".join(lines[header_index:]))
    df = pd.read_csv(csv_buffer)
    column = next((name for name in _SYMBOL_COLUMNS if name in df.columns), None)
    if column is None:
        raise ValueError(f"No symbol column found in {csv_path}")

    symbols = [
        _normalize_symbol(value)
        for value in df[column].dropna().tolist()
        if str(value).strip()
    ]
    return sorted(dict.fromkeys(symbols))


def load_symbols_from_manifest(manifest_path: str) -> list[str]:
    """Load all successful symbols from a parquet manifest."""
    if not manifest_path or not os.path.exists(manifest_path):
        return []

    df = pd.read_csv(manifest_path)
    if "status" in df.columns:
        df = df[df["status"] == "success"]
    column = "symbol" if "symbol" in df.columns else next(
        (name for name in _SYMBOL_COLUMNS if name in df.columns),
        None,
    )
    if column is None:
        raise ValueError(f"No symbol column found in manifest {manifest_path}")

    symbols = [
        _normalize_symbol(value)
        for value in df[column].dropna().tolist()
        if str(value).strip()
    ]
    return sorted(dict.fromkeys(symbols))


def load_excluded_symbols(config: RadarConfig) -> set[str]:
    """Load ETF/CEF blacklist symbols from config."""
    if not config.universe.exclude_etf_cef:
        return set()
    return set(load_symbol_csv(config.universe.exclude_symbol_csv))


def apply_symbol_exclusions(
    symbols: Sequence[str],
    excluded_symbols: Iterable[str],
    keep_symbols: Iterable[str] | None = None,
) -> list[str]:
    """Filter a symbol list while optionally preserving whitelisted keep symbols."""
    excluded = {_normalize_symbol(symbol) for symbol in excluded_symbols}
    keep = {_normalize_symbol(symbol) for symbol in (keep_symbols or [])}
    filtered = []
    for symbol in symbols:
        normalized = _normalize_symbol(symbol)
        if normalized in excluded and normalized not in keep:
            continue
        filtered.append(normalized)
    return list(dict.fromkeys(filtered))


def build_symbol_universe(
    config: RadarConfig,
    universe: str = "watchlist",
    include_benchmarks: bool | None = None,
) -> tuple[list[str], set[str]]:
    """Build the monitoring universe and return (symbols, excluded_symbols)."""
    ds = config.data_source
    if universe == "full_market":
        manifest_path = os.path.join(ds.historical_parquet_dir, "_manifest.csv")
        symbols = load_symbols_from_manifest(manifest_path)
        if not symbols:
            symbols = list(config.universe.custom_watchlist)
    elif universe == "core_500":
        manifest_path = os.path.join(ds.historical_parquet_dir, "_manifest.csv")
        symbols = load_symbols_from_manifest(manifest_path)[: config.universe.core_pool_size]
        if not symbols:
            symbols = list(config.universe.custom_watchlist)
    else:
        symbols = list(config.universe.custom_watchlist)

    excluded_symbols = load_excluded_symbols(config)

    if include_benchmarks is None:
        include_benchmarks = config.universe.keep_benchmark_symbols

    keep_symbols = config.feature.relative_benchmarks if include_benchmarks else []
    symbols = apply_symbol_exclusions(symbols, excluded_symbols, keep_symbols=keep_symbols)

    if include_benchmarks:
        for benchmark in config.feature.relative_benchmarks:
            normalized = _normalize_symbol(benchmark)
            if normalized not in symbols:
                symbols.append(normalized)

    return symbols, excluded_symbols
