from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE_ROOT = DEFAULT_REPO_ROOT.parent


@dataclass
class AlphaPaths:
    """Canonical file-system layout for the alpha validation pipeline."""

    repo_root: Path | str | None = None
    workspace_root: Path | str | None = None

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root or DEFAULT_REPO_ROOT).expanduser().resolve()
        self.workspace_root = Path(self.workspace_root or self.repo_root.parent).expanduser().resolve()
        self.data_root = self.repo_root / "data"
        self.raw_1m_root = self.data_root / "raw_1m"
        self.bars_5m_root = self.data_root / "bars_5m"
        self.bars_15m_root = self.data_root / "bars_15m"
        self.theme_candidates_root = self.data_root / "theme_candidates"
        self.signals_root = self.data_root / "leadlag_signals"
        self.backtest_root = self.data_root / "alpha_backtests"
        self.scanner_state_dir = self.repo_root / "realtime_dashboard" / "artifacts" / "scanner_state"
        self.universe_csv = self.workspace_root / "P123_Screen_0_20260606.csv"
        self.exclude_symbol_csv = self.workspace_root / "P123_ETFCEF.csv"

    def raw_1m_path(self, trade_date: str) -> Path:
        return self.raw_1m_root / f"date={trade_date}" / "bars_1m.parquet"

    def bars_path(self, trade_date: str, interval: str) -> Path:
        if interval == "5m":
            return self.bars_5m_root / f"date={trade_date}" / "bars_5m.parquet"
        if interval == "15m":
            return self.bars_15m_root / f"date={trade_date}" / "bars_15m.parquet"
        raise ValueError(f"Unsupported interval: {interval}")

    def theme_candidates_path(self, trade_date: str) -> Path:
        return self.theme_candidates_root / f"date={trade_date}" / "theme_candidates.parquet"

    def signals_path(self, trade_date: str) -> Path:
        return self.signals_root / f"date={trade_date}" / "leadlag_signals.parquet"

    def backtest_summary_path(self, trade_date: str) -> Path:
        return self.backtest_root / f"date={trade_date}" / "signal_backtest_summary.csv"

    def backtest_report_path(self, trade_date: str) -> Path:
        return self.backtest_root / f"date={trade_date}" / "signal_backtest_report.md"

    def ensure_parent(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


def load_universe_symbols(
    screen_csv: Path | str,
    exclude_csv: Path | str,
    keep_benchmarks: Iterable[str] | None = None,
    max_symbols: int | None = None,
) -> tuple[list[str], list[str]]:
    """Load the Portfolio123 stock universe and remove ETF/CEF symbols."""

    keep = {str(symbol).upper() for symbol in (keep_benchmarks or [])}
    screen = _read_portfolio123_csv(screen_csv)
    excluded_frame = _read_portfolio123_csv(exclude_csv)

    symbols = screen["Ticker"].astype(str).str.upper().str.strip()
    excluded = set(excluded_frame["Ticker"].astype(str).str.upper().str.strip())

    selected: list[str] = []
    removed: list[str] = []
    for symbol in symbols:
        if not symbol:
            continue
        if symbol in excluded and symbol not in keep:
            removed.append(symbol)
            continue
        if symbol not in selected:
            selected.append(symbol)

    if max_symbols is not None:
        selected = selected[:max_symbols]
    return selected, sorted(set(removed))


def _read_portfolio123_csv(path: Path | str) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    header_row = 0
    with source.open("r", encoding="utf-8-sig") as handle:
        for idx, line in enumerate(handle):
            if line.strip().startswith("Ticker,"):
                header_row = idx
                break
    frame = pd.read_csv(source, skiprows=header_row)
    if "Ticker" not in frame.columns:
        raise ValueError(f"Ticker column not found in {source}")
    return frame

