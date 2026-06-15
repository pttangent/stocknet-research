from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from stocknetv2.domain.snapshot.snapshot_clock import SnapshotClock


@dataclass(frozen=True)
class LegacySourceLayout:
    data_root: Path | str

    def __post_init__(self) -> None:
        data_root = Path(self.data_root).expanduser().resolve()
        object.__setattr__(self, "data_root", data_root)
        object.__setattr__(self, "bars_5m_root", data_root / "bars_5m")
        object.__setattr__(self, "trade_flow_1m_root", data_root / "trade_flow_1m")
        object.__setattr__(self, "features_1m_root", data_root / "features_1m")


@dataclass(frozen=True)
class LegacyDuckDBSource:
    database_path: Path | str
    source_timezone: str = "auto"

    def __post_init__(self) -> None:
        database_path = Path(self.database_path).expanduser().resolve()
        object.__setattr__(self, "database_path", database_path)


@dataclass(frozen=True)
class TradeDateInputs:
    trade_date: str
    bars_5m: pd.DataFrame
    trade_flow_1m: pd.DataFrame
    features_1m: pd.DataFrame
    data_version: str


class SourceProtocol(Protocol):
    pass


class MarketReadRepository:
    """Read legacy source partitions without leaking legacy business logic."""

    def __init__(
        self,
        source: LegacySourceLayout | LegacyDuckDBSource,
        symbol_limit: int | None = None,
    ) -> None:
        self._source = source
        self._symbol_limit = symbol_limit
        self._resolved_duckdb_source_timezone: str | None = None

    def list_available_trade_dates(self, dataset_name: str) -> list[str]:
        if isinstance(self._source, LegacyDuckDBSource):
            return self._list_available_trade_dates_from_duckdb(dataset_name)
        root = self._dataset_root(dataset_name)
        if not root.exists():
            return []

        trade_dates: list[str] = []
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("date="):
                trade_dates.append(child.name.split("=", 1)[1])
        return sorted(trade_dates)

    def load_trade_date_inputs(self, trade_date: str) -> TradeDateInputs:
        if isinstance(self._source, LegacyDuckDBSource):
            return self._load_trade_date_inputs_from_duckdb(trade_date)

        bars_5m = self._read_optional_parquet(self._source.bars_5m_root / f"date={trade_date}" / "bars_5m.parquet")
        trade_flow_1m = self._read_optional_parquet(
            self._source.trade_flow_1m_root / f"date={trade_date}" / "trade_flow_1m.parquet"
        )
        features_1m = self._read_optional_parquet(
            self._source.features_1m_root / f"date={trade_date}" / "features_1m.parquet"
        )
        bars_5m, trade_flow_1m, features_1m = self._apply_symbol_limit(
            bars_5m=bars_5m,
            trade_flow_1m=trade_flow_1m,
            features_1m=features_1m,
        )

        data_version = f"bars_5m:{trade_date}|trade_flow_1m:{trade_date}|features_1m:{trade_date}"
        if self._symbol_limit is not None:
            data_version = f"{data_version}|symbol_limit:{self._symbol_limit}"
        return TradeDateInputs(
            trade_date=trade_date,
            bars_5m=bars_5m,
            trade_flow_1m=trade_flow_1m,
            features_1m=features_1m,
            data_version=data_version,
        )

    def _dataset_root(self, dataset_name: str) -> Path:
        if dataset_name == "bars_5m":
            return self._source.bars_5m_root
        if dataset_name == "trade_flow_1m":
            return self._source.trade_flow_1m_root
        if dataset_name == "features_1m":
            return self._source.features_1m_root
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    @staticmethod
    def _read_optional_parquet(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    def _list_available_trade_dates_from_duckdb(self, dataset_name: str) -> list[str]:
        table_name = self._table_name_for_dataset(dataset_name)
        with duckdb.connect(str(self._source.database_path), read_only=True) as connection:
            rows = connection.execute(
                f"SELECT DISTINCT CAST(date AS VARCHAR) AS trade_date FROM {table_name} ORDER BY trade_date"
            ).fetchall()
        return [row[0] for row in rows]

    def _load_trade_date_inputs_from_duckdb(self, trade_date: str) -> TradeDateInputs:
        with duckdb.connect(str(self._source.database_path), read_only=True) as connection:
            self._resolved_duckdb_source_timezone = self._resolve_duckdb_source_timezone(connection, trade_date)
            limited_symbols = self._load_limited_symbols(connection, trade_date)
            filter_sql, filter_params = self._build_symbol_filter_sql(limited_symbols)
            bars_5m = connection.execute(
                f"""
                SELECT timestamp, close, symbol, date
                FROM bars_5m
                WHERE date = ?
                {filter_sql}
                """,
                [trade_date, *filter_params],
            ).df()
            trade_flow_raw = connection.execute(
                f"""
                SELECT ticker, minute, dollar_volume, imbalance_proxy, large_trade_dollar_volume, date
                FROM trade_flow_1m
                WHERE date = ?
                {filter_sql.replace('symbol', 'ticker')}
                """,
                [trade_date, *filter_params],
            ).df()
            features_raw = connection.execute(
                f"""
                SELECT symbol, timestamp, ret_1m_past, volume_z_proxy, large_trade_ratio, imbalance_proxy, date
                FROM features_1m
                WHERE date = ?
                {filter_sql}
                """,
                [trade_date, *filter_params],
            ).df()

        bars_5m = self._normalize_duckdb_timestamps(bars_5m, ["timestamp"]).sort_values(
            ["timestamp", "symbol"]
        ).reset_index(drop=True)
        trade_flow_1m = self._normalize_trade_flow_1m(trade_flow_raw).sort_values(
            ["timestamp", "symbol"]
        ).reset_index(drop=True)
        features_1m = self._normalize_features_1m(features_raw).sort_values(
            ["timestamp", "symbol"]
        ).reset_index(drop=True)
        data_version = f"duckdb:{self._source.database_path.name}:{trade_date}"
        if self._symbol_limit is not None:
            data_version = f"{data_version}:symbol_limit:{self._symbol_limit}"
        return TradeDateInputs(
            trade_date=trade_date,
            bars_5m=bars_5m,
            trade_flow_1m=trade_flow_1m,
            features_1m=features_1m,
            data_version=data_version,
        )

    @staticmethod
    def _table_name_for_dataset(dataset_name: str) -> str:
        if dataset_name == "bars_5m":
            return "bars_5m"
        if dataset_name == "trade_flow_1m":
            return "trade_flow_1m"
        if dataset_name == "features_1m":
            return "features_1m"
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    @staticmethod
    def _normalize_timestamps(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        normalized = frame.copy()
        for column in columns:
            if column in normalized.columns:
                normalized[column] = pd.to_datetime(normalized[column], utc=True)
        return normalized

    def _normalize_duckdb_timestamps(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        normalized = frame.copy()
        for column in columns:
            if column not in normalized.columns:
                continue
            series = pd.to_datetime(normalized[column], errors="coerce")
            if getattr(series.dt, "tz", None) is None:
                series = series.dt.tz_localize(self._duckdb_source_timezone()).dt.tz_convert("UTC")
            else:
                series = series.dt.tz_convert("UTC")
            normalized[column] = series
        return normalized

    def _normalize_trade_flow_1m(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        normalized = frame.rename(
            columns={
                "ticker": "symbol",
                "minute": "timestamp",
                "imbalance_proxy": "imbalance_z",
            }
        ).copy()
        normalized = self._normalize_duckdb_timestamps(normalized, ["timestamp"])
        normalized["flow_impulse_score"] = normalized.get("imbalance_z", 0.0).astype(float)
        if "large_trade_ratio_z" not in normalized.columns:
            dollar_volume = normalized.get("dollar_volume", 0.0).replace(0, pd.NA)
            normalized["large_trade_ratio_z"] = (
                normalized.get("large_trade_dollar_volume", 0.0).astype(float) / dollar_volume.astype(float)
            ).fillna(0.0)
        return normalized

    def _normalize_features_1m(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        normalized = frame.rename(
            columns={
                "ret_1m_past": "ret_1m",
                "volume_z_proxy": "volume_z_12",
                "imbalance_proxy": "imbalance_z",
                "large_trade_ratio": "large_trade_ratio_z",
            }
        ).copy()
        normalized = self._normalize_duckdb_timestamps(normalized, ["timestamp", "bar_end"])
        return normalized

    def _load_limited_symbols(self, connection: duckdb.DuckDBPyConnection, trade_date: str) -> list[str] | None:
        if self._symbol_limit is None:
            return None
        rows = connection.execute(
            """
            SELECT symbol
            FROM bars_5m
            WHERE date = ?
            GROUP BY symbol
            ORDER BY symbol
            LIMIT ?
            """,
            [trade_date, self._symbol_limit],
        ).fetchall()
        return [row[0] for row in rows]

    @staticmethod
    def _build_symbol_filter_sql(symbols: list[str] | None) -> tuple[str, list[str]]:
        if not symbols:
            return "", []
        placeholders = ", ".join(["?"] * len(symbols))
        return f"AND symbol IN ({placeholders})", symbols

    def _duckdb_source_timezone(self) -> str:
        if self._resolved_duckdb_source_timezone:
            return self._resolved_duckdb_source_timezone
        return self._source.source_timezone if isinstance(self._source, LegacyDuckDBSource) else "UTC"

    def _resolve_duckdb_source_timezone(
        self,
        connection: duckdb.DuckDBPyConnection,
        trade_date: str,
    ) -> str:
        if self._resolved_duckdb_source_timezone:
            return self._resolved_duckdb_source_timezone
        if self._source.source_timezone != "auto":
            return self._source.source_timezone

        snapshot_clock = SnapshotClock()
        session_open_utc = snapshot_clock.session_open_timestamp(trade_date).to_pydatetime()
        session_close_utc = snapshot_clock.iter_trade_date(trade_date)[-1].to_pydatetime()

        best_timezone = "UTC"
        best_count = -1
        for candidate in ("UTC", "Asia/Taipei"):
            candidate_zone = ZoneInfo(candidate)
            open_local = session_open_utc.astimezone(candidate_zone).time()
            close_local = session_close_utc.astimezone(candidate_zone).time()
            count = self._count_bars_in_local_session(
                connection=connection,
                trade_date=trade_date,
                session_open_local=open_local,
                session_close_local=close_local,
            )
            if count > best_count:
                best_timezone = candidate
                best_count = count
        return best_timezone

    @staticmethod
    def _count_bars_in_local_session(
        *,
        connection: duckdb.DuckDBPyConnection,
        trade_date: str,
        session_open_local: datetime.time,
        session_close_local: datetime.time,
    ) -> int:
        if session_open_local <= session_close_local:
            return int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM bars_5m
                    WHERE date = ?
                      AND CAST(timestamp AS TIME) BETWEEN ? AND ?
                    """,
                    [trade_date, session_open_local.isoformat(), session_close_local.isoformat()],
                ).fetchone()[0]
            )

        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM bars_5m
                WHERE date = ?
                  AND (CAST(timestamp AS TIME) >= ? OR CAST(timestamp AS TIME) <= ?)
                """,
                [trade_date, session_open_local.isoformat(), session_close_local.isoformat()],
            ).fetchone()[0]
        )

    def _apply_symbol_limit(
        self,
        *,
        bars_5m: pd.DataFrame,
        trade_flow_1m: pd.DataFrame,
        features_1m: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self._symbol_limit is None or bars_5m.empty or "symbol" not in bars_5m.columns:
            return bars_5m, trade_flow_1m, features_1m

        allowed_symbols = sorted(bars_5m["symbol"].dropna().astype(str).unique().tolist())[: self._symbol_limit]
        return (
            self._filter_frame_by_symbols(bars_5m, allowed_symbols),
            self._filter_frame_by_symbols(trade_flow_1m, allowed_symbols),
            self._filter_frame_by_symbols(features_1m, allowed_symbols),
        )

    @staticmethod
    def _filter_frame_by_symbols(frame: pd.DataFrame, allowed_symbols: list[str]) -> pd.DataFrame:
        if frame.empty or "symbol" not in frame.columns:
            return frame
        return frame[frame["symbol"].astype(str).isin(allowed_symbols)].copy()
