"""Intraday logging and CSV output for alerts and snapshots."""

import pandas as pd
import os
from datetime import datetime, date
from typing import Optional
import logging

try:
    from .config import OutputConfig
except ImportError:
    from config import OutputConfig

logger = logging.getLogger(__name__)


class IntradayLogger:
    """Log community snapshots, alerts, and members to disk."""

    def __init__(self, config: Optional[OutputConfig] = None):
        self.config = config or OutputConfig()
        self._today = date.today().isoformat()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create output directories for today."""
        for path_attr in ["bars_path", "snapshots_path", "members_path",
                          "alerts_path", "edges_path", "review_path"]:
            path = getattr(self.config, path_attr).format(date=self._today)
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def log_snapshots(self, snapshots_df: pd.DataFrame):
        """Append community snapshots to CSV."""
        if snapshots_df.empty:
            return

        path = self.config.snapshots_path.format(date=self._today)
        mode = "a" if os.path.exists(path) else "w"
        header = not os.path.exists(path) or os.path.getsize(path) == 0
        snapshots_df.to_csv(path, mode=mode, header=header, index=False)

    def log_members(self, members_df: pd.DataFrame):
        """Append member details to CSV."""
        if members_df.empty:
            return

        path = self.config.members_path.format(date=self._today)
        mode = "a" if os.path.exists(path) else "w"
        header = not os.path.exists(path) or os.path.getsize(path) == 0
        members_df.to_csv(path, mode=mode, header=header, index=False)

    def log_alerts(self, alerts_df: pd.DataFrame):
        """Append alerts to CSV."""
        if alerts_df.empty:
            return

        path = self.config.alerts_path.format(date=self._today)
        mode = "a" if os.path.exists(path) else "w"
        header = not os.path.exists(path) or os.path.getsize(path) == 0
        alerts_df.to_csv(path, mode=mode, header=header, index=False)

    def log_edges(self, edges_df: pd.DataFrame, timestamp: datetime):
        """Append edges with timestamp to CSV."""
        if edges_df.empty:
            return

        df = edges_df.copy()
        df["timestamp"] = timestamp

        path = self.config.edges_path.format(date=self._today)
        mode = "a" if os.path.exists(path) else "w"
        header = not os.path.exists(path) or os.path.getsize(path) == 0
        df.to_csv(path, mode=mode, header=header, index=False)

    def log_bars(self, bars_df: pd.DataFrame):
        """Append bars to parquet."""
        if bars_df.empty:
            return

        path = self.config.bars_path.format(date=self._today)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, bars_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp", "symbol"])
            combined.to_parquet(path, index=False)
        else:
            bars_df.to_parquet(path, index=False)

    def write_review(self, content: str):
        """Write intraday review markdown."""
        path = self.config.review_path.format(date=self._today)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def read_snapshots(self) -> pd.DataFrame:
        """Read all snapshots for today."""
        path = self.config.snapshots_path.format(date=self._today)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    def read_alerts(self) -> pd.DataFrame:
        """Read all alerts for today."""
        path = self.config.alerts_path.format(date=self._today)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    def read_members(self) -> pd.DataFrame:
        """Read all member data for today."""
        path = self.config.members_path.format(date=self._today)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    def read_edges(self) -> pd.DataFrame:
        """Read all edges for today."""
        path = self.config.edges_path.format(date=self._today)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()


class OneMinuteArchiveWriter:
    """Persist a traceable local 1-minute archive for later replay and research."""

    def __init__(self, config: Optional[OutputConfig] = None):
        self.config = config or OutputConfig()
        os.makedirs(self.config.archive_dir, exist_ok=True)

    def append_bars(self, bars_df: pd.DataFrame, provider: str = "yahoo", interval: str = "1m") -> None:
        """Append fetched bars to the archive, partitioned by session date."""
        if bars_df.empty:
            return

        df = bars_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["archive_fetched_at"] = pd.Timestamp.utcnow()
        df["archive_provider"] = provider
        df["archive_interval"] = interval
        df["trade_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

        for trade_date, trade_slice in df.groupby("trade_date", sort=True):
            path = self.config.archive_bars_path.format(date=trade_date)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            payload = trade_slice.drop(columns=["trade_date"]).sort_values(["symbol", "timestamp"])
            if os.path.exists(path):
                existing = pd.read_parquet(path)
                payload = pd.concat([existing, payload], ignore_index=True)
                payload = payload.sort_values(["symbol", "timestamp", "archive_fetched_at"])
                payload = payload.drop_duplicates(subset=["timestamp", "symbol"], keep="last")

            payload.to_parquet(path, index=False)
            self._append_manifest_row(
                trade_date=trade_date,
                payload=trade_slice,
                provider=provider,
                interval=interval,
            )

    def _append_manifest_row(
        self,
        trade_date: str,
        payload: pd.DataFrame,
        provider: str,
        interval: str,
    ) -> None:
        manifest_row = pd.DataFrame([{
            "trade_date": trade_date,
            "provider": provider,
            "interval": interval,
            "rows_written": int(len(payload)),
            "symbol_count": int(payload["symbol"].nunique()),
            "min_timestamp": payload["timestamp"].min(),
            "max_timestamp": payload["timestamp"].max(),
            "appended_at": pd.Timestamp.utcnow(),
        }])

        path = self.config.archive_manifest_path
        if os.path.exists(path):
            existing = pd.read_csv(path)
            combined = pd.concat([existing, manifest_row], ignore_index=True)
            combined.to_csv(path, index=False)
        else:
            manifest_row.to_csv(path, index=False)


class PartitionedParquetWriter:
    """Write symbol-partitioned parquet files compatible with HistoricalParquetFeed."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def write(self, bars_df: pd.DataFrame) -> int:
        if bars_df.empty:
            return 0

        df = bars_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        rows_written = 0
        for symbol, group in df.groupby("symbol", sort=True):
            target_dir = os.path.join(self.root_dir, f"symbol={symbol}")
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, "part-000.parquet")
            payload = group.sort_values("timestamp")
            if os.path.exists(target_path):
                existing = pd.read_parquet(target_path)
                payload = pd.concat([existing, payload], ignore_index=True)
                payload = payload.sort_values("timestamp")
                payload = payload.drop_duplicates(subset=["timestamp"], keep="last")
            payload.to_parquet(target_path, index=False)
            rows_written += len(group)
        return rows_written


class ScannerStateWriter:
    """Persist the latest headless scanner state for downstream consumers."""

    def __init__(self, config: Optional[OutputConfig] = None):
        self.config = config or OutputConfig()
        self.state_dir = os.path.join(self.config.artifact_dir, "scanner_state")
        os.makedirs(self.state_dir, exist_ok=True)

    def write_json(self, name: str, payload: dict) -> str:
        path = os.path.join(self.state_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            import json
            json.dump(payload, handle, indent=2, default=str)
        return path
