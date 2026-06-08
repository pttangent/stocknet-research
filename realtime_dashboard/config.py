"""Configuration for the realtime community monitoring radar."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import time
import os


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_STOCKNET_DIR = os.path.dirname(_MODULE_DIR)
_WORKSPACE_DIR = os.path.dirname(_STOCKNET_DIR)


@dataclass
class TimeConfig:
    """Market hours and warmup configuration."""
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    warmup_minutes: int = 15  # 09:30-09:45: collect only, no alerts
    scan_interval_seconds: int = 60  # 1m scan
    premarket_warmup: bool = True


@dataclass
class UniverseConfig:
    """Stock universe configuration."""
    core_pool_size: int = 500
    max_universe_size: int = 1000
    min_dollar_volume: float = 5_000_000  # min avg dollar volume
    exclude_etf_cef: bool = True
    keep_benchmark_symbols: bool = False
    exclude_symbol_csv: str = field(default_factory=lambda: os.path.join(
        _WORKSPACE_DIR, "P123_ETFCEF.csv"
    ))
    sectors_of_interest: List[str] = field(default_factory=lambda: [
        "Technology", "Financials", "Healthcare", "Energy",
        "Industrials", "Consumer Cyclical", "Communication Services"
    ])
    custom_watchlist: List[str] = field(default_factory=lambda: [
        # AI / Semiconductors
        "NVDA", "AMD", "AVGO", "TSM", "AMAT", "ACMR", "MRVL", "MU",
        # Big Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
        # Financials
        "JPM", "BAC", "GS", "MS", "BLK",
        # Energy / Nuclear
        "CCJ", "URA", "SMR", "OKLO", "NEE",
        # Space / Defense
        "LMT", "NOC", "RTX", "BA",
    ])


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    windows: List[int] = field(default_factory=lambda: [1, 5, 15])
    volume_zscore_lookback: int = 20
    vwap_enabled: bool = False
    relative_benchmarks: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])


@dataclass
class GraphConfig:
    """Graph construction parameters."""
    min_return_corr: float = 0.4
    min_volume_corr: float = 0.5
    min_directional_agreement: float = 0.7
    edge_weight_formula: str = "0.5*rc + 0.3*vc + 0.2*da"
    # Louvain / Leiden
    resolution: float = 1.0
    min_community_size: int = 4
    min_edge_density: float = 0.1


@dataclass
class ScoringConfig:
    """Community scoring weights."""
    radar_weights: Dict[str, float] = field(default_factory=lambda: {
        "coherence": 0.25,
        "volume_expansion": 0.20,
        "breadth": 0.20,
        "relative_return": 0.15,
        "edge_growth": 0.10,
        "member_stability": 0.10,
    })
    early_weights: Dict[str, float] = field(default_factory=lambda: {
        "volume_expansion": 0.30,
        "edge_growth": 0.25,
        "breadth": 0.20,
        "return_spike": 0.15,
        "coherence": 0.10,
    })
    confirmation_weights: Dict[str, float] = field(default_factory=lambda: {
        "coherence": 0.30,
        "member_stability": 0.25,
        "breadth": 0.20,
        "relative_return": 0.15,
        "volume_expansion": 0.10,
    })
    zscore_lookback_windows: int = 50


@dataclass
class AlertConfig:
    """Alert level thresholds."""
    # Level 1: 1m Early Alert
    l1_min_members: int = 4
    l1_min_coherence: float = 0.3
    l1_min_volume_expansion: float = 0.5
    l1_min_breadth: float = 0.55

    # Level 2: 1m Persistent
    l2_min_consecutive_windows: int = 3
    l2_min_member_stability: float = 0.4
    l2_min_coherence: float = 0.3

    # Level 3: 5m Pre-confirmed
    l3_min_coherence: float = 0.35
    l3_min_breadth: float = 0.55
    l3_min_members: int = 4

    # Level 4: 15m Confirmed
    l4_min_breadth: float = 0.6
    l4_min_coherence: float = 0.35
    l4_min_relative_return: float = 0.0
    l4_min_members: int = 4

    # Level 5: Expansion
    l5_min_member_growth: float = 0.15  # 15% member increase
    l5_min_edge_density_growth: float = 0.05

    # Decay detection
    decay_coherence_drop: float = 0.15
    decay_breadth_drop: float = 0.15
    decay_min_windows_since_peak: int = 2


@dataclass
class DashboardConfig:
    """Dashboard display settings."""
    top_communities: int = 20
    alert_feed_max: int = 50
    member_table_max: int = 50
    network_max_nodes: int = 30
    network_max_edges: int = 100
    auto_refresh_seconds: int = 60
    theme_colors: Dict[str, str] = field(default_factory=lambda: {
        "Level 0": "#6c757d",
        "Level 1": "#ffc107",
        "Level 2": "#fd7e14",
        "Level 3": "#17a2b8",
        "Level 4": "#28a745",
        "Level 5": "#dc3545",
        "Decay": "#6f42c1",
    })


@dataclass
class DataSourceConfig:
    """Live data source configuration."""
    provider: str = "yahoo"  # "yahoo", "alpaca", "polygon"
    interval: str = "1m"     # "1m", "5m", "15m"
    scan_mode: str = "chunked"  # "chunked", "full_parallel"
    lookback_days: int = 7   # How many days of history to fetch on init
    timeout_seconds: float = 15.0
    retries: int = 2
    max_workers: int = 32    # Concurrent fetch threads
    rate_limit_delay: float = 0.03  # Seconds between requests
    chunk_size: int = 200    # Symbols per scan (round-robin for large universes)
    # Historical parquet for warm-up
    historical_parquet_dir: str = field(default_factory=lambda: os.path.join(
        _STOCKNET_DIR, "artifacts", "parquet_5m_final"
    ))
    use_historical_warmup: bool = True
    warmup_lookback_bars: int = 100  # Number of historical bars to preload
    archive_subdir: str = "archive_1m"
    archive_metadata_name: str = "archive_manifest.csv"


@dataclass
class OutputConfig:
    """Output file paths."""
    base_dir: str = field(default_factory=lambda: os.path.join(
        _MODULE_DIR, "data"
    ))
    artifact_dir: str = field(default_factory=lambda: os.path.join(
        _MODULE_DIR, "artifacts"
    ))

    @property
    def bars_path(self) -> str:
        return os.path.join(self.base_dir, "{date}", "1m_bars.parquet")

    @property
    def snapshots_path(self) -> str:
        return os.path.join(self.base_dir, "{date}", "community_snapshots.csv")

    @property
    def members_path(self) -> str:
        return os.path.join(self.base_dir, "{date}", "community_members.csv")

    @property
    def alerts_path(self) -> str:
        return os.path.join(self.base_dir, "{date}", "live_alerts.csv")

    @property
    def edges_path(self) -> str:
        return os.path.join(self.base_dir, "{date}", "community_edges.csv")

    @property
    def review_path(self) -> str:
        return os.path.join(self.artifact_dir, "{date}", "intraday_review.md")

    @property
    def archive_dir(self) -> str:
        return os.path.join(self.base_dir, "archive_1m")

    @property
    def archive_bars_path(self) -> str:
        return os.path.join(self.archive_dir, "date={date}", "1m_bars.parquet")

    @property
    def archive_manifest_path(self) -> str:
        return os.path.join(self.archive_dir, "archive_manifest.csv")


@dataclass
class RadarConfig:
    """Master configuration object."""
    time: TimeConfig = field(default_factory=TimeConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)

    # Mode: "demo" | "live" | "hybrid" (historical warmup + live polling)
    mode: str = "demo"
    log_level: str = "INFO"


# Global default config instance
DEFAULT_CONFIG = RadarConfig()
