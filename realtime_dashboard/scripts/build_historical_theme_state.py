"""Build historical theme state from archived 15m parquet data.

This script replays historical community detection chronologically
and builds an active_theme_paths.json that can be used for warm-starting
the realtime scanner. It ensures theme continuity between historical
15m data and live realtime scans.

Usage:
    python build_historical_theme_state.py \
        --input-dir ../data/archive_15m_from_1m \
        --lookback-days 5 \
        --output-dir ../artifacts/theme_state
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import RadarConfig
from data_feed import HistoricalParquetFeed
from feature_engine import RollingFeatureEngine
from graph_builder import GraphBuilder
from community_detector import CommunityDetector
from scoring import CommunityScorer
from theme_state_manager import ThemeStateManager
from scripts.initialize_live_radar import aggregate_intraday

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build historical theme state from archived parquet."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with symbol=XXX/part-000.parquet files (e.g., archive_15m_from_1m).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for theme_state artifacts. Defaults to artifacts/theme_state.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=5,
        help="How many days back to replay.",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="15m",
        choices=["1m", "5m", "15m"],
        help="Time scale of the input data.",
    )
    parser.add_argument(
        "--universe-manifest",
        type=str,
        default=None,
        help="Optional manifest file to filter symbols.",
    )
    return parser.parse_args()


def load_symbols_from_manifest(manifest_path: str) -> list[str]:
    """Load symbol list from a manifest file (one symbol per line or CSV)."""
    if not os.path.exists(manifest_path):
        logger.warning("Manifest not found: %s", manifest_path)
        return []

    symbols = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Handle CSV format
            if "," in line:
                parts = [p.strip().upper() for p in line.split(",")]
                symbols.extend(parts)
            else:
                symbols.append(line.upper())
    return list(dict.fromkeys(symbols))


def load_all_bars(parquet_dir: str, symbols_filter: list[str] | None = None) -> pd.DataFrame:
    """Load all historical bars from a partitioned parquet directory."""
    all_data = []

    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    entries = os.listdir(parquet_dir)
    logger.info("Found %d entries in %s", len(entries), parquet_dir)

    for entry in entries:
        sym_dir = os.path.join(parquet_dir, entry)
        if not os.path.isdir(sym_dir):
            continue

        # Extract symbol from directory name
        sym = entry.replace("symbol=", "").replace("-", ".")
        if symbols_filter and sym not in symbols_filter:
            continue

        part_file = os.path.join(sym_dir, "part-000.parquet")
        if not os.path.exists(part_file):
            continue

        try:
            df = pd.read_parquet(part_file)
            if "symbol" not in df.columns:
                df["symbol"] = sym
            all_data.append(df)
            logger.debug("Loaded %d rows for %s", len(df), sym)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", part_file, exc)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    return combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def build_theme_state(
    bars_df: pd.DataFrame,
    frequency: str,
    theme_state_manager: ThemeStateManager,
) -> dict:
    """Replay community detection chronologically and build theme state."""
    config = RadarConfig()
    feature_engine = RollingFeatureEngine(config.feature)
    graph_builder = GraphBuilder(config.graph)
    detector = CommunityDetector(config.graph)
    scorer = CommunityScorer(config.scoring)

    timestamps = sorted(bars_df["timestamp"].unique())
    logger.info("Replaying %d timestamps for %s scale", len(timestamps), frequency)

    birth_count = 0
    continuation_count = 0
    revival_count = 0
    weak_count = 0

    for i, ts in enumerate(timestamps):
        # Use cumulative bars up to this timestamp so GraphBuilder has
        # enough history to compute return correlations.
        window_df = bars_df[bars_df["timestamp"] <= ts].copy()
        current_df = bars_df[bars_df["timestamp"] == ts].copy()
        if current_df.empty:
            continue

        feature_engine.ingest_bars(current_df)
        features_df = feature_engine.compute_features()
        if features_df.empty:
            continue

        nodes_df, edges_df = graph_builder.build_graph(features_df, window_df)
        if nodes_df.empty or edges_df.empty:
            continue

        communities_df, memberships_df = detector.detect(nodes_df, edges_df)
        if communities_df.empty:
            continue

        communities_df["level"] = 0
        communities_df["status"] = ""
        communities_df = scorer.score(communities_df, memberships_df, edges_df)

        ts_dt = pd.Timestamp(ts).to_pydatetime()
        communities_df = theme_state_manager.assign_and_update(
            timestamp=ts_dt,
            frequency=frequency,
            communities_df=communities_df,
            memberships_df=memberships_df,
        )

        # Count events
        for _, row in communities_df.iterrows():
            et = row.get("event_type", "")
            if et == "birth":
                birth_count += 1
            elif et == "continuation":
                continuation_count += 1
            elif et == "revival":
                revival_count += 1
            elif et == "weak_continuation":
                weak_count += 1

        if (i + 1) % 10 == 0 or i == len(timestamps) - 1:
            logger.info(
                "  Progress: %d/%d timestamps | paths=%d | births=%d cont=%d revival=%d weak=%d",
                i + 1,
                len(timestamps),
                len(theme_state_manager.paths),
                birth_count,
                continuation_count,
                revival_count,
                weak_count,
            )

    return {
        "timestamps_processed": len(timestamps),
        "total_paths": len(theme_state_manager.paths),
        "active_paths": len(theme_state_manager.get_active_paths()),
        "birth_count": birth_count,
        "continuation_count": continuation_count,
        "revival_count": revival_count,
        "weak_continuation_count": weak_count,
    }


def main() -> None:
    args = parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        config = RadarConfig()
        output_dir = os.path.join(config.output.artifact_dir, "theme_state")

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Load symbols filter if provided
    symbols_filter = None
    if args.universe_manifest:
        symbols_filter = load_symbols_from_manifest(args.universe_manifest)
        logger.info("Loaded %d symbols from manifest", len(symbols_filter))

    # Load historical bars
    logger.info("Loading historical bars from %s", args.input_dir)
    bars_df = load_all_bars(args.input_dir, symbols_filter=symbols_filter)

    if bars_df.empty:
        logger.error("No historical data found in %s", args.input_dir)
        sys.exit(1)

    logger.info(
        "Loaded %d rows across %d symbols | date range: %s to %s",
        len(bars_df),
        bars_df["symbol"].nunique(),
        bars_df["timestamp"].min(),
        bars_df["timestamp"].max(),
    )

    # Filter to lookback window
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.lookback_days)
    before_filter = len(bars_df)
    bars_df = bars_df[bars_df["timestamp"] >= cutoff]
    logger.info(
        "Filtered to lookback=%dd: %d -> %d rows",
        args.lookback_days,
        before_filter,
        len(bars_df),
    )

    if bars_df.empty:
        logger.error("No bars within lookback window")
        sys.exit(1)

    # Build theme state
    theme_state_manager = ThemeStateManager(state_dir=output_dir)
    stats = build_theme_state(
        bars_df=bars_df,
        frequency=args.frequency,
        theme_state_manager=theme_state_manager,
    )

    # Save final state
    theme_state_manager.save()
    logger.info("Theme state saved to %s", theme_state_manager.active_path)

    # Write summary report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": args.input_dir,
        "output_dir": output_dir,
        "lookback_days": args.lookback_days,
        "frequency": args.frequency,
        "stats": stats,
        "top_themes": [
            {
                "theme_path_id": p.theme_path_id,
                "state": p.state,
                "first_seen": p.first_seen,
                "last_seen": p.last_seen,
                "last_frequency": p.last_frequency,
                "member_count": len(p.last_members),
                "core_members": p.core_members[:10],
                "peak_radar_score": round(p.peak_radar_score, 4),
                "age_events": p.age_events,
            }
            for p in sorted(
                theme_state_manager.paths.values(),
                key=lambda x: x.peak_radar_score,
                reverse=True,
            )[:20]
        ],
    }

    report_path = os.path.join(output_dir, "historical_theme_state_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Report saved to %s", report_path)
    logger.info("=" * 60)
    logger.info("Historical theme state build complete!")
    logger.info("  Total paths: %d", stats["total_paths"])
    logger.info("  Active paths: %d", stats["active_paths"])
    logger.info("  Births: %d", stats["birth_count"])
    logger.info("  Continuations: %d", stats["continuation_count"])
    logger.info("  Revivals: %d", stats["revival_count"])
    logger.info("  Weak continuations: %d", stats["weak_continuation_count"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
