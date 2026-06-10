from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.config import AlphaPaths


def load_current_state(path: Path | str) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_theme_candidates_from_state(
    current_state: dict[str, Any],
    trade_date: str,
    min_member_count: int = 3,
) -> pd.DataFrame:
    """Normalize 5m theme candidates and mark whether 15m confirms them."""

    five_minute = current_state.get("five_minute", {}).get("top_communities", [])
    fifteen_minute = current_state.get("fifteen_minute", {}).get("top_communities", [])
    confirmation_index = {
        str(row.get("theme_path_id")): row
        for row in fifteen_minute
        if _matches_trade_date(row.get("timestamp"), trade_date)
    }

    rows: list[dict[str, Any]] = []
    for row in five_minute:
        if not _matches_trade_date(row.get("timestamp"), trade_date):
            continue
        theme_path_id = str(row.get("theme_path_id", "")).strip()
        if not theme_path_id:
            continue
        member_count = int(row.get("member_count", 0) or 0)
        if member_count < min_member_count:
            continue
        confirmation = confirmation_index.get(theme_path_id, {})
        confirmed = bool(confirmation)
        members = _coerce_members(row.get("members") or row.get("top_members") or "")
        rows.append(
            {
                "trade_date": trade_date,
                "signal_timestamp": pd.Timestamp(row["timestamp"], tz="UTC"),
                "theme_path_id": theme_path_id,
                "community_id": str(row.get("community_id", "")),
                "members": ",".join(members),
                "member_count": member_count,
                "confirmed_on_15m": confirmed,
                "confirmation_timestamp": pd.Timestamp(confirmation["timestamp"], tz="UTC") if confirmed else pd.NaT,
                "radar_score_5m": float(row.get("radar_score", 0.0) or 0.0),
                "confirmation_score_5m": float(row.get("confirmation_score", 0.0) or 0.0),
                "coherence_5m": float(row.get("coherence", 0.0) or 0.0),
                "breadth_5m": float(row.get("breadth", 0.0) or 0.0),
                "relative_return_5m": float(row.get("relative_return", 0.0) or 0.0),
                "volume_expansion_5m": float(row.get("volume_expansion", 0.0) or 0.0),
                "theme_score": (
                    0.40 * float(row.get("radar_score", 0.0) or 0.0)
                    + 0.30 * float(row.get("confirmation_score", 0.0) or 0.0)
                    + 0.20 * float(confirmation.get("confirmation_score", 0.0) or 0.0)
                    + 0.10 * float(confirmed)
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["confirmed_on_15m", "theme_score"], ascending=[False, False]).reset_index(drop=True)


def write_theme_candidates(frame: pd.DataFrame, paths: AlphaPaths, trade_date: str) -> Path:
    output_path = paths.ensure_parent(paths.theme_candidates_path(trade_date))
    frame.to_parquet(output_path, index=False)
    return output_path


def _matches_trade_date(raw_timestamp: Any, trade_date: str) -> bool:
    if raw_timestamp in (None, ""):
        return False
    timestamp = pd.Timestamp(raw_timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.date().isoformat() == trade_date


def _coerce_members(value: str) -> list[str]:
    return [member.strip().upper() for member in str(value).split(",") if member.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized theme candidates from scanner runtime state.")
    parser.add_argument("--date", required=True, help="Trade date partition in YYYY-MM-DD format.")
    parser.add_argument("--current-state", default="", help="Optional override path for scanner_state/current_state.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AlphaPaths()
    current_state_path = Path(args.current_state).expanduser().resolve() if args.current_state else paths.scanner_state_dir / "current_state.json"
    current_state = load_current_state(current_state_path)
    candidates = build_theme_candidates_from_state(current_state, args.date)
    output_path = write_theme_candidates(candidates, paths, args.date)
    print(f"Wrote {len(candidates)} theme candidates to {output_path}")


if __name__ == "__main__":
    main()
