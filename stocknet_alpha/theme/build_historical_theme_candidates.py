from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.config import AlphaPaths
from stocknet_alpha.theme.build_theme_candidates import write_theme_candidates
from stocknet_alpha.theme.historical_candidates import build_theme_candidates_from_market_data


def load_daily_market_inputs(
    paths: AlphaPaths,
    trade_date: str,
    *,
    bars_path: Path | str | None = None,
    trade_flow_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars_source = Path(bars_path).expanduser().resolve() if bars_path else paths.bars_path(trade_date, "5m")
    flow_source = Path(trade_flow_path).expanduser().resolve() if trade_flow_path else paths.trade_flow_1m_path(trade_date)
    bars = pd.read_parquet(bars_source)
    if flow_source.exists():
        trade_flow = pd.read_parquet(flow_source)
    else:
        trade_flow = pd.DataFrame(
            columns=[
                "ticker",
                "minute",
                "imbalance_proxy",
                "dollar_volume",
                "buy_vol_proxy",
                "sell_vol_proxy",
                "large_trade_dollar_volume",
                "off_exchange_volume",
                "volume",
                "trade_count",
            ]
        )
    return bars, trade_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build historical 5m theme candidates from bars_5m and trade_flow_1m.")
    parser.add_argument("--date", required=True, help="Trade date in YYYY-MM-DD format.")
    parser.add_argument("--bars-5m", default="", help="Optional override path for the 5m bars parquet.")
    parser.add_argument("--trade-flow-1m", default="", help="Optional override path for the 1m trade-flow parquet.")
    parser.add_argument("--lookback-bars", type=int, default=6)
    parser.add_argument("--top-symbols", type=int, default=200)
    parser.add_argument("--min-members", type=int, default=3)
    parser.add_argument("--min-theme-score", type=float, default=0.5)
    parser.add_argument("--min-pair-corr", type=float, default=0.6)
    parser.add_argument("--theme-path-score-method", default="jaccard", choices=["jaccard", "overlap_small"])
    parser.add_argument("--theme-path-min-overlap", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AlphaPaths()
    bars_5m, trade_flow_1m = load_daily_market_inputs(
        paths,
        args.date,
        bars_path=args.bars_5m or None,
        trade_flow_path=args.trade_flow_1m or None,
    )
    candidates = build_theme_candidates_from_market_data(
        bars_5m,
        trade_flow_1m,
        trade_date=args.date,
        lookback_bars=args.lookback_bars,
        top_symbols=args.top_symbols,
        min_members=args.min_members,
        min_theme_score=args.min_theme_score,
        min_pair_corr=args.min_pair_corr,
        theme_path_score_method=args.theme_path_score_method,
        theme_path_min_overlap=args.theme_path_min_overlap,
    )
    output_path = write_theme_candidates(candidates, paths, args.date)
    print(f"Wrote {len(candidates)} historical theme candidates to {output_path}")


if __name__ == "__main__":
    main()
