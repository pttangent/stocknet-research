from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.backtest.confirmation_relabel import relabel_evaluated_trades_file
from stocknet_alpha.config import AlphaPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild and relabel evaluated trade confirmations from causal theme candidates.")
    parser.add_argument("--evaluated-trades", required=True, help="Source evaluated_trades.parquet path.")
    parser.add_argument("--output-path", required=True, help="Target relabeled parquet path.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="Inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--lookback-bars", type=int, default=6)
    parser.add_argument("--top-symbols", type=int, default=60)
    parser.add_argument("--min-members", type=int, default=3)
    parser.add_argument("--min-theme-score", type=float, default=0.20)
    parser.add_argument("--min-pair-corr", type=float, default=0.55)
    parser.add_argument("--theme-path-score-method", default="jaccard", choices=["jaccard", "overlap_small"])
    parser.add_argument("--theme-path-min-overlap", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = relabel_evaluated_trades_file(
        args.evaluated_trades,
        output_path=args.output_path,
        paths=AlphaPaths(),
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_bars=args.lookback_bars,
        top_symbols=args.top_symbols,
        min_members=args.min_members,
        min_theme_score=args.min_theme_score,
        min_pair_corr=args.min_pair_corr,
        theme_path_score_method=args.theme_path_score_method,
        theme_path_min_overlap=args.theme_path_min_overlap,
    )
    print(f"Wrote relabeled evaluated trades to {output_path}")


if __name__ == "__main__":
    main()
