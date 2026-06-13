from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stocknet_alpha.config import AlphaPaths
from stocknet_alpha.data.us_market_data import initialize_market_database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the semantic research/backtest views in the shared U.S. market DuckDB.")
    parser.add_argument("--repo-root", default=str(ROOT_DIR), help="StockNet repository root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    paths = AlphaPaths(repo_root=repo_root)
    db_path = initialize_market_database(paths)
    print(db_path)


if __name__ == "__main__":
    main()
