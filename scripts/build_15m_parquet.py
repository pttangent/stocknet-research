#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocknetwork.run_metadata import create_run_context


YAHOO_CHART_BASE_URL = "https://query2.finance.yahoo.com/v8/finance/chart/"
DEFAULT_LOOKBACK_DAYS = 60
DEFAULT_INTERVAL = "15m"
DEFAULT_SLEEP_SECONDS = 0.35
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_RETRIES = 3
DEFAULT_WORKERS = 8
UTC = timezone.utc
MANIFEST_COLUMNS = [
    "source_symbol",
    "symbol",
    "status",
    "row_count",
    "min_timestamp",
    "max_timestamp",
    "exchange_timezone",
    "last_fetch_at",
    "error_message",
]
SUCCESS_COLUMNS = [
    "source_symbol",
    "symbol",
    "row_count",
    "min_timestamp",
    "max_timestamp",
    "exchange_timezone",
    "last_fetch_at",
]
FAILURE_COLUMNS = [
    "source_symbol",
    "symbol",
    "last_fetch_at",
    "error_message",
]


@dataclass(frozen=True)
class SymbolListRow:
    source_symbol: str
    symbol: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a symbol-partitioned parquet database of 15m Yahoo Finance bars."
    )
    parser.add_argument("--input", required=True, help="Path to the source CSV file with a Ticker column.")
    parser.add_argument("--output", required=True, help="Output directory for the parquet database.")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Yahoo chart interval. Default: 15m")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="How many calendar days to fetch. Default: 60",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Delay between symbol requests. Default: 0.35",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds. Default: 20",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Retry count for transient failures. Default: 3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for number of symbols to process, useful for validation.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=[],
        help="Optional explicit symbols to process. Values should match source tickers in the CSV.",
    )
    parser.add_argument(
        "--extra-symbols",
        nargs="*",
        default=["SPY", "QQQ", "IWM", "DIA"],
        help="Optional extra symbols to add on top of the CSV list. Default: SPY QQQ IWM DIA.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of concurrent fetch workers. Default: 8",
    )
    parser.add_argument("--run-id", default="", help="Optional explicit run identifier for metadata tracking.")
    parser.add_argument("--run-label", default="", help="Optional short label for this run.")
    parser.add_argument("--run-notes", default="", help="Optional notes for this run.")
    return parser.parse_args()


def normalize_yahoo_symbol(source_symbol: str) -> str:
    symbol = source_symbol.strip().upper()
    if not symbol:
        return symbol
    return symbol.replace(".", "-")


def load_symbol_rows(csv_path: Path) -> list[SymbolListRow]:
    rows: list[SymbolListRow] = []
    seen: set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(3):
            next(reader, None)

        header = next(reader, None)
        if not header:
            raise ValueError("CSV header row was not found after the intro lines.")

        header_index = {name.strip(): idx for idx, name in enumerate(header)}
        if "Ticker" not in header_index:
            raise ValueError("CSV does not contain a Ticker column.")

        ticker_idx = header_index["Ticker"]

        for raw_row in reader:
            if not raw_row or ticker_idx >= len(raw_row):
                continue

            source_symbol = raw_row[ticker_idx].strip().upper()
            if not source_symbol or source_symbol in seen:
                continue

            seen.add(source_symbol)
            rows.append(
                SymbolListRow(
                    source_symbol=source_symbol,
                    symbol=normalize_yahoo_symbol(source_symbol),
                )
            )

    return rows


def extend_symbol_rows(rows: list[SymbolListRow], extra_symbols: list[str]) -> list[SymbolListRow]:
    existing = {row.source_symbol for row in rows}
    extended_rows = list(rows)
    for raw_symbol in extra_symbols:
        source_symbol = raw_symbol.strip().upper()
        if not source_symbol or source_symbol in existing:
            continue
        extended_rows.append(
            SymbolListRow(
                source_symbol=source_symbol,
                symbol=normalize_yahoo_symbol(source_symbol),
            )
        )
        existing.add(source_symbol)
    return extended_rows


def build_chart_url(symbol: str, interval: str, lookback_days: int) -> str:
    params: dict[str, Any] = {
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits",
        "lang": "en-US",
        "region": "US",
    }

    # Yahoo's intraday endpoints are sensitive to the exact 60-day cutoff.
    # Prefer a range query for short intraday windows to avoid 422 boundary failures.
    if interval.endswith("m") and lookback_days <= 60:
        params["range"] = f"{lookback_days}d"
    else:
        now = datetime.now(UTC)
        params["period2"] = int(now.timestamp())
        params["period1"] = int((now - timedelta(days=lookback_days)).timestamp())

    query = urllib.parse.urlencode(params)
    return f"{YAHOO_CHART_BASE_URL}{urllib.parse.quote(symbol)}?{query}"


def candidate_lookback_days(lookback_days: int) -> list[int]:
    candidates = [lookback_days]
    if lookback_days >= 60:
        candidates.extend([59, 58])
    elif lookback_days > 1:
        candidates.append(lookback_days - 1)
    # Preserve order but remove duplicates.
    return list(dict.fromkeys(max(1, int(days)) for days in candidates))


def import_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "yfinance is required only for fallback fetches. Install it to enable fallback mode."
        ) from exc
    return yf


def fetch_json(url: str, timeout_seconds: float, retries: int) -> dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Origin": "https://finance.yahoo.com",
        "Referer": "https://finance.yahoo.com/",
    }
    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:  # noqa: BLE001
                body = ""

            message = f"HTTP {exc.code}: {exc.reason}"
            if body:
                try:
                    payload = json.loads(body)
                    description = (((payload.get("chart") or {}).get("error")) or {}).get("description")
                    if description:
                        message = f"{message} - {description}"
                except Exception:  # noqa: BLE001
                    message = f"{message} - {body[:200]}"

            last_error = RuntimeError(message)
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                break

        time.sleep(0.6 * attempt)

    if last_error is None:
        raise RuntimeError("Unknown request failure.")
    raise last_error


def result_to_frame(source_symbol: str, symbol: str, payload: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        description = error.get("description") or str(error)
        raise RuntimeError(description)

    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError("No chart result returned.")

    meta = result.get("meta") or {}
    exchange_timezone = meta.get("exchangeTimezoneName") or "UTC"
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    fetch_timestamp = datetime.now(UTC)
    records: list[dict[str, Any]] = []
    for index, unix_seconds in enumerate(timestamps):
        open_value = opens[index] if index < len(opens) else None
        high_value = highs[index] if index < len(highs) else None
        low_value = lows[index] if index < len(lows) else None
        close_value = closes[index] if index < len(closes) else None
        volume_value = volumes[index] if index < len(volumes) else None

        if any(value is None for value in (open_value, high_value, low_value, close_value)):
            continue

        records.append(
            {
                "source_symbol": source_symbol,
                "symbol": symbol,
                "timestamp": datetime.fromtimestamp(unix_seconds, tz=UTC),
                "open": float(open_value),
                "high": float(high_value),
                "low": float(low_value),
                "close": float(close_value),
                "volume": None if volume_value is None or (isinstance(volume_value, float) and math.isnan(volume_value)) else int(volume_value),
                "exchange_timezone": exchange_timezone,
                "fetch_date": fetch_timestamp,
            }
        )

    if not records:
        raise RuntimeError("No valid OHLC rows returned.")

    frame = pd.DataFrame.from_records(records)
    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    frame["volume"] = frame["volume"].astype("Int64")
    return frame, exchange_timezone


def yfinance_fallback_frame(source_symbol: str, symbol: str, interval: str, lookback_days: int) -> tuple[pd.DataFrame, str]:
    yf = import_yfinance()
    history = yf.Ticker(symbol).history(period=f"{lookback_days}d", interval=interval, auto_adjust=False, prepost=False)
    if history.empty:
        raise RuntimeError("yfinance fallback returned no data.")

    history = history.reset_index()
    datetime_column = "Datetime" if "Datetime" in history.columns else "Date"
    history = history.rename(
        columns={
            datetime_column: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    history = history[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    history["source_symbol"] = source_symbol
    history["symbol"] = symbol
    history["exchange_timezone"] = "UTC"
    history["fetch_date"] = pd.Timestamp.now(tz=UTC)
    history = history.dropna(subset=["open", "high", "low", "close"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    history["volume"] = history["volume"].fillna(0).astype("Int64")
    if history.empty:
        raise RuntimeError("yfinance fallback produced no valid OHLC rows.")
    return history.reset_index(drop=True), "UTC"


def write_symbol_partition(output_dir: Path, symbol: str, frame: pd.DataFrame) -> None:
    partition_dir = output_dir / f"symbol={symbol}"
    if partition_dir.exists():
        shutil.rmtree(partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(partition_dir / "part-000.parquet", index=False)


def write_run_outputs(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    success_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> None:
    manifest_frame = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    success_frame = pd.DataFrame(success_rows, columns=SUCCESS_COLUMNS)
    failure_frame = pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS)

    manifest_frame.to_parquet(output_dir / "_manifest.parquet", index=False)
    manifest_frame.to_csv(output_dir / "_manifest.csv", index=False)
    success_frame.to_csv(output_dir / "_success.csv", index=False)
    failure_frame.to_csv(output_dir / "_failed.csv", index=False)


def select_rows(all_rows: list[SymbolListRow], requested_symbols: list[str], limit: int) -> list[SymbolListRow]:
    filtered_rows = all_rows
    if requested_symbols:
        requested = {symbol.strip().upper() for symbol in requested_symbols if symbol.strip()}
        filtered_rows = [row for row in all_rows if row.source_symbol in requested]
        existing = {row.source_symbol for row in filtered_rows}
        for requested_symbol in requested:
            if requested_symbol not in existing:
                filtered_rows.append(
                    SymbolListRow(
                        source_symbol=requested_symbol,
                        symbol=normalize_yahoo_symbol(requested_symbol),
                    )
                )

    if limit > 0:
        filtered_rows = filtered_rows[:limit]

    return filtered_rows


def process_symbol(row: SymbolListRow, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    primary_errors: list[str] = []
    frame = None
    exchange_timezone = "UTC"
    fetch_source = ""
    for candidate_days in candidate_lookback_days(args.lookback_days):
        try:
            url = build_chart_url(row.symbol, args.interval, candidate_days)
            payload = fetch_json(url, timeout_seconds=args.timeout_seconds, retries=args.retries)
            frame, exchange_timezone = result_to_frame(row.source_symbol, row.symbol, payload)
            fetch_source = f"yahoo_chart_{candidate_days}d"
            break
        except Exception as primary_error:  # noqa: BLE001
            primary_errors.append(f"{candidate_days}d: {primary_error}")

    if frame is None:
        try:
            for candidate_days in candidate_lookback_days(args.lookback_days):
                try:
                    frame, exchange_timezone = yfinance_fallback_frame(
                        row.source_symbol,
                        row.symbol,
                        args.interval,
                        candidate_days,
                    )
                    fetch_source = f"yfinance_fallback_{candidate_days}d"
                    break
                except Exception as fallback_error:  # noqa: BLE001
                    primary_errors.append(f"yf-{candidate_days}d: {fallback_error}")
        except Exception:  # noqa: BLE001
            pass

    if frame is None:
        raise RuntimeError("; ".join(primary_errors))

    write_symbol_partition(output_dir, row.symbol, frame)
    return {
        "source_symbol": row.source_symbol,
        "symbol": row.symbol,
        "row_count": int(len(frame)),
        "min_timestamp": frame["timestamp"].min(),
        "max_timestamp": frame["timestamp"].max(),
        "exchange_timezone": exchange_timezone,
        "last_fetch_at": started_at,
        "fetch_source": fetch_source,
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    run_context = create_run_context(
        stage="build_parquet",
        output_dir=output_dir,
        args=args,
        inputs={"input_csv": input_path},
        run_id=args.run_id or None,
        run_label=args.run_label,
        notes=args.run_notes,
    )
    run_context.write_initial_metadata()

    all_rows = extend_symbol_rows(load_symbol_rows(input_path), args.extra_symbols)
    target_rows = select_rows(all_rows, args.symbols, args.limit)

    if not target_rows:
        run_context.write_validation({"input_exists": True, "selected_symbols": 0})
        run_context.write_summary({"status": "failed", "reason": "no_symbols_selected"})
        print("No symbols selected for processing.", file=sys.stderr)
        return 1

    manifest_rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    total = len(target_rows)
    max_workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_symbol, row, args, output_dir): (index, row)
            for index, row in enumerate(target_rows, start=1)
        }
        for future in as_completed(futures):
            index, row = futures[future]
            print(f"[{index}/{total}] Fetching {row.source_symbol} -> {row.symbol}", flush=True)
            try:
                success_record = future.result()
                success_rows.append(success_record)
                manifest_rows.append(
                    {
                        **success_record,
                        "status": "success",
                        "error_message": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                failure_record = {
                    "source_symbol": row.source_symbol,
                    "symbol": row.symbol,
                    "last_fetch_at": datetime.now(UTC),
                    "error_message": message,
                }
                failure_rows.append(failure_record)
                manifest_rows.append(
                    {
                        "source_symbol": row.source_symbol,
                        "symbol": row.symbol,
                        "row_count": 0,
                        "min_timestamp": pd.NaT,
                        "max_timestamp": pd.NaT,
                        "exchange_timezone": "",
                        "last_fetch_at": failure_record["last_fetch_at"],
                        "status": "failed",
                        "error_message": message,
                    }
                )
                print(f"  failed: {message}", file=sys.stderr, flush=True)

            time.sleep(max(args.sleep_seconds, 0.0))

    write_run_outputs(output_dir, manifest_rows, success_rows, failure_rows)
    run_context.write_validation(
        {
            "input_exists": True,
            "selected_symbols": total,
            "success_count": len(success_rows),
            "failure_count": len(failure_rows),
            "all_selected_symbols_normalized": all(bool(row.symbol) for row in target_rows),
        }
    )
    run_context.write_artifacts(
        {
            "manifest_csv": output_dir / "_manifest.csv",
            "manifest_parquet": output_dir / "_manifest.parquet",
            "success_csv": output_dir / "_success.csv",
            "failed_csv": output_dir / "_failed.csv",
        }
    )

    success_count = len(success_rows)
    failure_count = len(failure_rows)
    run_context.write_summary(
        {
            "status": "completed" if success_count > 0 else "failed",
            "success_count": success_count,
            "failure_count": failure_count,
            "output_dir": output_dir,
        }
    )
    print(f"Completed. Success: {success_count}, Failed: {failure_count}, Output: {output_dir}")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
