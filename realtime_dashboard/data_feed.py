"""Data feed abstraction for live or simulated intraday bars."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.parse
import urllib.error
import json
import logging
import time
import os
import math

logger = logging.getLogger(__name__)

UTC = timezone.utc
YAHOO_CHART_BASE_URL = "https://query2.finance.yahoo.com/v8/finance/chart/"


@dataclass
class Bar:
    """Single 1-minute or 5-minute bar."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
        }


class DataFeed:
    """Base class for data feeds."""

    def get_latest_bars(self, symbols: Optional[List[str]] = None) -> List[Bar]:
        raise NotImplementedError

    def get_historical_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        raise NotImplementedError


# ── Historical Parquet Feed ──

class HistoricalParquetFeed(DataFeed):
    """Read from existing partitioned parquet files (for warm-up/history)."""

    def __init__(self, parquet_dir: str, interval: str = "5m"):
        self.parquet_dir = parquet_dir
        self.interval = interval
        self._df: Optional[pd.DataFrame] = None
        self._symbols: List[str] = []

    def load_symbols(self, symbols: List[str]) -> pd.DataFrame:
        """Load historical data for given symbols from partitioned parquet."""
        all_data = []
        missing_symbols = []
        for sym in symbols:
            sym_norm = sym.replace("-", ".")  # Yahoo normalization reverse
            path = os.path.join(self.parquet_dir, f"symbol={sym_norm}")
            if not os.path.exists(path):
                # Try without normalization
                path = os.path.join(self.parquet_dir, f"symbol={sym}")
            if not os.path.exists(path):
                missing_symbols.append(sym)
                continue
            try:
                df = pd.read_parquet(path)
                df["symbol"] = sym  # Ensure consistent symbol
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {sym}: {e}")

        if missing_symbols:
            preview = ",".join(missing_symbols[:10])
            suffix = "" if len(missing_symbols) <= 10 else f" ... +{len(missing_symbols) - 10} more"
            logger.warning(
                "Missing warmup parquet for %s symbols under %s: %s%s",
                len(missing_symbols),
                self.parquet_dir,
                preview,
                suffix,
            )

        if all_data:
            self._df = pd.concat(all_data, ignore_index=True)
            if "timestamp" in self._df.columns:
                self._df["timestamp"] = pd.to_datetime(self._df["timestamp"])
            self._symbols = symbols
        else:
            self._df = pd.DataFrame()

        return self._df

    def get_latest_bars(self, symbols: Optional[List[str]] = None) -> List[Bar]:
        """Return the most recent bar for each symbol."""
        if self._df is None or self._df.empty:
            return []

        if symbols is None:
            symbols = self._df["symbol"].unique().tolist()

        bars = []
        for sym in symbols:
            sym_df = self._df[self._df["symbol"] == sym]
            if sym_df.empty:
                continue
            latest = sym_df.iloc[-1]
            bars.append(Bar(
                timestamp=pd.to_datetime(latest["timestamp"]),
                symbol=sym,
                open=float(latest.get("open", latest["close"])),
                high=float(latest.get("high", latest["close"])),
                low=float(latest.get("low", latest["close"])),
                close=float(latest["close"]),
                volume=int(latest.get("volume", 0)),
            ))
        return bars

    def get_historical_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if self._df is None:
            self.load_symbols(symbols)
        if self._df is None or self._df.empty:
            return pd.DataFrame()

        mask = (
            (self._df["timestamp"] >= start)
            & (self._df["timestamp"] <= end)
            & (self._df["symbol"].isin(symbols))
        )
        return self._df.loc[mask].copy()

    def get_all_timestamps(self) -> List[datetime]:
        if self._df is None:
            return []
        return sorted(self._df["timestamp"].unique())


# ── Yahoo Finance Live Feed ──

class YahooFinanceLiveFeed(DataFeed):
    """
    Live feed polling Yahoo Finance for latest intraday bars.

    Supports full-market scanning via round-robin chunking:
    - Large symbol lists are split into chunks
    - Each call to get_latest_bars() fetches one chunk
    - Chunks rotate so all symbols are covered over multiple scans
    """

    def __init__(
        self,
        interval: str = "1m",
        scan_mode: str = "chunked",
        lookback_days: int = 7,
        timeout: float = 15.0,
        retries: int = 2,
        max_workers: int = 32,
        rate_limit_delay: float = 0.03,
        chunk_size: int = 200,
    ):
        self.interval = interval
        self.scan_mode = scan_mode
        self.lookback_days = lookback_days
        self.timeout = timeout
        self.retries = retries
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.chunk_size = chunk_size
        self._last_fetch: Dict[str, datetime] = {}
        self._bar_cache: Dict[str, pd.DataFrame] = {}
        self._chunk_index: int = 0
        self._all_symbols: List[str] = []
        self._invalid_symbols: set[str] = set()

    def set_symbols(self, symbols: List[str]):
        """Set the full universe to scan."""
        self._all_symbols = [
            sym for sym in list(dict.fromkeys(symbols))
            if sym not in self._invalid_symbols
        ]

    def _get_next_chunk(self) -> List[str]:
        """Get the next chunk of symbols for round-robin fetching."""
        if not self._all_symbols:
            return []
        if self.scan_mode == "full_parallel" or self.chunk_size <= 0:
            return list(self._all_symbols)
        n = len(self._all_symbols)
        start = self._chunk_index * self.chunk_size
        if start >= n:
            self._chunk_index = 0
            start = 0
        end = min(start + self.chunk_size, n)
        chunk = self._all_symbols[start:end]
        self._chunk_index += 1
        return chunk

    def get_chunk_progress(self) -> tuple[int, int]:
        """Return (current_chunk, total_chunks)."""
        if not self._all_symbols:
            return 0, 0
        if self.scan_mode == "full_parallel" or self.chunk_size <= 0:
            return 1, 1
        total = (len(self._all_symbols) + self.chunk_size - 1) // self.chunk_size
        return self._chunk_index, total

    def _build_url(self, symbol: str, range_value: Optional[str] = None, interval: Optional[str] = None) -> str:
        params = {
            "interval": interval or self.interval,
            "includePrePost": "false",
            "events": "div,splits",
            "lang": "en-US",
            "region": "US",
        }
        # For live intraday polling, fetch today's bars only (fastest, most accurate)
        params["range"] = range_value or "1d"
        query = urllib.parse.urlencode(params)
        return f"{YAHOO_CHART_BASE_URL}{urllib.parse.quote(symbol)}?{query}"

    def _fetch_json(self, url: str) -> dict:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Origin": "https://finance.yahoo.com",
            "Referer": "https://finance.yahoo.com/",
        }
        request = urllib.request.Request(url, headers=headers)
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 422:
                    raise ValueError(f"HTTP 422 unsupported symbol or interval: {exc.reason}")
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(0.5 * attempt)
                    continue
                raise RuntimeError(f"HTTP {exc.code}: {exc.reason}")
            except Exception as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(0.3 * attempt)

        raise last_error or RuntimeError("Unknown request failure")

    def _parse_payload(self, symbol: str, payload: dict) -> pd.DataFrame:
        chart = payload.get("chart") or {}
        error = chart.get("error")
        if error:
            desc = error.get("description") or str(error)
            raise RuntimeError(desc)

        result = (chart.get("result") or [None])[0]
        if not result:
            raise RuntimeError("No chart result")

        meta = result.get("meta") or {}
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adjclose = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]

        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        adjcloses = adjclose.get("adjclose") or []

        if not timestamps:
            return pd.DataFrame()

        exchange_tz = meta.get("exchangeTimezoneName") or "America/New_York"
        tz = UTC  # Parse as UTC for consistency

        rows = []
        for i, ts in enumerate(timestamps):
            if ts is None:
                continue
            o = opens[i] if i < len(opens) and opens[i] is not None else None
            h = highs[i] if i < len(highs) and highs[i] is not None else None
            l = lows[i] if i < len(lows) and lows[i] is not None else None
            c = closes[i] if i < len(closes) and closes[i] is not None else None
            v = volumes[i] if i < len(volumes) and volumes[i] is not None else 0
            ac = adjcloses[i] if i < len(adjcloses) and adjcloses[i] is not None else c

            if c is None:
                continue

            # Use raw OHLC, but adjust close for splits if needed
            rows.append({
                "timestamp": datetime.fromtimestamp(ts, tz=tz),
                "symbol": symbol,
                "open": o if o is not None else c,
                "high": h if h is not None else c,
                "low": l if l is not None else c,
                "close": c,
                "volume": int(v),
                "adj_close": ac,
            })

        return pd.DataFrame(rows)

    def _fetch_symbol(
        self,
        symbol: str,
        range_value: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch latest bars for a single symbol."""
        url = self._build_url(symbol, range_value=range_value, interval=interval)
        payload = self._fetch_json(url)
        df = self._parse_payload(symbol, payload)
        return df

    def _range_from_interval(self, interval: str, lookback_days: int) -> str:
        if interval == "1m":
            return f"{max(1, min(lookback_days, 7))}d"
        if interval in {"2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
            capped = max(1, min(lookback_days, 60))
            if capped <= 5:
                return f"{capped}d"
            if capped <= 30:
                return "1mo"
            return "2mo"
        return "6mo"

    def fetch_recent_history(
        self,
        symbols: List[str],
        interval: Optional[str] = None,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch the latest Yahoo-supported history window for the given interval."""
        fetch_interval = interval or self.interval
        days = lookback_days or self.lookback_days
        range_value = self._range_from_interval(fetch_interval, days)
        all_data = []
        skipped = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_symbol, sym, range_value, fetch_interval): sym
                for sym in symbols
                if sym not in self._invalid_symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_data.append(df)
                except Exception as exc:
                    if isinstance(exc, ValueError) and "HTTP 422" in str(exc):
                        self._invalid_symbols.add(sym)
                        skipped += 1
                        logger.warning("Skipping unsupported Yahoo symbol for recent history: %s", sym)
                    else:
                        logger.warning("Failed recent history fetch for %s: %s", sym, exc)
        if skipped:
            self._all_symbols = [sym for sym in self._all_symbols if sym not in self._invalid_symbols]
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
            return combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        return pd.DataFrame()

    def get_latest_bars(self, symbols: Optional[List[str]] = None) -> List[Bar]:
        """Fetch latest complete bars for the next chunk of symbols (round-robin)."""
        # If symbols provided directly, use them; otherwise use round-robin chunk
        if symbols is not None and len(symbols) > 0:
            chunk = symbols
        else:
            chunk = self._get_next_chunk()

        if not chunk:
            return []

        bars = []
        errors = []
        new_bars_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_symbol, sym): sym
                for sym in chunk
                if sym not in self._invalid_symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    df = future.result()
                    if df.empty:
                        continue

                    cached = self._bar_cache.get(sym, pd.DataFrame())
                    if not cached.empty:
                        combined = pd.concat([cached, df], ignore_index=True)
                        combined = combined.drop_duplicates(subset=["timestamp"])
                        combined = combined.sort_values("timestamp")
                        combined = combined.tail(500)
                    else:
                        combined = df
                    self._bar_cache[sym] = combined

                    # Find latest COMPLETE bar
                    if len(combined) >= 2 and self._is_incomplete_bar(combined.iloc[-1]):
                        latest = combined.iloc[-2]
                    elif len(combined) >= 1:
                        latest = combined.iloc[-1]
                    else:
                        continue

                    prev_ts = self._last_fetch.get(sym)
                    curr_ts = latest["timestamp"]
                    if prev_ts is None or curr_ts > prev_ts:
                        bars.append(Bar(
                            timestamp=curr_ts,
                            symbol=sym,
                            open=float(latest["open"]),
                            high=float(latest["high"]),
                            low=float(latest["low"]),
                            close=float(latest["close"]),
                            volume=int(latest["volume"]),
                        ))
                        new_bars_count += 1

                    self._last_fetch[sym] = curr_ts

                except Exception as e:
                    if isinstance(e, ValueError) and "HTTP 422" in str(e):
                        self._invalid_symbols.add(sym)
                        self._all_symbols = [symbol for symbol in self._all_symbols if symbol != sym]
                        logger.warning("Skipping unsupported Yahoo symbol in live scan: %s", sym)
                    else:
                        errors.append(f"{sym}: {e}")
                        logger.warning(f"Failed to fetch {sym}: {e}")

        if errors and len(errors) > len(chunk) * 0.3:
            logger.error(f"High error rate: {len(errors)}/{len(chunk)} symbols failed")

        if self.scan_mode == "full_parallel" or self.chunk_size <= 0:
            logger.info(
                "Fetched %s new bars from full parallel scan across %s symbols",
                new_bars_count,
                len(chunk),
            )
        else:
            logger.info(
                "Fetched %s new bars from chunk %s/%s",
                new_bars_count,
                self._chunk_index,
                self.get_chunk_progress()[1],
            )
        return bars

    def _is_incomplete_bar(self, row: pd.Series) -> bool:
        """Check if a bar is still forming (incomplete)."""
        if row.get("volume", 0) == 0:
            return True
        o, h, l, c = row.get("open"), row.get("high"), row.get("low"), row.get("close")
        if o == h == l == c:
            return True
        # Also check if timestamp aligns with interval boundary
        ts = row.get("timestamp")
        if hasattr(ts, "minute"):
            minute = ts.minute
            second = ts.second
            if self.interval == "1m":
                # 1m bars should have second == 0
                if second != 0:
                    return True
            elif self.interval == "5m":
                # 5m bars should have minute % 5 == 0 and second == 0
                if minute % 5 != 0 or second != 0:
                    return True
            elif self.interval == "15m":
                # 15m bars should have minute % 15 == 0 and second == 0
                if minute % 15 != 0 or second != 0:
                    return True
        return False

    def get_historical_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch historical bars for warm-up."""
        lookback_days = max(1, math.ceil((end - start).total_seconds() / 86400))
        df = self.fetch_recent_history(symbols, interval=self.interval, lookback_days=lookback_days)
        if df.empty:
            return df
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        return df.loc[mask].copy()

    def get_cached_bars(self, symbol: str) -> pd.DataFrame:
        """Get cached bar history for a symbol."""
        return self._bar_cache.get(symbol, pd.DataFrame()).copy()

    def get_all_cached(self) -> pd.DataFrame:
        """Get all cached bars across all symbols."""
        if not self._bar_cache:
            return pd.DataFrame()
        return pd.concat(self._bar_cache.values(), ignore_index=True)

    def preload_from_historical(self, historical_df: pd.DataFrame) -> None:
        """Pre-populate cache from historical data for warm-up."""
        if historical_df.empty:
            return
        for sym, group in historical_df.groupby("symbol"):
            group = group.sort_values("timestamp")
            self._bar_cache[sym] = group.copy()


# ── Simulated Data Feed (for testing) ──

class SimulatedDataFeed(DataFeed):
    """
    Simulated intraday data feed for testing.
    Generates synthetic 1m bars with correlated movements within
    predefined 'themes' to simulate real community behavior.
    """

    THEMES = {
        "AI_Semiconductors": {
            "symbols": ["NVDA", "AMD", "AVGO", "TSM", "AMAT", "MRVL", "MU"],
            "base_corr": 0.65,
            "volatility": 0.008,
            "drift": 0.0002,
        },
        "Big_Tech": {
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
            "base_corr": 0.55,
            "volatility": 0.006,
            "drift": 0.0001,
        },
        "Financials": {
            "symbols": ["JPM", "BAC", "GS", "MS", "BLK", "WFC", "C"],
            "base_corr": 0.50,
            "volatility": 0.005,
            "drift": 0.00005,
        },
        "Nuclear_Energy": {
            "symbols": ["CCJ", "SMR", "OKLO", "NEE", "EXC", "SO"],
            "base_corr": 0.60,
            "volatility": 0.012,
            "drift": 0.0003,
        },
        "Space_Defense": {
            "symbols": ["LMT", "NOC", "RTX", "BA", "HII", "GD"],
            "base_corr": 0.45,
            "volatility": 0.006,
            "drift": 0.0001,
        },
        "Healthcare": {
            "symbols": ["JNJ", "PFE", "UNH", "ABBV", "LLY", "MRK", "TMO"],
            "base_corr": 0.35,
            "volatility": 0.004,
            "drift": 0.00005,
        },
        "Retail_Consumer": {
            "symbols": ["WMT", "COST", "HD", "NKE", "SBUX", "MCD", "TGT"],
            "base_corr": 0.30,
            "volatility": 0.004,
            "drift": 0.00003,
        },
        "Energy_Oil": {
            "symbols": ["XOM", "CVX", "COP", "OXY", "SLB", "MPC"],
            "base_corr": 0.55,
            "volatility": 0.007,
            "drift": 0.0001,
        },
    }

    def __init__(
        self,
        seed: int = 42,
        start_price: float = 100.0,
        start_time: Optional[datetime] = None,
    ):
        self.rng = np.random.RandomState(seed)
        self.start_price = start_price
        self.start_time = start_time or datetime(2026, 6, 8, 9, 30, 0)
        self._prices: Dict[str, float] = {}
        self._volumes: Dict[str, float] = {}
        self._init_prices()

    def _init_prices(self):
        """Initialize prices for all symbols."""
        for theme, info in self.THEMES.items():
            for sym in info["symbols"]:
                self._prices[sym] = self.start_price * (1 + self.rng.randn() * 0.3)
                self._volumes[sym] = 100_000 + self.rng.exponential(500_000)

    def _get_all_symbols(self) -> List[str]:
        symbols = []
        for info in self.THEMES.values():
            symbols.extend(info["symbols"])
        return list(set(symbols))

    def _generate_returns(self, timestamp: datetime) -> Dict[str, float]:
        """Generate correlated returns for all symbols."""
        # Market-wide factor
        market_return = self.rng.randn() * 0.003

        # Theme-specific shocks
        theme_shocks = {}
        minute_of_day = (timestamp.hour - 9) * 60 + timestamp.minute - 30

        if 30 <= minute_of_day <= 60:
            theme_shocks["AI_Semiconductors"] = self.rng.randn() * 0.005 + 0.003
        elif 90 <= minute_of_day <= 120:
            theme_shocks["Nuclear_Energy"] = self.rng.randn() * 0.006 + 0.004
        elif 150 <= minute_of_day <= 180:
            theme_shocks["Big_Tech"] = self.rng.randn() * 0.004 + 0.002
        elif 210 <= minute_of_day <= 240:
            theme_shocks["Financials"] = self.rng.randn() * 0.004 + 0.003
        elif 270 <= minute_of_day <= 300:
            theme_shocks["Energy_Oil"] = self.rng.randn() * 0.005 + 0.003

        returns = {}
        for theme, info in self.THEMES.items():
            theme_factor = self.rng.randn() * info["volatility"]
            shock = theme_shocks.get(theme, 0.0)

            for sym in info["symbols"]:
                individual = self.rng.randn() * info["volatility"] * 0.5
                ret = (
                    individual
                    + theme_factor * info["base_corr"]
                    + market_return * 0.3
                    + info["drift"]
                    + shock
                )
                returns[sym] = ret

        return returns

    def get_latest_bars(self, symbols: Optional[List[str]] = None) -> List[Bar]:
        """Generate bars for the current simulated minute."""
        if symbols is None:
            symbols = self._get_all_symbols()

        now = self.start_time
        returns = self._generate_returns(now)
        bars = []

        for sym in symbols:
            if sym not in self._prices:
                self._prices[sym] = self.start_price * (1 + self.rng.randn() * 0.3)
                self._volumes[sym] = 100_000 + self.rng.exponential(500_000)

            price = self._prices[sym]
            ret = returns.get(sym, self.rng.randn() * 0.005)

            new_price = price * (1 + ret)
            high = max(price, new_price) * (1 + abs(self.rng.randn() * 0.001))
            low = min(price, new_price) * (1 - abs(self.rng.randn() * 0.001))

            base_vol = self._volumes[sym]
            vol_multiplier = 1.0 + abs(ret) * 50
            volume = int(base_vol * vol_multiplier * (0.5 + self.rng.exponential(0.5)))

            bar = Bar(
                timestamp=now,
                symbol=sym,
                open=price,
                high=high,
                low=low,
                close=new_price,
                volume=volume,
                vwap=(price + new_price) / 2 * (1 + self.rng.randn() * 0.0005),
            )
            bars.append(bar)

            self._prices[sym] = new_price
            self._volumes[sym] = base_vol * 0.9 + volume * 0.1

        self.start_time += timedelta(minutes=1)
        return bars

    def get_historical_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        saved_time = self.start_time
        saved_prices = dict(self._prices)

        self.start_time = start
        self._prices = dict(saved_prices)

        all_bars = []
        current = start
        while current <= end:
            bars = self.get_latest_bars(symbols)
            all_bars.extend([b.to_dict() for b in bars])
            current = self.start_time

        self.start_time = saved_time
        self._prices = saved_prices

        return pd.DataFrame(all_bars)

    def step(self, n_minutes: int = 1) -> pd.DataFrame:
        """Advance simulation by n minutes and return bars."""
        all_bars = []
        for _ in range(n_minutes):
            bars = self.get_latest_bars()
            all_bars.extend([b.to_dict() for b in bars])
        return pd.DataFrame(all_bars)
