"""US stock signal scanner. Runs all four scans by default, sharing the universe:

uptrend
    Catches stocks at the very beginning of a potential uptrend:
      - Quiet base for several weeks (modest return, low drawup)
      - Wide bullish candle today (close > open, gain above threshold)
      - Volume thrust (today's vol >> recent avg vol)
      - Close above SMA50 (trend confirmation)

earnings
    Surfaces companies that reported earnings on the previous trading day
    (or within --start/--end window) with a positive surprise >= --min-surprise %.

revisions
    Catches upward-revision acceleration: 7-day net-upgrade pace vs 30-day
    baseline (volume-thrust analog for sell-side coverage). Requires >= --min-up7d
    upward revisions in last 7d to filter noise.

targets
    Catches broad price-target hikes: >= --min-target-raisers distinct firms
    raised target in the last --target-lookback-days, with median raise % >=
    --min-target-raise. Tickers that show up in BOTH revisions and targets are
    the highest-conviction picks.

NOTE: Yahoo's revision/upgrade APIs are current snapshots — --start/--end is
ignored for `revisions` and `targets`. They always apply to the most recent window.

Common filters: --start/--end date range, --min-market-cap, --refresh-tickers.
Use --mode {uptrend|earnings|revisions|targets} to run a single scan instead of all.

Run:
    poetry run python -m algorithmic_trading.analysis.us_stock_scanner
    poetry run python -m algorithmic_trading.analysis.us_stock_scanner --mode revisions --min-revision-accel 3
"""

import argparse
import contextlib
import datetime as dt
import io
import calendar
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import certifi
import pandas as pd

from curl_cffi import requests as curl_requests
from tqdm import tqdm


# Single browser-impersonating session. Yahoo's rate limit keys on TLS/JA3
# fingerprints — a real-Chrome-looking session bypasses it. Explicit CA bundle
# via certifi avoids the "error setting certificate verify locations" failures
# we saw under high concurrency.
_SESSION: curl_requests.Session | None = None


def _get_session() -> curl_requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = curl_requests.Session(
            impersonate="chrome",
            verify=certifi.where(),
        )
    return _SESSION


VERBOSE = False
SCAN_START: dt.date | None = None  # inclusive; if None, only scan most recent bar
SCAN_END: dt.date | None = None    # inclusive


def _configure_logging(verbose: bool):
    """Quiet yfinance + urllib3 unless --verbose."""
    if verbose:
        return
    for name in ("yfinance", "urllib3", "peewee", "requests"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate = False


@contextlib.contextmanager
def _maybe_silence_stderr():
    """yfinance prints some failures straight to stderr; swallow them when not verbose."""
    if VERBOSE:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield


# --- scan parameters ---------------------------------------------------------
LOOKBACK_DAYS = 120         # how much daily history to pull per ticker
BASE_DAYS = 20              # window used to characterize the "base"
VOL_AVG_DAYS = 20           # window for average-volume comparison
MIN_VOL_RATIO = 1.7         # today's vol / avg vol must be >= this
MIN_DAILY_GAIN_PCT = 3.0    # today's % gain (close vs prev close) >= this
MAX_BASE_RETURN_PCT = 20.0  # base-window return must be <= this (still quiet)
MAX_BASE_DRAWUP_PCT = 30.0  # high/low ratio over base <= this (no prior rip)
SMA_LEN = 50                # confirmation MA

FETCH_WORKERS = 8       # controlled concurrency for per-ticker chart requests
FETCH_TIMEOUT = 15      # seconds per HTTP call
MAX_RETRIES = 3
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# Minimum number of trading days required to evaluate the setup on a single bar.
MIN_REQUIRED_BARS = max(BASE_DAYS, VOL_AVG_DAYS, SMA_LEN) + 2

# When the user passes --start/--end, we silently pull extra history before
# SCAN_START so SMA50 + base + volume windows have enough context regardless of
# how short the user's window is. 90 calendar days ≈ 63 trading days, comfortably
# covers MIN_REQUIRED_BARS even after holidays.
DATA_BUFFER_CALENDAR_DAYS = 90


def _is_common_stock(ticker: str) -> bool:
    """Heuristic: keep US common shares, drop SPAC units/rights/warrants and preferreds.

    Yahoo conventions we filter:
        - suffix 'U' / 'UN'        -> SPAC units
        - suffix 'R'  / 'RT'       -> rights
        - suffix 'W'  / 'WS'       -> warrants
        - contains '-P' or '^'     -> preferred shares (e.g. BAC-PL, BAC^L)
        - contains '.' or '='      -> class shares / when-issued
    Slash-tickers (BRK/A) are rewritten to dash form by `_normalize_ticker`.
    """
    if not ticker:
        return False
    # preferreds, when-issued, oddball symbols
    if "^" in ticker or "=" in ticker or "$" in ticker:
        return False
    if "-P" in ticker or ticker.endswith("-PR") or ticker.endswith("-PA"):
        return False
    # SPAC paraphernalia
    if ticker.endswith(("WS", "RT", "UN")):
        return False
    if len(ticker) >= 4 and ticker[-1] in {"U", "R", "W"} and ticker[-2].isalpha():
        # only strip when the suffix looks like a unit/right/warrant tag.
        # 4+ letters with one of these suffixes is the typical SPAC pattern (e.g. AACBU, AACBR).
        # Real 4-letter commons ending in U/R/W are rare; this trades a little recall for a lot of noise.
        return False
    return True


def _normalize_ticker(ticker: str) -> str:
    """Yahoo uses dash for class shares (BRK-A). NASDAQ uses '.' (BRK.A), CSV files '/'."""
    return ticker.replace("/", "-").replace(".", "-").strip().upper()


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
TICKER_CACHE_TTL_SECONDS = 24 * 3600
TICKER_CACHE_FILENAME = ".us_tickers_cache.txt"

QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
QUOTE_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
CRUMB_BOOTSTRAP_URL = "https://finance.yahoo.com/quote/AAPL/"
MARKETCAP_BATCH_SIZE = 100
MARKETCAP_CACHE_TTL_SECONDS = 7 * 24 * 3600
QUOTE_METRICS_CACHE_FILENAME = ".us_quote_metrics_cache.json"

DEFAULT_MIN_DAILY_VOLUME = 200_000
MIN_VOLUME_LOOKBACK_DAYS = 30          # trading days to check
MIN_VOLUME_CACHE_FILENAME = ".us_min_volumes_cache.json"
MIN_VOLUME_CACHE_TTL_SECONDS = 24 * 3600

ETF_TICKERS_CACHE_FILENAME = ".us_etfs_cache.txt"
ETF_ASSETS_CACHE_FILENAME = ".us_etf_assets_cache.json"
ETF_ASSETS_CACHE_TTL_SECONDS = 7 * 24 * 3600
ETF_LIQUIDITY_PREFILTER_VOLUME = 50_000   # skip AUM fetch for ETFs below this avg vol
DEFAULT_MIN_ETF_ASSETS = 15_000_000_000   # $15B

DEFAULT_SQUEEZE_BARS = 6                 # bars of tight consolidation required before breakout
DEFAULT_SQUEEZE_MAX_RANGE_PCT = 4.0      # high-low spread over those bars must be <= this % of midpoint
DEFAULT_SQUEEZE_BREAKOUT_PCT = 1.5       # today's close must clear the consol range by at least this %
DEFAULT_SQUEEZE_DIRECTION = "both"       # 'up', 'down', or 'both'
DEFAULT_SQUEEZE_TIMEFRAME = "1d"         # '1d' (default), '4h', '1h'
SQUEEZE_TIMEFRAMES = ("1d", "4h", "1h")
DEFAULT_SQUEEZE_MAX_LOOKBACK = 5         # try breakout at position -1, -2, ..., up to this many bars back
DEFAULT_SQUEEZE_MIN_BARS = 3             # shortest consolidation we'll accept (per candidate breakout)

DEFAULT_MIN_SURPRISE_PCT = 10.0
DEFAULT_MIN_REVISION_ACCEL = 2.0     # 7d pace must be >= this x 30d pace
DEFAULT_MIN_UP7D = 3                  # absolute min upward revisions in last 7d (noise filter)
DEFAULT_TARGET_LOOKBACK_DAYS = 7
DEFAULT_MIN_TARGET_RAISERS = 3        # require this many distinct firms hiking target
DEFAULT_MIN_TARGET_RAISE_PCT = 30.0    # median hike % across raises must be >= this


def _parse_nasdaq_symdir(text: str, is_nasdaq: bool) -> list[str]:
    """Parse a NASDAQ Trader pipe-delimited symbol directory file.

    nasdaqlisted.txt columns: Symbol|Name|MktCat|TestIssue|FinStatus|LotSize|ETF|...
    otherlisted.txt   columns: ACTSymbol|Name|Exchange|CQSSym|ETF|LotSize|TestIssue|...
    Both files end with a 'File Creation Time:...' footer line we must skip.
    """
    out = []
    lines = text.splitlines()
    for line in lines[1:]:  # skip header
        if not line or line.startswith("File Creation Time"):
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        if is_nasdaq:
            symbol, _name, _mkt, test_issue, fin_status, _lot, etf = parts[:7]
            if test_issue == "Y" or etf == "Y" or fin_status != "N":
                continue
        else:
            symbol, _name, _exchange, _cqs, etf, _lot, test_issue = parts[:7]
            if test_issue == "Y" or etf == "Y":
                continue
        if symbol:
            out.append(symbol)
    return out


def _fetch_us_tickers_from_nasdaq() -> list[str] | None:
    """Pull live US common-stock tickers from NASDAQ Trader directory files."""
    session = _get_session()
    tickers: set[str] = set()
    try:
        for url, is_nasdaq in ((NASDAQ_LISTED_URL, True), (OTHER_LISTED_URL, False)):
            r = session.get(url, timeout=FETCH_TIMEOUT)
            r.raise_for_status()
            tickers.update(_parse_nasdaq_symdir(r.text, is_nasdaq))
    except Exception as e:
        if VERBOSE:
            print(f"NASDAQ ticker fetch failed: {e}")
        return None
    return sorted(tickers)


def _ticker_cache_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), TICKER_CACHE_FILENAME)


def _load_ticker_cache(max_age_seconds: int) -> list[str] | None:
    path = _ticker_cache_path()
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > max_age_seconds:
        return None
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _save_ticker_cache(tickers: list[str]):
    with open(_ticker_cache_path(), "w") as f:
        f.write("\n".join(tickers) + "\n")


def get_tickers(force_refresh: bool = False):
    """Get US common-stock tickers, preferring fresh NASDAQ directory data.

    Resolution order:
        1. Cached NASDAQ directory (refreshed every 24h, or now if --refresh-tickers).
        2. Live fetch from NASDAQ Trader.
        3. Local us_stocks.txt fallback (legacy).
        4. Tiny hardcoded default list.

    Always passes the result through _is_common_stock and _normalize_ticker.
    """
    raw: list[str] | None = None
    source = ""

    if not force_refresh:
        cached = _load_ticker_cache(TICKER_CACHE_TTL_SECONDS)
        if cached is not None:
            raw, source = cached, "cache"

    if raw is None:
        fetched = _fetch_us_tickers_from_nasdaq()
        if fetched:
            _save_ticker_cache(fetched)
            raw, source = fetched, "NASDAQ live"

    if raw is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "us_stocks.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                raw = [line.strip() for line in f if line.strip()]
            source = "us_stocks.txt fallback"

    if raw is None:
        raw = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "IBM",
            "AMD", "ASML", "PLTR", "RKLB", "FIX", "QQQ", "VOO", "SCHD", "SCHG",
        ]
        source = "hardcoded default"

    normalized = [_normalize_ticker(t) for t in raw]
    cleaned = [t for t in normalized if _is_common_stock(t)]
    # de-dupe, keep order
    seen = set()
    out = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            out.append(t)
    if VERBOSE:
        print(f"tickers loaded from {source}: {len(raw)} raw -> {len(out)} after filter")
    return out


_CRUMB: str | None = None


def _get_crumb() -> str | None:
    """Yahoo's v7 quote endpoint requires a per-session crumb. Fetch once, cache."""
    global _CRUMB
    if _CRUMB:
        return _CRUMB
    session = _get_session()
    try:
        # Bootstrap cookies first — without these, getcrumb returns 401.
        session.get(CRUMB_BOOTSTRAP_URL, timeout=FETCH_TIMEOUT)
        r = session.get(CRUMB_URL, timeout=FETCH_TIMEOUT)
        if r.status_code == 200 and r.text.strip():
            _CRUMB = r.text.strip()
            return _CRUMB
    except Exception as e:
        if VERBOSE:
            print(f"crumb fetch failed: {e}")
    return None


def _fetch_quote_metrics_batch(tickers: list[str], crumb: str) -> dict[str, dict]:
    """Pull marketCap + averageDailyVolume3Month for a batch via v7 quote endpoint."""
    session = _get_session()
    r = session.get(
        QUOTE_URL,
        params={"symbols": ",".join(tickers), "crumb": crumb},
        timeout=FETCH_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    quotes = (data.get("quoteResponse") or {}).get("result", [])
    out: dict[str, dict] = {}
    for q in quotes:
        sym = q.get("symbol")
        if not sym:
            continue
        out[sym] = {
            "market_cap": float(q["marketCap"]) if q.get("marketCap") else None,
            "avg_volume": float(q["averageDailyVolume3Month"]) if q.get("averageDailyVolume3Month") else None,
        }
    return out


def fetch_quote_metrics(tickers: list[str]) -> dict[str, dict]:
    """Bulk-fetch market cap + avg volume for `tickers`. Skips entries Yahoo doesn't return."""
    crumb = _get_crumb()
    if not crumb:
        if VERBOSE:
            print("could not obtain Yahoo crumb — quote metrics unavailable")
        return {}
    out: dict[str, dict] = {}
    batches = [tickers[i:i + MARKETCAP_BATCH_SIZE] for i in range(0, len(tickers), MARKETCAP_BATCH_SIZE)]
    for batch in tqdm(batches, desc="quote metrics"):
        for attempt in range(MAX_RETRIES):
            try:
                out.update(_fetch_quote_metrics_batch(batch, crumb))
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    if VERBOSE:
                        print(f"quote metrics batch failed: {e}")
                else:
                    time.sleep(0.5 * (2 ** attempt))
    return out


def _quote_metrics_cache_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), QUOTE_METRICS_CACHE_FILENAME)


def _load_quote_metrics_cache(max_age_seconds: int) -> dict[str, dict] | None:
    path = _quote_metrics_cache_path()
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > max_age_seconds:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_quote_metrics_cache(metrics: dict[str, dict]):
    with open(_quote_metrics_cache_path(), "w") as f:
        json.dump(metrics, f)


def get_quote_metrics(tickers: list[str], force_refresh: bool = False) -> dict[str, dict]:
    """Resolve {market_cap, avg_volume} for `tickers` (cached 7d), fetching missing ones live."""
    cached: dict[str, dict] = {}
    if not force_refresh:
        loaded = _load_quote_metrics_cache(MARKETCAP_CACHE_TTL_SECONDS)
        if loaded is not None:
            cached = loaded

    missing = [t for t in tickers if t not in cached]
    if missing:
        if VERBOSE:
            print(f"quote metrics: {len(cached)} cached, {len(missing)} to fetch")
        fresh = fetch_quote_metrics(missing)
        cached.update(fresh)
        _save_quote_metrics_cache(cached)
    elif VERBOSE:
        print(f"quote metrics: all {len(tickers)} served from cache")
    return cached


def fetch_min_volume(ticker: str, days: int = MIN_VOLUME_LOOKBACK_DAYS) -> float | None:
    """Min daily share volume over the last `days` trading days (excluding today)."""
    end = dt.date.today()
    start = end - dt.timedelta(days=days * 2)  # weekends/holidays buffer
    df = fetch_history(ticker, start, end)
    if df is None or df.empty:
        return None
    recent = df["Volume"].tail(days)
    if recent.empty:
        return None
    return float(recent.min())


def _min_volume_cache_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), MIN_VOLUME_CACHE_FILENAME)


def _load_min_volume_cache(max_age_seconds: int) -> dict[str, float] | None:
    path = _min_volume_cache_path()
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > max_age_seconds:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_min_volume_cache(vols: dict[str, float]):
    with open(_min_volume_cache_path(), "w") as f:
        json.dump(vols, f)


def get_min_volumes(tickers: list[str], force_refresh: bool = False) -> dict[str, float]:
    """Cached min-daily-volume-in-last-30-days for `tickers`. Cache TTL 24h."""
    cached: dict[str, float] = {}
    if not force_refresh:
        loaded = _load_min_volume_cache(MIN_VOLUME_CACHE_TTL_SECONDS)
        if loaded is not None:
            cached = loaded

    missing = [t for t in tickers if t not in cached]
    if missing:
        if VERBOSE:
            print(f"min volumes: {len(cached)} cached, {len(missing)} to fetch")
        results: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
            futures = {ex.submit(fetch_min_volume, t): t for t in missing}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="min volumes"):
                ticker = futures[fut]
                try:
                    val = fut.result()
                    if val is not None:
                        results[ticker] = val
                except Exception as e:
                    if VERBOSE:
                        print(f"min volume {ticker}: {e}")
        cached.update(results)
        _save_min_volume_cache(cached)
    elif VERBOSE:
        print(f"min volumes: all {len(tickers)} served from cache")
    return cached


# ----- ETF universe + AUM -----

def _fetch_etfs_from_nasdaq() -> list[str] | None:
    """Pull ETF tickers from NASDAQ Trader directory files."""
    session = _get_session()
    etfs: set[str] = set()
    try:
        for url, is_nasdaq in ((NASDAQ_LISTED_URL, True), (OTHER_LISTED_URL, False)):
            r = session.get(url, timeout=FETCH_TIMEOUT)
            r.raise_for_status()
            for line in r.text.splitlines()[1:]:
                if not line or line.startswith("File Creation Time"):
                    continue
                parts = line.split("|")
                if len(parts) < 7:
                    continue
                if is_nasdaq:
                    sym, _name, _mkt, test, _fin, _lot, etf_flag = parts[:7]
                else:
                    sym, _name, _ex, _cqs, etf_flag, _lot, test = parts[:7]
                if etf_flag == "Y" and test != "Y" and sym:
                    etfs.add(sym)
    except Exception as e:
        if VERBOSE:
            print(f"NASDAQ ETF fetch failed: {e}")
        return None
    return sorted(etfs)


def _etf_tickers_cache_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ETF_TICKERS_CACHE_FILENAME)


def get_etfs(force_refresh: bool = False) -> list[str]:
    """Cached ETF ticker list from NASDAQ Trader (24h TTL, mirrors get_tickers)."""
    path = _etf_tickers_cache_path()
    if not force_refresh and os.path.exists(path) and time.time() - os.path.getmtime(path) < TICKER_CACHE_TTL_SECONDS:
        with open(path, "r") as f:
            raw = [line.strip() for line in f if line.strip()]
    else:
        fetched = _fetch_etfs_from_nasdaq() or []
        if fetched:
            with open(path, "w") as f:
                f.write("\n".join(fetched) + "\n")
        raw = fetched
    # ETF symbols use the same normalization rules; but skip the common-stock filter
    # since legitimate ETFs can have 4-letter L/W/U-ending tickers (TQQQ, SQQQ...).
    seen = set()
    out = []
    for t in raw:
        n = _normalize_ticker(t)
        if n and "^" not in n and "$" not in n and "=" not in n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def fetch_etf_total_assets(ticker: str) -> float | None:
    """Net assets (AUM) for an ETF via quoteSummary.defaultKeyStatistics.totalAssets."""
    crumb = _get_crumb()
    if not crumb:
        return None
    session = _get_session()
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(
                QUOTE_SUMMARY_URL.format(ticker=ticker),
                params={"modules": "defaultKeyStatistics", "crumb": crumb},
                timeout=FETCH_TIMEOUT,
            )
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            ks = ((data.get("quoteSummary") or {}).get("result") or [{}])[0].get("defaultKeyStatistics") or {}
            ta = (ks.get("totalAssets") or {}).get("raw")
            return float(ta) if ta is not None else None
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                if VERBOSE:
                    print(f"ETF assets fetch {ticker}: {e}")
                return None
            time.sleep(0.5 * (2 ** attempt))
    return None


def _etf_assets_cache_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ETF_ASSETS_CACHE_FILENAME)


def _load_etf_assets_cache(max_age_seconds: int) -> dict[str, float] | None:
    path = _etf_assets_cache_path()
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > max_age_seconds:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_etf_assets_cache(assets: dict[str, float]):
    with open(_etf_assets_cache_path(), "w") as f:
        json.dump(assets, f)


def get_etf_assets(tickers: list[str], force_refresh: bool = False) -> dict[str, float]:
    """Cached ETF net-assets (AUM) lookup. Fetches missing per-ticker in parallel."""
    cached: dict[str, float] = {}
    if not force_refresh:
        loaded = _load_etf_assets_cache(ETF_ASSETS_CACHE_TTL_SECONDS)
        if loaded is not None:
            cached = loaded

    missing = [t for t in tickers if t not in cached]
    if missing:
        if VERBOSE:
            print(f"ETF assets: {len(cached)} cached, {len(missing)} to fetch")
        results: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
            futures = {ex.submit(fetch_etf_total_assets, t): t for t in missing}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="ETF assets"):
                ticker = futures[fut]
                try:
                    val = fut.result()
                    if val is not None:
                        results[ticker] = val
                except Exception as e:
                    if VERBOSE:
                        print(f"ETF assets {ticker}: {e}")
        cached.update(results)
        _save_etf_assets_cache(cached)
    elif VERBOSE:
        print(f"ETF assets: all {len(tickers)} served from cache")
    return cached


def _to_unix(d: dt.date) -> int:
    return int(dt.datetime.combine(d, dt.time.min, tzinfo=dt.timezone.utc).timestamp())


def _resolve_fetch_window() -> tuple[dt.date, dt.date]:
    """Return (start, end) for the chart fetch, including SMA/base buffer when ranged."""
    if SCAN_START and SCAN_END:
        return SCAN_START - dt.timedelta(days=DATA_BUFFER_CALENDAR_DAYS), SCAN_END
    end = dt.date.today()
    return end - dt.timedelta(days=LOOKBACK_DAYS), end


def fetch_history(
    ticker: str, start: dt.date, end: dt.date, interval: str = "1d",
) -> pd.DataFrame | None:
    """Hit Yahoo's chart endpoint directly via curl_cffi. Returns OHLCV DataFrame or None.

    Supported intervals: '1d' (default), '1h', '4h'. For '4h' we fetch 1h bars and
    resample. Intraday bars older than ~2 years aren't available from Yahoo.
    """
    # 4h not natively supported — fetch 1h and resample below.
    fetch_interval = "1h" if interval == "4h" else interval
    session = _get_session()
    params = {
        "period1": _to_unix(start),
        "period2": _to_unix(end + dt.timedelta(days=1)),
        "interval": fetch_interval,
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(
                CHART_URL.format(ticker=ticker), params=params, timeout=FETCH_TIMEOUT
            )
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                if VERBOSE:
                    print(f"fetch {ticker}: {e}")
                return None
            time.sleep(0.5 * (2 ** attempt))
    else:
        return None

    result = (data.get("chart") or {}).get("result")
    if not result:
        return None
    chart = result[0]
    timestamps = chart.get("timestamp")
    indicators = (chart.get("indicators") or {}).get("quote")
    if not timestamps or not indicators:
        return None
    quote = indicators[0]
    df = pd.DataFrame({
        "Open": quote.get("open"),
        "High": quote.get("high"),
        "Low": quote.get("low"),
        "Close": quote.get("close"),
        "Volume": quote.get("volume"),
    }, index=pd.to_datetime(timestamps, unit="s"))
    df.index.name = "Date"
    df = df.dropna()

    if interval == "4h":
        # Resample 1h → 4h. Yahoo returns timestamps at the start of each hourly bar.
        df = df.resample("4h", origin="start_day").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
        }).dropna()

    if interval == "1d":
        # Drop today's bar — during market hours Yahoo returns a partial candle.
        today = dt.date.today()
        df = df[df.index.date < today]
    else:
        # Intraday: drop the last bar if its window isn't fully closed yet,
        # and drop any zero-volume bars (after-hours stubs Yahoo returns).
        interval_hours = {"1h": 1, "4h": 4}[interval]
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=interval_hours)
        df = df[df.index <= cutoff]
        df = df[df["Volume"] > 0]

    return df


def evaluate(ticker, df):
    """Return a result dict if `ticker` matches the early-uptrend setup, else None.

    Evaluates the LAST bar of `df` as the candidate day. All indicators (including
    SMA50) are computed from the price series in-memory — no extra network calls.
    """
    if df is None or df.empty or len(df) < MIN_REQUIRED_BARS:
        return None
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < MIN_REQUIRED_BARS:
        return None

    today = df.iloc[-1]
    prev = df.iloc[-2]

    # 1) Today: wide bullish candle.
    daily_gain_pct = (today["Close"] - prev["Close"]) / prev["Close"] * 100
    if daily_gain_pct < MIN_DAILY_GAIN_PCT or today["Close"] <= today["Open"]:
        return None

    # 2) Volume thrust.
    avg_vol = df["Volume"].iloc[-(VOL_AVG_DAYS + 1):-1].mean()
    if avg_vol <= 0:
        return None
    vol_ratio = today["Volume"] / avg_vol
    if vol_ratio < MIN_VOL_RATIO:
        return None

    # 3) The base was quiet — we want to catch the START of a move, not chase one.
    base = df["Close"].iloc[-(BASE_DAYS + 1):-1]
    base_return_pct = (base.iloc[-1] - base.iloc[0]) / base.iloc[0] * 100
    base_drawup_pct = (base.max() - base.min()) / base.min() * 100
    if base_return_pct > MAX_BASE_RETURN_PCT:
        return None
    if base_drawup_pct > MAX_BASE_DRAWUP_PCT:
        return None

    # 4) Trend confirmation — close above SMA50.
    sma = df["Close"].rolling(SMA_LEN).mean().iloc[-1]
    if pd.isna(sma) or today["Close"] <= sma:
        return None

    return {
        "Ticker": ticker,
        "Date": df.index[-1].date(),
        "Close": round(float(today["Close"]), 2),
        "Gain %": round(float(daily_gain_pct), 2),
        "Vol x Avg": round(float(vol_ratio), 2),
        "Base Return %": round(float(base_return_pct), 2),
        "Base Drawup %": round(float(base_drawup_pct), 2),
        f"SMA{SMA_LEN}": round(float(sma), 2),
    }


def _prepare_universe(
    refresh_tickers: bool,
    min_market_cap_usd: float | None,
    refresh_market_caps: bool,
    min_daily_volume: float | None = None,
    min_etf_assets: float | None = DEFAULT_MIN_ETF_ASSETS,
) -> list[str]:
    """Fetch tickers, apply universe-level filters.

    Stocks: filtered by market cap (cap from cached v7 quote).
    ETFs (optional): filtered by AUM via quoteSummary.totalAssets.
    Min daily volume applies to the combined universe.
    """
    stocks = get_tickers(force_refresh=refresh_tickers)

    if min_market_cap_usd is not None:
        metrics = get_quote_metrics(stocks, force_refresh=refresh_market_caps)
        before = len(stocks)
        stocks = [
            t for t in stocks
            if (metrics.get(t, {}).get("market_cap") or 0) >= min_market_cap_usd
        ]
        print(
            f"Market-cap filter: kept {len(stocks)}/{before} stocks "
            f"with cap >= {_format_market_cap(min_market_cap_usd)}"
        )

    etfs_kept: list[str] = []
    if min_etf_assets is not None:
        all_etfs = get_etfs()
        # Pre-filter: only fetch AUM for ETFs with meaningful trading activity.
        # A $15B+ AUM ETF effectively always trades >50k shares/day on average.
        etf_quote_metrics = get_quote_metrics(all_etfs, force_refresh=refresh_market_caps)
        liquid_etfs = [
            t for t in all_etfs
            if (etf_quote_metrics.get(t, {}).get("avg_volume") or 0) >= ETF_LIQUIDITY_PREFILTER_VOLUME
        ]
        aum = get_etf_assets(liquid_etfs)
        etfs_kept = [t for t in liquid_etfs if (aum.get(t) or 0) >= min_etf_assets]
        print(
            f"ETF filter: kept {len(etfs_kept)}/{len(all_etfs)} ETFs "
            f"with assets >= {_format_market_cap(min_etf_assets)} "
            f"(after liquidity pre-filter: {len(liquid_etfs)} fetched for AUM)"
        )

    tickers = stocks + etfs_kept

    if min_daily_volume is not None:
        min_vols = get_min_volumes(tickers)
        before = len(tickers)
        tickers = [t for t in tickers if (min_vols.get(t) or 0) >= min_daily_volume]
        print(
            f"Min-daily-volume filter: kept {len(tickers)}/{before} tickers "
            f"with min volume >= {int(min_daily_volume):,} over last "
            f"{MIN_VOLUME_LOOKBACK_DAYS} trading days"
        )

    return tickers


def _previous_trading_day(today: dt.date | None = None) -> dt.date:
    """Most recent weekday strictly before today (skips Sat/Sun, ignores holidays)."""
    d = (today or dt.date.today()) - dt.timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= dt.timedelta(days=1)
    return d


def fetch_earnings_data(ticker: str) -> dict | None:
    """Most-recent quarterly earnings record with release date + surprise."""
    crumb = _get_crumb()
    if not crumb:
        return None
    session = _get_session()
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(
                QUOTE_SUMMARY_URL.format(ticker=ticker),
                params={"modules": "earnings", "crumb": crumb},
                timeout=FETCH_TIMEOUT,
            )
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            result = (data.get("quoteSummary") or {}).get("result") or []
            if not result:
                return None
            quarterly = ((result[0].get("earnings") or {}).get("earningsChart") or {}).get("quarterly") or []
            return quarterly[-1] if quarterly else None
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                if VERBOSE:
                    print(f"earnings fetch {ticker}: {e}")
                return None
            time.sleep(0.5 * (2 ** attempt))
    return None


def evaluate_earnings(
    ticker: str, record: dict | None, min_surprise_pct: float
) -> dict | None:
    """Match if surprise >= threshold AND release date is in window
    (or == previous trading day when no window is set)."""
    if not record:
        return None
    # surprisePct in the `earnings` module is a string like "5.54"
    try:
        surprise_pct = float(record.get("surprisePct"))
    except (TypeError, ValueError):
        return None
    if surprise_pct < min_surprise_pct:
        return None

    reported_ts = (record.get("reportedDate") or {}).get("raw")
    if reported_ts is None:
        return None
    reported_date = dt.datetime.fromtimestamp(reported_ts, tz=dt.timezone.utc).date()

    if SCAN_START and SCAN_END:
        if not (SCAN_START <= reported_date <= SCAN_END):
            return None
    else:
        if reported_date != _previous_trading_day():
            return None

    actual = (record.get("actual") or {}).get("raw")
    estimate = (record.get("estimate") or {}).get("raw")
    if actual is None or estimate is None:
        return None

    return {
        "Ticker": ticker,
        "Reported": reported_date,
        "Quarter": record.get("date"),
        "EPS Actual": round(float(actual), 3),
        "EPS Estimate": round(float(estimate), 3),
        "Surprise %": round(surprise_pct, 2),
    }


def scan_uptrend(tickers: list[str]):
    if SCAN_START and SCAN_END:
        print(
            f"Scanning {len(tickers)} tickers using data window "
            f"{SCAN_START} → {SCAN_END} (evaluating last bar in window)..."
        )
    else:
        print(f"Scanning {len(tickers)} tickers (latest bar) for early-uptrend setups...")

    start, end = _resolve_fetch_window()
    matches = []
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = {ex.submit(fetch_history, t, start, end): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="scanning"):
            ticker = futures[fut]
            df = fut.result()
            if df is None or df.empty:
                failed += 1
                continue
            success += 1
            try:
                hit = evaluate(ticker, df)
                if hit:
                    matches.append(hit)
            except Exception as e:
                if VERBOSE:
                    print(f"evaluate {ticker}: {e}")

    total = success + failed
    pct = (success / total * 100) if total else 0.0
    print(f"\nDownload success: {success}/{total} ({pct:.1f}%) — {failed} failed.")
    if not matches:
        print("No matches.")
        return pd.DataFrame()
    out = pd.DataFrame(matches).sort_values(
        by=["Vol x Avg", "Gain %"], ascending=False
    ).reset_index(drop=True)
    print(f"\n{len(out)} match(es):")
    print(out.to_string(index=False))
    return out


def _check_squeeze_at(
    ticker: str,
    df,
    breakout_pos: int,
    consol_bars: int,
    max_range_pct: float,
    breakout_pct: float,
    direction: str,
    bars_ago: int,
):
    """Run the squeeze gates with `breakout_pos` (negative iloc) as the breakout bar."""
    today = df.iloc[breakout_pos]
    prev = df.iloc[breakout_pos - 1]
    # consol bars are the `consol_bars` bars immediately before the breakout
    consol_start = breakout_pos - consol_bars
    consol = df.iloc[consol_start:breakout_pos]

    consol_high = float(consol["High"].max())
    consol_low = float(consol["Low"].min())
    mid = (consol_high + consol_low) / 2
    if mid <= 0:
        return None
    consol_range_pct = (consol_high - consol_low) / mid * 100
    if consol_range_pct > max_range_pct:
        return None

    today_vol = float(today["Volume"])
    prev_vol = float(prev["Volume"])
    if prev_vol <= 0 or today_vol <= prev_vol:
        return None

    today_close = float(today["Close"])
    today_open = float(today["Open"])
    up_pct = (today_close - consol_high) / consol_high * 100
    down_pct = (consol_low - today_close) / consol_low * 100

    direction_str = None
    breakout_pct_val = 0.0
    if direction in ("up", "both") and up_pct >= breakout_pct and today_close > today_open:
        direction_str = "UP"
        breakout_pct_val = up_pct
    elif direction in ("down", "both") and down_pct >= breakout_pct and today_close < today_open:
        direction_str = "DOWN"
        breakout_pct_val = down_pct
    if direction_str is None:
        return None

    return {
        "Ticker": ticker,
        "Date": df.index[breakout_pos].date() if hasattr(df.index[breakout_pos], "date") else df.index[breakout_pos],
        "Bars Ago": bars_ago,
        "Direction": direction_str,
        "Consol Range %": round(consol_range_pct, 2),
        "Consol Low": round(consol_low, 2),
        "Consol High": round(consol_high, 2),
        "Breakout %": round(breakout_pct_val, 2),
        "Vol vs Prev": round(today_vol / prev_vol, 2),
        "Close": round(today_close, 2),
    }


def evaluate_squeeze(
    ticker: str,
    df,
    consol_bars: int,
    max_range_pct: float,
    breakout_pct: float,
    direction: str,
    max_lookback: int = DEFAULT_SQUEEZE_MAX_LOOKBACK,
    min_consol_bars: int = DEFAULT_SQUEEZE_MIN_BARS,
):
    """Detect a tight consolidation followed by a breakout in the last few bars.

    Two-dimensional sliding search:
      - Candidate breakout position slides back from -1 to -(1 + max_lookback).
      - For each candidate, consolidation length tries the LONGEST tight stretch
        immediately before it (from `consol_bars` down to `min_consol_bars`).
        Bigger consolidations win — if a disruptive big bar sits just before the
        consolidation, the shorter window still passes.

    Returns the first hit (most recent breakout, longest valid consolidation).
    Match criteria per attempt (B = breakout bar, L = consolidation length):
      - (max High - min Low) over L bars / midpoint <= max_range_pct
      - B closes outside that range by >= breakout_pct
      - B's volume > bar immediately before B's volume
      - bullish bar for 'up', bearish bar for 'down'
    """
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    min_consol_bars = max(2, min_consol_bars)

    for bars_ago in range(max_lookback + 1):
        breakout_pos = -1 - bars_ago
        # Try the longest consolidation first; shrink if it fails.
        for consol_len in range(consol_bars, min_consol_bars - 1, -1):
            if len(df) < consol_len + bars_ago + 2:
                continue
            hit = _check_squeeze_at(
                ticker, df, breakout_pos,
                consol_len, max_range_pct, breakout_pct, direction,
                bars_ago,
            )
            if hit:
                hit["Consol Bars"] = consol_len
                return hit
    return None


def scan_squeeze(
    tickers: list[str],
    consol_bars: int,
    max_range_pct: float,
    breakout_pct: float,
    direction: str,
    timeframe: str = DEFAULT_SQUEEZE_TIMEFRAME,
    max_lookback: int = DEFAULT_SQUEEZE_MAX_LOOKBACK,
    min_consol_bars: int = DEFAULT_SQUEEZE_MIN_BARS,
):
    print(
        f"Scanning {len(tickers)} tickers on {timeframe} bars for "
        f"{min_consol_bars}-{consol_bars} bar tight consolidation → {direction} breakout "
        f"(range <= {max_range_pct}%, breakout >= {breakout_pct}%, "
        f"breakout within last {max_lookback + 1} bars)..."
    )
    if SCAN_START and SCAN_END:
        print(f"  Using data window {SCAN_START} → {SCAN_END} (evaluating last bar in window)")

    start, end = _resolve_fetch_window()
    # Yahoo caps intraday history at ~730 days; truncate start if needed.
    if timeframe != "1d":
        max_start = dt.date.today() - dt.timedelta(days=720)
        if start < max_start:
            start = max_start

    matches = []
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = {ex.submit(fetch_history, t, start, end, timeframe): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="squeeze"):
            ticker = futures[fut]
            df = fut.result()
            if df is None or df.empty:
                failed += 1
                continue
            success += 1
            try:
                hit = evaluate_squeeze(
                    ticker, df,
                    consol_bars, max_range_pct, breakout_pct, direction,
                    max_lookback=max_lookback, min_consol_bars=min_consol_bars,
                )
                if hit:
                    matches.append(hit)
            except Exception as e:
                if VERBOSE:
                    print(f"evaluate_squeeze {ticker}: {e}")

    total = success + failed
    pct = (success / total * 100) if total else 0.0
    print(f"\nSqueeze fetch success: {success}/{total} ({pct:.1f}%) — {failed} failed.")
    if not matches:
        print("No matches.")
        return pd.DataFrame()
    out = pd.DataFrame(matches).sort_values(
        by=["Breakout %", "Vol vs Prev"], ascending=False
    ).reset_index(drop=True)
    print(f"\n{len(out)} match(es):")
    print(out.to_string(index=False))
    return out


def scan_earnings(tickers: list[str], min_surprise_pct: float):
    if SCAN_START and SCAN_END:
        print(
            f"Scanning {len(tickers)} tickers for earnings beats >= {min_surprise_pct}% "
            f"released in {SCAN_START} → {SCAN_END}..."
        )
    else:
        ptd = _previous_trading_day()
        print(
            f"Scanning {len(tickers)} tickers for earnings beats >= {min_surprise_pct}% "
            f"released on previous trading day ({ptd})..."
        )

    matches = []
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = {ex.submit(fetch_earnings_data, t): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="earnings"):
            ticker = futures[fut]
            record = fut.result()
            if record is None:
                failed += 1
                continue
            success += 1
            try:
                hit = evaluate_earnings(ticker, record, min_surprise_pct)
                if hit:
                    matches.append(hit)
            except Exception as e:
                if VERBOSE:
                    print(f"evaluate_earnings {ticker}: {e}")

    total = success + failed
    pct = (success / total * 100) if total else 0.0
    print(f"\nEarnings fetch success: {success}/{total} ({pct:.1f}%) — {failed} failed.")
    if not matches:
        print("No matches.")
        return pd.DataFrame()
    out = pd.DataFrame(matches).sort_values(
        by=["Surprise %"], ascending=False
    ).reset_index(drop=True)
    print(f"\n{len(out)} match(es):")
    print(out.to_string(index=False))
    return out


def fetch_revisions_data(ticker: str) -> dict | None:
    """Fetch the earningsTrend module — contains analyst EPS revision counts."""
    crumb = _get_crumb()
    if not crumb:
        return None
    session = _get_session()
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(
                QUOTE_SUMMARY_URL.format(ticker=ticker),
                params={"modules": "earningsTrend", "crumb": crumb},
                timeout=FETCH_TIMEOUT,
            )
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            result = (data.get("quoteSummary") or {}).get("result") or []
            return result[0] if result else None
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                if VERBOSE:
                    print(f"revisions fetch {ticker}: {e}")
                return None
            time.sleep(0.5 * (2 ** attempt))
    return None


def evaluate_revisions(
    ticker: str,
    data: dict | None,
    min_acceleration: float,
    min_up7d: int,
) -> dict | None:
    """Catch analyst-revision acceleration: 7-day net upgrade pace vs 30-day baseline.

    Mirrors the volume-thrust idea — we want recent activity well above the
    trailing average. Output is the +1q (next-quarter) revision behaviour, which
    is the most actively-updated horizon.
    """
    if not data:
        return None
    trend = (data.get("earningsTrend") or {}).get("trend") or []
    next_q = next((t for t in trend if t.get("period") == "+1q"), None)
    if next_q is None:
        return None
    revs = next_q.get("epsRevisions") or {}

    def _ct(key: str) -> int:
        v = (revs.get(key) or {}).get("raw")
        return int(v) if v is not None else 0

    up7, down7 = _ct("upLast7days"), _ct("downLast7days")
    up30, down30 = _ct("upLast30days"), _ct("downLast30days")

    # Noise filter: need a meaningful absolute count of recent upgrades.
    if up7 < min_up7d:
        return None
    # Net-positive momentum required this week.
    net7 = up7 - down7
    if net7 <= 0:
        return None

    pace7 = net7 / 7
    pace30 = (up30 - down30) / 30
    # Smooth the baseline so a zero/negative 30d pace doesn't divide by zero.
    # 0.05 ≈ 1.5 net upgrades per month — small but non-trivial.
    acceleration = pace7 / max(pace30, 0.05)
    if acceleration < min_acceleration:
        return None

    return {
        "Ticker": ticker,
        "Up 7d": up7,
        "Down 7d": down7,
        "Up 30d": up30,
        "Down 30d": down30,
        "Acceleration": round(acceleration, 2),
    }


def scan_revisions(
    tickers: list[str],
    min_acceleration: float,
    min_up7d: int,
):
    print(
        f"Scanning {len(tickers)} tickers for revision acceleration "
        f">= {min_acceleration}x (min {min_up7d} upgrades in last 7d)..."
    )
    if SCAN_START and SCAN_END:
        print(
            "  NOTE: --start/--end is ignored for this scan — Yahoo's revision API "
            "is a current snapshot, not a historical series."
        )

    matches = []
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = {ex.submit(fetch_revisions_data, t): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="revisions"):
            ticker = futures[fut]
            data = fut.result()
            if data is None:
                failed += 1
                continue
            success += 1
            try:
                hit = evaluate_revisions(ticker, data, min_acceleration, min_up7d)
                if hit:
                    matches.append(hit)
            except Exception as e:
                if VERBOSE:
                    print(f"evaluate_revisions {ticker}: {e}")

    total = success + failed
    pct = (success / total * 100) if total else 0.0
    print(f"\nGrowth fetch success: {success}/{total} ({pct:.1f}%) — {failed} failed.")
    if not matches:
        print("No matches.")
        return pd.DataFrame()
    out = pd.DataFrame(matches).sort_values(
        by=["Acceleration", "Up 7d"], ascending=False
    ).reset_index(drop=True)
    print(f"\n{len(out)} match(es):")
    print(out.to_string(index=False))
    return out


def fetch_target_data(ticker: str) -> dict | None:
    """Pull upgrade/downgrade history + current price in one quoteSummary call."""
    crumb = _get_crumb()
    if not crumb:
        return None
    session = _get_session()
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(
                QUOTE_SUMMARY_URL.format(ticker=ticker),
                params={"modules": "upgradeDowngradeHistory,price", "crumb": crumb},
                timeout=FETCH_TIMEOUT,
            )
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            result = (data.get("quoteSummary") or {}).get("result") or []
            if not result:
                return None
            res = result[0]
            history = (res.get("upgradeDowngradeHistory") or {}).get("history") or []
            cur_price = ((res.get("price") or {}).get("regularMarketPrice") or {}).get("raw")
            return {"history": history, "current_price": cur_price}
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                if VERBOSE:
                    print(f"target fetch {ticker}: {e}")
                return None
            time.sleep(0.5 * (2 ** attempt))
    return None


def evaluate_target_hikes(
    ticker: str,
    data: dict | None,
    lookback_days: int,
    min_raisers: int,
    min_raise_pct: float,
) -> dict | None:
    """Surface broad analyst price-target hikes in the last `lookback_days`.

    Requires at least `min_raisers` distinct firms raised target AND the median
    raise % across those events is >= `min_raise_pct`. Filtering by distinct
    firms avoids one analyst getting double-counted via multiple actions.
    """
    if not data:
        return None
    history = data.get("history") or []
    current_price = data.get("current_price")
    if not history:
        return None
    cutoff = time.time() - lookback_days * 86400
    # firm -> (% hike, current target $) for that firm's latest action in window
    raises_by_firm: dict[str, tuple[float, float]] = {}
    for ev in history:
        if (ev.get("epochGradeDate") or 0) < cutoff:
            continue
        if ev.get("priceTargetAction") != "Raises":
            continue
        firm = ev.get("firm")
        cur = ev.get("currentPriceTarget")
        prior = ev.get("priorPriceTarget")
        if not firm or cur is None or prior is None or prior <= 0:
            continue
        pct = (cur - prior) / prior * 100
        # Keep latest hike per firm (history is newest-first, so first wins).
        raises_by_firm.setdefault(firm, (pct, float(cur)))

    if len(raises_by_firm) < min_raisers:
        return None
    pcts = sorted(p for p, _ in raises_by_firm.values())
    targets = sorted(t for _, t in raises_by_firm.values())
    n = len(pcts)
    median_pct = pcts[n // 2] if n % 2 == 1 else (pcts[n // 2 - 1] + pcts[n // 2]) / 2
    if median_pct < min_raise_pct:
        return None

    median_target = targets[n // 2] if n % 2 == 1 else (targets[n // 2 - 1] + targets[n // 2]) / 2

    out = {
        "Ticker": ticker,
        "Firms Raised": n,
        "Median Raise %": round(median_pct, 1),
        "Max Raise %": round(max(pcts), 1),
        "Min Raise %": round(min(pcts), 1),
        "Max Target $": round(max(targets), 2),
        "Min Target $": round(min(targets), 2),
    }
    if current_price and current_price > 0:
        out["Price $"] = round(float(current_price), 2)
        out["Upside Median %"] = round((median_target - current_price) / current_price * 100, 1)
        out["Upside Max %"] = round((max(targets) - current_price) / current_price * 100, 1)
    return out


def scan_target_hikes(
    tickers: list[str],
    lookback_days: int,
    min_raisers: int,
    min_raise_pct: float,
):
    print(
        f"Scanning {len(tickers)} tickers for analyst price-target hikes "
        f"(>= {min_raisers} firms in last {lookback_days}d, median raise >= {min_raise_pct}%)..."
    )
    if SCAN_START and SCAN_END:
        print(
            "  NOTE: --start/--end is ignored for this scan — upgradeDowngradeHistory "
            "uses the last N days from now, not a historical window."
        )

    matches = []
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = {ex.submit(fetch_target_data, t): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="targets"):
            ticker = futures[fut]
            data = fut.result()
            if data is None:
                failed += 1
                continue
            success += 1
            try:
                hit = evaluate_target_hikes(
                    ticker, data, lookback_days, min_raisers, min_raise_pct
                )
                if hit:
                    matches.append(hit)
            except Exception as e:
                if VERBOSE:
                    print(f"evaluate_target_hikes {ticker}: {e}")

    total = success + failed
    pct = (success / total * 100) if total else 0.0
    print(f"\nTarget fetch success: {success}/{total} ({pct:.1f}%) — {failed} failed.")
    if not matches:
        print("No matches.")
        return pd.DataFrame()
    out = pd.DataFrame(matches).sort_values(
        by=["Median Raise %", "Firms Raised"], ascending=False
    ).reset_index(drop=True)
    print(f"\n{len(out)} match(es):")
    print(out.to_string(index=False))
    return out


# ---------- Calendar helpers (earnings + economic releases) ----------

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """Date of the n-th `weekday` (0=Mon, 4=Fri) in given month."""
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    last = dt.date(year, month, calendar.monthrange(year, month)[1])
    offset = (last.weekday() - weekday) % 7
    return last - dt.timedelta(days=offset)


def _nth_business_day(year: int, month: int, n: int) -> dt.date:
    """The n-th Mon-Fri of the month (ignores US holidays — close enough)."""
    d = dt.date(year, month, 1)
    bd = 0
    while True:
        if d.weekday() < 5:
            bd += 1
            if bd == n:
                return d
        d += dt.timedelta(days=1)


# Hardcoded FOMC rate-decision announcement days (day 2 of each meeting).
# Update annually from https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_ANNOUNCEMENT_DATES = [
    dt.date(2026, 1, 28),  dt.date(2026, 3, 18),  dt.date(2026, 4, 29),
    dt.date(2026, 6, 17),  dt.date(2026, 7, 29),  dt.date(2026, 9, 16),
    dt.date(2026, 11, 4),  dt.date(2026, 12, 16),
    dt.date(2027, 1, 27),  dt.date(2027, 3, 17),  dt.date(2027, 4, 28),
]


def compute_economic_releases(days_ahead: int = 30) -> list[dict]:
    """Major US economic releases in [today, today + days_ahead].

    Dates use well-known cadences (e.g. NFP = 1st Friday of month). They're
    accurate to within ±1 day for most releases; FOMC dates are exact since
    they're hardcoded. Federal holidays are not accounted for.
    """
    today = dt.date.today()
    end = today + dt.timedelta(days=days_ahead)
    events: list[dict] = []

    for d in FOMC_ANNOUNCEMENT_DATES:
        if today <= d <= end:
            events.append({"Date": d, "Event": "FOMC Rate Decision", "Source": "scheduled"})

    # Walk months overlapping the window
    year, month = today.year, today.month
    while dt.date(year, month, 1) <= end:
        monthly = {
            "Non-Farm Payrolls":      _nth_weekday(year, month, 4, 1),  # 1st Fri
            "ISM Manufacturing PMI":  _nth_business_day(year, month, 1),
            "ISM Services PMI":       _nth_business_day(year, month, 3),
            "CPI Report":             _nth_weekday(year, month, 2, 2),  # 2nd Wed
            "PPI Report":             _nth_weekday(year, month, 3, 2),  # 2nd Thu
            "Retail Sales":           _nth_weekday(year, month, 1, 3),  # 3rd Tue
            "PCE Deflator":           _last_weekday(year, month, 4),    # last Fri
        }
        for name, date in monthly.items():
            if today <= date <= end:
                events.append({"Date": date, "Event": name, "Source": "approx"})
        if month in (1, 4, 7, 10):
            gdp = _last_weekday(year, month, 4)
            if today <= gdp <= end:
                events.append({"Date": gdp, "Event": "GDP Advance Estimate", "Source": "approx"})
        # advance
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)

    # Initial Jobless Claims — every Thursday
    d = today + dt.timedelta(days=(3 - today.weekday()) % 7)  # next Thursday
    while d <= end:
        events.append({"Date": d, "Event": "Initial Jobless Claims", "Source": "weekly"})
        d += dt.timedelta(days=7)

    events.sort(key=lambda e: (e["Date"], e["Event"]))
    return events


_EARNINGS_CALENDAR_CACHE: dict[str, tuple[float, dict | None]] = {}
EARNINGS_CALENDAR_CACHE_TTL_SECONDS = 24 * 3600


def fetch_next_earnings(ticker: str) -> dict | None:
    """Next reported earnings date + analyst EPS estimate for `ticker`.

    Returns None if Yahoo has no scheduled date. In-memory TTL cache (24h).
    """
    cached = _EARNINGS_CALENDAR_CACHE.get(ticker)
    if cached and time.time() - cached[0] < EARNINGS_CALENDAR_CACHE_TTL_SECONDS:
        return cached[1]

    crumb = _get_crumb()
    if not crumb:
        return None
    session = _get_session()
    try:
        r = session.get(
            QUOTE_SUMMARY_URL.format(ticker=ticker),
            params={"modules": "calendarEvents", "crumb": crumb},
            timeout=FETCH_TIMEOUT,
        )
        if r.status_code != 200:
            _EARNINGS_CALENDAR_CACHE[ticker] = (time.time(), None)
            return None
        data = r.json()
        res = (data.get("quoteSummary") or {}).get("result") or [{}]
        ce = (res[0].get("calendarEvents") or {}).get("earnings") or {}
        dates = ce.get("earningsDate") or []
        if not dates or (dates[0] or {}).get("raw") is None:
            _EARNINGS_CALENDAR_CACHE[ticker] = (time.time(), None)
            return None
        d = dt.datetime.fromtimestamp(dates[0]["raw"], tz=dt.timezone.utc).date()
        est = (ce.get("earningsAverage") or {}).get("raw")
        out = {
            "Ticker": ticker,
            "Date": d,
            "EPS Estimate": round(float(est), 2) if est is not None else None,
            "Is Estimated Date": ce.get("isEarningsDateEstimate", True),
        }
        _EARNINGS_CALENDAR_CACHE[ticker] = (time.time(), out)
        return out
    except Exception as e:
        if VERBOSE:
            print(f"earnings calendar fetch {ticker}: {e}")
        return None


def get_upcoming_earnings(tickers: list[str], days_ahead: int = 30) -> list[dict]:
    """Bulk fetch — returns earnings scheduled in [today, today + days_ahead]."""
    today = dt.date.today()
    cutoff = today + dt.timedelta(days=days_ahead)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = {ex.submit(fetch_next_earnings, t): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="earnings cal"):
            data = fut.result()
            if data and today <= data["Date"] <= cutoff:
                rows.append(data)
    rows.sort(key=lambda r: (r["Date"], r["Ticker"]))
    return rows


def _print_cross_scan_summary(results: dict[str, pd.DataFrame]):
    """Show tickers that appear in 2+ scan results, with which scans caught them."""
    ticker_scans: dict[str, list[str]] = {}
    for scan_name, df in results.items():
        if df is None or df.empty:
            continue
        for ticker in df["Ticker"]:
            ticker_scans.setdefault(ticker, []).append(scan_name)

    multi = {t: s for t, s in ticker_scans.items() if len(s) >= 2}
    print("\n=== CROSS-SCAN SUMMARY (tickers caught by 2+ scanners) ===")
    if not multi:
        print("No tickers appear in 2+ scans.")
        return
    rows = sorted(
        ({"Ticker": t, "# Scans": len(s), "Caught By": ", ".join(sorted(s))}
         for t, s in multi.items()),
        key=lambda r: (-r["# Scans"], r["Ticker"]),
    )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


def scan(
    mode: str = "all",
    refresh_tickers: bool = False,
    min_market_cap_usd: float | None = None,
    refresh_market_caps: bool = False,
    min_daily_volume: float | None = None,
    min_etf_assets: float | None = DEFAULT_MIN_ETF_ASSETS,
    min_surprise_pct: float = DEFAULT_MIN_SURPRISE_PCT,
    min_revision_accel: float = DEFAULT_MIN_REVISION_ACCEL,
    min_up7d: int = DEFAULT_MIN_UP7D,
    target_lookback_days: int = DEFAULT_TARGET_LOOKBACK_DAYS,
    min_target_raisers: int = DEFAULT_MIN_TARGET_RAISERS,
    min_target_raise_pct: float = DEFAULT_MIN_TARGET_RAISE_PCT,
    squeeze_bars: int = DEFAULT_SQUEEZE_BARS,
    squeeze_max_range_pct: float = DEFAULT_SQUEEZE_MAX_RANGE_PCT,
    squeeze_breakout_pct: float = DEFAULT_SQUEEZE_BREAKOUT_PCT,
    squeeze_direction: str = DEFAULT_SQUEEZE_DIRECTION,
    squeeze_timeframe: str = DEFAULT_SQUEEZE_TIMEFRAME,
    squeeze_max_lookback: int = DEFAULT_SQUEEZE_MAX_LOOKBACK,
    squeeze_min_bars: int = DEFAULT_SQUEEZE_MIN_BARS,
):
    tickers = _prepare_universe(
        refresh_tickers, min_market_cap_usd, refresh_market_caps, min_daily_volume,
        min_etf_assets=min_etf_assets,
    )
    results: dict[str, pd.DataFrame] = {}
    if mode in ("all", "uptrend"):
        print("\n=== UPTREND SCAN ===")
        results["uptrend"] = scan_uptrend(tickers)
    if mode in ("all", "squeeze"):
        print("\n=== TIGHT-CONSOLIDATION BREAKOUT SCAN ===")
        results["squeeze"] = scan_squeeze(
            tickers, squeeze_bars, squeeze_max_range_pct,
            squeeze_breakout_pct, squeeze_direction, squeeze_timeframe,
            max_lookback=squeeze_max_lookback, min_consol_bars=squeeze_min_bars,
        )
    if mode in ("all", "earnings"):
        print("\n=== EARNINGS SURPRISE SCAN ===")
        results["earnings"] = scan_earnings(tickers, min_surprise_pct)
    if mode in ("all", "revisions"):
        print("\n=== UPWARD REVISION ACCELERATION SCAN ===")
        results["revisions"] = scan_revisions(tickers, min_revision_accel, min_up7d)
    if mode in ("all", "targets"):
        print("\n=== PRICE-TARGET HIKE SCAN ===")
        results["targets"] = scan_target_hikes(
            tickers, target_lookback_days, min_target_raisers, min_target_raise_pct
        )
    if len(results) >= 2:
        _print_cross_scan_summary(results)


_MARKETCAP_UNITS = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def _parse_market_cap(s: str) -> float:
    """Parse '5B', '100M', '1.5T', '500K' into raw USD value."""
    s = s.strip().upper()
    if not s or s[-1] not in _MARKETCAP_UNITS:
        raise argparse.ArgumentTypeError(
            f"market cap must end in K/M/B/T (got {s!r}); e.g. 5B for $5 billion"
        )
    try:
        value = float(s[:-1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid market cap number {s[:-1]!r}") from e
    return value * _MARKETCAP_UNITS[s[-1]]


def _format_market_cap(usd: float) -> str:
    for suffix, mult in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if usd >= mult:
            return f"${usd / mult:g}{suffix}"
    return f"${usd:.0f}"


def _parse_date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"date must be YYYY-MM-DD (got {s!r})") from e


def _validate_range(start: dt.date, end: dt.date) -> str | None:
    """Return an error message if the range is unusable, else None."""
    if end < start:
        return f"--end ({end}) must be on or after --start ({start})."
    if start > dt.date.today():
        return f"--start ({start}) is in the future."
    return None


def main():
    global VERBOSE, SCAN_START, SCAN_END
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show yfinance download/HTTP errors (suppressed by default).",
    )
    parser.add_argument(
        "--start", type=_parse_date, default=None,
        help="Start of scan range (inclusive, YYYY-MM-DD). Requires --end.",
    )
    parser.add_argument(
        "--end", type=_parse_date, default=None,
        help="End of scan range (inclusive, YYYY-MM-DD). Defaults to today when --start is set.",
    )
    parser.add_argument(
        "--refresh-tickers", action="store_true",
        help="Force-refresh the US ticker list from NASDAQ Trader, ignoring the 24h cache.",
    )
    parser.add_argument(
        "--min-market-cap", type=_parse_market_cap, default=None, metavar="N[KMBT]",
        help="Restrict to tickers with market cap >= this. Suffix required: K/M/B/T (e.g. 5B for $5 billion, 100M for $100 million).",
    )
    parser.add_argument(
        "--refresh-market-caps", action="store_true",
        help="Force-refresh market caps from Yahoo, ignoring the 7-day cache.",
    )
    parser.add_argument(
        "--min-daily-volume", type=float, default=DEFAULT_MIN_DAILY_VOLUME, metavar="SHARES",
        help=f"Restrict to tickers whose MIN daily volume in the last {MIN_VOLUME_LOOKBACK_DAYS} trading days >= this many shares (default {DEFAULT_MIN_DAILY_VOLUME:,}; pass 0 to disable). Stricter than averages — filters out stocks with quiet days.",
    )
    parser.add_argument(
        "--min-etf-assets", type=_parse_market_cap, default=None, metavar="N[KMBT]",
        help=f"Minimum ETF net assets. ETFs are always included in the universe; raise this (e.g. 999T) to effectively exclude them. Default: ${DEFAULT_MIN_ETF_ASSETS / 1e9:g}B.",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "uptrend", "earnings", "revisions", "targets", "squeeze"),
        default="all",
        help="Which scan(s) to run: 'all' (default), 'uptrend', 'earnings', 'revisions', 'targets', or 'squeeze'.",
    )
    parser.add_argument(
        "--min-surprise", type=float, default=DEFAULT_MIN_SURPRISE_PCT, metavar="PCT",
        help=f"For --mode earnings: minimum positive EPS surprise %% to include (default {DEFAULT_MIN_SURPRISE_PCT}).",
    )
    parser.add_argument(
        "--min-revision-accel", type=float, default=DEFAULT_MIN_REVISION_ACCEL, metavar="N",
        help=f"For --mode revisions: min ratio of 7-day net-upgrade pace vs 30-day pace (default {DEFAULT_MIN_REVISION_ACCEL}).",
    )
    parser.add_argument(
        "--min-up7d", type=int, default=DEFAULT_MIN_UP7D, metavar="N",
        help=f"For --mode revisions: noise filter — require at least this many upward revisions in last 7d (default {DEFAULT_MIN_UP7D}).",
    )
    parser.add_argument(
        "--target-lookback-days", type=int, default=DEFAULT_TARGET_LOOKBACK_DAYS, metavar="N",
        help=f"For --mode targets: window in days for recent price-target hikes (default {DEFAULT_TARGET_LOOKBACK_DAYS}).",
    )
    parser.add_argument(
        "--min-target-raisers", type=int, default=DEFAULT_MIN_TARGET_RAISERS, metavar="N",
        help=f"For --mode targets: min distinct firms that raised target in window (default {DEFAULT_MIN_TARGET_RAISERS}).",
    )
    parser.add_argument(
        "--min-target-raise", type=float, default=DEFAULT_MIN_TARGET_RAISE_PCT, metavar="PCT",
        help=f"For --mode targets: median %% hike across raises must be >= this (default {DEFAULT_MIN_TARGET_RAISE_PCT}).",
    )
    parser.add_argument(
        "--squeeze-bars", type=int, default=DEFAULT_SQUEEZE_BARS, metavar="N",
        help=f"For --mode squeeze: bars of tight consolidation required (default {DEFAULT_SQUEEZE_BARS}).",
    )
    parser.add_argument(
        "--squeeze-max-range", type=float, default=DEFAULT_SQUEEZE_MAX_RANGE_PCT, metavar="PCT",
        help=f"For --mode squeeze: max consolidation range as %% of midpoint (default {DEFAULT_SQUEEZE_MAX_RANGE_PCT}).",
    )
    parser.add_argument(
        "--squeeze-breakout", type=float, default=DEFAULT_SQUEEZE_BREAKOUT_PCT, metavar="PCT",
        help=f"For --mode squeeze: min %% the breakout bar must clear the consolidation by (default {DEFAULT_SQUEEZE_BREAKOUT_PCT}).",
    )
    parser.add_argument(
        "--squeeze-direction", choices=("up", "down", "both"),
        default=DEFAULT_SQUEEZE_DIRECTION,
        help=f"For --mode squeeze: which breakout direction(s) to catch (default {DEFAULT_SQUEEZE_DIRECTION}).",
    )
    parser.add_argument(
        "--squeeze-timeframe", choices=SQUEEZE_TIMEFRAMES,
        default=DEFAULT_SQUEEZE_TIMEFRAME,
        help=f"For --mode squeeze: candle interval. '4h'/'1h' use intraday data (capped at ~720 days of history). Default: {DEFAULT_SQUEEZE_TIMEFRAME}.",
    )
    parser.add_argument(
        "--squeeze-max-lookback", type=int, default=DEFAULT_SQUEEZE_MAX_LOOKBACK, metavar="N",
        help=f"For --mode squeeze: catch breakouts that fired up to N bars ago (default {DEFAULT_SQUEEZE_MAX_LOOKBACK}, 0 = strictly the last bar).",
    )
    parser.add_argument(
        "--squeeze-min-bars", type=int, default=DEFAULT_SQUEEZE_MIN_BARS, metavar="N",
        help=f"For --mode squeeze: shortest acceptable consolidation length (default {DEFAULT_SQUEEZE_MIN_BARS}). The scan tries from --squeeze-bars down to this value.",
    )
    args = parser.parse_args()

    VERBOSE = args.verbose
    _configure_logging(VERBOSE)

    if args.start or args.end:
        start = args.start
        end = args.end or dt.date.today()
        if start is None:
            parser.error("--end given without --start")
        err = _validate_range(start, end)
        if err:
            print(err, file=sys.stderr)
            sys.exit(2)
        SCAN_START = start
        SCAN_END = end

    scan(
        mode=args.mode,
        refresh_tickers=args.refresh_tickers,
        min_market_cap_usd=args.min_market_cap,
        refresh_market_caps=args.refresh_market_caps,
        min_daily_volume=(args.min_daily_volume if args.min_daily_volume > 0 else None),
        min_etf_assets=(args.min_etf_assets if args.min_etf_assets is not None else DEFAULT_MIN_ETF_ASSETS),
        min_surprise_pct=args.min_surprise,
        min_revision_accel=args.min_revision_accel,
        min_up7d=args.min_up7d,
        target_lookback_days=args.target_lookback_days,
        min_target_raisers=args.min_target_raisers,
        min_target_raise_pct=args.min_target_raise,
        squeeze_bars=args.squeeze_bars,
        squeeze_max_range_pct=args.squeeze_max_range,
        squeeze_breakout_pct=args.squeeze_breakout,
        squeeze_direction=args.squeeze_direction,
        squeeze_timeframe=args.squeeze_timeframe,
        squeeze_max_lookback=args.squeeze_max_lookback,
        squeeze_min_bars=args.squeeze_min_bars,
    )


if __name__ == "__main__":
    main()
