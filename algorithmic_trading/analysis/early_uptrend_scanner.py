"""Scan for stocks at the very beginning of a potential uptrend.

The setup we target (e.g. IBM breaking out of a multi-month base on a wide-range
green candle with the highest volume in weeks):

    1. The stock has been quiet — sideways or down — for several weeks.
       We require its 50-day SMA to be roughly flat / not strongly trending up
       AND its return over the base window to be modest, so we don't pick up
       stocks that already ripped.
    2. The most recent session is a wide bullish candle (close > open, gain
       above a threshold).
    3. That day's volume is meaningfully larger than the recent average
       volume — the "thrust" that often marks the start of a new leg.
    4. The candle closes above its 50-day SMA (stage-2-ish confirmation).

Run:
    poetry run python -m algorithmic_trading.analysis.early_uptrend_scanner
"""

import argparse
import contextlib
import datetime as dt
import io
import logging
import os
import sys
import time
import pandas as pd
import yfinance as yf

from multiprocessing import cpu_count, set_start_method
from tqdm.contrib.concurrent import process_map


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

BATCH_SIZE = 200
MAX_RETRIES = 3

# Minimum number of trading days required to evaluate the setup on a single bar.
# SMA50 comes from Yahoo's API, so we only need enough bars for the base & volume windows.
MIN_REQUIRED_BARS = max(BASE_DAYS, VOL_AVG_DAYS) + 2


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
    # preferreds, when-issued, class quirks
    if "^" in ticker or "=" in ticker or "." in ticker:
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
    """Yahoo uses dash for class shares (BRK-A, not BRK/A)."""
    return ticker.replace("/", "-").strip().upper()


def get_tickers():
    """Read tickers from us_stocks.txt next to this file, fall back to defaults.

    Filters to clean US common stocks (drops SPAC units/rights/warrants, preferreds,
    when-issued lines) and normalizes class-share notation (BRK/A -> BRK-A).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "us_stocks.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            raw = [line.strip() for line in f if line.strip()]
    else:
        raw = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "IBM",
            "AMD", "ASML", "PLTR", "RKLB", "FIX", "LB", "EUAD",
            "QQQ", "VOO", "SCHD", "SCHG",
        ]

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
        print(f"loaded {len(raw)} raw tickers -> kept {len(out)} after US-common-stock filter")
    return out


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def download_batch(tickers):
    """Daily bars for a batch with simple retry. Honors SCAN_START/SCAN_END if set."""
    kwargs = dict(
        tickers=tickers,
        interval="1d",
        group_by="ticker",
        progress=False,
        auto_adjust=False,
        threads=True,
    )
    if SCAN_START and SCAN_END:
        # yfinance `end` is exclusive — add one day so SCAN_END is included.
        kwargs["start"] = SCAN_START.isoformat()
        kwargs["end"] = (SCAN_END + dt.timedelta(days=1)).isoformat()
    else:
        kwargs["period"] = f"{LOOKBACK_DAYS}d"

    for attempt in range(MAX_RETRIES):
        try:
            with _maybe_silence_stderr():
                return yf.download(**kwargs)
        except Exception as e:
            wait = 2 ** attempt
            if VERBOSE:
                print(f"download attempt {attempt + 1} failed: {e} — retry in {wait}s")
            time.sleep(wait)
    return None


def _fetch_sma50(ticker):
    """Pull Yahoo's 50-day moving average via fast_info. Snapshot of CURRENT value."""
    try:
        val = yf.Ticker(ticker).fast_info["fiftyDayAverage"]
        return float(val) if val is not None else None
    except Exception as e:
        if VERBOSE:
            print(f"fast_info SMA fetch failed for {ticker}: {e}")
        return None


def evaluate(ticker, df):
    """Return a result dict if `ticker` matches the early-uptrend setup, else None.

    Evaluates the LAST bar of `df` as the candidate day. Cheap filters run first;
    the SMA50 (an HTTP call to Yahoo's fast_info) runs only if everything else passes.
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

    # 4) Trend confirmation — close above SMA50 (fetched from Yahoo API).
    sma = _fetch_sma50(ticker)
    if sma is None or today["Close"] <= sma:
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


def scan_batch(batch):
    data = download_batch(batch)
    if data is None:
        return []

    matches = []
    for ticker in batch:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    continue
                df = data[ticker]
            else:
                df = data
            hit = evaluate(ticker, df)
            if hit:
                matches.append(hit)
        except Exception as e:
            if VERBOSE:
                print(f"error on {ticker}: {e}")
    return matches


def scan():
    tickers = get_tickers()
    if SCAN_START and SCAN_END:
        print(
            f"Scanning {len(tickers)} tickers using data window "
            f"{SCAN_START} → {SCAN_END} (evaluating last bar in window)..."
        )
    else:
        print(f"Scanning {len(tickers)} tickers (latest bar) for early-uptrend setups...")

    batches = list(batched(tickers, BATCH_SIZE))
    workers = max(cpu_count() - 1, 1)
    results = process_map(scan_batch, batches, max_workers=workers)

    matches = [m for batch in results for m in batch]
    if not matches:
        print("No matches.")
        return pd.DataFrame()

    df = pd.DataFrame(matches).sort_values(
        by=["Vol x Avg", "Gain %"], ascending=False
    ).reset_index(drop=True)
    print(f"\n{len(df)} match(es):")
    print(df.to_string(index=False))
    return df


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
    # We need ~MIN_REQUIRED_BARS trading days inside the window. Estimate
    # trading days as ~5/7 of calendar days; require a small safety margin.
    calendar_days = (end - start).days + 1
    est_trading_days = int(calendar_days * 5 / 7)
    if est_trading_days < MIN_REQUIRED_BARS:
        min_calendar = int(MIN_REQUIRED_BARS * 7 / 5) + 5  # round up + margin
        return (
            f"Date range is too short: {calendar_days} calendar days "
            f"(~{est_trading_days} trading days), but the {BASE_DAYS}-day base + "
            f"{VOL_AVG_DAYS}-day volume window need at least {MIN_REQUIRED_BARS} "
            f"trading days. Use a range of at least ~{min_calendar} calendar days."
        )
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
        if end < dt.date.today():
            print(
                f"warning: --end {end} is in the past, but Yahoo's fast_info SMA{SMA_LEN} "
                f"reflects the CURRENT value, not the value as of {end}. "
                f"Results for backdated windows will compare {end}'s close against today's SMA.",
                file=sys.stderr,
            )

    scan()


if __name__ == "__main__":
    set_start_method("fork")  # MacOS multiprocessing
    main()
