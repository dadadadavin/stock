"""
historical_fetcher.py

Standalone module that fetches historical OHLCV (and related) data from yfinance
with flexible period/start-end inputs, automatic interval selection (user mapping),
optional explicit interval override, JSON-ready output, and TTL cache.

- Primary function: fetch_history(...)
- Default cache: in-memory TTLCache (auto-expire)
- Optional Redis support if you later provide a Redis client (pluggable)

Usage example (CLI):
    python historical_fetcher.py --ticker BBCA.JK --period 5y
    python historical_fetcher.py --ticker BBCA.JK --start 2020-01-01 --end 2021-12-31 --interval 1d
"""

from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta, date
import time
import os
import json
import argparse

import pandas as pd
import yfinance as yf
from cachetools import TTLCache

# Optional: import redis only if you configure it later
try:
    import redis
except Exception:
    redis = None

# -------------------------
# CONFIG
# -------------------------
DEFAULT_CACHE_TTL_SECONDS = int(os.getenv("HIST_CACHE_TTL", "600"))  # default 10 minutes
DEFAULT_CACHE_MAXSIZE = int(os.getenv("HIST_CACHE_MAXSIZE", "500"))

# max intraday days for 1m interval (Yahoo generally limits 1m to ~7 days)
MAX_INTRADAY_DAYS_1M = 7

# Create an in-memory TTL cache by default
_memory_cache = TTLCache(maxsize=DEFAULT_CACHE_MAXSIZE, ttl=DEFAULT_CACHE_TTL_SECONDS)

# You can later set a redis client via set_redis_client(...)
_redis_client = None
_redis_prefix = "hist_fetcher:"


# -------------------------
# Interval mapping (user-provided mapping)
# -------------------------
USER_INTERVAL_MAPPING = {
    "1d": "1m",     # one-day -> 1 minute
    "5d": "15m",
    "1mo": "1h",
    "3mo": "1d",
    "6mo": "1d",
    "1y": "1d",
    "2y": "1d",
    "5y": "1wk",
    "10y": "1mo",
    "ytd": "1d",
    "max": "1mo",
}

# Helper thresholds in days for automatic interval selection when given start/end
# If range_days <= threshold -> use corresponding interval
RANGE_TO_INTERVAL = [
    (1, "1m"),
    (5, "15m"),
    (31, "1h"),
    (180, "1d"),  # up to ~6 months
    (365 * 2, "1d"),  # up to 2 years: daily
    (365 * 5, "1wk"),  # up to 5 years
    (99999, "1mo"),  # beyond -> monthly
]


# -------------------------
# Utilities
# -------------------------
def set_redis_client(r: "redis.Redis", prefix: str = "hist_fetcher:"):
    """Optional: set a redis client to use as cache backend (must be configured by caller)."""
    global _redis_client, _redis_prefix
    _redis_client = r
    _redis_prefix = prefix


def _make_cache_key(ticker: str, period: Optional[str], start: Optional[str], end: Optional[str], interval: Optional[str], include_index: bool) -> str:
    return f"{ticker}|period={period or ''}|start={start or ''}|end={end or ''}|interval={interval or ''}|idx={int(include_index)}"


def _store_cache(key: str, payload: Dict[str, Any]):
    """Store in Redis if available else in-memory TTLCache"""
    if _redis_client:
        try:
            # store as JSON string with expire (in seconds)
            _redis_client.setex(f"{_redis_prefix}{key}", DEFAULT_CACHE_TTL_SECONDS, json.dumps(payload, default=str))
            return True
        except Exception as e:
            # fallback to memory cache
            print("Redis cache store failed, falling back to memory cache:", e)
    _memory_cache[key] = payload
    return True


def _load_cache(key: str) -> Optional[Dict[str, Any]]:
    if _redis_client:
        try:
            val = _redis_client.get(f"{_redis_prefix}{key}")
            if val:
                # decode bytes
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                return json.loads(val)
        except Exception as e:
            # fallback to memory
            print("Redis cache load failed:", e)

    return _memory_cache.get(key)


def _parse_iso_date(s: str) -> date:
    # Accept YYYY-MM-DD or full ISO; raises ValueError on failure
    return pd.to_datetime(s).date()


def _parse_period_to_days(period: str) -> int:
    """Parse period strings like '1d','5d','1mo','3mo','1y','5y' into approximate days"""
    p = period.lower().strip()
    if p == "ytd":
        today = date.today()
        start = date(today.year, 1, 1)
        return (today - start).days or 1
    if p == "max":
        return 365 * 50
    # number + unit
    import re
    m = re.fullmatch(r"(\d+)(d|mo|m|y|w)", p)
    if not m:
        raise ValueError(f"Invalid period format: {period}")
    num = int(m.group(1))
    unit = m.group(2)
    if unit == "d":
        return num
    if unit in ("mo", "m"):
        return num * 30
    if unit == "w":
        return num * 7
    if unit == "y":
        return num * 365
    raise ValueError("Unknown period unit")


def _choose_interval_by_range_days(days: int) -> str:
    for thresh, interval in RANGE_TO_INTERVAL:
        if days <= thresh:
            return interval
    return "1mo"


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


# -------------------------
# Core function
# -------------------------
def fetch_history(
    ticker: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: Optional[str] = None,
    include_index: bool = True,
    adjust_close: bool = True,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Fetch historical OHLCV for a ticker.
    - ticker: e.g., 'BBCA.JK'
    - period: '1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max' OR None
    - start/end: ISO date strings (YYYY-MM-DD) OR None. If provided, they take precedence over period.
    - interval: optional override interval like '1m','15m','1h','1d','1wk','1mo'
    - include_index: include original timestamp index in each row as 'Date' (ISO str)
    - adjust_close: include 'Adj Close' if available
    - force_refresh: ignore cache and fetch fresh
    Returns a JSON-serializable dict with metadata + data list.
    """

    # 1) validate ticker
    if not ticker or not isinstance(ticker, str):
        raise ValueError("ticker must be a non-empty string")
    ticker = _normalize_ticker(ticker)

    # 2) build cache key and check cache
    cache_key = _make_cache_key(ticker, period, start, end, interval, include_index)
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None:
            cached["meta"]["cache_hit"] = True
            return cached

    # 3) determine what to request
    use_interval = None
    yf_kwargs: Dict[str, Any] = {}
    # if start/end provided, use them (yfinance supports start/end)
    if start:
        try:
            start_dt = _parse_iso_date(start)
        except Exception as e:
            raise ValueError(f"Invalid 'start' date: {e}")
        if end:
            try:
                end_dt = _parse_iso_date(end)
            except Exception as e:
                raise ValueError(f"Invalid 'end' date: {e}")
        else:
            end_dt = date.today()

        # days span (at least 1)
        span_days = max(1, (end_dt - start_dt).days)
        guessed_interval = _choose_interval_by_range_days(span_days)
        use_interval = interval or guessed_interval

        # yfinance accepts start/end as strings, give ISO format
        yf_kwargs["start"] = start_dt.isoformat()
        yf_kwargs["end"] = (end_dt + timedelta(days=1)).isoformat()  # include end day by adding one day
    else:
        # no start provided -> use period
        if period is None:
            # default to 1y
            period = "1y"
        # if user mapping present
        period_lower = period.lower()
        if interval:
            use_interval = interval
        else:
            if period_lower in USER_INTERVAL_MAPPING:
                use_interval = USER_INTERVAL_MAPPING[period_lower]
            else:
                # try parse period numeric
                try:
                    days = _parse_period_to_days(period_lower)
                    use_interval = _choose_interval_by_range_days(days)
                except Exception:
                    use_interval = "1d"

        yf_kwargs["period"] = period_lower

    # 4) Validate interval vs limitations (e.g., 1m limited)
    # If requested 1m but period/span too large (> MAX_INTRADAY_DAYS_1M), degrade interval automatically
    if use_interval == "1m":
        # compute approximate days for request
        if "start" in yf_kwargs:
            # start/end mode
            start_dt = pd.to_datetime(yf_kwargs["start"]).date()
            end_dt = pd.to_datetime(yf_kwargs["end"]).date()
            days_span = (end_dt - start_dt).days
        else:
            # period mode
            try:
                days_span = _parse_period_to_days(yf_kwargs.get("period", "1d"))
            except Exception:
                days_span = 0
        if days_span > MAX_INTRADAY_DAYS_1M:
            # degrade to 15m
            prev_interval = use_interval
            use_interval = "15m"
            # we keep going; warn in meta

    # final interval selection
    chosen_interval = use_interval

    # 5) perform yf.download
    try:
        # yfinance expects interval string like '1m','15m','1h','1d','1wk','1mo'
        # we call download with either (tickers, period=..., interval=...) or (tickers, start=..., end=..., interval=...)
        df = yf.download(ticker, interval=chosen_interval, progress=False, **yf_kwargs)
    except Exception as e:
        raise RuntimeError(f"yfinance fetch failed: {e}")

    is_complete = True
    if df is None or df.empty:
        # no data
        result = {
            "meta": {
                "ticker": ticker,
                "requested": {"period": period, "start": start, "end": end, "interval": interval},
                "chosen_interval": chosen_interval,
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "data_points": 0,
                "cache_hit": False,
                "is_complete": False,
                "notes": "No rows returned by yfinance",
            },
            "data": [],
        }
        _store_cache(cache_key, result)
        return result

    # 6) Ensure DataFrame has expected columns; standardize column names
    # yfinance returns columns: ['Open','High','Low','Close','Adj Close','Volume'] (possibly multiindex for multiple tickers)
    # For single ticker, simple columns; for safety, handle multiindex by selecting ticker slice
    if isinstance(df.columns, pd.MultiIndex):
        # multi-ticker result (shouldn't happen for single ticker), select first level that matches ticker
        if ticker in df.columns.levels[1]:
            df = df.xs(ticker, axis=1, level=1)
        else:
            # fallback: take first column set
            df = df.iloc[:, df.columns.get_level_values(0) == ticker]

    # Reset index and convert Timestamp to ISO str
    df = df.reset_index()
    # Rename index column to Date if needed
    if "Date" not in df.columns:
        # the index might be 'Datetime' etc., ensure consistent name
        df = df.rename(columns={df.columns[0]: "Date"})

    # Convert Date to ISO strings (preserving timezone info if present)
    def _to_iso(dt):
        if pd.isna(dt):
            return None
        try:
            ts = pd.to_datetime(dt)
            # convert to ISO 8601 string (UTC if tz-aware)
            if ts.tzinfo is None:
                return ts.isoformat()
            else:
                return ts.tz_convert("UTC").isoformat()
        except Exception:
            return str(dt)

    df["Date"] = df["Date"].apply(_to_iso)

    # 7) If user asked adjust_close False and 'Adj Close' present, drop or keep as requested
    # Keep columns present for completeness
    available_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]

    # 8) Optionally resample to requested interval if requested interval coarse and yfinance returned finer data
    # For instance, we might request '1d' but get '1m' (rare since we pass interval), but implement safe resampling if needed.
    # Note: resampling requires DatetimeIndex; choose not to resample by default (yfinance respects requested interval).
    # We'll keep raw rows.

    # 9) Build result list of dicts
    records = []
    for _, row in df.iterrows():
        rec = {"Date": row["Date"]}
        # populate numeric fields if present (convert numpy types to python native)
        for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if c in df.columns:
                val = row.get(c)
                # convert numpy types
                if pd.isna(val):
                    rec[c] = None
                else:
                    # volume might be float but really int; coerce
                    if c == "Volume":
                        try:
                            rec[c] = int(val)
                        except Exception:
                            rec[c] = float(val)
                    else:
                        rec[c] = float(val)
        records.append(rec)

    # 10) metadata
    meta = {
        "ticker": ticker,
        "requested": {"period": period, "start": start, "end": end, "interval": interval},
        "chosen_interval": chosen_interval,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "data_points": len(records),
        "cache_hit": False,
        "is_complete": is_complete,
    }

    result = {"meta": meta, "data": records}

    # 11) store in cache
    try:
        _store_cache(cache_key, result)
    except Exception as e:
        # don't crash on cache issues
        print("Cache store error:", e)

    return result


# -------------------------
# CLI / quick test
# -------------------------
def _cli():
    parser = argparse.ArgumentParser(description="Fetch historical stock data using yfinance (with cache)")
    parser.add_argument("--ticker", required=True, help="Ticker e.g. BBCA.JK")
    parser.add_argument("--period", required=False, help="period like 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max")
    parser.add_argument("--start", required=False, help="start date YYYY-MM-DD (overrides period)")
    parser.add_argument("--end", required=False, help="end date YYYY-MM-DD (overrides period)")
    parser.add_argument("--interval", required=False, help="explicit interval override (1m,15m,1h,1d,1wk,1mo)")
    parser.add_argument("--json", action="store_true", help="output raw JSON to stdout")
    parser.add_argument("--outfile", required=False, help="write JSON to file")
    parser.add_argument("--force", action="store_true", help="force refresh (ignore cache)")

    args = parser.parse_args()

    res = fetch_history(
        ticker=args.ticker,
        period=args.period,
        start=args.start,
        end=args.end,
        interval=args.interval,
        force_refresh=args.force,
    )
    if args.json or args.outfile:
        s = json.dumps(res, indent=2, default=str)
        if args.outfile:
            with open(args.outfile, "w", encoding="utf-8") as f:
                f.write(s)
            print(f"Wrote JSON to {args.outfile}")
        else:
            print(s)
    else:
        # pretty print brief summary
        meta = res.get("meta", {})
        print("TICKER:", meta.get("ticker"))
        print("REQUESTED:", meta.get("requested"))
        print("CHOSEN INTERVAL:", meta.get("chosen_interval"))
        print("POINTS:", meta.get("data_points"))
        print("FETCHED AT:", meta.get("fetched_at"))
        print("--- FIRST 5 ROWS ---")
        for r in res.get("data", [])[:5]:
            print(r)
        print("...")

if __name__ == "__main__":
    _cli()