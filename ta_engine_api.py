# ta_engine_final.py
"""
TA Engine Final (compact output option)
- yfinance fetch
- ta package indicators (EMA, MACD, RSI, Bollinger, OBV)
- robust window selection
- normalization to -1..1 with z-score fallback
- weightable combination -> 1..10000 score + BUY/HOLD/SELL
- presets: last5h, last1day, last3day, last5d and manual ranges YYYY-MM-DD,YYYY-MM-DD
- JSON-serializable output ready for FastAPI
- New: downsampling support to limit returned points for frontend efficiency
"""

from datetime import datetime
import math
import warnings
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

warnings.simplefilter(action="ignore", category=FutureWarning)


# ---------------- presets / helpers ----------------
def choose_short_preset(opt: str):
    o = (opt or "").lower().strip()
    if o == "last5h":
        return {"period": "1d", "interval": "5m", "slice_hours": 5}
    if o == "last1day":
        return {"period": "2d", "interval": "15m", "slice_days": 1}
    if o == "last3day":
        return {"period": "7d", "interval": "15m", "slice_days": 3}
    if o == "last5d":
        return {"period": "60d", "interval": "1d", "slice_days": 5}
    return None


def slice_df_by_last(df: pd.DataFrame, hours=None, days=None):
    if df is None or df.empty:
        return df
    last_ts = pd.to_datetime(df.index[-1])
    if hours is not None:
        cutoff = last_ts - pd.Timedelta(hours=hours)
    elif days is not None:
        cutoff = last_ts - pd.Timedelta(days=days)
    else:
        return df
    return df[df.index >= cutoff]


def normalize_ohlcv(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None:
        return pd.DataFrame()
    # flatten multiindex
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = [
            "_".join([str(x) for x in col if x is not None]).strip()
            for col in df_raw.columns.values
        ]
    cols = list(df_raw.columns)

    def find(keywords):
        for c in cols:
            name = str(c).lower()
            for k in keywords:
                if k in name:
                    return c
        return None

    mapping = {
        "Open": find(["open"]),
        "High": find(["high"]),
        "Low": find(["low"]),
        "Close": find(["adj close", "close"]),
        "Volume": find(["volume"]),
    }

    df = pd.DataFrame(index=df_raw.index)
    for k, v in mapping.items():
        if v is None:
            if k == "Volume":
                df["Volume"] = 0
            else:
                raise ValueError(f"Column {k} not found. Available columns: {cols}")
        else:
            s = df_raw[v]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            df[k] = pd.to_numeric(s, errors="coerce")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna(subset=["Close"])
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df


# ---------------- downsampling helpers ----------------
def downsample_uniform(df: pd.DataFrame, max_points: int):
    """Uniformly sample rows from df to at most max_points (preserves first & last)."""
    if df is None:
        return df
    n = len(df)
    if n <= max_points or max_points < 2:
        return df.copy()
    # select indices uniformly and ensure last index included
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    # unique and sorted
    idx = np.unique(idx)
    return df.iloc[idx]


def resample_ohlcv(df: pd.DataFrame, rule: str):
    """Resample OHLCV to a rule like '1D', '1H' (requires DatetimeIndex)."""
    if df is None or df.empty:
        return df
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    # ensure datetime index and sorted
    df2 = df.copy()
    df2.index = pd.to_datetime(df2.index)
    df2 = df2.sort_index()
    res = df2.resample(rule).agg(agg).dropna(subset=["Close"])
    # forward/backfill if needed for short gaps
    res = res.fillna(method="ffill").fillna(method="bfill")
    return res


# ---------------- robust window selection ----------------
def choose_windows(n: int, pref=None):
    if pref is None:
        pref = {
            "ema_short": 12,
            "ema_long": 26,
            "rsi": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_sig": 9,
            "bb_w": 20,
        }
    max_window = max(2, n - 1)

    def cap_pref(key, default):
        p = pref.get(key, default) if isinstance(pref, dict) else default
        w = min(p, max_window)
        return max(2, w)

    ema_s = cap_pref("ema_short", 12)
    ema_l = cap_pref("ema_long", 26)
    if ema_l <= ema_s:
        ema_l = min(max_window, ema_s + 1) if max_window > ema_s else ema_s
    rsi_w = cap_pref("rsi", 14)
    macd_fast = cap_pref("macd_fast", 12)
    macd_slow = cap_pref("macd_slow", 26)
    if macd_slow <= macd_fast:
        macd_slow = min(max_window, macd_fast + 1) if max_window > macd_fast else macd_fast
    macd_sig = cap_pref("macd_sig", 9)
    bb_w = cap_pref("bb_w", 20)
    return ema_s, ema_l, rsi_w, macd_fast, macd_slow, macd_sig, bb_w


# ---------------- compute indicators ----------------
def compute_short_indicators(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n < 2:
        df_out = df.copy()
        for col in ["ema_short", "ema_long", "rsi", "macd_hist", "macd_line", "macd_signal", "bb_h", "bb_l", "bb_m", "obv"]:
            df_out[col] = np.nan
        return df_out

    ema_s, ema_l, rsi_w, macd_fast, macd_slow, macd_sig, bb_w = choose_windows(n)

    close = df["Close"]
    vol = df["Volume"]

    try:
        ema_short = EMAIndicator(close, window=ema_s).ema_indicator()
        ema_long = EMAIndicator(close, window=ema_l).ema_indicator()
    except Exception:
        ema_short = close.rolling(window=ema_s, min_periods=1).mean()
        ema_long = close.rolling(window=ema_l, min_periods=1).mean()

    try:
        rsi = RSIIndicator(close, window=rsi_w).rsi()
    except Exception:
        rsi = pd.Series(50.0, index=df.index)

    try:
        macd_obj = MACD(close, window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_sig)
        macd_hist = macd_obj.macd_diff()
        macd_line = macd_obj.macd()
        macd_signal = macd_obj.macd_signal()
    except Exception:
        macd_hist = pd.Series(0.0, index=df.index)
        macd_line = pd.Series(0.0, index=df.index)
        macd_signal = pd.Series(0.0, index=df.index)

    try:
        bb = BollingerBands(close, window=bb_w)
        bb_h = bb.bollinger_hband()
        bb_l = bb.bollinger_lband()
        bb_m = bb.bollinger_mavg()
    except Exception:
        bb_h = close.copy()
        bb_l = close.copy()
        bb_m = close.copy()

    try:
        obv = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    except Exception:
        obv = vol.cumsum()

    out = df.copy()
    out["ema_short"] = ema_short
    out["ema_long"] = ema_long
    out["rsi"] = rsi.fillna(50)
    out["macd_hist"] = macd_hist.fillna(0)
    out["macd_line"] = macd_line.fillna(0)
    out["macd_signal"] = macd_signal.fillna(0)
    out["bb_h"] = bb_h
    out["bb_l"] = bb_l
    out["bb_m"] = bb_m
    out["obv"] = obv.fillna(0)
    return out


# ---------------- normalization helpers ----------------
def zscore_safe(series, window=30):
    if series is None or len(series) == 0:
        return None
    try:
        s = pd.Series(series).astype(float)
        if len(s) < 2:
            return None
        tail = s.tail(window) if len(s) >= window else s
        mean = tail.mean()
        std = tail.std()
        if std is None or std == 0 or np.isnan(std):
            return None
        z = (s.iloc[-1] - mean) / (std + 1e-12)
        return float(z)
    except Exception:
        return None


def norm_ema(ema_s, ema_l):
    if ema_l == 0 or pd.isna(ema_s) or pd.isna(ema_l):
        return 0.0
    raw = (ema_s - ema_l) / (abs(ema_l) + 1e-9)
    return float(np.tanh(raw * 6.0))


def norm_macd(macd_hist, df):
    if pd.isna(macd_hist):
        return 0.0
    if "macd_hist" not in df.columns:
        std = 1.0
    else:
        std = df["macd_hist"].tail(30).std()
        if std is None or np.isnan(std) or std == 0:
            std = 1e-6
    return float(np.tanh(macd_hist / (std * 1.2)))


def norm_rsi(rsi):
    if pd.isna(rsi):
        return 0.0
    v = (rsi - 50.0) / 50.0
    if rsi > 90:
        v -= 0.6
    if rsi < 10:
        v += 0.6
    return float(max(-1.0, min(1.0, v)))


def norm_bb(close, bb_m, bb_h, bb_l):
    try:
        width = max(1e-9, (bb_h - bb_l))
        pos = (close - bb_m) / (width / 2.0)
        return float(np.tanh(pos))
    except Exception:
        return 0.0


def norm_obv(df):
    try:
        if "obv" not in df.columns or len(df["obv"]) < 2:
            return 0.0
        last = float(df["obv"].iloc[-1])
        if len(df["obv"]) >= 7:
            prev = float(df["obv"].iloc[-7])
        else:
            prev = float(df["obv"].iloc[0])
        slope = (last - prev) / (abs(prev) + 1e-9)
        return float(np.tanh(math.copysign(math.log1p(abs(slope) + 1e-12), slope) * 3.0))
    except Exception:
        return 0.0


def norm_volume(df):
    try:
        n = len(df)
        if n == 0:
            return 0.0
        if n >= 10:
            avg_vol = df["Volume"].tail(30).mean()
        else:
            avg_vol = df["Volume"].mean()
        vol_now = float(df["Volume"].iloc[-1])
        vol_rel = 0.0 if avg_vol == 0 or pd.isna(avg_vol) else (vol_now / avg_vol - 1.0)
        return float(np.tanh(vol_rel * 2.0))
    except Exception:
        return 0.0


# ---------------- scoring ----------------
def score_combination(df_ind: pd.DataFrame, weights: dict = None, debug=False):
    default_weights = {"ema": 0.28, "macd": 0.22, "rsi": 0.16, "bb": 0.08, "obv": 0.14, "vol": 0.12}
    if weights:
        w = default_weights.copy()
        w.update(weights)
    else:
        w = default_weights

    row = df_ind.iloc[-1]
    ema_s = row.get("ema_short", np.nan)
    ema_l = row.get("ema_long", np.nan)
    macd_hist = row.get("macd_hist", np.nan)
    rsi = row.get("rsi", np.nan)
    bb_h = row.get("bb_h", np.nan)
    bb_l = row.get("bb_l", np.nan)
    bb_m = row.get("bb_m", np.nan)

    ema_score = norm_ema(ema_s, ema_l)
    macd_score = norm_macd(macd_hist, df_ind)
    rsi_score = norm_rsi(rsi)
    bb_score = norm_bb(row["Close"], bb_m, bb_h, bb_l)
    obv_score = norm_obv(df_ind)
    vol_score = norm_volume(df_ind)

    final_raw = (
        w["ema"] * ema_score
        + w["macd"] * macd_score
        + w["rsi"] * rsi_score
        + w["bb"] * bb_score
        + w["obv"] * obv_score
        + w["vol"] * vol_score
    )

    final = max(-1.0, min(1.0, final_raw))
    score_int = int(((final + 1.0) / 2.0) * 9999) + 1
    score_int = max(1, min(10000, score_int))
    confidence = float(abs(final))
    p_up = float((final + 1.0) / 2.0)

    contributions = {
        "ema_score": float(ema_score),
        "macd_score": float(macd_score),
        "rsi_score": float(rsi_score),
        "bb_score": float(bb_score),
        "obv_score": float(obv_score),
        "vol_score": float(vol_score),
        "final_raw": float(final_raw),
        "p_up": p_up,
        "confidence": confidence,
    }

    if debug:
        contributions["weights"] = w

    return score_int, contributions


# ---------------- sanitize output ----------------
def sanitize_value(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        try:
            return v.item()
        except Exception:
            try:
                return float(v)
            except Exception:
                pass
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, (list, tuple)):
        return [sanitize_value(x) for x in v]
    if isinstance(v, np.ndarray):
        return [sanitize_value(x) for x in v.tolist()]
    try:
        return float(v)
    except Exception:
        try:
            return str(v)
        except Exception:
            return None


def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return sanitize_value(obj)


# ---------------- top-level get_ta_json ----------------
def get_ta_json(
    ticker: str,
    range_opt: str = "last5h",
    weights: dict = None,
    save_csv: str = None,
    debug=False,
    # New params controlling returned data size:
    max_points: int = 200,
    downsample: bool = True,
    downsample_method: str = "uniform",  # 'uniform' | 'resample'
    resample_rule: str = "1D",  # used only if downsample_method == 'resample'
):
    params = choose_short_preset(range_opt)

    def fetch_with_fallback(ticker_sym, params, manual_range=None):
        try:
            if manual_range:
                start, end = manual_range
                df_raw = yf.download(ticker_sym, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
            else:
                period = params.get("period")
                interval = params.get("interval", "1d")
                df_raw = yf.download(ticker_sym, period=period, interval=interval, progress=False, auto_adjust=True)
                if "slice_hours" in params:
                    df_raw = slice_df_by_last(df_raw, hours=params["slice_hours"])
                elif "slice_days" in params:
                    df_raw = slice_df_by_last(df_raw, days=params["slice_days"])
            return df_raw
        except Exception as e:
            raise RuntimeError(f"Error fetching data: {e}")

    manual = None
    if params is None:
        if "," in (range_opt or ""):
            parts = [s.strip() for s in range_opt.split(",")]
            if len(parts) == 2:
                manual = (parts[0], parts[1])
            else:
                raise ValueError("Manual range must be 'YYYY-MM-DD,YYYY-MM-DD'")
        else:
            raise ValueError("Unrecognized range. Use preset or manual YYYY-MM-DD,YYYY-MM-DD")

    df_raw = fetch_with_fallback(ticker, params, manual_range=manual)

    required_min_bars = 30
    if df_raw is None or df_raw.empty or len(df_raw) < required_min_bars:
        try:
            if manual:
                start_dt = pd.to_datetime(manual[0]) - pd.Timedelta(days=max(60, required_min_bars))
                end_dt = pd.to_datetime(manual[1]) + pd.Timedelta(days=1)
                df_raw = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False, auto_adjust=True)
            else:
                interval = params.get("interval", "1d")
                df_raw = yf.download(ticker, period="120d", interval=interval, progress=False, auto_adjust=True)
                if params and "slice_days" in params:
                    df_raw = slice_df_by_last(df_raw, days=params["slice_days"])
                if params and "slice_hours" in params:
                    df_raw = slice_df_by_last(df_raw, hours=params["slice_hours"])
        except Exception as e:
            raise RuntimeError(f"Failed extended fetch: {e}")

    if df_raw is None or df_raw.empty:
        raise RuntimeError("No data returned. Check ticker, range, market hours, or interval.")

    # Normalize
    df = normalize_ohlcv(df_raw)
    if df.empty:
        raise RuntimeError("Normalized data empty after cleaning.")

    # If user wants resample-based reduction, do it BEFORE indicators
    if downsample and downsample_method == "resample" and resample_rule:
        df = resample_ohlcv(df, resample_rule)

    # compute indicators on df (either original or resampled)
    df_ind = compute_short_indicators(df)

    # If user wants uniform downsampling (after computing indicators), apply it
    if downsample and downsample_method == "uniform" and max_points and len(df_ind) > max_points:
        df_ind_reduced = downsample_uniform(df_ind, max_points)
        # align OHLCV to sampled timestamps
        df_reduced_ohlcv = df.loc[df_ind_reduced.index]
        df_ind = df_ind_reduced
        df = df_reduced_ohlcv
    else:
        # still possibly limit if resample produced too many points
        if downsample and max_points and len(df_ind) > max_points:
            df_ind = downsample_uniform(df_ind, max_points)
            df = df.loc[df_ind.index]

    # reliability: fraction of non-null indicator values in last row
    last_row = df_ind.iloc[-1]
    indicator_fields = ["ema_short", "ema_long", "rsi", "macd_hist", "bb_h", "bb_l", "bb_m", "obv"]
    valid_count = sum(1 for f in indicator_fields if not pd.isna(last_row.get(f, np.nan)))
    reliability = float(valid_count / len(indicator_fields))

    # compute score (uses df_ind)
    score_int, contributions = score_combination(df_ind, weights=weights, debug=debug)

    # Decision rule
    if reliability < 0.4:
        contributions["confidence"] = float(contributions.get("confidence", 0.0) * reliability)
        score_int = int((5000 + score_int) / 2)
        decision = "HOLD"
    else:
        if score_int > 6000:
            decision = "BUY"
        elif score_int < 4000:
            decision = "SELL"
        else:
            decision = "HOLD"

    # Build OHLCV list (from df which may be reduced)
    ohlcv = []
    for ts, row in df[["Open", "High", "Low", "Close", "Volume"]].iterrows():
        ohlcv.append(
            {
                "timestamp": pd.Timestamp(ts).isoformat(),
                "open": sanitize_value(row["Open"]) if not pd.isna(row["Open"]) else None,
                "high": sanitize_value(row["High"]) if not pd.isna(row["High"]) else None,
                "low": sanitize_value(row["Low"]) if not pd.isna(row["Low"]) else None,
                "close": sanitize_value(row["Close"]) if not pd.isna(row["Close"]) else None,
                "volume": sanitize_value(row["Volume"]) if not pd.isna(row["Volume"]) else None,
            }
        )

    # Build indicator arrays (from df_ind)
    indicators = {
        "ema_short": [sanitize_value(x) for x in df_ind["ema_short"].tolist()],
        "ema_long": [sanitize_value(x) for x in df_ind["ema_long"].tolist()],
        "rsi": [sanitize_value(x) for x in df_ind["rsi"].tolist()],
        "macd_hist": [sanitize_value(x) for x in df_ind["macd_hist"].tolist()],
        "bb_h": [sanitize_value(x) for x in df_ind["bb_h"].tolist()],
        "bb_l": [sanitize_value(x) for x in df_ind["bb_l"].tolist()],
        "bb_m": [sanitize_value(x) for x in df_ind["bb_m"].tolist()],
        "obv": [sanitize_value(x) for x in df_ind["obv"].tolist()],
    }

    # top contributions (numeric only)
    contrib_pairs = [
        (k, abs(v), v) for k, v in contributions.items() if isinstance(v, (int, float, np.floating, np.integer))
    ]
    contrib_pairs.sort(key=lambda x: x[1], reverse=True)
    top_contrib = [{"name": p[0], "signed_value": sanitize_value(p[2])} for p in contrib_pairs[:4]]

    result = {
        "ticker": str(ticker).upper(),
        "range": range_opt,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "score": int(score_int),
        "decision": decision,
        "confidence": sanitize_value(contributions.get("confidence", 0.0)),
        "p_up": sanitize_value(contributions.get("p_up", 0.5)),
        "contributions": sanitize(contributions),
        "top_contributions": sanitize(top_contrib),
        "ohlcv": sanitize(ohlcv),
        "indicators": sanitize(indicators),
        "reliability": float(reliability),
        "points_returned": int(len(df_ind)),
    }

    if save_csv:
        try:
            df_ind.to_csv(save_csv)
        except Exception:
            pass

    return result


# ---------------- CLI ----------------
if __name__ == "__main__":
    import json

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    rng = sys.argv[2] if len(sys.argv) > 2 else "last5d"
    # optional: third arg can be max_points
    maxp = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    out = get_ta_json(ticker, rng, max_points=maxp, downsample=True, downsample_method="uniform")
    print(json.dumps(out, indent=2))