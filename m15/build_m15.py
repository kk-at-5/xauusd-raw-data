"""
build_m15.py — ONE-COMMAND deterministic rebuild of the 15M artifacts.
Run from a fresh clone:  python3 m15/build_m15.py
Produces (in m15/data/):  xauusd_m15.csv, xauusd_m15.pkl, dev_candles.pkl
and SELF-VERIFIES against Cycle-1's frozen invariants before writing.

Provenance: the 15M series is a pure OHLC aggregate of the SHA-verified 1M bulk
in ctrader_m1/. Nothing here is estimated. If any invariant fails, it aborts —
a failed invariant means the 1M bulk changed and downstream work must stop.

Invariants (Cycle-1, frozen):
  full 15M bars = 326,542 · 0 dup / 0 non-mono / 0 OHLC breach
  holdout (trading-day, close-date labelled) = 35,176 bars / 384 days
  8/21 unfiltered baseline 2016-2024 = 10,096 trades @ +$0.0137/trade gross
"""
import pandas as pd, numpy as np, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import c4_core as C  # frozen library (unchanged); provides day_blocks + oracle-validated feature/cross

HERE = os.path.dirname(os.path.abspath(__file__))
CT   = os.path.join(HERE, "..", "ctrader_m1")
OUT  = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)

DEV_START = pd.Timestamp("2015-12-01")   # 2015-12 seed (EMA burn-in), mirrors load_dev
DEV_END   = pd.Timestamp("2025-01-01")   # exclusive; HOLDOUT sealed by construction

def load_1m_full():
    files  = sorted(glob.glob(os.path.join(CT, "xauusd_m1_*.csv")))
    qfiles = sorted(glob.glob(os.path.join(CT, "quarantine_pre2016", "xauusd_m1_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in (qfiles + files)], ignore_index=True)
    df = df.drop_duplicates("timestamp_utc").sort_values("timestamp_utc").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df.datetime_utc)
    return df

def clean(df):
    dup = int(df.timestamp_utc.duplicated().sum()); mono = bool(df.timestamp_utc.is_monotonic_increasing)
    bad = int((~((df.low<=df.open)&(df.low<=df.close)&(df.high>=df.open)&(df.high>=df.close)&(df.low<=df.high))).sum())
    return dup, mono, bad

def resample_15m(m1):
    ts = m1.timestamp_utc.values.astype(np.int64)
    b = (ts // 900) * 900
    g = m1.groupby(b, sort=True)
    m15 = pd.DataFrame({
        "timestamp_utc": g.timestamp_utc.first().index.values.astype(np.int64),
        "open": g.open.first().values, "high": g.high.max().values,
        "low": g.low.min().values, "close": g.close.last().values})
    m15["dt"] = pd.to_datetime(m15.timestamp_utc, unit="s")
    m15["datetime_utc"] = m15.dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    return m15.sort_values("timestamp_utc").reset_index(drop=True)

def emas(close):
    out8, out21 = np.empty(len(close)), np.empty(len(close)); k8,k21 = 2/9, 2/22
    out8[:8]=np.nan; out8[7]=close[:8].mean(); out21[:21]=np.nan; out21[20]=close[:21].mean()
    e=out8[7]
    for i in range(8,len(close)): e=close[i]*k8+e*(1-k8); out8[i]=e
    e=out21[20]
    for i in range(21,len(close)): e=close[i]*k21+e*(1-k21); out21[i]=e
    return out8, out21

def main():
    m1 = load_1m_full()
    d,mo,b = clean(m1)
    assert (d,mo,b)==(0,True,0), f"1M bulk audit failed {(d,mo,b)} (expected 0/True/0)"
    print(f"[1M]  rows={len(m1):,}  audit 0/OK/0")

    m15 = resample_15m(m1)
    d,mo,b = clean(m15)
    assert (d,mo,b)==(0,True,0), f"15M audit failed {(d,mo,b)}"
    assert len(m15)==326542, f"15M bar count {len(m15):,} != 326,542"
    print(f"[15M] rows={len(m15):,}  audit 0/OK/0  -> 326,542 OK")

    # holdout invariant via derived trading day (close-date labelled)
    day_id,_,_,close_date,_,_,is_day = C.day_blocks(m15)
    cd = close_date[day_id]
    ho = (cd>=pd.Timestamp("2025-01-01")) & (cd<=pd.Timestamp("2026-06-29"))
    ho = np.asarray(ho)
    assert int(ho.sum())==35176, f"holdout bars {int(ho.sum())} != 35,176"
    assert pd.Series(day_id[ho]).nunique()==384, "holdout days != 384"
    print(f"[holdout] 35,176 bars / 384 days OK")

    # dev_candles (seed + 2016-2024), EMAs on 15M closes, holdout sealed
    dev = m15[(m15.dt>=DEV_START)&(m15.dt<DEV_END)].reset_index(drop=True).copy()
    assert not (dev.dt>=DEV_END).any(), "HOLDOUT LEAK"
    dev["ema8"], dev["ema21"] = emas(dev.close.values)

    # 8/21 baseline invariant
    ci,dr = C.detect_crosses_vec(dev.ema8.values, dev.ema21.values)
    F = C.build_features_vec(dev, ci, dr)
    yr = pd.DatetimeIndex(dev.dt.values[F.confirm_idx.values]).year
    ev = F[(~F.censored)&(yr>=2016)&(yr<=2024)]
    n, g = len(ev), float(ev["emacross_pnl_$"].mean())
    assert n==10096, f"8/21 trades {n} != 10,096"
    assert abs(g-0.0137)<5e-4, f"8/21 gross {g:+.4f} != +0.0137"
    print(f"[8/21] {n:,} trades @ {g:+.4f}/trade gross OK")

    m15.to_pickle(os.path.join(OUT,"xauusd_m15.pkl"))
    m15.to_csv(os.path.join(OUT,"xauusd_m15.csv"), index=False)
    dev.to_pickle(os.path.join(OUT,"dev_candles.pkl"))
    print(f"\n-> wrote m15/data/: xauusd_m15.pkl, xauusd_m15.csv, dev_candles.pkl")
    print("ALL INVARIANTS PASSED — artifacts safe to use.")

if __name__ == "__main__":
    main()
