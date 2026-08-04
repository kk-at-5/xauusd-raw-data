"""
C4 STEP 1 — LOAD DEV SET + DERIVE BOUNDARIES + COMPUTE EMAs
==========================================================
Dev window: 2016-01-01 .. 2024-12-31  (F1..F8 train+test).
HOLDOUT 2025-01-01..2026-06-29 and EXCISED 2026-06-30..2026-07-31 are
NEVER READ. Seed burn-in comes from quarantine_pre2016 (2015-12) only.

EMA spec (rulebook): standard EMA on close, k=2/9 and 2/22, SMA-seeded then
rolled, >=200 candles of seed before any range of interest, rolls continuously
across ALL gaps, never reseeded.
"""
import pandas as pd, numpy as np, glob, os, sys

REPO = "/home/claude/repo"
CT   = os.path.join(REPO, "ctrader_m1")
OUT  = "/home/claude/work"

DEV_START = pd.Timestamp("2016-01-01")
DEV_END   = pd.Timestamp("2025-01-01")   # exclusive
SEED_FILE = os.path.join(CT, "quarantine_pre2016", "xauusd_m1_201512.csv")

def load():
    files = sorted(glob.glob(os.path.join(CT, "xauusd_m1_*.csv")))
    keep = []
    for f in files:
        ym = os.path.basename(f).replace("xauusd_m1_", "").replace(".csv", "")
        y = int(ym[:4])
        if 2016 <= y <= 2024:
            keep.append(f)
    assert not any("2025" in os.path.basename(k) or "2026" in os.path.basename(k)
                   for k in keep), "HOLDOUT LEAK"
    print(f"[load] active dev files: {len(keep)}  ({os.path.basename(keep[0])} .. "
          f"{os.path.basename(keep[-1])})")
    seed = pd.read_csv(SEED_FILE)
    print(f"[load] seed file: {os.path.basename(SEED_FILE)}  rows={len(seed)}")
    dfs = [seed] + [pd.read_csv(f) for f in keep]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("timestamp_utc")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df.datetime_utc)
    df = df[df.dt < DEV_END].reset_index(drop=True)
    return df

def emas(close):
    """SMA-seeded EMA, rolled forward. Seed length = period (standard)."""
    out8, out21 = np.empty(len(close)), np.empty(len(close))
    k8, k21 = 2/9, 2/22
    s8 = close[:8].mean(); s21 = close[:21].mean()
    out8[:8] = np.nan; out8[7] = s8
    out21[:21] = np.nan; out21[20] = s21
    e = s8
    for i in range(8, len(close)):
        e = close[i]*k8 + e*(1-k8); out8[i] = e
    e = s21
    for i in range(21, len(close)):
        e = close[i]*k21 + e*(1-k21); out21[i] = e
    return out8, out21

def derive_day_boundaries(df):
    """Data-derived: no hardcoded session times (rulebook 02 Aug)."""
    ts = df.timestamp_utc.values
    d = np.diff(ts)
    gapsz = d[d > 60]
    print("\n=== GAP DISTRIBUTION (deriving day boundary, not hardcoding) ===")
    for lo, hi, lab in [(60,300,"1-5 min"),(300,1800,"5-30 min"),
                        (1800,3000,"30-50 min"),(3000,5400,"50-90 min"),
                        (5400,21600,"1.5-6 h"),(21600,10**9,">6 h")]:
        n = int(((gapsz>=lo)&(gapsz<hi)).sum())
        print(f"  {lab:>10s} : {n:7d}")
    return d

def main():
    df = load()
    print(f"[load] rows={len(df)}  {df.dt.iloc[0]} .. {df.dt.iloc[-1]}")
    # structural sanity
    assert df.timestamp_utc.is_monotonic_increasing
    assert df.timestamp_utc.duplicated().sum() == 0
    bad = df[~((df.low<=df.open)&(df.low<=df.close)&(df.high>=df.open)&
               (df.high>=df.close)&(df.low<=df.high))]
    print(f"[sanity] dupes=0 monotonic=True OHLC-breaches={len(bad)}")
    derive_day_boundaries(df)
    e8, e21 = emas(df.close.values)
    df["ema8"], df["ema21"] = e8, e21
    seed_ok = (~np.isnan(e21)).argmax()
    print(f"[ema] first valid ema21 at row {seed_ok} ({df.dt.iloc[seed_ok]})")
    n_seed = int((df.dt < DEV_START).sum())
    print(f"[ema] seed candles before 2016-01-01: {n_seed}  (need >=200)")
    df.to_pickle(os.path.join(OUT, "dev_candles.pkl"))
    print(f"-> dev_candles.pkl  ({len(df)} rows)")

if __name__ == "__main__":
    main()
