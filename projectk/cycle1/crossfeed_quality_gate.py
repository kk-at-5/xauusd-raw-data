#!/usr/bin/env python3
"""
PROJECT K — CYCLE 1, STEP 5: CROSS-FEED + DATA-QUALITY GATES
============================================================
A source is NOT A LEVER until it clears these. Run before any M1 computation.

GATE 1 — CROSS-FEED (XAUUSD only)
    XAUUSD is the ONLY symbol for which both H1 and the SHA-manifested 1-minute
    ground truth exist. Comparing them is therefore the only available proof that
    cTrader's H1 aggregation is hour-aligned in UTC and price-identical to M1.
    If it fails, the EURUSD/USDJPY H1 files are discarded — they inherit their
    trustworthiness from this test and have no independent evidence.

GATE 2 — DATA QUALITY, PER SOURCE
    Structure (duplicates, monotonicity, OHLC breaches, hour-grid alignment) and
    coverage of the bars M1 actually requires, against XAUUSD's tradeable days.

SCOPE. Prices are examined on DEV ONLY (2016-01-01..2022-12-31). Holdout rows are
counted for structural presence and never read as prices. The seal is a date range
binding every source, not only XAUUSD.

Usage
  python3 crossfeed_quality_gate.py <ctrader_m1_dir> <h1_dir> <seal_module_dir>
"""
import sys, os, glob, datetime as dt
import numpy as np
import pandas as pd

REQUIRED_HOURS = (0, 2, 16)          # signal start, signal end / entry, exit
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
FAILURES = []


def note_fail(msg):
    FAILURES.append(msg)
    print(f"    *** FAIL: {msg}")


def epoch_at(d, hh):
    return int((dt.datetime(d.year, d.month, d.day, hh, tzinfo=dt.timezone.utc) - EPOCH).total_seconds())


def load_h1(path):
    df = pd.read_csv(path)
    df["timestamp_utc"] = df["timestamp_utc"].astype("int64")
    return df


def structural(name, df):
    """Duplicates, monotonicity, OHLC breaches, hour-grid alignment.

    Re-derived here rather than trusted from the bot's own summary: a dumper
    reporting on itself is not an independent check.
    """
    print(f"\n  [{name}] structure")
    ts = df["timestamp_utc"].values
    dup = int((np.diff(ts) == 0).sum())
    nonmono = int((np.diff(ts) < 0).sum())
    offgrid = int((ts % 3600 != 0).sum())
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    breach = int(((h < l) | (o > h) | (o < l) | (c > h) | (c < l)).sum())
    print(f"    rows {len(df):,}  duplicates {dup}  non-monotonic {nonmono}  "
          f"off-hour-grid {offgrid}  OHLC breaches {breach}")
    for label, v in (("duplicate timestamps", dup), ("non-monotonic timestamps", nonmono),
                     ("timestamps off the exact hour grid", offgrid), ("OHLC breaches", breach)):
        if v:
            note_fail(f"{name}: {v} {label}")


def coverage(name, df, dev_days, hold_days):
    """Do the bars M1 requires actually exist, on every dev tradeable day?"""
    print(f"\n  [{name}] coverage of required bars")
    have = set(df["timestamp_utc"].values.tolist())
    for hh in REQUIRED_HOURS:
        missing = [d for d in dev_days if epoch_at(d, hh) not in have]
        pct = 100.0 * (len(dev_days) - len(missing)) / len(dev_days)
        print(f"    {hh:02d}:00Z  present on {len(dev_days)-len(missing):,}/{len(dev_days):,} dev days ({pct:.3f}%)")
        if missing:
            print(f"             missing: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    # holdout: structural presence only. No price column is read.
    hmiss = sum(1 for d in hold_days for hh in REQUIRED_HOURS if epoch_at(d, hh) not in have)
    print(f"    HOLDOUT (structural presence only, prices NOT examined): "
          f"{len(hold_days)*len(REQUIRED_HOURS)-hmiss:,}/{len(hold_days)*len(REQUIRED_HOURS):,} required bars present")


def gate1_crossfeed(m1, h1, dev_days):
    """XAUUSD H1 bar opens vs the 1-minute ground truth, on dev days only."""
    print("\n" + "=" * 78)
    print("GATE 1 — CROSS-FEED: XAUUSD H1 opens vs ctrader_m1/ (DEV only)")
    print("=" * 78)
    m1o = dict(zip(m1["timestamp_utc"].values, m1["open"].values))
    h1o = dict(zip(h1["timestamp_utc"].values, h1["open"].values))
    displaced = []
    for hh in REQUIRED_HOURS:
        both = exact = 0
        mism = []
        no_m1 = []
        for d in dev_days:
            t = epoch_at(d, hh)
            a, b = m1o.get(t), h1o.get(t)
            if a is None:
                no_m1.append(d)
                continue
            if b is None:
                mism.append((d, a, None))
                continue
            both += 1
            if abs(a - b) < 1e-9:
                exact += 1
            else:
                mism.append((d, a, b))
        pct = 100.0 * exact / both if both else 0.0
        print(f"\n  {hh:02d}:00Z  comparable {both:,}  EXACT {exact:,} ({pct:.4f}%)")
        if mism:
            note_fail(f"{hh:02d}:00Z: {len(mism)} price mismatch(es) — first {mism[:5]}")
        if no_m1:
            print(f"          {len(no_m1)} day(s) have NO M1 bar at this exact minute: {no_m1}")
            for d in no_m1:
                displaced.append((d, hh))
    return displaced


def report_displacement(m1, h1, displaced):
    """Where the exact minute is absent, what does the H1 open actually represent?"""
    if not displaced:
        return
    print("\n" + "=" * 78)
    print("MINUTE-DISPLACEMENT REPORT (not a mismatch — a semantic difference)")
    print("=" * 78)
    print("  An H1 bar's open is the first traded price AT OR AFTER the hour, not the")
    print("  price AT the hour. Where the exact minute bar is missing, the H1 endpoint")
    print("  is therefore displaced by however long the quote gap lasted.")
    h1o = dict(zip(h1["timestamp_utc"].values, h1["open"].values))
    for d, hh in displaced:
        t = epoch_at(d, hh)
        w = m1[(m1.timestamp_utc >= t) & (m1.timestamp_utc < t + 3600)]
        if len(w) == 0:
            print(f"    {d} {hh:02d}:00Z  NO M1 bar anywhere in the hour; H1 open {h1o.get(t)}")
            continue
        first = int(w.timestamp_utc.iloc[0])
        mins = (first - t) // 60
        agree = abs(float(w.open.iloc[0]) - float(h1o.get(t))) < 1e-9
        print(f"    {d} {hh:02d}:00Z  first M1 bar +{mins} min, open {w.open.iloc[0]}  "
              f"H1 open {h1o.get(t)}  {'consistent' if agree else 'INCONSISTENT'}")


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    m1dir, h1dir, sealdir = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, sealdir)
    import holdout_seal as H

    print("Loading 1-minute ground truth ...")
    m1 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(m1dir, "xauusd_m1_*.csv")))],
                   ignore_index=True)
    m1["timestamp_utc"] = m1["timestamp_utc"].astype("int64")

    _, _, tdays = H._trading_days(m1["timestamp_utc"].values)
    dev = [d for d in tdays if H.DEV_START <= d <= H.DEV_END]
    hold = [d for d in tdays if H.HOLDOUT_START <= d <= H.HOLDOUT_END]
    H.assert_no_leak(dev, "DEV")            # the C9 guard, on the evaluation set
    print(f"  M1 rows {len(m1):,}   dev tradeable days {len(dev):,}   holdout {len(hold):,}   "
          f"assert_no_leak(DEV) PASSED")

    sources = {}
    for sym in ("xauusd", "eurusd", "usdjpy"):
        p = os.path.join(h1dir, f"{sym}_hour.csv")
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        sources[sym] = load_h1(p)

    displaced = gate1_crossfeed(m1, sources["xauusd"], dev)
    report_displacement(m1, sources["xauusd"], displaced)

    print("\n" + "=" * 78)
    print("GATE 2 — DATA QUALITY, PER SOURCE")
    print("=" * 78)
    for sym, df in sources.items():
        structural(sym.upper(), df)
        coverage(sym.upper(), df, dev, hold)

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"GATES FAILED — {len(FAILURES)} problem(s). These sources are NOT levers.")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("GATES PASSED. Sources are admissible under the pre-registration.")
    print("Note any displacement rows above: they are a RULE question, not a data fault.")


if __name__ == "__main__":
    main()
