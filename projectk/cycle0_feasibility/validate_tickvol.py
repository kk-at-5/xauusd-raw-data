#!/usr/bin/env python3
"""
PROJECT K — TASK 2B VALIDATOR: tick-volume companion vs ctrader_m1 ground truth
===============================================================================
The rulebook's DATA DISCIPLINE requires CROSS-FEED VALIDATION BEFORE MIXING:
a newly-obtained series is a DIFFERENT series until proven consistent on an
overlap window. This script is that proof.

WHAT IT CHECKS
  1. TIMESTAMP AGREEMENT  — every bar in the overlap exists in both, in the
     same order, with no extras on either side. Proves same feed, same grid.
  2. CLOSE AGREEMENT      — closes match exactly. Proves the broker has NOT
     revised any historical bar since the 2026-08-01 dump. A timestamp match
     alone would not catch a silent price revision.
  3. TICK VOLUME SANITY   — zero-volume count, distribution, and whether the
     LEVEL is stable across years. cTrader tick volume counts price updates,
     not contracts; a feed-throttling change would show as a level shift and
     would invalidate any multi-year comparison built on it.

EXIT STATUS: prints PASS or FAIL per check. Any FAIL means do not use the
companion until understood — never work around it.

Run: python3 validate_tickvol.py <ctrader_m1_dir> <ctrader_m1_tickvol_dir>
"""
import sys, os, glob
import numpy as np, pandas as pd


def load(dirpath, pattern, cols):
    files = sorted(glob.glob(os.path.join(dirpath, pattern)))
    if not files:
        sys.exit(f"no files matching {pattern} in {dirpath}")
    return pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True), len(files)


def main(base_dir, tv_dir):
    print("=" * 74)
    print("TASK 2B CROSS-FEED VALIDATION")
    print("=" * 74)

    base, nb = load(base_dir, "xauusd_m1_*.csv", ["timestamp_utc", "close"])
    tv, nt = load(tv_dir, "xauusd_tv_*.csv", ["timestamp_utc", "close", "tick_volume"])
    print(f"ground truth : {len(base):,} rows from {nb} files")
    print(f"companion    : {len(tv):,} rows from {nt} files")

    # The companion was dumped later, so it extends further. Compare the OVERLAP only.
    lo = max(base.timestamp_utc.min(), tv.timestamp_utc.min())
    hi = min(base.timestamp_utc.max(), tv.timestamp_utc.max())
    b = base[(base.timestamp_utc >= lo) & (base.timestamp_utc <= hi)].reset_index(drop=True)
    t = tv[(tv.timestamp_utc >= lo) & (tv.timestamp_utc <= hi)].reset_index(drop=True)
    print(f"\noverlap      : {pd.to_datetime(lo, unit='s')} -> {pd.to_datetime(hi, unit='s')}")
    print(f"               {len(b):,} ground-truth rows vs {len(t):,} companion rows")
    print(f"companion extends {(tv.timestamp_utc.max() - hi) / 86400:.1f} days beyond ground truth")

    fails = []

    # --- CHECK 1: timestamps ---
    print("\n--- CHECK 1: TIMESTAMP AGREEMENT ---")
    if len(b) != len(t):
        print(f"  FAIL: row counts differ ({len(b):,} vs {len(t):,})")
        sb, st = set(b.timestamp_utc), set(t.timestamp_utc)
        print(f"        in ground truth only: {len(sb - st):,}")
        print(f"        in companion only   : {len(st - sb):,}")
        fails.append("timestamps")
    elif not (b.timestamp_utc.values == t.timestamp_utc.values).all():
        n = int((b.timestamp_utc.values != t.timestamp_utc.values).sum())
        print(f"  FAIL: {n:,} timestamps differ at the same row index")
        fails.append("timestamps")
    else:
        print(f"  PASS: {len(b):,} timestamps match exactly, in order")

    # --- CHECK 2: closes ---
    print("\n--- CHECK 2: CLOSE AGREEMENT (revision check) ---")
    if "timestamps" in fails:
        print("  SKIPPED: cannot compare prices when the grids differ")
    else:
        d = np.abs(b.close.values - t.close.values)
        n = int((d > 1e-9).sum())
        if n == 0:
            print(f"  PASS: all {len(b):,} closes identical. No historical revision.")
        else:
            print(f"  FAIL: {n:,} closes differ ({100*n/len(b):.4f}%), max |diff| = {d.max():.4f}")
            idx = np.where(d > 1e-9)[0][:5]
            for i in idx:
                print(f"        {pd.to_datetime(b.timestamp_utc[i], unit='s')}  "
                      f"base {b.close[i]:.2f}  companion {t.close[i]:.2f}")
            fails.append("closes")

    # --- CHECK 3: tick volume ---
    print("\n--- CHECK 3: TICK VOLUME SANITY ---")
    v = tv.tick_volume.values
    z = int((v <= 0).sum())
    print(f"  zero-volume bars : {z:,} ({100*z/len(v):.4f}%)")
    print(f"  mean/median/p99  : {v.mean():.1f} / {np.median(v):.0f} / {np.percentile(v,99):.0f} ticks per bar")
    print(f"  max              : {v.max():,.0f}")

    yr = pd.to_datetime(tv.timestamp_utc, unit="s").dt.year
    med = tv.groupby(yr).tick_volume.median()
    print("\n  MEDIAN ticks/bar BY YEAR (a level shift = feed-throttling change,")
    print("  which would invalidate multi-year comparison of raw volume):")
    for y, m in med.items():
        bar = "#" * int(min(50, m / max(1, med.max()) * 50))
        print(f"     {y}: {m:>7.0f}  {bar}")
    ratio = med.max() / max(1e-9, med.min())
    print(f"\n  max/min year ratio: {ratio:.1f}x")
    if ratio > 5:
        print("  WARN: large level shift across years. Raw tick-volume LEVELS are not")
        print("        comparable across the sample. Normalise within-regime before use.")

    print("\n" + "=" * 74)
    if fails:
        print(f"VALIDATION FAILED: {', '.join(fails)}")
        print("The companion is NOT proven consistent. Do not use it until resolved.")
        sys.exit(1)
    print("VALIDATION PASSED: same feed, same grid, no revisions.")
    print("The tick-volume companion may be joined to ctrader_m1/ on timestamp_utc.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: validate_tickvol.py <ctrader_m1_dir> <ctrader_m1_tickvol_dir>")
    main(sys.argv[1], sys.argv[2])
