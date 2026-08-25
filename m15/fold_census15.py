"""
TASK 0 (Cycle 2) step 3 — verify the 15M FOLD STRUCTURE data-health claims
(~23,600 bars / ~258 days per fold, missing-rate <0.05%) using a correct
900-second expected grid. The committed build_dataset.py bakes a 60-second
grid (//60, arange(...,60)); running it unmodified on 15M would miscount
expected/missing, so this is the STEP=900 analog, clearly a Cycle-2 adaptation
(the frozen 1M file is NOT edited).
"""
import pandas as pd, numpy as np, sys, os
sys.path.insert(0, "/home/claude/repo/code")
import c4_core as C

W = "/home/claude/work"
m15 = pd.read_pickle(os.path.join(W, "xauusd_m15.pkl"))
STEP = 900

# derive trading-day blocks on the full series (close-date labelled)
day_id, starts, ends, close_date, gi, gs, is_day = C.day_blocks(m15)
blk = pd.DataFrame({"day_id": np.arange(day_id.max()+1), "start": starts, "end": ends,
                    "close_date": close_date})
ts = m15.timestamp_utc.values.astype(np.int64)
blk["ncand"]    = blk.end - blk.start + 1
blk["expected"] = (ts[blk.end.values] - ts[blk.start.values]) // STEP + 1   # 15M grid
blk["missing"]  = blk.expected - blk.ncand
blk["year"]     = blk.close_date.dt.year

FOLDS = [(1,2017),(2,2018),(3,2019),(4,2020),(5,2021),(6,2022),(7,2023),(8,2024)]
print("=== 15M WALK-FORWARD FOLD CENSUS (test years; expected grid = 900s) ===")
print(f"{'fold':>4} {'test':>5} {'days':>5} {'bars':>7} {'expected':>9} {'missing':>8} {'miss%':>7}")
rows = []
for fid, te in FOLDS:
    b = blk[blk.year == te]
    days = len(b); bars = int(b.ncand.sum())
    exp = int(b.expected.sum()); miss = int(b.missing.sum())
    pct = 100*miss/exp if exp else 0.0
    rows.append((fid, te, days, bars, exp, miss, pct))
    print(f"F{fid:>3} {te:>5} {days:>5} {bars:>7,} {exp:>9,} {miss:>8,} {pct:>6.3f}%")

miss_max = max(r[6] for r in rows)
print(f"\nmax fold missing-rate: {miss_max:.3f}%   (Cycle-2 stated <0.05%)  "
      f"OK={miss_max < 0.05}")
print(f"days/fold range: {min(r[2] for r in rows)}..{max(r[2] for r in rows)}  "
      f"(Cycle-2 stated ~258)")
print(f"bars/fold range: {min(r[3] for r in rows):,}..{max(r[3] for r in rows):,}  "
      f"(Cycle-2 stated ~23,600)")

# holdout fold-health for completeness
h = blk[(blk.close_date >= '2025-01-01') & (blk.close_date <= '2026-06-29')]
print(f"\nholdout: days={len(h)} bars={int(h.ncand.sum()):,} "
      f"missing={int(h.missing.sum())} ({100*h.missing.sum()/h.expected.sum():.3f}%)")
