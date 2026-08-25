"""
CYCLE 2 · TASK 2 — UNFILTERED 8/21 BASELINE on the 8 walk-forward test folds.
The bar Fresh Compression must beat OUT-OF-SAMPLE.

Folds: train<=Y -> test Y+1, test years 2017..2024 (8 folds). 8/21 is a FIXED
pair (nothing is fitted), so each test year is out-of-sample by construction;
the walk-forward structure supplies (a) per-fold sign-stability and (b) the
purge/embargo at fold seams. 2016 = train-only (not scored).

Costs: engine_fast (canonical-equivalent) — spread (session median, once/round
trip) + commission (30/1e6/side) + swap (22:00 UTC, Wed triple). Gross = zero-cost
bid open-to-open (emacross_pnl_$). Holdout NEVER read.

Purge/embargo: drop any trade whose [entry_day, exit_day] overlaps W trading days
either side of a fold seam (year boundary). W widened to ceil(p99.5 duration) if
train-fold p99.5 trade duration > 1 trading day (computed on 2016-2023, never holdout).
"""
import sys; sys.path.insert(0, "/home/claude/repo/code")
import pandas as pd, numpy as np, c4_core as C, engine_fast as EF
from scipy import stats

dev = pd.read_pickle("/home/claude/work/dev_candles.pkl")
day_id = C.day_blocks(dev)[0]
dev = EF.attach(dev)

ci, dr = C.detect_crosses_vec(dev.ema8.values, dev.ema21.values)
F = C.build_features_vec(dev, ci, dr)
comp = F[~F.censored].copy()                      # completed trades only
cost = EF.run(dev, comp, rule="naive").reset_index(drop=True)
comp = comp.reset_index(drop=True)
comp["net"]        = cost.net.values
comp["gross_cost"] = cost.gross_exec.values       # exec gross (spread embedded)
comp["commission"] = cost.commission.values
comp["swap"]       = cost.swap.values
comp["gross0"]     = comp["emacross_pnl_$"].values  # zero-cost bid open-to-open
comp["entry_day"]  = day_id[comp.entry_idx.values]
comp["exit_day"]   = day_id[comp.exit_idx.values]
comp["dur_days"]   = comp.exit_day - comp.entry_day + 1
comp["entry_dt"]   = pd.to_datetime(dev.dt.values[comp.entry_idx.values])
comp["yr"]         = comp.entry_dt.dt.year
comp["ym"]         = comp.entry_dt.dt.to_period("M")

# ---------- embargo width from TRAIN EDA (2016-2023), never holdout ----------
train = comp[(comp.yr >= 2016) & (comp.yr <= 2023)]
p995 = float(np.percentile(train.dur_days.values, 99.5))
W = max(1, int(np.ceil(p995)))
print(f"[embargo EDA] train(2016-2023) trade duration: median={np.median(train.dur_days):.0f}d "
      f"p99.5={p995:.1f}d -> embargo W = {W} trading day(s) each side of every seam")

# ---------- fold seams = first day_id of each year 2017..2025 ----------
firstday = {}
for y in range(2016, 2026):
    m = comp.entry_dt.dt.year.values == y
    if m.any(): firstday[y] = int(comp.entry_day[m].min())
seams = [firstday[y] for y in range(2017, 2026) if y in firstday]

def purged(ed_in, ed_out):
    for b in seams:
        if not (ed_out < b - W or ed_in > b + W):   # interval overlaps [b-W, b+W]
            return True
    return False
comp["purged"] = [purged(a, b) for a, b in zip(comp.entry_day, comp.exit_day)]

# ---------- per-fold table (test years 2017..2024, post-purge) ----------
print("\n=== UNFILTERED 8/21 — PER-FOLD (test years, post-purge) ===")
print(f"{'fold':>4} {'test':>5} {'n':>6} {'gross0/tr':>10} {'net/tr':>9} "
      f"{'net_total':>10} {'sign':>5}  {'mo+/mo':>8}")
rows = []
for fid, Y in enumerate(range(2017, 2025), 1):
    f = comp[(comp.yr == Y) & (~comp.purged)]
    n = len(f); g0 = f.gross0.mean(); nt = f.net.mean(); tot = f.net.sum()
    mo = f.groupby("ym").net.sum(); mopos = int((mo > 0).sum()); mon = len(mo)
    sign = "+" if tot > 0 else "-"
    rows.append(dict(fold=fid, Y=Y, n=n, g0=g0, nt=nt, tot=tot, sign=sign,
                     mopos=mopos, mon=mon))
    print(f"F{fid:>3} {Y:>5} {n:>6} {g0:>+10.4f} {nt:>+9.4f} {tot:>+10.2f} "
          f"{sign:>5}  {mopos:>3}/{mon:<3}")

R = pd.DataFrame(rows)
pooled = comp[(comp.yr >= 2017) & (comp.yr <= 2024) & (~comp.purged)]
npre = int(((comp.yr>=2017)&(comp.yr<=2024)).sum()); npost = len(pooled)
print(f"\n  purge dropped {npre-npost} of {npre} test-fold trades "
      f"({100*(npre-npost)/npre:.1f}%) at seams; {npost} scored.")

# ---------- sign-stability ----------
kpos = int((R.tot > 0).sum())
bt = stats.binomtest(kpos, 8, 0.5, alternative="two-sided")
print(f"\n=== SIGN-STABILITY (NET across 8 folds) ===")
print(f"  net-positive folds: {kpos}/8   two-sided binomial p (vs coin-flip null) = {bt.pvalue:.3f}")

# ---------- pooled net + gross ----------
print(f"\n=== POOLED (2017-2024, post-purge) ===")
print(f"  trades={len(pooled):,}")
print(f"  mean GROSS (zero-cost)  = {pooled.gross0.mean():+.4f}/trade  "
      f"total {pooled.gross0.sum():+.2f}")
print(f"  mean NET (full costs)   = {pooled.net.mean():+.4f}/trade  "
      f"total {pooled.net.sum():+.2f}")
print(f"  cost decomposition/trade: spread+dir embedded; "
      f"commission {-pooled.commission.mean():+.4f}, swap {pooled.swap.mean():+.4f}")

# ---------- OOS monthly consistency ----------
mo_all = pooled.groupby("ym").net.sum()
print(f"\n=== OOS MONTHLY CONSISTENCY ===")
print(f"  months net-positive: {int((mo_all>0).sum())}/{len(mo_all)} "
      f"({100*(mo_all>0).mean():.1f}%)   (need >50% to be a majority)")

# ---------- drawdown (chronological pooled net equity, per 1 oz) ----------
eq = pooled.sort_values("entry_dt").net.cumsum().values
peak = np.maximum.accumulate(eq); dd = peak - eq
print(f"\n=== DRAWDOWN (pooled net equity, per 1 oz = 0.01 lot) ===")
print(f"  final equity {eq[-1]:+.2f}  peak {peak.max():+.2f}  max drawdown {dd.max():.2f} $/oz")

comp.to_pickle("/home/claude/work/baseline_ledger_dev.pkl")
print(f"\n-> saved baseline_ledger_dev.pkl ({len(comp):,} completed trades w/ net + purge tags)")
