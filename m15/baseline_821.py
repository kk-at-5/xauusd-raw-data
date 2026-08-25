"""
TASK 0 (Cycle 2) step 4 — reproduce the UNFILTERED 8/21 baseline on the
reconstructed 15M dev set. Anchor to beat/confirm: Cycle-1 stated
~$0.014/trade gross over ~10,096 trades, a NON-survivor.
Two independent code paths (oracle-validated build_features_vec AND the
canonical pair_kernel gross sweep) must agree.
"""
import pandas as pd, numpy as np, sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/repo/code")
import c4_core as C

W = "/home/claude/work"
dev = pd.read_pickle(os.path.join(W, "dev_candles.pkl"))

# trading-day close_date -> year (build_dataset convention)
day_id, starts, ends, close_date, gi, gs, is_day = C.day_blocks(dev)
dev["day_id"] = day_id
dev["close_date"] = close_date[day_id]
dev["year"] = dev.close_date.dt.year

# ---- PATH 1: oracle-validated build_features_vec ----
ci, dr = C.detect_crosses_vec(dev.ema8.values, dev.ema21.values)
F = C.build_features_vec(dev, ci, dr)
F["year"] = dev.close_date.dt.year.values[F.confirm_idx.values]
ev = F[(~F.censored) & (F.year >= 2016) & (F.year <= 2024)]
n1 = len(ev); mean1 = ev["emacross_pnl_$"].mean()
print("=== PATH 1: build_features_vec (oracle-validated) ===")
print(f"  completed 8/21 trades 2016-2024 : {n1:,}")
print(f"  mean GROSS $/trade (zero-cost)  : {mean1:+.4f}")
print(f"  total crosses (all years)       : {len(ci):,}  completed all-yr: {int((~F.censored).sum()):,}")

# ---- PATH 2: canonical pair_kernel gross sweep (independent path) ----
p2_ok = False
try:
    import pair_kernel as PK
    o = dev.open.values.astype(np.float64)
    ef = dev.ema8.values.astype(np.float64)
    es = dev.ema21.values.astype(np.float64)
    yrs = dev.year.values
    # month/year index arrays; pre-2016 -> -1 so kernel excludes them (open_m>=0 gate)
    ymkey = dev.close_date.dt.year.values * 12 + (dev.close_date.dt.month.values - 1)
    uniq_m = {v: i for i, v in enumerate(sorted(set(ymkey[yrs >= 2016].tolist())))}
    mon = np.array([uniq_m.get(v, -1) if y >= 2016 else -1 for v, y in zip(ymkey, yrs)], dtype=np.int64)
    uniq_y = {v: i for i, v in enumerate(sorted(set(yrs[yrs >= 2016].tolist())))}
    yr = np.array([uniq_y.get(v, -1) if v >= 2016 else -1 for v in yrs], dtype=np.int64)
    n_mon, n_yr = len(uniq_m), len(uniq_y)
    mon_sum = np.zeros(n_mon); mon_n = np.zeros(n_mon, np.int64)
    yr_sum = np.zeros(n_yr);  yr_n = np.zeros(n_yr, np.int64)
    start = int((~np.isnan(es)).argmax())   # first valid ema21
    ntr, tot, totdur, totsq = PK.sweep_pair(o, ef, es, mon, yr, start, n_mon, n_yr,
                                            mon_sum, mon_n, yr_sum, yr_n)
    print("\n=== PATH 2: pair_kernel.sweep_pair (canonical gross sweep) ===")
    print(f"  completed 8/21 trades 2016-2024 : {ntr:,}")
    print(f"  mean GROSS $/trade (zero-cost)  : {tot/ntr:+.4f}")
    p2_ok = True
except Exception as e:
    print(f"\n[PATH 2 skipped: {type(e).__name__}: {e}]")

# ---- per-year gross (context: where any pseudo-edge lives) ----
print("\n=== PER-YEAR GROSS (8/21, zero-cost, completed) ===")
g = ev.groupby("year")["emacross_pnl_$"].agg(["size", "mean", "sum"])
for y, r in g.iterrows():
    print(f"  {int(y)}: n={int(r['size']):5d}  mean={r['mean']:+.4f}  sum={r['sum']:+9.2f}")

# ---- anchor check ----
print("\n=== ANCHOR CHECK vs Cycle-1 (8/21 non-survivor) ===")
print(f"  trades  : COMPUTED {n1:,}   STATED ~10,096   diff {n1-10096:+d}")
print(f"  gross/tr: COMPUTED {mean1:+.4f}   STATED ~+0.014")
if p2_ok:
    print(f"  PATH1==PATH2 trades: {n1==ntr}   gross agree: {abs(mean1-tot/ntr)<1e-9}")
