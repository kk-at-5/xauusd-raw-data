"""
CYCLE 2 · TASK 3 — FIT + APPLY FRESH-COMPRESSION FILTER (walk-forward, budgeted)
+ MATCHED-EXPOSURE NULL + FREEZE BATTERY.

=========================== PRE-REGISTRATION (LOCKED) ==========================
SUBSTRATE  : 8/21 EMA cross. FIXED CONSTANT (a priori, never fit).
FILTER     : admit a cross iff compression_measure(window) <= T  ("coiled/fresh").
MEASURE    : {gap = max|ema8-ema21|, range = max(high)-min(low)} over pre-cross
             window [c-X, c-1]  (Task-1 firewall-clean, leakage-proven).
X (bars)   : {4, 8, 12, 20}.
T          : FITTED per train fold as a PERCENTILE of the train-fold measure
             distribution, percentile in {p10, p20, p30}  (self-normalising ->
             handles the price-level non-stationarity found in Task 1).
GRID       : 2 measures x 4 X x 3 percentiles = 24 combos. COMPARISON BUDGET = 24.
FIT/APPLY  : test fold Y in 2017..2024; train = crosses entered <= Y-1 (expanding,
             2016..Y-1). tau = percentile(train measure). Apply to strictly-future
             test fold Y. Fit-set != apply-set in TIME (past vs future); identical
             in CONDITIONING (same 8/21 cross population, same measure). Confirmed.
PURGE      : same as Task 2 — drop trades overlapping W=3 trading days either side
             of a fold seam (train p99.5 duration=2.2d). Applied to BOTH arms.
GAP WINDOWS: KEEP windows that span weekend/day gaps (measure over the X preceding
             bars in index order), consistent with the project's EMAs-roll-across-
             gaps convention. A within-session variant would be a 2nd mechanism ->
             out of scope (one mechanism at a time). DECLARED.
SUCCESS    : NOT "less negative than baseline". A combo can freeze only if it
             (i) is net-positive after full costs, (ii) is up-sign-stable across
             folds, (iii) BEATS the matched-exposure null, (iv) >50% positive
             months + positive/near-positive years + survivable drawdown,
             (v) power (holdout MDE < train effect), (vi) no single day >50% of
             effect, (vii) one-sentence mechanism. NO-FREEZE is the honest prior.
HOLDOUT    : SEALED. Opened only if a combo clears the ENTIRE battery in-sample.
================================================================================
"""
import sys; sys.path.insert(0, "/home/claude/repo/code")
import pandas as pd, numpy as np, c4_core as C, engine_fast as EF
from scipy import stats

dev = pd.read_pickle("/home/claude/work/dev_candles.pkl")
day_id = C.day_blocks(dev)[0]
dev = EF.attach(dev)
ema8, ema21 = dev.ema8.values, dev.ema21.values
high, low, close = dev.high.values, dev.low.values, dev.close.values
gap = np.abs(ema8 - ema21)

ci, dr = C.detect_crosses_vec(ema8, ema21)
F = C.build_features_vec(dev, ci, dr)
comp = F[~F.censored].copy().reset_index(drop=True)
cost = EF.run(dev, comp, rule="naive").reset_index(drop=True)
comp["net"]   = cost.net.values
comp["gross0"]= comp["emacross_pnl_$"].values
comp["entry_idx_"] = comp.entry_idx.values
comp["entry_day"]  = day_id[comp.entry_idx.values]
comp["exit_day"]   = day_id[comp.exit_idx.values]
comp["dur_days"]   = comp.exit_day - comp.entry_day + 1
comp["entry_dt"]   = pd.to_datetime(dev.dt.values[comp.entry_idx.values])
comp["yr"] = comp.entry_dt.dt.year
comp["ym"] = comp.entry_dt.dt.to_period("M")
comp["entry_date"] = comp.entry_dt.dt.date

XS=[4,8,12,20]
def wmeasure(cidx, X, arr_is_gap=True):
    offs=np.arange(1,X+1); W=cidx[:,None]-offs[None,:]; valid=W>=0; Wc=np.clip(W,0,None)
    if arr_is_gap:
        v=np.where(valid, gap[Wc], np.nan); return np.nanmax(v,axis=1)
    h=np.where(valid, high[Wc], np.nan); l=np.where(valid, low[Wc], np.nan)
    return np.nanmax(h,axis=1)-np.nanmin(l,axis=1)
for X in XS:
    comp[f"gap_X{X}"] = wmeasure(comp.confirm_idx.values, X, True)
    comp[f"rng_X{X}"] = wmeasure(comp.confirm_idx.values, X, False)

# purge (W=3) — same seams as Task 2
W=3
firstday={y:int(comp.entry_day[comp.yr.values==y].min()) for y in range(2016,2026)
          if (comp.yr.values==y).any()}
seams=[firstday[y] for y in range(2017,2026) if y in firstday]
def purged(a,b):
    return any(not(b<s-W or a>s+W) for s in seams)
comp["purged"]=[purged(a,b) for a,b in zip(comp.entry_day,comp.exit_day)]

print(__doc__)

# ---------- unfiltered baseline (bar to beat), post-purge test folds ----------
base = comp[(comp.yr>=2017)&(comp.yr<=2024)&(~comp.purged)]
base_npt = base.net.mean()
print(f"BASELINE (unfiltered, OOS): n={len(base):,}  net/trade={base_npt:+.4f}  "
      f"0/8 folds+  44.8% months+\n")

# ---------- 24-combo walk-forward funnel ----------
YEARS=range(2017,2025)
def run_combo(col, pct):
    fold_tot={}; fold_n={}; adm_parts=[]
    for Y in YEARS:
        tr=comp[(comp.yr>=2016)&(comp.yr<=Y-1)]
        tau=np.percentile(tr[col].values, pct)
        te=comp[(comp.yr==Y)&(~comp.purged)]
        adm=te[te[col].values<=tau]
        fold_tot[Y]=adm.net.sum(); fold_n[Y]=len(adm); adm_parts.append(adm)
    pooled=pd.concat(adm_parts)
    kpos=sum(1 for Y in YEARS if fold_tot[Y]>0)
    mo=pooled.groupby("ym").net.sum(); mopos=100*(mo>0).mean() if len(mo) else 0
    return dict(n=len(pooled), npt=pooled.net.mean(), tot=pooled.net.sum(),
                kpos=kpos, mopos=mopos, minfold_n=min(fold_n.values()),
                pooled=pooled, fold_tot=fold_tot)

print("=== 24-COMBO FUNNEL (walk-forward, T fitted per train fold) ===")
print(f"{'measure':>7} {'X':>3} {'T':>4} {'n':>5} {'n/fold_min':>10} {'net/tr':>8} "
      f"{'net_tot':>9} {'k/8+':>5} {'mo%+':>6} {'>base?':>7}")
rows=[]
for meas,pref in [("gap","gap"),("range","rng")]:
    for X in XS:
        col=f"{pref}_X{X}"
        for pct in [10,20,30]:
            r=run_combo(col,pct); r.update(meas=meas,X=X,pct=pct,col=col)
            rows.append(r)
            beat = "yes" if r["npt"]>base_npt else "no"
            print(f"{meas:>7} {X:>3} p{pct:<3} {r['n']:>5} {r['minfold_n']:>10} "
                  f"{r['npt']:>+8.4f} {r['tot']:>+9.1f} {r['kpos']:>4}/8 "
                  f"{r['mopos']:>5.1f}% {beat:>7}")

# ---------- screen for freeze candidates ----------
cand=[r for r in rows if (r["tot"]>0 and r["kpos"]>=5 and r["mopos"]>50 and r["npt"]>0)]
print(f"\n=== FREEZE-CANDIDATE SCREEN (net-positive + >=5/8 folds + >50% months) ===")
print(f"  combos passing first screen: {len(cand)} of 24")

# ---------- matched-exposure null on the BEST combo by net/trade (illustrative + mandatory if candidate) ----------
best=max(rows,key=lambda r:r["npt"])
print(f"\n=== MATCHED-EXPOSURE NULL — best combo by net/trade: "
      f"{best['meas']} X{best['X']} p{best['pct']} (net/tr {best['npt']:+.4f}, n {best['n']}) ===")
pool_unf=base.reset_index(drop=True)
nf=best["n"]; mu=best["npt"]; rng=np.random.default_rng(1); R=3000
# size-matched null
idx=np.arange(len(pool_unf)); szmeans=np.array([pool_unf.net.values[rng.choice(idx,nf,replace=False)].mean() for _ in range(R)])
pct_sz=100*(szmeans<mu).mean()
# duration-matched null: match the filtered set's dur_days histogram
fb=best["pooled"].dur_days.values
bins=np.array([1,2,3,5,8,13,21,10**9]); fb_bin=np.digitize(fb,bins); need=pd.Series(fb_bin).value_counts().to_dict()
unf_bin=np.digitize(pool_unf.dur_days.values,bins)
pools={b:idx[unf_bin==b] for b in need}
def draw_dur():
    picks=[]
    for b,k in need.items():
        p=pools.get(b,np.array([],int))
        if len(p)>=k: picks.append(rng.choice(p,k,replace=False))
        elif len(p)>0: picks.append(rng.choice(p,k,replace=True))
    return np.concatenate(picks) if picks else np.array([],int)
durmeans=np.array([pool_unf.net.values[draw_dur()].mean() for _ in range(R)])
pct_dur=100*(durmeans<mu).mean()
print(f"  filtered net/trade      : {mu:+.4f}")
print(f"  size-matched null mean  : {szmeans.mean():+.4f}  (filtered percentile in null: {pct_sz:.1f}%)")
print(f"  duration-matched null   : {durmeans.mean():+.4f}  (filtered percentile in null: {pct_dur:.1f}%)")
print(f"  -> compression selects better than a same-size random subset? "
      f"{'YES' if pct_sz>95 else 'NO'}")
print(f"  -> ... beyond the holding-period confound (duration-matched)?     "
      f"{'YES' if pct_dur>95 else 'NO'}")

# ---------- verdict ----------
print(f"\n=== CYCLE-2 VERDICT ===")
any_netpos=any(r["npt"]>0 for r in rows)
print(f"  any combo net-positive/trade? {any_netpos}")
print(f"  freeze candidates: {len(cand)}")
print(f"  best combo beats matched-exposure null? {(pct_sz>95 and pct_dur>95)}")
print(f"  VERDICT: {'FREEZE CANDIDATE -> proceed to full battery' if cand and pct_dur>95 else 'NO FREEZE — holdout stays SEALED'}")
