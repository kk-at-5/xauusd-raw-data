"""
C4 STEP 2 — full dev feature table + walk-forward structure + diagnostics.
"""
import numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/repo/code"); sys.path.insert(0, "/home/claude/work")
import c4_core as C, engine_fast as EF
from scipy import stats

W = "/home/claude/work/"
df = pd.read_pickle(W+"dev_candles.pkl")
n = len(df)
print(f"=== DEV CANDLES === {n:,} rows  {df.dt.iloc[0]} .. {df.dt.iloc[-1]}")

# ---------------- trading-day blocks (derived) ----------------
day_id, starts, ends, close_date, gi, gs, is_day = C.day_blocks(df)
nday = day_id.max()+1
print(f"=== TRADING DAYS === {nday} blocks (labelled by CLOSE date) "
      f"| boundary gaps used: {int(is_day.sum())} of {len(gi)} gaps>1min")
blk = pd.DataFrame({"day_id": np.arange(nday), "start": starts, "end": ends,
                    "close_date": close_date})
blk["ncand"] = blk.end - blk.start + 1
blk["expected"] = ((df.timestamp_utc.values[blk.end.values] -
                    df.timestamp_utc.values[blk.start.values])//60 + 1)
blk["missing"] = blk.expected - blk.ncand
blk["year"] = blk.close_date.dt.year
print(blk.groupby("year").agg(days=("day_id","size"), cand=("ncand","sum"),
      exp=("expected","sum"), miss=("missing","sum")).assign(
      miss_pct=lambda x: (100*x["miss"]/x["exp"]).round(3)).to_string())
df["day_id"] = day_id
df["close_date"] = blk.close_date.values[day_id]
df["year"] = df.close_date.dt.year

# ---------------- crosses + features ----------------
ci, dr = C.detect_crosses_vec(df.ema8.values, df.ema21.values)
F = C.build_features_vec(df, ci, dr)
print(f"\n=== LAYER A/B/C === crosses={len(ci):,} trades={len(F):,} "
      f"BULL={(dr=='BULL').sum():,} BEAR={(dr=='BEAR').sum():,}")
alt = (F.direction.values[:-1] != F.direction.values[1:]).all()
print(f"[audit] strict alternation={alt}  censored={int(F.censored.sum())} (expect 1)"
      f"  entry_only={int(F.entry_only.sum())} (expect 1)"
      f"  MFE<0={(F['mfe_$']<0).sum()}  MAE<0={(F['mae_$']<0).sum()} (expect 0)")
F["close_date"] = df.close_date.values[F.confirm_idx.values]
F["year"] = F.close_date.dt.year
F["day_id"] = df.day_id.values[F.confirm_idx.values]
F["exit_day_id"] = np.where(F.censored, -1, df.day_id.values[np.clip(F.exit_idx,0,n-1)])

# ---------------- E1 normalizer: mean_range_prior_14 (C3 D1/D2 convention) ----
h,l,c = df.high.values, df.low.values, df.close.values
tsv = df.timestamp_utc.values
gap_before = np.zeros(n, bool); gap_before[1:] = np.diff(tsv) != 60
tr = h-l
prev_c = np.roll(c,1)
tr_full = np.maximum.reduce([h-l, np.abs(h-prev_c), np.abs(l-prev_c)])
tr = np.where(gap_before, h-l, tr_full); tr[0] = h[0]-l[0]
atr14 = pd.Series(tr).shift(1).rolling(14, min_periods=14).mean().values
F["mean_range_prior_14"] = atr14[F.confirm_idx.values]
F["gap_norm"] = F["gap_at_confirm_$"]/F["mean_range_prior_14"]

# ---------------- walk-forward folds + purge ----------------
FOLDS = [(1,2016,2017),(2,2017,2018),(3,2018,2019),(4,2019,2020),
         (5,2020,2021),(6,2021,2022),(7,2022,2023),(8,2023,2024)]
seam_years = list(range(2016,2025))          # boundaries 2016/17 ... 2024/25
purge_days = set()
for y in seam_years:
    a = blk[blk.year==y]; b = blk[blk.year==y+1]
    if len(a): purge_days.add(int(a.day_id.iloc[-1]))
    if len(b): purge_days.add(int(b.day_id.iloc[0]))
print(f"\n=== PURGE/EMBARGO === seams={len(seam_years)} purged trading days={len(purge_days)}")
# a trade is dropped if its entry->exit interval overlaps any purged day
ed = np.where(F.censored, F.day_id.values, F.exit_day_id.values)
ov = np.zeros(len(F), bool)
pd_arr = np.array(sorted(purge_days))
for j,(a,b) in enumerate(zip(F.day_id.values, ed)):
    lo,hi = (a,b) if a<=b else (b,a)
    k = np.searchsorted(pd_arr, lo)
    if k < len(pd_arr) and pd_arr[k] <= hi: ov[j] = True
F["purged"] = ov | F.censored.values
print(f"trades purged (incl. 1 censored): {int(F.purged.sum())} of {len(F):,} "
      f"({100*F.purged.mean():.3f}%)")

# ---------------- embargo-width check (train folds only) ----------------
tr_pool = F[(~F.purged) & (F.year<=2023)]
p995 = np.percentile(tr_pool.duration_candles, 99.5)
med_day = blk[blk.year<=2023].ncand.median()
print(f"[embargo check] train-pool p99.5 duration = {p995:.0f} candles; "
      f"median trading day = {med_day:.0f} candles -> "
      f"{'WIDEN REQUIRED' if p995>med_day else 'one-day embargo SUFFICIENT'}")

# ---------------- missing-minute / quarantine diagnostic ----------------
print("\n=== PER-FOLD QUARANTINE DIAGNOSTIC (mechanical) ===")
present_min = set(tsv.tolist())
rows=[]
for fid, tr_end, te in FOLDS:
    b = blk[blk.year==te]
    exp_tot = int(b.expected.sum()); miss_tot = int(b.missing.sum())
    lo_t = tsv[b.start.iloc[0]]; hi_t = tsv[b.end.iloc[-1]]
    # build expected grid for the test year, block by block
    grids=[]
    for _,r in b.iterrows():
        grids.append(np.arange(tsv[int(r.start)], tsv[int(r.end)]+60, 60))
    grid = np.concatenate(grids)
    pres = np.isin(grid, tsv)
    cts = tsv[F.confirm_idx.values[(F.year.values==te)]]
    near = np.zeros(len(grid), bool)
    gi_pos = np.searchsorted(grid, cts)
    for p in gi_pos:
        near[max(0,p-21):min(len(grid),p+22)] = True
    a = int((~pres & near).sum()); b_ = int((pres & near).sum())
    c_ = int((~pres & ~near).sum()); d_ = int((pres & ~near).sum())
    p_near = a/(a+b_) if (a+b_) else 0.0
    p_all  = (a+c_)/len(grid)
    R = p_near/p_all if p_all>0 else np.nan
    if a+c_>0:
        chi2,pv,_,_ = stats.chi2_contingency([[a,b_],[c_,d_]])
    else:
        pv = 1.0
    rate = 100*miss_tot/exp_tot
    q = (R>=1.50 and pv<0.01) or (rate>2.00)
    rows.append(dict(fold=f"F{fid}", test=te, exp=exp_tot, miss=miss_tot,
                     miss_pct=round(rate,3), R=round(R,3), p=pv, quarantine=q))
Q = pd.DataFrame(rows)
Q["p"] = Q.p.map(lambda x: f"{x:.2e}")
print(Q.to_string(index=False))
print(f"USABLE FOLDS: {int((~Q.quarantine).sum())} of 8")

F.to_pickle(W+"F_dev.pkl"); blk.to_pickle(W+"blocks.pkl"); Q.to_pickle(W+"quar.pkl")
df[["timestamp_utc","dt","open","high","low","close","ema8","ema21","day_id","year"]].to_pickle(W+"df_slim.pkl")
print("\n-> F_dev.pkl / blocks.pkl / quar.pkl / df_slim.pkl")
