"""
CYCLE 3 · EXPLORATION LEDGER (2016-2019, outcomes VISIBLE).
Substrates: 8/21 (fast) + 50/200 (slow), a-priori fixed. No pair sweep.
Firewall: features use bars with index <= confirm bar c only (confirmation-time,
known at c's close). Labels (mfe/mae/net/etc.) are outcomes, tagged, never inputs.
Validation (2020-2024) + holdout are NOT touched here.
"""
import sys; sys.path.insert(0, "/home/claude/repo/code")
import pandas as pd, numpy as np, c4_core as C, engine_fast as EF

dev = pd.read_pickle("/home/claude/work/dev_candles.pkl")
day_id = C.day_blocks(dev)[0]
dev = EF.attach(dev)
o,h,l,cl = dev.open.values, dev.high.values, dev.low.values, dev.close.values

def ema(close, n):
    k=2/(n+1); out=np.full(len(close),np.nan); out[n-1]=close[:n].mean(); e=out[n-1]
    for i in range(n,len(close)): e=close[i]*k+e*(1-k); out[i]=e
    return out
dev["ema50"]=ema(cl,50); dev["ema200"]=ema(cl,200)

# true range (needs prev close) for ATR
prev_close=np.concatenate([[cl[0]], cl[:-1]])
tr=np.maximum.reduce([h-l, np.abs(h-prev_close), np.abs(l-prev_close)])
logret=np.concatenate([[0.0], np.diff(np.log(cl))])

def win_end_at_c(cidx, L, arr, fn):
    """stat over [c-L+1, c] inclusive of confirm bar."""
    offs=np.arange(L-1,-1,-1); W=cidx[:,None]-offs[None,:]; valid=W>=0; Wc=np.clip(W,0,None)
    v=np.where(valid, arr[Wc], np.nan); return fn(v,axis=1)
def win_prior(cidx, N, arr, fn):
    """stat over [c-N, c-1] strictly before confirm bar."""
    offs=np.arange(N,0,-1); W=cidx[:,None]-offs[None,:]; valid=W>=0; Wc=np.clip(W,0,None)
    v=np.where(valid, arr[Wc], np.nan); return fn(v,axis=1)

WINDOWS=[4,8,20]
body=np.abs(cl-o); upper=h-np.maximum(o,cl); lower=np.minimum(o,cl)-l; wick=upper+lower

def build_substrate(name, fast_ema, slow_ema):
    ci, dr = C.detect_crosses_vec(fast_ema, slow_ema)
    # trades: cross k -> entry k+1 open, exit at next cross's entry (naive opposite cross)
    ent = ci+1
    exit_conf = np.concatenate([ci[1:], [len(cl)-1]])   # next cross confirm (last censored)
    exi = exit_conf+1
    censored = np.zeros(len(ci), bool); censored[-1]=True
    exi = np.clip(exi, 0, len(cl)-1)
    F = pd.DataFrame({"cross_seq":np.arange(len(ci)), "confirm_idx":ci, "entry_idx":ent,
                      "exit_idx":exi, "direction":np.asarray(dr),
                      "censored":censored})
    comp = F[~F.censored].copy().reset_index(drop=True)
    cost = EF.run(dev, comp, rule="naive").reset_index(drop=True)
    c = comp.confirm_idx.values
    # ---- OUTCOMES (visible) ----
    comp["net"]=cost.net.values
    comp["gross0"]=np.where(comp.direction=="BULL", o[comp.exit_idx.values]-o[comp.entry_idx.values],
                                                     o[comp.entry_idx.values]-o[comp.exit_idx.values])
    comp["dur_bars"]=comp.exit_idx.values-comp.entry_idx.values
    comp["dur_days"]=day_id[comp.exit_idx.values]-day_id[comp.entry_idx.values]+1
    comp["win"]=(comp.net>0).astype(int)
    comp["entry_dt"]=pd.to_datetime(dev.dt.values[comp.entry_idx.values])
    comp["year"]=comp.entry_dt.dt.year; comp["month"]=comp.entry_dt.dt.to_period("M")
    # ---- CONFIRMATION-TIME FEATURES (inputs; index <= c) ----
    from fe_pipeline import session_of
    comp["session"]=[session_of(pd.Timestamp(x).hour) for x in dev.dt.values[c]]
    comp["gap_at_confirm"]=np.abs(fast_ema[c]-slow_ema[c])
    comp["close_vs_slow"]=cl[c]-slow_ema[c]
    comp["body_confirm"]=body[c]
    comp["body_dominant"]=(body[c]>wick[c]).astype(int)
    # windowed (pre-cross / as-of-confirm)
    for N in WINDOWS:
        comp[f"atr_{N}"]=win_end_at_c(c,N,tr,np.nanmean)
        comp[f"rvol_{N}"]=win_end_at_c(c,N,logret,np.nanstd)
        comp[f"meanrange_{N}"]=win_prior(c,N,(h-l),np.nanmean)
        comp[f"mom_{N}"]=body[c]/win_prior(c,N,body,np.nanmean)
        comp[f"body_over_atr_{N}"]=body[c]/comp[f"atr_{N}"]
        comp[f"gap_over_atr_{N}"]=comp["gap_at_confirm"]/comp[f"atr_{N}"]
        comp[f"fslope_{N}"]=fast_ema[c]-fast_ema[np.clip(c-N,0,None)]
        comp[f"sslope_{N}"]=slow_ema[c]-slow_ema[np.clip(c-N,0,None)]
        comp[f"gapmax_pre_{N}"]=win_prior(c,N,np.abs(fast_ema-slow_ema),np.nanmax)  # fresh-compression
    # crosses_in_prior_M and candles_since_prev_cross (cross-count, STEP-independent)
    csp=np.diff(ci, prepend=ci[0]); comp["cands_since_prev_cross"]=csp[~F.censored.values]
    allc=ci
    for M in WINDOWS:
        cim=np.array([np.sum((allc>=cc-M)&(allc<cc)) for cc in c])
        comp[f"crosses_in_{M}"]=cim
    comp["substrate"]=name
    # ---- FIREWALL: max index any feature touched == c ; must be < entry ----
    maxidx = c  # every windowed op ends at c; slopes/atr/gap all <= c
    comp["_fw_ok"]=(maxidx < comp.entry_idx.values)
    return comp, dict(crosses=len(ci), fw_fail=int((~comp["_fw_ok"]).sum()))

f8,  s8  = dev.ema8.values,  dev.ema21.values
f50, s50 = dev.ema50.values, dev.ema200.values
c821, m821 = build_substrate("8/21", f8, s8)
c50,  m50  = build_substrate("50/200", f50, s50)
led = pd.concat([c821, c50], ignore_index=True)

# restrict to EXPLORATION years 2016-2019 (leave 2020-2024 sealed for validation)
expl = led[(led.year>=2016)&(led.year<=2019)].copy()
print("=== FIREWALL ===")
print(f"  8/21   crosses={m821['crosses']}  firewall_fail={m821['fw_fail']}")
print(f"  50/200 crosses={m50['crosses']}   firewall_fail={m50['fw_fail']}")
print(f"  -> all features index <= confirm < entry: {(led['_fw_ok'].all())}")

print("\n=== EXPLORATION LEDGER (2016-2019) ===")
for name,g in expl.groupby("substrate"):
    print(f"  {name:>7}: {len(g):5d} completed trades | net/tr {g.net.mean():+.4f} | "
          f"win% {100*g.win.mean():.1f} | median dur {g.dur_days.median():.0f}d")

led.drop(columns=["_fw_ok"]).to_pickle("/home/claude/work/explore_ledger_full.pkl")
expl.drop(columns=["_fw_ok"]).to_pickle("/home/claude/work/explore_ledger_2016_2019.pkl")
expl.drop(columns=["_fw_ok"]).to_csv("/home/claude/work/explore_ledger_2016_2019.csv", index=False)
print(f"\n-> saved explore_ledger_2016_2019.(pkl|csv)  rows={len(expl)}  cols={expl.shape[1]}")
