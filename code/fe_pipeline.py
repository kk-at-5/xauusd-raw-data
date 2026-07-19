"""
XAUUSD EMA 8/21 — Phase 2 Feature Engineering Pipeline (CONSOLIDATED)
====================================================================
One CC-runnable file: LOAD -> SANITY -> LAYER A (crosses) -> LAYER B (path)
-> LAYER C (threshold-free features + labels). Writes a single per-cross
feature table to the output dir.

CORE MODEL — one continuous STOP-AND-REVERSE chain:
  * crosses strictly alternate BULL/BEAR (a sign-flip cannot repeat its sign)
  * each cross k EXITS trade k-1 and ENTERS trade k at the SAME price
    (open of the candle after the cross confirms)
  * daily gaps AND weekend gaps are NOT exits — a position is carried across
    them until the TRUE opposite cross prints (this session or a later one)
  * "flat at day-end / before weekend" is a Layer-D STRATEGY overlay, tested
    later, never a measurement default (File 0 Rule 5)

TWO true boundary exceptions (dataset-level, NOT day-level):
  * FIRST cross of the whole dataset  -> entry-only (closes no prior trade)
  * LAST open trade of the whole dataset -> right-censored (opposite cross
    has not printed yet)

Firewall: FEATURES known at confirmation candle i (pre-entry); LABELS use the
post-entry path. MAE/MFE/exit are labels. No strategy, no costs here.
Portable pandas, file-in/file-out. Environment parity (sandbox == CC box).
"""
import pandas as pd, numpy as np, glob, os

# ---- CONFIG (CC: edit these two paths for your box) -----------------
RAW_DIR = "/mnt/user-data/uploads"
OUT_DIR = "/home/claude"
SPLIT_DATE = None        # e.g. "2026-07-20" — seals test set. None during dev.
N_MOM      = 10          # momentum context window (placeholder, NOT frozen)
STEP       = 60          # seconds between 1-min candles
DAILY_GAP_MAX_S = 6*3600 # gaps longer than this are weekend/holiday (not daily)
K8, K21    = 2/9, 2/22
RAW_SCHEMA = ["timestamp_utc","datetime_utc","open","high","low","close","ema8","ema21"]

SESSIONS = [("Asian",22,7),("London",7,13),("NY_Overlap",13,16),("NY",16,22)]
def session_of(hour):
    for name,a,b in SESSIONS:
        if a<b and a<=hour<b: return name
        if a>b and (hour>=a or hour<b): return name
    return "Other"

# ---- LOAD & STITCH --------------------------------------------------
def load_all(folder):
    files=sorted(glob.glob(os.path.join(folder,"*.csv")))
    dfs=[]
    for f in files:
        d=pd.read_csv(f); assert list(d.columns)==RAW_SCHEMA, f"schema {os.path.basename(f)}"
        dfs.append(d)
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("timestamp_utc")
    return df.sort_values("timestamp_utc").reset_index(drop=True)

# ---- GAP MAP --------------------------------------------------------
def gap_map(df):
    """Index i (i>=1) -> gap seconds BEFORE candle i, if >STEP. Classify."""
    step=df["timestamp_utc"].diff()
    gaps={}
    for i in df.index[1:]:
        s=int(step.iloc[i])
        if s!=STEP:
            gaps[i]=("weekend" if s>DAILY_GAP_MAX_S else "daily", s)
    return gaps

# ---- SANITY ---------------------------------------------------------
def sanity(df,gaps):
    print("=== GLOBAL SANITY (data validation) ===")
    print(f"[1] rows: {len(df)}   dupes: {df.timestamp_utc.duplicated().sum()}   "
          f"increasing: {df.timestamp_utc.is_monotonic_increasing}")
    bad=df[~((df.low<=df.open)&(df.low<=df.close)&(df.high>=df.open)&
             (df.high>=df.close)&(df.low<=df.high))]
    print(f"[2] OHLC-inconsistent rows: {len(bad)}")
    dcount=sum(1 for k in gaps.values() if k[0]=="daily")
    wcount=sum(1 for k in gaps.values() if k[0]=="weekend")
    print(f"[3] gaps: {len(gaps)}  (daily={dcount}, weekend/holiday={wcount})")
    for i,(kind,s) in gaps.items():
        print(f"      {kind:8s} {s:6d}s ({s//60:4d}m) "
              f"{df.datetime_utc.iloc[i-1]} -> {df.datetime_utc.iloc[i]}")

# ---- LAYER A: CROSS DETECTION (continuous series) -------------------
def detect_crosses(df):
    e8,e21=df.ema8.values,df.ema21.values; n=len(df); rows=[]
    for i in range(1,n):
        bull=(e8[i-1]<=e21[i-1]) and (e8[i]>e21[i])
        bear=(e8[i-1]>=e21[i-1]) and (e8[i]<e21[i])
        if not(bull or bear): continue
        rows.append({"confirm_idx":i,"direction":"BULL" if bull else "BEAR"})
    c=pd.DataFrame(rows).reset_index(drop=True)
    c["cross_seq"]=range(1,len(c)+1)
    return c

# ---- LAYER B + C: stop-and-reverse chain over continuous path -------
def build_features(df,crosses,gaps):
    o,h,l,cl=df.open.values,df.high.values,df.low.values,df.close.values
    e8,e21=df.ema8.values,df.ema21.values
    dt=df.datetime_utc.values; n=len(df)
    last_idx=n-1
    def gaps_in(a,b):   # count/classify gaps with index in (a, b]
        gg=[gaps[i] for i in gaps if a<i<=b]
        return len(gg), any(k=="weekend" for k,_ in gg)
    out=[]
    C=crosses.reset_index(drop=True)
    for k in range(len(C)):
        i=int(C.confirm_idx.iloc[k]); direction=C.direction.iloc[k]
        ei=i+1
        entry_available = ei<=last_idx
        entry = o[ei] if entry_available else np.nan
        # confirmation-candle FEATURES (pre-entry)
        body=abs(cl[i]-o[i]); wick=(h[i]-max(o[i],cl[i]))+(min(o[i],cl[i])-l[i])
        gap_confirm=abs(e8[i]-e21[i])
        nu=min(N_MOM,i); pr=slice(i-nu,i)
        avg_body=np.mean(np.abs(cl[pr]-o[pr])) if nu>0 else np.nan
        mom=(body/avg_body) if (avg_body and avg_body>0) else np.nan
        # exit = next cross (opposite) -> stop-and-reverse
        is_last=(k==len(C)-1)
        if not is_last and entry_available:
            jc=int(C.confirm_idx.iloc[k+1]); assert C.direction.iloc[k+1]!=direction
            xi=jc+1; exit_price=o[xi] if xi<=last_idx else np.nan
            win_end=jc; censored=False
        else:
            jc=None; xi=None; exit_price=np.nan; win_end=last_idx; censored=True
        # path window [ei .. win_end] inclusive
        if entry_available:
            hi=h[ei:win_end+1].max(); lo=l[ei:win_end+1].min()
            clo=cl[ei:win_end+1]
            if direction=="BULL":
                mfe=hi-entry; mae=entry-lo
                mfe_i=ei+int(np.argmax(h[ei:win_end+1])); mae_i=ei+int(np.argmin(l[ei:win_end+1]))
                pnl=(exit_price-entry) if not censored else np.nan
            else:
                mfe=entry-lo; mae=hi-entry
                mfe_i=ei+int(np.argmin(l[ei:win_end+1])); mae_i=ei+int(np.argmax(h[ei:win_end+1]))
                pnl=(entry-exit_price) if not censored else np.nan
            gspan,wspan=gaps_in(ei,win_end)
            dur=win_end-ei+1
            rec={
              "cross_seq":int(C.cross_seq.iloc[k]),"direction":direction,
              "confirm_time_utc":str(dt[i]),"session":session_of(pd.Timestamp(dt[i]).hour),
              "entry_time_utc":str(dt[ei]),"entry_price":round(float(entry),2),
              # FEATURES
              "gap_at_confirm_$":round(float(gap_confirm),4),
              "body_$":round(float(body),2),"body_dominant":bool(body>wick),
              "momentum_ratio":round(float(mom),3) if pd.notna(mom) else np.nan,
              "mom_n_used":nu,
              # LABELS
              "exit_time_utc":str(dt[xi]) if (not censored and xi<=last_idx) else None,
              "exit_price":round(float(exit_price),2) if not censored else np.nan,
              "mfe_$":round(float(mfe),2),"mae_$":round(float(mae),2),
              "emacross_pnl_$":round(float(pnl),2) if not censored else np.nan,
              "duration_candles":int(dur),
              "candles_to_mfe":int(mfe_i-ei),"candles_to_mae":int(mae_i-ei),
              "fav_first":bool((mfe_i-ei)<(mae_i-ei)),
              # CHAIN / BOUNDARY FLAGS
              "gaps_spanned":int(gspan),"weekend_spanned":bool(wspan),
              "gap_unobserved":bool(gspan>0),
              "entry_only":bool(k==0),"censored":bool(censored),
              "zero_path":bool(dur<=1),
            }
            out.append(rec)
    return pd.DataFrame(out)

# ---- SELF-AUDIT -----------------------------------------------------
def audit(F,crosses):
    print("\n=== SELF-AUDIT (invariants) ===")
    print(f"crosses: {len(crosses)}   trades: {len(F)}")
    print(f"completed: {(~F.censored).sum()}   censored: {F.censored.sum()} (expect 1)")
    print(f"entry_only flags: {F.entry_only.sum()} (expect 1)")
    d=F.direction.values
    print(f"strict alternation: {all(d[i]!=d[i+1] for i in range(len(d)-1))}")
    print(f"MFE<0: {(F['mfe_$']<0).sum()}   MAE<0: {(F['mae_$']<0).sum()} (both expect 0)")
    # STOP-AND-REVERSE link: exit[k] must equal entry[k+1] (same candle open)
    comp=F[~F.censored].reset_index(drop=True)
    link_ok=True
    Fi=F.reset_index(drop=True)
    for k in range(len(Fi)-1):
        if not Fi.censored.iloc[k]:
            ex=Fi.exit_price.iloc[k]; en=Fi.entry_price.iloc[k+1]
            if pd.notna(ex) and abs(ex-en)>1e-9: link_ok=False
    print(f"stop-and-reverse link (exit[k]==entry[k+1]): {link_ok}")
    print(f"trades spanning >=1 gap: {(F.gaps_spanned>0).sum()}  "
          f"weekend-spanning: {F.weekend_spanned.sum()}")

# ---- C3 TRACK-1 FEATURES (frozen per C3_TRACK1_FEATURE_DEFINITIONS.md, 19 Jul 2026) ----
# D1 gap-aware TR: at the first post-gap candle TR = high-low (prev-close terms dropped)
# D2 atr windows: PRIOR N candles EXCLUDING confirmation candle; strict NaN at burn-in
# D4 shock_state cut-points FITTED ON THE 13-DAY EXPLORATION POOL and frozen:
SHOCK_T1, SHOCK_T2 = 0.8706, 1.0236   # vol_ratio terciles, logged 19 Jul 2026

def build_c3_track1(df, crosses, F, refit_shock=False):
    h,l,c,o = df.high.values, df.low.values, df.close.values, df.open.values
    gaps = gap_map(df); n=len(df)
    gap_before = np.zeros(n, bool)
    for i in gaps: gap_before[i]=True
    tr = np.empty(n); tr[0]=h[0]-l[0]
    for j in range(1,n):
        tr[j] = h[j]-l[j] if gap_before[j] else \
                max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1]))
    trs = pd.Series(tr)
    atr14 = trs.shift(1).rolling(14, min_periods=14).mean().values
    atr60 = trs.shift(1).rolling(60, min_periods=60).mean().values
    ci = crosses.confirm_idx.values
    body = np.abs(c[ci]-o[ci])
    gapc = np.abs(df.ema8.values[ci]-df.ema21.values[ci])
    cs = pd.Series(c)
    T = pd.DataFrame({
        "cross_seq": F.cross_seq,
        "mom_vol":         body/atr14[ci],
        "vol_ratio":       atr14[ci]/atr60[ci],
        "displacement_15": np.abs(c[ci]-cs.shift(15).values[ci])/atr60[ci],
        "displacement_60": np.abs(c[ci]-cs.shift(60).values[ci])/atr60[ci],
        "gap_norm":        gapc/atr14[ci]})
    t1,t2 = ((np.nanpercentile(T.vol_ratio,[100/3,200/3])) if refit_shock
             else (SHOCK_T1, SHOCK_T2))
    T["shock_state"] = pd.cut(T.vol_ratio, [-np.inf,t1,t2,np.inf],
                              labels=["LOW","MID","HIGH"])
    print(f"C3 Track-1: shock cut-points {'REFIT' if refit_shock else 'frozen'} "
          f"t1={t1:.4f} t2={t2:.4f}   NaN rows: {int(T.mom_vol.isna().sum())}/"
          f"{int(T.displacement_60.isna().sum())} (atr14/atr60 burn-in)")
    return T.merge(F, on="cross_seq")

def main(raw_dir=RAW_DIR,out_dir=OUT_DIR):
    df=load_all(raw_dir); gaps=gap_map(df)
    sanity(df,gaps)
    crosses=detect_crosses(df)
    print(f"\n=== LAYER A === crosses: {len(crosses)}  "
          f"BULL={(crosses.direction=='BULL').sum()} BEAR={(crosses.direction=='BEAR').sum()}")
    F=build_features(df,crosses,gaps)
    audit(F,crosses)
    if SPLIT_DATE:
        F["split"]=np.where(pd.to_datetime(F.confirm_time_utc)<pd.Timestamp(SPLIT_DATE),"train","test")
        print(f"\nsplit_date={SPLIT_DATE}: train={sum(F.split=='train')} test={sum(F.split=='test')}")
    else:
        F["split"]="unassigned"
    outp=os.path.join(out_dir,"tranche1_features.csv"); F.to_csv(outp,index=False)
    print(f"\n-> {outp}   ({len(F)} rows, {len(F.columns)} cols)")
    T=build_c3_track1(df,crosses,F)
    outt=os.path.join(out_dir,"c3_track1_features.csv"); T.to_csv(outt,index=False)
    print(f"-> {outt}   ({len(T)} rows, {len(T.columns)} cols)")
    return df,crosses,F

if __name__=="__main__":
    main()
