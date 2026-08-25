"""
CYCLE 3 · COARSE-PAIR GATE (reserved, fired on Cycle-3 NO-FREEZE).
Pre-registered 6 Fibonacci-spaced pairs, scored on 2016-2019 naive P/L with a
concentration + year-stability screen. Carry a pair to validation ONLY if it
shows robust (distributed, year-stable) positive net. Result: 0/6 -> closure.
Holdout untouched; 2020-2024 unspent.
"""
import sys; sys.path.insert(0,"/home/claude/repo/code")
import pandas as pd, numpy as np, c4_core as C, engine_fast as EF
dev = pd.read_pickle("/home/claude/work/dev_candles.pkl")
dev = EF.attach(dev); cl=dev.close.values; o=dev.open.values
def ema(close,n):
    k=2/(n+1); out=np.full(len(close),np.nan); out[n-1]=close[:n].mean(); e=out[n-1]
    for i in range(n,len(close)): e=close[i]*k+e*(1-k); out[i]=e
    return out
E={n:ema(cl,n) for n in [5,8,13,20,21,34,50,55,100,200]}; YEARS=[2016,2017,2018,2019]
def eval_pair(fn,sn):
    ci,dr=C.detect_crosses_vec(E[fn],E[sn])
    ent=ci+1; exi=np.clip(np.concatenate([ci[1:],[len(cl)-1]])+1,0,len(cl)-1)
    cens=np.zeros(len(ci),bool); cens[-1]=True
    F=pd.DataFrame({"cross_seq":np.arange(len(ci)),"entry_idx":ent,"exit_idx":exi,
                    "direction":np.asarray(dr),"censored":cens})
    comp=F[~F.censored].copy().reset_index(drop=True)
    comp["net"]=EF.run(dev,comp,"naive").reset_index(drop=True).net.values
    comp["gross0"]=np.where(comp.direction=="BULL",o[comp.exit_idx]-o[comp.entry_idx],
                                                    o[comp.entry_idx]-o[comp.exit_idx])
    comp["year"]=pd.to_datetime(dev.dt.values[comp.entry_idx.values]).year
    d=comp[(comp.year>=2016)&(comp.year<=2019)]; tot=d.net.sum()
    extop5=(tot-d.net.nlargest(5).sum())/max(len(d)-5,1)
    yp=sum(1 for y in YEARS if d[d.year==y].net.mean()>0)
    return dict(n=len(d),gross0=d.gross0.mean(),net=d.net.mean(),extop5=extop5,yrs=yp)
for cls,fn,sn in [("fast",5,20),("fast",8,21),("fast",13,34),("fast",21,55),("slow",50,200),("slow",100,200)]:
    r=eval_pair(fn,sn); rob=(r["net"]>0 and r["extop5"]>0 and r["yrs"]>=3 and r["n"]>=50)
    print(f"{fn}/{sn:<3} {cls} n={r['n']:5d} gross0={r['gross0']:+.3f} net={r['net']:+.3f} "
          f"extop5={r['extop5']:+.3f} yrs+={r['yrs']}/4 ROBUST={'YES' if rob else 'no'}")
