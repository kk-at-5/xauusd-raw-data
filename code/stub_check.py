"""
STUB RULE (rulebook, Raw Data [Added - 02 Aug 2026]) applied to the DEV set.
"strip only zero-range bars that are the final bar of a trading day timestamped
 past the era/weekday-normal close ... discriminator is final-bar + overshoot,
 NOT a fixed +1 min (one stub overshoots +3)"

Era-normal close is DERIVED, never hardcoded. Quarter-modal FAILS across a
DST switch inside a quarter (2023-03-17 sits in summer hours but 2023Q1 is
winter-dominated). Rolling local mode over the nearest same-weekday-class
trading days tracks every boundary era AND both DST switches.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/claude/repo/code"); sys.path.insert(0, "/home/claude/work")
import c4_core as C

W = "/home/claude/work/"
WIN = 15   # nearest same-class days each side; FIXED CONSTANT, structural only

def find_stubs(df, verbose=True):
    day_id, starts, ends, close_date, gi, gs, is_day = C.day_blocks(df)
    L = pd.DataFrame({"idx": ends})
    dt = pd.DatetimeIndex(df.dt.values[ends])
    L["tod"] = dt.hour * 60 + dt.minute
    L["dow"] = dt.dayofweek
    L["wclass"] = np.where(L.dow == 4, "FRI", "MON_THU")
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    L["zero"] = (h[ends] == l[ends]) & (h[ends] == o[ends]) & (h[ends] == c[ends])
    L["ts"] = dt

    # rolling local modal tod within weekday class
    # normal = modal tod among same-weekday-class days IN THE SAME US-DST
    # REGIME (market close follows New York). US DST is a documented FACT in
    # the rulebook ("BST values hold until 01 Nov 2026 (US DST end)"), not a
    # hardcoded session boundary. Without regime-splitting, a window straddling
    # a switch mislabels either the +3 stub (2023-03-17) or a genuine winter
    # close (2023-12-08).
    def us_dst(t):
        y = t.year
        mar = pd.Timestamp(year=y, month=3, day=1)
        start = mar + pd.Timedelta(days=(6 - mar.dayofweek) % 7 + 7)   # 2nd Sun Mar
        nov = pd.Timestamp(year=y, month=11, day=1)
        end = nov + pd.Timedelta(days=(6 - nov.dayofweek) % 7)         # 1st Sun Nov
        return start <= t.normalize() < end
    L["dst"] = [us_dst(t) for t in L.ts]
    norm = np.empty(len(L))
    for wc in ["FRI", "MON_THU"]:
        for rg in [True, False]:
            m = ((L.wclass == wc) & (L.dst == rg)).values
            if not m.any(): continue
            sub = L.tod.values[m]
            nm = np.empty(len(sub))
            for j in range(len(sub)):
                a, b = max(0, j - WIN), min(len(sub), j + WIN + 1)
                v, ct = np.unique(sub[a:b], return_counts=True)
                nm[j] = v[ct.argmax()]
            norm[m] = nm
    L["norm"] = norm
    L["stub"] = L.zero & (L.tod > L.norm)
    S = L[L.stub]
    if verbose:
        print(f"=== STUB RULE (dev set) ===")
        print(f"  trading days={len(L)}  zero-range final bars={int(L.zero.sum())}"
              f"  STUBS={len(S)}")
        for _, r in S.iterrows():
            print(f"    {r.ts}  {r.wclass:8s} tod={int(r.tod)} "
                  f"era-normal={int(r.norm)}  overshoot=+{int(r.tod-r.norm)}m")
    return L, S

if __name__ == "__main__":
    df = pd.read_pickle(W + "dev_candles.pkl")
    L, S = find_stubs(df)
    exp = ["2023-03-17","2024-04-30","2024-05-01","2024-05-29","2024-06-04",
           "2024-06-10","2024-07-11","2024-07-31","2024-08-02","2024-08-16",
           "2024-09-05","2024-09-17"]
    got = sorted(S.ts.dt.strftime("%Y-%m-%d").tolist())
    print(f"\n  reconciles with independent full-set scan (12 dev stubs)? {got == exp}")
    if got != exp:
        print("   expected:", exp); print("   got     :", got)

    # ---- rebuild WITHOUT the stubs, recompute EMAs, recount ----
    keep = np.ones(len(df), bool); keep[S.idx.values] = False
    d2 = df[keep].reset_index(drop=True).copy()
    from load_dev import emas
    e8, e21 = emas(d2.close.values)
    d2["ema8"], d2["ema21"] = e8, e21
    vi1, vd1 = C.detect_crosses_vec(df.ema8.values, df.ema21.values)
    vi2, vd2 = C.detect_crosses_vec(d2.ema8.values, d2.ema21.values)
    print(f"\n=== EFFECT OF STRIPPING {len(S)} STUB BARS ===")
    print(f"  candles  : {len(df):,} -> {len(d2):,}   (-{len(df)-len(d2)})")
    print(f"  crosses  : {len(vi1):,} -> {len(vi2):,}  (delta {len(vi2)-len(vi1):+d})")
    d2.to_pickle(W + "dev_candles_stubstripped.pkl")
    print(f"-> dev_candles_stubstripped.pkl")
