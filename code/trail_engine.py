"""
TRAIL ENGINE for C5 exit families X3 / X4.

Canonical engine.simulate_trade supports a SCALAR trail. C5 needs a PER-TRADE
trail distance (ATR-scaled, per the pre-registered normalization rule), plus an
"armed" variant matching the declared X3 ("rides baseline until price goes
favorable, THEN trails at best - D").

ORACLE CHAIN (two steps, both exact-equality):
  step 1: slow reference (python loop) with armed=False  ==  canonical
          simulate_trade for {trail}, {stop}, {stop+trail}
  step 2: numba fast kernel  ==  slow reference, on every trade
Only after both pass is the fast kernel used at scale.

Semantics inherited from canonical (frozen assumptions A1-A5):
  A2 adverse-first: stop checked before any arming/updating on that candle
  A3 arming/updating happens at candle CLOSE, effective from the NEXT candle
  A4 gap-through: stop fills at WORSE of (level, open)
  A5 no trigger before opposite cross -> exit at opposite cross entry-candle open
"""
import numpy as np, pandas as pd, sys
from numba import njit
sys.path.insert(0, "/home/claude/repo/code"); sys.path.insert(0, "/home/claude/work")
import engine as EN
import engine_fast as EF

COMM = EN.COMMISSION_RATE_PER_SIDE


def slow_trade(is_bull, ei, xe, o, h, l, c, sp, stop=None, trail=None, armed=False):
    """Reference implementation. Mirrors canonical loop ordering exactly."""
    sp_e = sp[ei]
    entry_exec = o[ei] + sp_e if is_bull else o[ei]
    stop_lv = None
    if stop is not None:
        stop_lv = (entry_exec - stop) if is_bull else (entry_exec + stop)
    trail_ref = None
    k = ei; kind = 0; exit_exec = None; exit_k = xe      # kind 0=cross 1=stop
    while k <= xe:
        sp_k = sp[k]
        if k == xe:
            exit_exec = o[xe] if is_bull else o[xe] + sp_k
            kind = 0; exit_k = xe; break
        if stop_lv is not None:
            if is_bull:
                trig = l[k] <= stop_lv; fill = min(o[k], stop_lv)
            else:
                trig = h[k] + sp_k >= stop_lv; fill = max(o[k] + sp_k, stop_lv)
            if trig:
                exit_exec = fill; kind = 1; exit_k = k; break
        if trail is not None:
            ref = c[k] if is_bull else c[k] + sp_k
            take = True
            if armed:   # X3: only start banking once price is favorable
                take = (ref > entry_exec) if is_bull else (ref < entry_exec)
            if take:
                trail_ref = ref if trail_ref is None else \
                            (max(trail_ref, ref) if is_bull else min(trail_ref, ref))
            if trail_ref is not None:
                cand = (trail_ref - trail) if is_bull else (trail_ref + trail)
                stop_lv = cand if stop_lv is None else \
                          (max(stop_lv, cand) if is_bull else min(stop_lv, cand))
        k += 1
    gross = (exit_exec - entry_exec) if is_bull else (entry_exec - exit_exec)
    return entry_exec, exit_exec, gross, kind, exit_k


@njit(cache=True)
def _kernel(isb, ei, xe, o, h, l, c, sp, stop, trail, use_stop, use_trail, armed):
    n = len(ei)
    entry_x = np.empty(n); exit_x = np.empty(n); gross = np.empty(n)
    kind = np.zeros(n, np.int64); exk = np.empty(n, np.int64)
    for j in range(n):
        b = isb[j]; a = ei[j]; z = xe[j]
        sp_e = sp[a]
        ex = o[a] + sp_e if b else o[a]
        has_stop = use_stop[j]
        stop_lv = (ex - stop[j]) if b else (ex + stop[j])
        has_ref = False; tref = 0.0
        k = a; kd = 0; xx = 0.0; xk = z
        while k <= z:
            sp_k = sp[k]
            if k == z:
                xx = o[z] if b else o[z] + sp_k
                kd = 0; xk = z; break
            if has_stop:
                if b:
                    trig = l[k] <= stop_lv
                    fill = o[k] if o[k] < stop_lv else stop_lv
                else:
                    trig = (h[k] + sp_k) >= stop_lv
                    fill = (o[k] + sp_k) if (o[k] + sp_k) > stop_lv else stop_lv
                if trig:
                    xx = fill; kd = 1; xk = k; break
            if use_trail[j]:
                ref = c[k] if b else c[k] + sp_k
                take = True
                if armed:
                    take = (ref > ex) if b else (ref < ex)
                if take:
                    if not has_ref:
                        tref = ref; has_ref = True
                    else:
                        if b:
                            if ref > tref: tref = ref
                        else:
                            if ref < tref: tref = ref
                if has_ref:
                    cand = (tref - trail[j]) if b else (tref + trail[j])
                    if not has_stop:
                        stop_lv = cand; has_stop = True
                    else:
                        if b:
                            if cand > stop_lv: stop_lv = cand
                        else:
                            if cand < stop_lv: stop_lv = cand
            k += 1
        entry_x[j] = ex; exit_x[j] = xx; kind[j] = kd; exk[j] = xk
        gross[j] = (xx - ex) if b else (ex - xx)
    return entry_x, exit_x, gross, kind, exk


def run_trail(df, F, stop=None, trail=None, armed=False):
    """stop/trail may be scalar, per-trade array, or None."""
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    sp = df._sp; ct, cw = df._ct, df._cw
    ns = df.dt.values.astype("datetime64[ns]").astype(np.int64)
    ei = F.entry_idx.values.astype(np.int64); xe = F.exit_idx.values.astype(np.int64)
    isb = (F.direction.values == "BULL")
    n = len(ei)
    def arr(x):
        if x is None: return np.zeros(n), np.zeros(n, np.bool_)
        v = np.full(n, float(x)) if np.isscalar(x) else np.asarray(x, float)
        return v, np.ones(n, np.bool_)
    sv, su = arr(stop); tv, tu = arr(trail)
    ex_, xx_, gr, kd, xk = _kernel(isb, ei, xe, o, h, l, c, sp, sv, tv, su, tu, armed)
    comm = (ex_ + xx_) * COMM
    nn = EF.swap_nights_vec(ns[ei], ns[xk], ct, cw)
    swp = nn * np.where(isb, EN.SWAP_LONG_PER_NIGHT, EN.SWAP_SHORT_PER_NIGHT)
    return pd.DataFrame({"cross_seq": F.cross_seq.values, "exit_idx_run": xk,
                         "exit_kind": np.where(kd == 1, "stop", "cross"),
                         "gross_exec": gr, "commission": comm, "swap": swp,
                         "net": gr - comm + swp})


if __name__ == "__main__":
    import c4_core as C
    df = pd.read_pickle("/home/claude/work/dev_candles.pkl")
    s = df[(df.dt >= pd.Timestamp("2023-06-01")) & (df.dt < pd.Timestamp("2023-08-01"))
           ].reset_index(drop=True).copy()
    s["datetime_utc"] = s.dt.astype(str)
    vi, vd = C.detect_crosses_vec(s.ema8.values, s.ema21.values)
    Fs = C.build_features_vec(s, vi, vd); Fs = Fs[~Fs.censored].reset_index(drop=True)
    s = EF.attach(s)
    o, h, l, c = s.open.values, s.high.values, s.low.values, s.close.values
    sp = s._sp; ts = pd.DatetimeIndex(s.dt.values)

    print("=== STEP 1: slow reference (armed=False) vs CANONICAL simulate_trade ===")
    for lab, kw_slow, kw_can in [
        ("trail 1.00",         dict(trail=1.00),            dict(trail=1.00)),
        ("trail 0.30",         dict(trail=0.30),            dict(trail=0.30)),
        ("stop 2.00",          dict(stop=2.00),             dict(rule="fixed", stop=2.00)),
        ("stop 2.00+trail 1.0",dict(stop=2.00, trail=1.00), dict(rule="fixed", stop=2.00, trail=1.00)),
    ]:
        bad = 0
        for r in Fs.itertuples():
            b = r.direction == "BULL"
            ex, xx, gr, kd, xk = slow_trade(b, int(r.entry_idx), int(r.exit_idx),
                                            o, h, l, c, sp, armed=False, **kw_slow)
            can = EN.simulate_trade(r.direction, int(r.entry_idx), int(r.exit_idx),
                                    o, h, l, c, ts, **kw_can)
            if (abs(can["gross_exec"] - gr) > 1e-9 or can["exit_idx"] != xk):
                bad += 1
        print(f"  [{lab:22s}] trades={len(Fs)} mismatches={bad} -> {'PASS' if bad==0 else 'FAIL'}")

    print("\n=== STEP 2: numba kernel vs slow reference (incl. armed & per-trade D) ===")
    rng = np.random.default_rng(7)
    for lab, kw in [("trail 1.00 armed=F", dict(trail=1.00, armed=False)),
                    ("trail 1.00 armed=T", dict(trail=1.00, armed=True)),
                    ("stop2+trail1 armed=T", dict(stop=2.00, trail=1.00, armed=True)),
                    ("per-trade random D", dict(trail=rng.uniform(0.2, 3.0, len(Fs)),
                                                stop=rng.uniform(1.0, 6.0, len(Fs)), armed=True))]:
        R = run_trail(s, Fs, **kw)
        sv = kw.get("stop"); tv = kw.get("trail"); am = kw.get("armed", False)
        bad = 0
        for j, r in enumerate(Fs.itertuples()):
            b = r.direction == "BULL"
            st = None if sv is None else (sv if np.isscalar(sv) else sv[j])
            tl = None if tv is None else (tv if np.isscalar(tv) else tv[j])
            ex, xx, gr, kd, xk = slow_trade(b, int(r.entry_idx), int(r.exit_idx),
                                            o, h, l, c, sp, stop=st, trail=tl, armed=am)
            if abs(R.gross_exec.iloc[j] - gr) > 1e-9 or R.exit_idx_run.iloc[j] != xk:
                bad += 1
        print(f"  [{lab:22s}] trades={len(Fs)} mismatches={bad} -> {'PASS' if bad==0 else 'FAIL'}")
