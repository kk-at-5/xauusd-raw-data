"""
dx_engine.py — C7 DYNAMIC-EXIT ENGINE (NEW; not previously in the repo)
======================================================================
trail_engine.py expresses a FIXED-DISTANCE trail (best - D). C7's DX1 is a
PERCENTAGE GIVE-BACK and DX2 is an EARLY-PROMISE-SCALED adverse stop; both
read the trade's OWN developing state, so both need a different kernel.

DECLARED SEMANTICS (pre-registered; no lookahead — every input at candle k
uses only information available up to and including candle k, and every
level takes effect from candle k+1 per A3).

  mfe_so_far(k)  long : max over [ei..k] of ( h[j] - entry_exec )
                 short: max over [ei..k] of ( entry_exec - (l[j] + sp[j]) )
                 (intra-candle EXTREME — the rulebook's excursion definition;
                  ask-side corrected for shorts)

  DX1 GIVE-BACK (fitted f):
      exit when unrealised(k) <= f * mfe_so_far(k)
      implemented as a ratcheting level  L = entry_exec +/- f*mfe_ref
      ARMED ONLY once mfe_ref > 0 (a trade that never went favorable has
      nothing to give back; it rides to the naive exit). Level updated at
      candle CLOSE, effective NEXT candle (A3). Triggers intra-candle,
      adverse-first (A2), fills at WORSE of (level, open) (A4).

  DX2 ADVERSE-DEVELOPMENT (fitted m, FIXED CONSTANT W):
      early_fav = mfe_so_far(ei + W - 1)          # first W candles
      from candle ei+W onward: exit when mae_so_far(k) >= m * early_fav
      implemented as a per-trade stop at distance D = m*early_fav from
      entry_exec, armed at close of candle ei+W-1, effective from ei+W.
      If early_fav <= 0 the rule NEVER ARMS (no promise -> no turn).

  Inherited unchanged from canonical engine.py: A1 (entry candle evaluated
  with full range), A2 (adverse-first), A3 (arm/update at close, effective
  next candle), A4 (gap-through fills at the worse of level/open for stops),
  A5 (no trigger -> exit at opposite-cross entry-candle open), A6 (trade list
  fixed by the cross list), A7 (slippage = 0). Time-stop retains the C4
  Case-8 ordering: time exit at OPEN is checked BEFORE any intra-candle path.

ORACLE CHAIN (both steps exact-equality, run in __main__):
  step 1: slow reference == canonical engine.simulate_trade on the shared
          paths {naive, fixed stop, fixed trail, stop+trail, time stop}
  step 2: numba kernel == slow reference, trade-by-trade, including DX1/DX2
          and per-trade random parameter vectors
  step 3: hand-computed synthetic cases for DX1 and DX2 (File 0 Rule 1)
"""
import numpy as np, pandas as pd, sys
from numba import njit
sys.path.insert(0, "/home/claude/repo/code"); sys.path.insert(0, "/home/claude/work")
import engine as EN
import engine_fast as EF

COMM = EN.COMMISSION_RATE_PER_SIDE
KIND_CROSS, KIND_STOP, KIND_TIME = 0, 1, 2


# ----------------------------------------------------------------- slow ref
def slow_trade(is_bull, ei, xe, o, h, l, c, sp,
               static_stop=None, trail=None, armed=False,
               dx1_f=None, dx2_m=None, dx2_W=5, tmax=None):
    """Reference implementation. Mirrors the canonical loop ordering exactly."""
    entry_exec = o[ei] + sp[ei] if is_bull else o[ei]

    stop_lv = None
    has_stop = False
    if static_stop is not None:
        stop_lv = (entry_exec - static_stop) if is_bull else (entry_exec + static_stop)
        has_stop = True

    trail_ref = None
    mfe_ref = None                      # running best favorable excursion ($)
    k = ei
    kind, exit_exec, exit_k = KIND_CROSS, None, xe
    while k <= xe:
        sp_k = sp[k]
        if k == xe:                                     # A5 naive exit
            exit_exec = o[xe] if is_bull else o[xe] + sp_k
            kind, exit_k = KIND_CROSS, xe
            break
        if tmax is not None and k - ei >= tmax:         # Case-8 ordering
            exit_exec = o[k] if is_bull else o[k] + sp_k
            kind, exit_k = KIND_TIME, k
            break
        if has_stop:                                    # A2 adverse-first
            if is_bull:
                trig = l[k] <= stop_lv
                fill = min(o[k], stop_lv)               # A4 worse
            else:
                trig = h[k] + sp_k >= stop_lv
                fill = max(o[k] + sp_k, stop_lv)        # A4 worse
            if trig:
                exit_exec, kind, exit_k = fill, KIND_STOP, k
                break
        # ---- A3: all updates at candle CLOSE, effective from candle k+1 ----
        fav = (h[k] - entry_exec) if is_bull else (entry_exec - (l[k] + sp_k))
        mfe_ref = fav if mfe_ref is None else max(mfe_ref, fav)

        if dx1_f is not None and mfe_ref > 0.0:
            cand = (entry_exec + dx1_f * mfe_ref) if is_bull else \
                   (entry_exec - dx1_f * mfe_ref)
            if not has_stop:
                stop_lv, has_stop = cand, True
            else:
                stop_lv = max(stop_lv, cand) if is_bull else min(stop_lv, cand)

        if dx2_m is not None and (k - ei) == dx2_W - 1 and mfe_ref > 0.0:
            D = dx2_m * mfe_ref
            cand = (entry_exec - D) if is_bull else (entry_exec + D)
            if not has_stop:
                stop_lv, has_stop = cand, True
            else:
                stop_lv = max(stop_lv, cand) if is_bull else min(stop_lv, cand)

        if trail is not None:
            ref = c[k] if is_bull else c[k] + sp_k
            take = True
            if armed:
                take = (ref > entry_exec) if is_bull else (ref < entry_exec)
            if take:
                trail_ref = ref if trail_ref is None else \
                            (max(trail_ref, ref) if is_bull else min(trail_ref, ref))
            if trail_ref is not None:
                cand = (trail_ref - trail) if is_bull else (trail_ref + trail)
                if not has_stop:
                    stop_lv, has_stop = cand, True
                else:
                    stop_lv = max(stop_lv, cand) if is_bull else min(stop_lv, cand)
        k += 1

    gross = (exit_exec - entry_exec) if is_bull else (entry_exec - exit_exec)
    return entry_exec, exit_exec, gross, kind, exit_k


# --------------------------------------------------------------- fast kernel
@njit(cache=True)
def _kernel(isb, ei, xe, o, h, l, c, sp,
            sstop, use_sstop, trail, use_trail, armed,
            f1, use_f1, m2, use_m2, W2, tmax, use_tmax):
    n = len(ei)
    entry_x = np.empty(n); exit_x = np.empty(n); gross = np.empty(n)
    kind = np.zeros(n, np.int64); exk = np.empty(n, np.int64)
    for j in range(n):
        b = isb[j]; a = ei[j]; z = xe[j]
        ex = o[a] + sp[a] if b else o[a]
        has_stop = use_sstop[j]
        if b:
            stop_lv = ex - sstop[j]
        else:
            stop_lv = ex + sstop[j]
        has_tref = False; tref = 0.0
        has_mfe = False; mfe = 0.0
        k = a; kd = KIND_CROSS; xx = 0.0; xk = z
        while k <= z:
            sp_k = sp[k]
            if k == z:
                xx = o[z] if b else o[z] + sp_k
                kd = KIND_CROSS; xk = z; break
            if use_tmax[j] and (k - a) >= tmax[j]:
                xx = o[k] if b else o[k] + sp_k
                kd = KIND_TIME; xk = k; break
            if has_stop:
                if b:
                    trig = l[k] <= stop_lv
                    fill = o[k] if o[k] < stop_lv else stop_lv
                else:
                    ao = o[k] + sp_k
                    trig = (h[k] + sp_k) >= stop_lv
                    fill = ao if ao > stop_lv else stop_lv
                if trig:
                    xx = fill; kd = KIND_STOP; xk = k; break
            if b:
                fav = h[k] - ex
            else:
                fav = ex - (l[k] + sp_k)
            if not has_mfe:
                mfe = fav; has_mfe = True
            elif fav > mfe:
                mfe = fav

            if use_f1[j] and mfe > 0.0:
                if b:
                    cand = ex + f1[j] * mfe
                else:
                    cand = ex - f1[j] * mfe
                if not has_stop:
                    stop_lv = cand; has_stop = True
                else:
                    if b:
                        if cand > stop_lv: stop_lv = cand
                    else:
                        if cand < stop_lv: stop_lv = cand

            if use_m2[j] and (k - a) == (W2 - 1) and mfe > 0.0:
                D = m2[j] * mfe
                if b:
                    cand = ex - D
                else:
                    cand = ex + D
                if not has_stop:
                    stop_lv = cand; has_stop = True
                else:
                    if b:
                        if cand > stop_lv: stop_lv = cand
                    else:
                        if cand < stop_lv: stop_lv = cand

            if use_trail[j]:
                ref = c[k] if b else c[k] + sp_k
                take = True
                if armed:
                    take = (ref > ex) if b else (ref < ex)
                if take:
                    if not has_tref:
                        tref = ref; has_tref = True
                    else:
                        if b:
                            if ref > tref: tref = ref
                        else:
                            if ref < tref: tref = ref
                if has_tref:
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


def run_dx(df, F, static_stop=None, trail=None, armed=False,
           dx1_f=None, dx2_m=None, dx2_W=5, tmax=None):
    """Scalars, per-trade arrays, or None for every rule parameter."""
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    sp = df._sp; ct, cw = df._ct, df._cw
    ns = df.dt.values.astype("datetime64[ns]").astype(np.int64)
    ei = F.entry_idx.values.astype(np.int64)
    xe = F.exit_idx.values.astype(np.int64)
    isb = (F.direction.values == "BULL")
    n = len(ei)

    def arr(x, dtype=float):
        if x is None:
            return np.zeros(n, dtype), np.zeros(n, np.bool_)
        v = np.full(n, x, dtype) if np.isscalar(x) else np.asarray(x, dtype)
        return v, np.ones(n, np.bool_)

    sv, su = arr(static_stop); tv, tu = arr(trail)
    fv, fu = arr(dx1_f);       mv, mu = arr(dx2_m)
    kv, ku = arr(tmax, np.int64)

    ex_, xx_, gr, kd, xk = _kernel(isb, ei, xe, o, h, l, c, sp,
                                   sv, su, tv, tu, armed,
                                   fv, fu, mv, mu, int(dx2_W), kv, ku)
    comm = (ex_ + xx_) * COMM
    nn = EF.swap_nights_vec(ns[ei], ns[xk], ct, cw)
    swp = nn * np.where(isb, EN.SWAP_LONG_PER_NIGHT, EN.SWAP_SHORT_PER_NIGHT)
    kindstr = np.where(kd == KIND_STOP, "stop", np.where(kd == KIND_TIME, "time", "cross"))
    return pd.DataFrame({"cross_seq": F.cross_seq.values,
                         "entry_idx": ei, "exit_idx_run": xk,
                         "exit_kind": kindstr,
                         "held": xk - ei,
                         "entry_exec": ex_, "exit_exec": xx_,
                         "gross_exec": gr, "commission": comm, "swap": swp,
                         "net": gr - comm + swp})


# --------------------------------------------------------------- oracle chain
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
    ALL = True

    print("=== STEP 1: slow reference vs CANONICAL engine.simulate_trade ===")
    for lab, kw_slow, kw_can in [
        ("naive (A5)",           dict(),                            dict()),
        ("fixed stop 2.00",      dict(static_stop=2.00),            dict(rule="fixed", stop=2.00)),
        ("fixed stop 0.50",      dict(static_stop=0.50),            dict(rule="fixed", stop=0.50)),
        ("trail 1.00",           dict(trail=1.00),                  dict(trail=1.00)),
        ("stop 2.00 + trail 1.0",dict(static_stop=2.00, trail=1.00),dict(rule="fixed", stop=2.00, trail=1.00)),
        ("time stop 30",         dict(tmax=30),                     dict(tmax=30)),
        ("time 10 + stop 1.00",  dict(tmax=10, static_stop=1.00),   dict(rule="fixed", stop=1.00, tmax=10)),
    ]:
        bad = 0
        for r in Fs.itertuples():
            b = r.direction == "BULL"
            ex, xx, gr, kd, xk = slow_trade(b, int(r.entry_idx), int(r.exit_idx),
                                            o, h, l, c, sp, **kw_slow)
            can = EN.simulate_trade(r.direction, int(r.entry_idx), int(r.exit_idx),
                                    o, h, l, c, ts, **kw_can)
            if abs(can["gross_exec"] - gr) > 1e-9 or can["exit_idx"] != xk:
                bad += 1
        ALL &= (bad == 0)
        print(f"  [{lab:22s}] trades={len(Fs)} mismatches={bad} -> {'PASS' if bad==0 else 'FAIL'}")

    print("\n=== STEP 2: numba kernel vs slow reference (DX1 / DX2 included) ===")
    rng = np.random.default_rng(20260804)
    nF = len(Fs)
    cases = [
        ("naive",                dict()),
        ("fixed stop 2.00",      dict(static_stop=2.00)),
        ("trail 1.00 armed=T",   dict(trail=1.00, armed=True)),
        ("time stop 4",          dict(tmax=4)),
        ("DX1 f=0.00",           dict(dx1_f=0.00)),
        ("DX1 f=0.30",           dict(dx1_f=0.30)),
        ("DX1 f=0.90",           dict(dx1_f=0.90)),
        ("DX2 m=1.0 W=5",        dict(dx2_m=1.0, dx2_W=5)),
        ("DX2 m=3.0 W=5",        dict(dx2_m=3.0, dx2_W=5)),
        ("DX1 per-trade random", dict(dx1_f=rng.uniform(0.0, 0.95, nF))),
        ("DX2 per-trade random", dict(dx2_m=rng.uniform(0.3, 6.0, nF), dx2_W=5)),
        ("DX1+DX2+stop mixed",   dict(dx1_f=rng.uniform(0.0, 0.9, nF),
                                      dx2_m=rng.uniform(0.5, 4.0, nF),
                                      static_stop=rng.uniform(1.0, 8.0, nF), dx2_W=5)),
    ]
    for lab, kw in cases:
        R = run_dx(s, Fs, **kw)
        bad = 0
        for j, r in enumerate(Fs.itertuples()):
            b = r.direction == "BULL"
            kws = {}
            for key in ("static_stop", "trail", "dx1_f", "dx2_m", "tmax"):
                v = kw.get(key)
                if v is None: continue
                kws[key] = v if np.isscalar(v) else v[j]
            kws["armed"] = kw.get("armed", False)
            kws["dx2_W"] = kw.get("dx2_W", 5)
            ex, xx, gr, kd, xk = slow_trade(b, int(r.entry_idx), int(r.exit_idx),
                                            o, h, l, c, sp, **kws)
            if (abs(R.gross_exec.iloc[j] - gr) > 1e-12 or R.exit_idx_run.iloc[j] != xk
                    or abs(R.entry_exec.iloc[j] - ex) > 1e-12):
                bad += 1
        ALL &= (bad == 0)
        print(f"  [{lab:22s}] trades={nF} mismatches={bad} -> {'PASS' if bad==0 else 'FAIL'}")

    print("\n=== STEP 3: HAND-COMPUTED SYNTHETIC CASES (File 0 Rule 1) ===")
    tsx = pd.DatetimeIndex(pd.date_range("2026-07-15 08:00", periods=10, freq="min"))
    spx = np.full(10, 0.08)                      # London band, constant

    def chk(name, got, want, tol=1e-9):
        global ALL
        good = abs(got - want) <= tol
        ALL &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: got {got:.4f} want {want:.4f}")

    # D1 LONG DX1 f=0.50.  entry_exec = 100 + 0.08 = 100.08
    #   k0: h=100.30 -> mfe = 0.22 -> level = 100.08 + .5*.22 = 100.19  (from k1)
    #   k1: l=100.25 > 100.19 no trigger; h=100.88 -> mfe = 0.80
    #        -> level = 100.08 + .40 = 100.48 (from k2)
    #   k2: l=100.20 <= 100.48 -> STOP, fill = min(open 100.80, 100.48) = 100.48
    o1 = np.array([100.00,100.28,100.80,100.0,100.0,100.0,100.0,100.0,100.,100.])
    h1 = np.array([100.30,100.88,100.85,100.0,100.0,100.0,100.0,100.0,100.,100.])
    l1 = np.array([ 99.95,100.25,100.20,100.0,100.0,100.0,100.0,100.0,100.,100.])
    c1 = np.array([100.28,100.80,100.30,100.0,100.0,100.0,100.0,100.0,100.,100.])
    r = slow_trade(True, 0, 6, o1, h1, l1, c1, spx, dx1_f=0.50)
    chk("D1 long DX1 f=.5 fill", r[1], 100.48)
    chk("D1 long DX1 f=.5 gross", r[2], 100.48 - 100.08)
    chk("D1 exit kind == stop", float(r[3]), float(KIND_STOP))
    chk("D1 exit idx", float(r[4]), 2.0)

    # D2 same path, f=0.90 -> level after k1 = 100.08+.9*.80 = 100.80
    #   but level after k0 = 100.08+.9*.22 = 100.278; k1 low 100.25 <= 100.278
    #   -> triggers at k1, fill = min(open 100.28, 100.278) = 100.278
    r = slow_trade(True, 0, 6, o1, h1, l1, c1, spx, dx1_f=0.90)
    chk("D2 long DX1 f=.9 fill", r[1], 100.278)
    chk("D2 exit idx", float(r[4]), 1.0)

    # D3 no-arm case: trade never goes favorable (all highs <= entry_exec)
    #   -> DX1 never arms -> naive exit at o[3] = 99.50
    o3 = np.array([100.00,99.90,99.70,99.50,99.0,99.0,99.0,99.0,99.,99.])
    h3 = np.array([100.02,99.95,99.75,99.55,99.0,99.0,99.0,99.0,99.,99.])
    l3 = np.array([ 99.80,99.65,99.45,99.40,99.0,99.0,99.0,99.0,99.,99.])
    c3 = np.array([ 99.90,99.70,99.50,99.45,99.0,99.0,99.0,99.0,99.,99.])
    r = slow_trade(True, 0, 3, o3, h3, l3, c3, spx, dx1_f=0.50)
    chk("D3 never-favorable -> naive exit", r[1], 99.50)
    chk("D3 exit kind == cross", float(r[3]), float(KIND_CROSS))

    # D4 A4 gap-through: level 100.48 from k1 close, k2 OPENS at 100.10
    o4 = o1.copy(); o4[2] = 100.10
    l4 = l1.copy(); l4[2] = 100.05
    r = slow_trade(True, 0, 6, o4, h1, l4, c1, spx, dx1_f=0.50)
    chk("D4 gap-through fills at open (worse)", r[1], 100.10)

    # D5 SHORT DX1 f=0.50. entry_exec = bid open = 100.00; exits on ASK.
    #   k0: ask_low = 99.70+0.08 = 99.78 -> mfe = 100.00-99.78 = 0.22
    #        -> level = 100.00 - .11 = 99.89 ... wait: short level = ex - f*mfe
    #        = 100.00 - 0.11 = 99.89 is BELOW entry; trigger when ask_high >= lv
    #        ask_high at k1 = 99.95+.08 = 100.03 >= 99.89 -> STOP at k1
    #        fill = max(ask_open 99.88+... , lv). ask_open k1 = 99.80+.08 = 99.88
    #        -> fill = max(99.88, 99.89) = 99.89
    o5 = np.array([100.00, 99.80, 99.90,100.0,100.0,100.0,100.0,100.0,100.,100.])
    h5 = np.array([100.05, 99.95,100.00,100.0,100.0,100.0,100.0,100.0,100.,100.])
    l5 = np.array([ 99.70, 99.75, 99.85,100.0,100.0,100.0,100.0,100.0,100.,100.])
    c5 = np.array([ 99.80, 99.90, 99.95,100.0,100.0,100.0,100.0,100.0,100.,100.])
    r = slow_trade(False, 0, 6, o5, h5, l5, c5, spx, dx1_f=0.50)
    chk("D5 short DX1 fill (ask terms)", r[1], 99.89)
    chk("D5 short DX1 gross", r[2], 100.00 - 99.89)

    # D6 DX2 W=3, m=2.0, LONG. entry_exec = 100.08
    #   mfe over k0..k2 (extremes): h = 100.30,100.40,100.35 -> best 100.40
    #     -> early_fav = 0.32 ; D = 0.64 ; level = 100.08-0.64 = 99.44 from k3
    #   k3 low 99.40 <= 99.44 -> STOP, fill = min(open 99.60, 99.44) = 99.44
    o6 = np.array([100.00,100.25,100.30,99.60,100.0,100.0,100.0,100.0,100.,100.])
    h6 = np.array([100.30,100.40,100.35,99.70,100.0,100.0,100.0,100.0,100.,100.])
    l6 = np.array([ 99.95,100.20,99.55, 99.40,100.0,100.0,100.0,100.0,100.,100.])
    c6 = np.array([100.25,100.30, 99.60,99.50,100.0,100.0,100.0,100.0,100.,100.])
    r = slow_trade(True, 0, 6, o6, h6, l6, c6, spx, dx2_m=2.0, dx2_W=3)
    chk("D6 DX2 W=3 m=2 fill", r[1], 99.44)
    chk("D6 DX2 exit idx", float(r[4]), 3.0)

    # D7 DX2 does not arm before ei+W: same path, W=3, but candle2 low 99.55
    #    would breach a level armed at k1 (not allowed). Confirm no exit at k2.
    chk("D7 DX2 no pre-arm exit at k2", float(r[4] != 2), 1.0)

    # D8 DX2 early_fav <= 0 -> never arms -> naive exit at o[3]=99.50
    r = slow_trade(True, 0, 3, o3, h3, l3, c3, spx, dx2_m=2.0, dx2_W=3)
    chk("D8 DX2 no promise -> naive exit", r[1], 99.50)

    print(f"\nDX ENGINE ORACLE CHAIN: {'CLEARED' if ALL else 'BLOCKED'}")
