"""
C5 PRE-RUN ORACLE GATE (extends validate_vs_oracle.py)
======================================================
A) FEATURE/CROSS equality vs canonical fe_pipeline across all 3 boundary eras,
   with the terminal-cross edge condition made EXPLICIT rather than silent.
B) ENGINE equality: engine_fast.run vs canonical engine.simulate_trade,
   trade-by-trade, for naive / stop / time rules. (validate_engine.py was not
   in the C4 handover upload -> re-proved here from scratch.)
Any unexplained mismatch aborts the cycle.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/claude/repo/code"); sys.path.insert(0, "/home/claude/work")
import fe_pipeline as FP, engine as EN
import c4_core as C, engine_fast as EF

df = pd.read_pickle("/home/claude/work/dev_candles.pkl")
FEATCOLS = ["cross_seq","direction","entry_price","gap_at_confirm_$","body_$",
            "body_dominant","momentum_ratio","mom_n_used","exit_price","mfe_$",
            "mae_$","emacross_pnl_$","duration_candles","candles_to_mfe",
            "candles_to_mae","fav_first","gaps_spanned","weekend_spanned",
            "gap_unobserved","entry_only","censored","zero_path","session"]

def feat_slice(lab, a, b):
    s = df[(df.dt >= a) & (df.dt < b)].reset_index(drop=True).copy()
    s["datetime_utc"] = s.dt.astype(str)
    can_c = FP.detect_crosses(s)
    vi, vd = C.detect_crosses_vec(s.ema8.values, s.ema21.values)
    ok1 = np.array_equal(can_c.confirm_idx.values, vi) and np.array_equal(can_c.direction.values, vd)
    edge = bool(vi[-1] + 1 > len(s) - 1)          # terminal cross has no entry candle
    canF = FP.build_features(s, can_c, FP.gap_map(s)); vecF = C.build_features_vec(s, vi, vd)
    bad = []
    for col in FEATCOLS:
        A, B = canF[col].values, vecF[col].values
        eq = (np.allclose(A.astype(float), B.astype(float), atol=1e-9, equal_nan=True)
              if A.dtype.kind == "f" else
              np.array_equal(np.asarray(A).astype(str), np.asarray(B).astype(str)))
        if not eq: bad.append(col)
    expected = ["censored"] if edge else []
    verdict = "PASS" if bad == expected else "FAIL"
    print(f"  [{lab:26s}] crosses={len(vi):5d} trades={len(vecF):5d} "
          f"detect={'OK' if ok1 else 'FAIL'} edge={str(edge):5s} "
          f"mismatch={bad if bad else 'NONE'} -> {verdict}")
    return ok1 and bad == expected

print("=== A) FEATURE / CROSS ORACLE EQUALITY ===")
print("    (edge=True => terminal cross lands on the last candle; canonical mislabels")
print("     the orphaned prior trade censored=False while emitting exit_price=NaN.")
print("     Documented divergence, vectorized label is the conservative one.)")
slices = [("2016-03 era :00/:58", "2016-03-01", "2016-04-01"),
          ("2017-09 era :00/:58", "2017-09-01", "2017-10-01"),
          ("2020-03 covid vol",   "2020-03-01", "2020-04-01"),
          ("2023-06 era :02/:56", "2023-06-01", "2023-07-01"),
          ("2024-06 era :02/:58", "2024-06-01", "2024-07-01")]
rA = all(feat_slice(l, pd.Timestamp(a), pd.Timestamp(b)) for l, a, b in slices)

# terminal condition on the ACTUAL run set
vi_full, _ = C.detect_crosses_vec(df.ema8.values, df.ema21.values)
run_edge = vi_full[-1] + 1 > len(df) - 1
print(f"    FULL DEV SET terminal-cross edge condition present? {run_edge} "
      f"(last cross idx {vi_full[-1]}, last candle idx {len(df)-1})")

print("\n=== B) ENGINE ORACLE EQUALITY (engine_fast vs canonical simulate_trade) ===")
s = df[(df.dt >= pd.Timestamp("2023-06-01")) & (df.dt < pd.Timestamp("2023-08-01"))
       ].reset_index(drop=True).copy()
s["datetime_utc"] = s.dt.astype(str)
vi, vd = C.detect_crosses_vec(s.ema8.values, s.ema21.values)
Fs = C.build_features_vec(s, vi, vd)
Fs = Fs[~Fs.censored].reset_index(drop=True)
s = EF.attach(s)
o, h, l, c = s.open.values, s.high.values, s.low.values, s.close.values
ts = pd.DatetimeIndex(s.dt.values)

def engine_case(lab, kw_fast, kw_can):
    R = EF.run(s, Fs, **kw_fast)
    bad = 0; first = None
    for j, r in enumerate(Fs.itertuples()):
        can = EN.simulate_trade(r.direction, int(r.entry_idx), int(r.exit_idx),
                                o, h, l, c, ts, **kw_can)
        got = R.iloc[j]
        same = (abs(can["net"] - got.net) < 1e-9 and
                abs(can["gross_exec"] - got.gross_exec) < 1e-9 and
                abs(can["commission"] - got.commission) < 1e-9 and
                abs(can["swap"] - got.swap) < 1e-9 and
                can["exit_idx"] == got.exit_idx_run and
                can["exit_kind"] == got.exit_kind)
        if not same:
            bad += 1
            if first is None: first = (j, can, dict(got))
    print(f"  [{lab:22s}] trades={len(Fs)} mismatches={bad} -> {'PASS' if bad==0 else 'FAIL'}")
    if first: print("     first mismatch:", first)
    return bad == 0

rB = True
rB &= engine_case("naive (A5)",      dict(rule="naive"),            dict())
rB &= engine_case("fixed stop $2.00", dict(rule="stop", stop=2.00), dict(rule="fixed", stop=2.00))
rB &= engine_case("fixed stop $0.50", dict(rule="stop", stop=0.50), dict(rule="fixed", stop=0.50))
rB &= engine_case("time stop 30",     dict(rule="time", tmax=30),   dict(tmax=30))

print(f"\nC5 ORACLE GATE: A={'PASS' if rA else 'FAIL'}  B={'PASS' if rB else 'FAIL'}  "
      f"=> {'CLEARED' if (rA and rB) else 'BLOCKED'}")
