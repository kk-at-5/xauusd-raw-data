"""ORACLE VALIDATION — vectorized C4 routines vs canonical repo functions.
Exact-equality on two slices from different boundary eras. Any mismatch aborts.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/claude/repo/code")
sys.path.insert(0, "/home/claude/work")
import fe_pipeline as FP
import c4_core as C

df = pd.read_pickle("/home/claude/work/dev_candles.pkl")

def slice_test(lab, a, b):
    s = df[(df.dt >= a) & (df.dt < b)].reset_index(drop=True).copy()
    s["datetime_utc"] = s.dt.astype(str)
    # ---- crosses
    can_c = FP.detect_crosses(s)
    vi, vd = C.detect_crosses_vec(s.ema8.values, s.ema21.values)
    ok1 = np.array_equal(can_c.confirm_idx.values, vi) and \
          np.array_equal(can_c.direction.values, vd)
    # ---- features
    gaps = FP.gap_map(s)
    can_F = FP.build_features(s, can_c, gaps)
    vec_F = C.build_features_vec(s, vi, vd)
    checks = {}
    for col in ["cross_seq","direction","entry_price","gap_at_confirm_$","body_$",
                "body_dominant","momentum_ratio","mom_n_used","exit_price","mfe_$",
                "mae_$","emacross_pnl_$","duration_candles","candles_to_mfe",
                "candles_to_mae","fav_first","gaps_spanned","weekend_spanned",
                "gap_unobserved","entry_only","censored","zero_path","session"]:
        a_ = can_F[col].values; b_ = vec_F[col].values
        if a_.dtype.kind == "f":
            checks[col] = np.allclose(a_.astype(float), b_.astype(float),
                                      atol=1e-9, equal_nan=True)
        else:
            checks[col] = np.array_equal(np.asarray(a_).astype(str),
                                         np.asarray(b_).astype(str))
    t_ok = (can_F.confirm_time_utc.values ==
            pd.Series(vec_F.confirm_time_utc).astype(str).values).all()
    bad = [k for k,v in checks.items() if not v]
    print(f"[{lab}] rows={len(s)} crosses={len(vi)} trades={len(vec_F)} "
          f"| cross-detect={'OK' if ok1 else 'FAIL'} | times={'OK' if t_ok else 'FAIL'} "
          f"| feature cols mismatched: {bad if bad else 'NONE'}")
    return ok1 and t_ok and not bad

r1 = slice_test("2016-03 (era :00/:58)", pd.Timestamp("2016-03-01"), pd.Timestamp("2016-04-01"))
r2 = slice_test("2023-06 (era :02/:56)", pd.Timestamp("2023-06-01"), pd.Timestamp("2023-07-01"))
r3 = slice_test("2024-06 (era :02/:58)", pd.Timestamp("2024-06-01"), pd.Timestamp("2024-07-01"))
print("ORACLE VALIDATION:", "PASSED" if (r1 and r2 and r3) else "FAILED")
