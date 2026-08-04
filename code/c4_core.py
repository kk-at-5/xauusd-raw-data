"""
C4 CORE — day blocks, cross detection, feature build (vectorized), engine.
Every vectorized routine is validated for EXACT equality against the
canonical repo functions on a held slice before being used at scale.
Canonical files are read-only oracles; they are never edited.
"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "/home/claude/repo/code")
import fe_pipeline as FP
import engine as EN

STEP = 60
ROLLOVER_H = 22          # documented FACT (rulebook Cost: swap time 22:00 UTC)
DAY_GAP_MIN_S = 1800     # 30 min — empirical valley in the gap distribution
WEEKEND_S = 6*3600       # canonical DAILY_GAP_MAX_S

# ---------------------------------------------------------------- day blocks
def day_blocks(df):
    """Derived rule: a gap starts a new trading day iff
         gap >= 30 min  AND  (gap spans the 22:00 UTC rollover OR gap >= 6h).
       Trading day is LABELLED BY ITS CLOSE DATE (rulebook, 03 Aug lock)."""
    ts = df.timestamp_utc.values
    d = np.diff(ts)
    gi = np.where(d > STEP)[0] + 1          # candle index AFTER each gap
    gs = d[gi-1]
    prev = df.dt.values[gi-1]; nxt = df.dt.values[gi]
    prev = pd.DatetimeIndex(prev); nxt = pd.DatetimeIndex(nxt)
    roll = prev.normalize() + pd.Timedelta(hours=ROLLOVER_H)
    roll = np.where(roll.values <= prev.values,
                    (roll + pd.Timedelta(days=1)).values, roll.values)
    spans = (roll > prev.values) & (roll <= nxt.values)
    is_day = (gs >= DAY_GAP_MIN_S) & (spans | (gs >= WEEKEND_S))
    starts = np.concatenate([[0], gi[is_day]])
    day_id = np.zeros(len(df), dtype=np.int32)
    day_id[starts[1:]] = 1
    day_id = np.cumsum(day_id)
    ends = np.concatenate([starts[1:]-1, [len(df)-1]])
    close_date = pd.DatetimeIndex(df.dt.values[ends]).normalize()
    return day_id, starts, ends, close_date, gi, gs, is_day

# ------------------------------------------------------- cross detection
def detect_crosses_vec(e8, e21):
    bull = (e8[:-1] <= e21[:-1]) & (e8[1:] > e21[1:])
    bear = (e8[:-1] >= e21[:-1]) & (e8[1:] < e21[1:])
    idx = np.where(bull | bear)[0] + 1
    direction = np.where(bull[idx-1], "BULL", "BEAR")
    return idx, direction

# ------------------------------------------------------- feature build
def build_features_vec(df, ci, direction, N_MOM=10):
    o,h,l,c = df.open.values, df.high.values, df.low.values, df.close.values
    e8,e21 = df.ema8.values, df.ema21.values
    n = len(df); last = n-1
    # drop a final cross that has no entry candle
    if ci[-1] + 1 > last:
        ci = ci[:-1]; direction = direction[:-1]
    K = len(ci); ei = ci + 1
    entry = o[ei]
    body = np.abs(c[ci]-o[ci])
    wick = (h[ci]-np.maximum(o[ci],c[ci])) + (np.minimum(o[ci],c[ci])-l[ci])
    gapc = np.abs(e8[ci]-e21[ci])
    ab = np.abs(c-o)
    avg_body = pd.Series(ab).shift(1).rolling(N_MOM, min_periods=1).mean().values[ci]
    with np.errstate(invalid="ignore", divide="ignore"):
        mom = np.where(avg_body > 0, body/avg_body, np.nan)
    # windows: trade k path = [ei_k .. win_end_k]; win_end = ci_{k+1} (last: n-1)
    win_end = np.empty(K, dtype=np.int64)
    win_end[:-1] = ci[1:]; win_end[-1] = last
    censored = np.zeros(K, bool); censored[-1] = True
    xi = np.full(K, -1, dtype=np.int64); xi[:-1] = ci[1:]+1
    ok_exit = (~censored) & (xi <= last)
    exit_price = np.where(ok_exit, o[np.clip(xi,0,last)], np.nan)
    # segment reductions over disjoint windows [ei_k, win_end_k]
    starts = ei; stops = win_end + 1
    assert np.all(starts[1:] == stops[:-1]), "windows must tile contiguously"
    lo_b, hi_b = starts[0], stops[-1]
    hh = np.maximum.reduceat(h[lo_b:hi_b], starts-lo_b)
    ll = np.minimum.reduceat(l[lo_b:hi_b], starts-lo_b)
    ar = np.arange(lo_b, hi_b)
    BIG = np.int64(2**62)
    fmax = np.minimum.reduceat(np.where(h[lo_b:hi_b] == np.repeat(hh, stops-starts),
                                        ar, BIG), starts-lo_b)
    fmin = np.minimum.reduceat(np.where(l[lo_b:hi_b] == np.repeat(ll, stops-starts),
                                        ar, BIG), starts-lo_b)
    isb = direction == "BULL"
    mfe = np.where(isb, hh-entry, entry-ll)
    mae = np.where(isb, entry-ll, hh-entry)
    mfe_i = np.where(isb, fmax, fmin); mae_i = np.where(isb, fmin, fmax)
    pnl = np.where(censored, np.nan, np.where(isb, exit_price-entry, entry-exit_price))
    # gaps (canonical semantics: ANY step != 60 counts; >6h = weekend)
    d = np.diff(df.timestamp_utc.values)
    gidx = np.where(d != STEP)[0] + 1
    gwk = d[gidx-1] > WEEKEND_S
    cum_all = np.zeros(n+1, np.int32); cum_all[gidx+1] = 1; cum_all = np.cumsum(cum_all)
    cum_wk  = np.zeros(n+1, np.int32); cum_wk[gidx[gwk]+1] = 1; cum_wk = np.cumsum(cum_wk)
    gspan = cum_all[win_end+1] - cum_all[ei+1]
    wspan = (cum_wk[win_end+1] - cum_wk[ei+1]) > 0
    dt = df.dt.values
    hours = pd.DatetimeIndex(dt).hour.values
    sess_map = np.array([FP.session_of(x) for x in range(24)])
    F = pd.DataFrame({
        "cross_seq": np.arange(1, K+1),
        "confirm_idx": ci, "entry_idx": ei, "win_end": win_end, "exit_idx": xi,
        "direction": direction,
        "confirm_time_utc": dt[ci], "session": sess_map[hours[ci]],
        "entry_time_utc": dt[ei], "entry_price": np.round(entry, 2),
        "gap_at_confirm_$": np.round(gapc, 4),
        "body_$": np.round(body, 2), "body_dominant": body > wick,
        "momentum_ratio": np.round(mom, 3), "mom_n_used": np.minimum(N_MOM, ci),
        "exit_time_utc": np.where(ok_exit, dt[np.clip(xi,0,last)], np.datetime64("NaT")),
        "exit_price": np.round(exit_price, 2),
        "mfe_$": np.round(mfe, 2), "mae_$": np.round(mae, 2),
        "emacross_pnl_$": np.round(pnl, 2),
        "duration_candles": win_end-ei+1,
        "candles_to_mfe": mfe_i-ei, "candles_to_mae": mae_i-ei,
        "fav_first": (mfe_i-ei) < (mae_i-ei),
        "gaps_spanned": gspan, "weekend_spanned": wspan,
        "gap_unobserved": gspan > 0,
        "entry_only": np.arange(K) == 0, "censored": censored,
        "zero_path": (win_end-ei+1) <= 1,
        "candles_since_prev_cross": np.concatenate([[-1], np.diff(ci)]),
    })
    return F
