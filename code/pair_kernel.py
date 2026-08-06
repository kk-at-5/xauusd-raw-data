"""
PAIR KERNEL — MA-pair cross sweep (gross, zero-cost) with monthly/yearly accumulators.
Semantics MUST match c4_core.detect_crosses_vec + build_features_vec exactly for the
8/21 pair (oracle-validated before any sweep runs).

Cross:  BULL if ef[i-1] <= es[i-1] and ef[i] >  es[i]
        BEAR if ef[i-1] >= es[i-1] and ef[i] <  es[i]
Entry:  open[i+1]. Exit: open[j+1] where j = next (opposite) cross. Last trade censored.
EMA:    SMA-seeded over the first `period` closes, then rolled, k = 2/(period+1),
        never reseeded across gaps (rulebook).
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def ema(close, period, out):
    k = 2.0 / (period + 1.0)
    s = 0.0
    for i in range(period):
        s += close[i]
        out[i] = np.nan
    e = s / period
    out[period - 1] = e
    for i in range(period, close.shape[0]):
        e = close[i] * k + e * (1.0 - k)
        out[i] = e
    return out


@njit(cache=True, fastmath=False)
def sweep_pair(o, ef, es, mon, yr, start, n_mon, n_yr,
               mon_sum, mon_n, yr_sum, yr_n):
    """Zero-cost gross sweep. Returns (n_trades, sum_gross, sum_dur, sum_sq)."""
    for m in range(n_mon):
        mon_sum[m] = 0.0
        mon_n[m] = 0
    for y in range(n_yr):
        yr_sum[y] = 0.0
        yr_n[y] = 0
    n = o.shape[0]
    open_ei = -1
    open_bull = False
    open_m = -1
    open_y = -1
    ntr = 0
    tot = 0.0
    totsq = 0.0
    totdur = 0.0
    for i in range(start, n - 1):
        a0 = ef[i - 1] - es[i - 1]
        a1 = ef[i] - es[i]
        bull = (a0 <= 0.0) and (a1 > 0.0)
        bear = (a0 >= 0.0) and (a1 < 0.0)
        if not (bull or bear):
            continue
        px = o[i + 1]
        if open_ei >= 0:
            g = px - o[open_ei] if open_bull else o[open_ei] - px
            if open_m >= 0:                      # pre-2016 rows excluded from eval
                ntr += 1
                tot += g
                totsq += g * g
                totdur += (i - open_ei + 1)
                mon_sum[open_m] += g
                mon_n[open_m] += 1
                yr_sum[open_y] += g
                yr_n[open_y] += 1
        open_ei = i + 1
        open_bull = bull
        open_m = mon[i]
        open_y = yr[i]
    return ntr, tot, totdur, totsq
