"""
engine_fast.py — performance-equivalent re-expression of code/engine.py.
Semantics are IDENTICAL by construction and verified trade-by-trade against
the canonical simulate_trade (see validate_engine.py). Canonical is the oracle.
Covers exactly the rules C4 needs: naive (A5), fixed stop (A2/A4), time stop (A6+Case8).
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/claude/repo/code")
import engine as EN
from fe_pipeline import session_of

COMM = EN.COMMISSION_RATE_PER_SIDE
SPREAD_MEDIAN = EN.SPREAD_MEDIAN

def spread_array(dt_index):
    hours = pd.DatetimeIndex(dt_index).hour.values
    tbl = np.array([SPREAD_MEDIAN[session_of(x)] for x in range(24)])
    return tbl[hours]

def swap_table(dt_index):
    """Charge points: every Mon-Fri 22:00 UTC, weight 3 on Wed. Returns
       (sorted charge ns array, cumulative weight array)."""
    d0 = pd.DatetimeIndex(dt_index).normalize()
    days = pd.date_range(d0[0] - pd.Timedelta(days=2), d0[-1] + pd.Timedelta(days=2),
                         freq="D")
    t = days + pd.Timedelta(hours=EN.SWAP_HOUR_UTC)
    w = np.where(t.dayofweek < 5, np.where(t.dayofweek == 2, 3, 1), 0)
    keep = w > 0
    return t[keep].values.astype("datetime64[ns]").astype(np.int64), np.cumsum(w[keep])

def swap_nights_vec(entry_ns, exit_ns, ct, cw):
    """count in (entry, exit] — matches canonical strict-greater start."""
    a = np.searchsorted(ct, entry_ns, side="right")   # first charge > entry
    b = np.searchsorted(ct, exit_ns,  side="right")   # first charge > exit
    lo = np.where(a > 0, cw[np.clip(a-1, 0, len(cw)-1)], 0)
    lo = np.where(a > 0, lo, 0)
    hi = np.where(b > 0, cw[np.clip(b-1, 0, len(cw)-1)], 0)
    return hi - lo

def run(df, F, rule="naive", stop=None, tmax=None):
    """F must be completed trades only. Returns per-trade cost decomposition."""
    o, h, l = df.open.values, df.high.values, df.low.values
    sp = df._sp; ct, cw = df._ct, df._cw
    ns = df.dt.values.astype("datetime64[ns]").astype(np.int64)
    ei = F.entry_idx.values.astype(np.int64)
    xe = F.exit_idx.values.astype(np.int64)
    isb = (F.direction.values == "BULL")
    entry_exec = np.where(isb, o[ei] + sp[ei], o[ei])

    if rule == "naive":
        k = xe.copy(); kind = np.full(len(ei), "cross", object)
    elif rule == "time":
        k = np.minimum(ei + tmax, xe)
        kind = np.where(k < xe, "time", "cross").astype(object)
    elif rule == "stop":
        lv = np.where(isb, entry_exec - stop, entry_exec + stop)
        k = xe.copy(); kind = np.full(len(ei), "cross", object)
        for j in range(len(ei)):
            a, b = ei[j], xe[j]
            if a >= b: continue
            if isb[j]:
                seg = l[a:b] <= lv[j]
            else:
                seg = (h[a:b] + sp[a:b]) >= lv[j]
            if seg.any():
                k[j] = a + int(seg.argmax()); kind[j] = "stop"
    else:
        raise ValueError(rule)

    stopped = np.array([x == "stop" for x in kind])
    if rule == "stop":
        lvv = np.where(isb, entry_exec - stop, entry_exec + stop)
        fill = np.where(isb, np.minimum(o[k], lvv), np.maximum(o[k] + sp[k], lvv))
    else:
        fill = np.where(isb, o[k], o[k] + sp[k])
    exit_exec = np.where(stopped, fill, np.where(isb, o[k], o[k] + sp[k]))

    gross = np.where(isb, exit_exec - entry_exec, entry_exec - exit_exec)
    comm = (entry_exec + exit_exec) * COMM
    n = swap_nights_vec(ns[ei], ns[k], ct, cw)
    swp = n * np.where(isb, EN.SWAP_LONG_PER_NIGHT, EN.SWAP_SHORT_PER_NIGHT)
    return pd.DataFrame({
        "cross_seq": F.cross_seq.values, "exit_kind": kind, "exit_idx_run": k,
        "entry_exec": entry_exec, "exit_exec": exit_exec, "gross_exec": gross,
        "commission": comm, "swap": swp, "net": gross - comm + swp})

def attach(df):
    df._sp = spread_array(df.dt.values)
    df._ct, df._cw = swap_table(df.dt.values)
    return df
