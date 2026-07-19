"""
engine.py — TRUE BID/ASK EXECUTION ENGINE (canonical, repo: code/engine.py)
XAUUSD 1M · 0.01 lot = 1 oz · raw data is BID OHLC

Authority: PROJECT_REBIRTH_RULES.md (Cost, Raw Data / bid-ask spec,
Objective 2 re-entry semantics) + 08_SPREAD_MODEL_V1.md (session-median
spread, once per round trip, ask-side-fill session convention).

EXECUTABLE-PRICE MODEL
  LONG :  enter at Ask = bid_open + spread(entry candle)
          exit  at Bid
          stop level referenced to the Ask entry price, triggers on Bid
  SHORT:  enter at Bid = bid_open
          exit  at Ask = bid + spread(exit candle)
          stop level referenced to the Bid entry price, triggers on Ask
  Spread is therefore paid exactly ONCE per round trip by construction.

FROZEN INTRA-CANDLE ASSUMPTIONS (conservative; every result depends on these)
  A1  Entry candle is evaluated for stops/TP with its full range.
  A2  ADVERSE-FIRST inside any candle: if both a stop and a TP level lie
      within the candle's range, the STOP fills (pessimistic ordering).
  A3  Arming/updating (breakeven arm, trail update) happens at candle CLOSE
      and takes effect from the NEXT candle. No same-candle arm-then-exit.
  A4  Gap-through: stops fill at the WORSE of (level, open);
      take-profits fill at the BETTER of (level, open).
  A5  If nothing triggers before the opposite cross, exit at the opposite
      cross's entry candle open (executable side) — the naive exit.
  A6  Stopped-out trades do NOT re-enter before the next cross
      (rulebook Objective 2): the trade list is fixed by the cross list;
      exit rules change exits only.
  A7  Slippage = 0 everywhere (deferred by rulebook; never silent — it is
      an explicit constant below).

COSTS
  commission = (entry_exec_px + exit_exec_px) * 30/1e6   (per-oz, per side)
  swap: −0.5465/night long, +0.3726/night short; nights counted as 22:00 UTC
        crossings; Wednesday triple; weekend crossings do not count.
"""
import numpy as np, pandas as pd
from fe_pipeline import session_of   # single source of truth for bands

SPREAD_MEDIAN = {"Asian":0.09, "London":0.08, "NY_Overlap":0.08, "NY":0.08}
COMMISSION_RATE_PER_SIDE = 30.0/1_000_000
SWAP_LONG_PER_NIGHT, SWAP_SHORT_PER_NIGHT = -0.5465, +0.3726
SWAP_HOUR_UTC = 22
SLIPPAGE = 0.0   # deferred (rulebook); explicit, never silent

def spread_of(ts): return SPREAD_MEDIAN[session_of(ts.hour)]

def swap_nights(entry_ts, exit_ts):
    """22:00 UTC crossings between entry and exit; Wed triple; weekends skip."""
    nights = 0
    t = entry_ts.normalize() + pd.Timedelta(hours=SWAP_HOUR_UTC)
    if t <= entry_ts: t += pd.Timedelta(days=1)
    while t <= exit_ts:
        if t.dayofweek < 5:                      # Mon..Fri 22:00 charge points
            nights += 3 if t.dayofweek == 2 else 1   # Wednesday triple
        t += pd.Timedelta(days=1)
    return nights

def swap_oz(direction, entry_ts, exit_ts):
    n = swap_nights(entry_ts, exit_ts)
    return n * (SWAP_LONG_PER_NIGHT if direction=="BULL" else SWAP_SHORT_PER_NIGHT)

# ----------------------------------------------------------------------
# CORE: simulate one trade window under an exit rule
# window = [ei, xe]  where ei = entry candle idx (i+1 of the confirming
# cross) and xe = the opposite cross's entry candle idx (naive exit candle).
# Returns executable P&L and full cost decomposition.
# ----------------------------------------------------------------------
def simulate_trade(direction, ei, xe, o, h, l, c, ts, rule="naive",
                   stop=None, be_trigger=None, trail=None, tp=None, tmax=None):
    long_ = direction == "BULL"
    sp_e = spread_of(ts[ei])
    entry_exec = o[ei] + sp_e if long_ else o[ei]

    # stop level in TRIGGER terms (bid for long, bid-equivalent for short)
    stop_lv = None
    if stop is not None:
        stop_lv = (entry_exec - stop) if long_ else (entry_exec + stop)  # short: ask-terms level
    be_armed = False
    trail_ref = None   # highest bid close (long) / lowest ask close (short)

    k = ei
    exit_kind, exit_exec, exit_k = "cross", None, xe
    while k <= xe:
        sp_k = spread_of(ts[k])
        if k == xe:                                   # A5 naive exit at open
            exit_exec = o[xe] if long_ else o[xe] + sp_k
            exit_kind = "cross"; exit_k = xe
            break
        # -- time stop: exits AT OPEN, hence checked BEFORE any intra-candle
        #    stop/TP path on this candle --------------------------------------
        if tmax is not None and k - ei >= tmax:
            exit_exec = o[k] if long_ else o[k] + sp_k
            exit_kind, exit_k = "time", k
            break
        # -- A2 adverse-first: stop check ---------------------------------
        if stop_lv is not None:
            if long_:
                trig = l[k] <= stop_lv
                fill = min(o[k], stop_lv)                        # A4 worse
            else:  # short stop in ask terms: ask_high = h+sp
                trig = h[k] + sp_k >= stop_lv
                fill = max(o[k] + sp_k, stop_lv)                 # A4 worse
            if trig:
                exit_exec, exit_kind, exit_k = fill, "stop", k
                break
        # -- take-profit --------------------------------------------------
        if tp is not None:
            lv = entry_exec + tp if long_ else entry_exec - tp   # exec terms
            if long_:
                trig = h[k] >= lv;            fill = max(o[k], lv)     # A4 better
            else:  # short TP on ask low = l+sp
                trig = l[k] + sp_k <= lv;     fill = min(o[k] + sp_k, lv)
            if trig:
                exit_exec, exit_kind, exit_k = fill, "tp", k
                break
        # -- A3 close-of-candle arming/updating, effective next candle ----
        if rule == "breakeven" and not be_armed and be_trigger is not None:
            fav = (h[k] - entry_exec) if long_ else (entry_exec - (l[k] + sp_k))
            if fav >= be_trigger:
                be_armed = True
                stop_lv = entry_exec                 # scratch level, both sides
        if trail is not None:
            ref = c[k] if long_ else c[k] + sp_k
            trail_ref = ref if trail_ref is None else \
                        (max(trail_ref, ref) if long_ else min(trail_ref, ref))
            cand = (trail_ref - trail) if long_ else (trail_ref + trail)
            stop_lv = cand if stop_lv is None else \
                      (max(stop_lv, cand) if long_ else min(stop_lv, cand))
        k += 1

    gross = (exit_exec - entry_exec) if long_ else (entry_exec - exit_exec)
    comm = (entry_exec + exit_exec) * COMMISSION_RATE_PER_SIDE
    swp = swap_oz(direction, ts[ei], ts[exit_k])
    return {"exit_kind": exit_kind, "exit_idx": exit_k,
            "entry_exec": entry_exec, "exit_exec": exit_exec,
            "gross_exec": gross, "commission": comm, "swap": swp,
            "net": gross - comm + swp}

def simulate_family(df, F, **rule_kw):
    """Run one exit-rule config over every completed trade in F."""
    o,h,l,c = df.open.values, df.high.values, df.low.values, df.close.values
    ts = pd.to_datetime(df.datetime_utc).values
    ts = pd.DatetimeIndex(ts)
    idx_of = {t: i for i, t in enumerate(df.timestamp_utc.values)}
    rows = []
    comp = F[~F.censored]
    for r in comp.itertuples():
        ei = idx_of[int(pd.Timestamp(r.entry_time_utc).timestamp())]
        xe = idx_of[int(pd.Timestamp(r.exit_time_utc).timestamp())]
        # entry/exit times in F are candle-open stamps of i+1 candles
        out = simulate_trade(r.direction, ei, xe, o, h, l, c, ts, **rule_kw)
        out["cross_seq"] = r.cross_seq
        rows.append(out)
    R = pd.DataFrame(rows)
    return R

# ----------------------------------------------------------------------
# SELF-TEST BATTERY — synthetic candles, every expected value HAND-COMPUTED
# (constant London spread 0.08 via fixed timestamps 08:00+)
# ----------------------------------------------------------------------
def _selftest():
    ts = pd.DatetimeIndex(pd.date_range("2026-07-15 08:00", periods=8, freq="min"))
    ok = True
    def chk(name, got, want, tol=1e-9):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: got {got:.4f} want {want:.4f}")

    # Case 1 — LONG fixed stop hit exactly.
    # o=[100,...] entry_ask=100.08; stop=1.00 -> level 99.08; candle2 low 99.00
    o = np.array([100,100.2,100.1,100,100,100,100,100.])
    h = np.array([100.3,100.4,100.2,100,100,100,100,100.])
    l = np.array([ 99.9,100.0, 99.0,100,100,100,100,100.])
    c = np.array([100.2,100.1, 99.5,100,100,100,100,100.])
    r = simulate_trade("BULL",0,5,o,h,l,c,ts,rule="fixed",stop=1.00)
    chk("C1 long stop fill", r["exit_exec"], 99.08)
    chk("C1 long stop gross", r["gross_exec"], -1.00)

    # Case 2 — SHORT fixed stop, ask-terms trigger.
    # entry_bid=100; stop=1.00 -> ask level 101.00; trigger bid h>=100.92
    # candle1 h=100.95 -> fill ask = max(o+0.08, 101.00) = max(100.28,101)=101
    o2 = np.array([100,100.2,100,100,100,100,100,100.])
    h2 = np.array([100.1,100.95,100,100,100,100,100,100.])
    l2 = np.array([99.9,100.0,100,100,100,100,100,100.])
    c2 = np.array([100,100.9,100,100,100,100,100,100.])
    r = simulate_trade("BEAR",0,5,o2,h2,l2,c2,ts,rule="fixed",stop=1.00)
    chk("C2 short stop gross", r["gross_exec"], -1.00)

    # Case 3 — gap-through: long stop level 99.08, candle2 OPENS at 98.50
    o3 = np.array([100,100.2,98.5,100,100,100,100,100.])
    h3 = np.array([100.3,100.4,98.8,100,100,100,100,100.])
    l3 = np.array([99.9,100.0,98.2,100,100,100,100,100.])
    c3 = np.array([100.2,100.1,98.6,100,100,100,100,100.])
    r = simulate_trade("BULL",0,5,o3,h3,l3,c3,ts,rule="fixed",stop=1.00)
    chk("C3 gap-through fills at open", r["exit_exec"], 98.50)

    # Case 4 — breakeven scratch: trigger 0.50 armed at close of candle1
    # (h=100.60 >= 100.58), effective candle2+; candle3 low touches 100.08
    o4 = np.array([100,100.3,100.5,100.3,100,100,100,100.])
    h4 = np.array([100.2,100.6,100.6,100.4,100,100,100,100.])
    l4 = np.array([99.95,100.2,100.3,100.05,100,100,100,100.])
    c4 = np.array([100.1,100.5,100.4,100.1,100,100,100,100.])
    r = simulate_trade("BULL",0,6,o4,h4,l4,c4,ts,rule="breakeven",be_trigger=0.50)
    chk("C4 breakeven scratch gross", r["gross_exec"], 0.0)
    chk("C4 breakeven net = -commission", r["net"],
        -(100.08+100.08)*COMMISSION_RATE_PER_SIDE)

    # Case 5 — TP gap fills BETTER: long tp=0.50 -> level 100.58,
    # candle2 opens 100.90 -> fill at 100.90
    o5 = np.array([100,100.3,100.9,100,100,100,100,100.])
    h5 = np.array([100.2,100.4,101.0,100,100,100,100,100.])
    l5 = np.array([99.95,100.2,100.8,100,100,100,100,100.])
    c5 = np.array([100.1,100.3,100.9,100,100,100,100,100.])
    r = simulate_trade("BULL",0,5,o5,h5,l5,c5,ts,tp=0.50)
    chk("C5 TP gap fills at open", r["exit_exec"], 100.90)

    # Case 6 — naive == deduction identity: long, exit candle5 open 100.40
    o6 = np.array([100,100.1,100.2,100.3,100.35,100.4,100,100.])
    h6 = o6+0.05; l6 = o6-0.05; c6 = o6+0.02
    r = simulate_trade("BULL",0,5,o6,h6,l6,c6,ts)
    chk("C6 naive gross == bid_pnl - spread", r["gross_exec"], (100.40-100.00)-0.08)

    # Case 7 — swap: Tue 18:00 -> Thu 02:00 crosses Tue22 + Wed22(x3) = 4 nights
    n = swap_nights(pd.Timestamp("2026-07-14 18:00"), pd.Timestamp("2026-07-16 02:00"))
    chk("C7 swap nights (Wed triple)", n, 4)
    # Fri 18:00 -> Mon 02:00: Fri22 charges, Sat/Sun skip = 1
    n = swap_nights(pd.Timestamp("2026-07-17 18:00"), pd.Timestamp("2026-07-20 02:00"))
    chk("C7 swap weekend skip", n, 1)

    # Case 8 — time-exit-at-open PRIORITY over same-candle stop:
    # tmax=2 -> exit at open of candle2 (100.30) even though candle2's low
    # (99.00) pierces the 99.08 stop level.
    o8 = np.array([100,100.2,100.3,100,100,100,100,100.])
    h8 = np.array([100.3,100.4,100.5,100,100,100,100,100.])
    l8 = np.array([ 99.9,100.0, 99.0,100,100,100,100,100.])
    c8 = np.array([100.2,100.1,100.2,100,100,100,100,100.])
    r = simulate_trade("BULL",0,5,o8,h8,l8,c8,ts,rule="fixed",stop=1.00,tmax=2)
    chk("C8 time exit beats same-candle stop", r["exit_exec"], 100.30)
    chk("C8 exit kind is time", 1.0 if r["exit_kind"]=="time" else 0.0, 1.0)

    print(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    return ok

if __name__ == "__main__":
    _selftest()
