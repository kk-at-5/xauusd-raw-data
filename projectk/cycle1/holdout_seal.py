#!/usr/bin/env python3
"""
PROJECT K — CYCLE 1, TASK 1: THE HOLDOUT SEAL
=============================================
Defines and ENFORCES the irreversible three-way partition of the XAUUSD record.

    QUARANTINE  ..2015-12-31   pre-2016 = a structurally different measurement
                               instrument (drifting session boundaries, Friday
                               overnight and Sunday pre-open bars). READ-ONLY,
                               indicator burn-in ONLY, never an evaluation row.
    DEV         2016-01-01 .. 2022-12-31   free to explore, fit, iterate.
    HOLDOUT     2023-01-01 .. 2026-06-29   SEALED. Opened ONCE, at the very end,
                               only after the full freeze battery passes in-sample.
    EXCISED     2026-06-30 .. 2026-07-31   inherited contamination. Never any pool.

WHY A SCRIPT AND NOT A CONVENTION
  Predecessor cycle C9 found the 2015/16 boundary missing from a hand-maintained
  seam list in build_dataset.py, which let 1,540 quarantined trades into the
  fit/eval pool. That is a WHITELIST failure: an enumerated list can silently omit
  an entry. This module replaces the whitelist with a RANGE ASSERTION, which
  cannot. Every evaluation set must pass through assert_no_leak().

WHAT THE SEAL CAN AND CANNOT DO
  Sole operator, own repo: this cannot PREVENT looking at the holdout. It makes
  looking visible, deliberate and logged. That is its entire function.

THE SEAL IS A DATE RANGE OVER EVERY SOURCE, NOT JUST XAUUSD.
  Cross-asset series (XAGUSD, EURUSD, USDJPY, XAUEUR, US500, XTIUSD) extracted in
  later tasks are subject to the SAME walls. Exploring a correlate across the
  holdout window is a leak.

Usage
  python3 holdout_seal.py seal   /path/to/repo/ctrader_m1   # once. writes manifest.
  python3 holdout_seal.py verify /path/to/repo/ctrader_m1   # any time after.
  python3 holdout_seal.py status

  from holdout_seal import load_dev, load_burnin, assert_no_leak
"""
import sys, os, glob, json, hashlib, datetime as dt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# THE PARTITION — ratified 31/08/2026. Changing any line below invalidates the
# seal and every result derived under it.
# ----------------------------------------------------------------------------
BURNIN_START   = dt.date(2015, 12, 1)    # read-only, state seeding only
QUARANTINE_END = dt.date(2015, 12, 31)
DEV_START      = dt.date(2016, 1, 1)
DEV_END        = dt.date(2022, 12, 31)
HOLDOUT_START  = dt.date(2023, 1, 1)
HOLDOUT_END    = dt.date(2026, 6, 29)
EXCISED_START  = dt.date(2026, 6, 30)
EXCISED_END    = dt.date(2026, 7, 31)

WINDOW_OPEN_MIN  = 2 * 60     # 02:00Z
WINDOW_CLOSE_MIN = 16 * 60    # 16:00Z
BREAK_MIN        = 20         # gap >= this marks a session boundary
MAIN_SESSION_H   = 12         # a session shorter than this is not a trading day

# Frozen invariants. Asserted on every run; any mismatch aborts.
INVARIANTS = dict(
    active_bars    = 3_730_800,
    sessions       = 2_747,
    main_sessions  = 2_728,
    tradeable_days = 2_721,
    first_bar      = "2016-01-03 23:00:00",
    last_bar       = "2026-07-31 20:57:00",
    dev_days       = 1_801,
    holdout_days   =   896,
    excised_days   =    24,
)

SEAL_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HOLDOUT_SEAL.json")
UNSEAL_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UNSEAL_LOG.txt")
UNSEAL_TOKEN = "I-AM-OPENING-THE-HOLDOUT-ONCE-AND-FOREVER"


# ----------------------------------------------------------------------------
# Loading and canonicalisation
# ----------------------------------------------------------------------------
def _load_raw(datadir):
    files = sorted(glob.glob(os.path.join(datadir, "xauusd_m1_*.csv")))
    if not files:
        sys.exit(f"no monthly CSVs found in {datadir}")
    df = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    df["timestamp_utc"] = df["timestamp_utc"].astype("int64")
    ts = df["timestamp_utc"].values
    assert (np.diff(ts) > 0).all(), "timestamps not strictly monotonic"
    return df


def _canonical_hash(df):
    """SHA-256 over integer cents, NOT over file bytes.

    File-level hashing would break on any re-chunking or reformatting of the
    monthly CSVs, and the holdout/excised boundary falls INSIDE 202606.csv.
    Prices are exact to $0.01 (verified: max 2 decimal places), so integer
    cents is lossless and invariant to text formatting.
    """
    h = hashlib.sha256()
    cents = {c: np.rint(df[c].astype(float).values * 100).astype("int64")
             for c in ("open", "high", "low", "close")}
    ts = df["timestamp_utc"].values
    for i in range(len(df)):
        h.update(f"{ts[i]},{cents['open'][i]},{cents['high'][i]},"
                 f"{cents['low'][i]},{cents['close'][i]}\n".encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Session derivation — identical logic to session_calendar_proof.py (Cycle 0)
# ----------------------------------------------------------------------------
def _trading_days(ts):
    """Return the set of dates whose session fully contains 02:00-16:00Z.

    A trading day is labelled by the UTC date its session CLOSES on, because the
    session opens the previous evening (22:00Z summer / 23:00Z winter).
    """
    cut = np.where(np.diff(ts) >= BREAK_MIN * 60)[0]
    s = np.concatenate(([ts[0]], ts[cut + 1]))
    e = np.concatenate((ts[cut], [ts[-1]]))
    S = pd.DataFrame({"start": pd.to_datetime(s, unit="s", utc=True),
                      "end":   pd.to_datetime(e, unit="s", utc=True)})
    S["hours"] = (S.end - S.start).dt.total_seconds() / 3600
    M = S[S.hours >= MAIN_SESSION_H].copy()

    day = M["end"].dt.normalize()
    dsec = day.astype("int64").values
    # UNIT TRAP (Cycle 0 error #5): datetime64[s].astype('int64') gives SECONDS.
    assert 1.4e9 < dsec[0] < 2.5e9, "UNIT CHECK FAILED: expected epoch SECONDS"
    st, en = M["start"].astype("int64").values, M["end"].astype("int64").values
    ok = (dsec + WINDOW_OPEN_MIN * 60 >= st) & (dsec + WINDOW_CLOSE_MIN * 60 <= en)
    return len(S), len(M), pd.Series(day.dt.date.values)[ok].tolist()


def _pool_of(d):
    if d < DEV_START:       return "QUARANTINE"
    if d <= DEV_END:        return "DEV"
    if d <= HOLDOUT_END:    return "HOLDOUT"
    return "EXCISED"


# ----------------------------------------------------------------------------
# THE WALL — the C9 fix, expressed as a range assertion
# ----------------------------------------------------------------------------
def assert_no_leak(dates, pool):
    """Raise if ANY evaluation row falls outside its declared pool.

    Call this on every fit set, every evaluation set, every fold. It is cheap and
    it is the only thing standing between you and a repeat of the C9 leak.
    """
    bounds = {"DEV": (DEV_START, DEV_END), "HOLDOUT": (HOLDOUT_START, HOLDOUT_END)}
    if pool not in bounds:
        raise ValueError(f"{pool} is not an evaluable pool")
    lo, hi = bounds[pool]
    bad = [d for d in dates if not (lo <= d <= hi)]
    if bad:
        raise AssertionError(
            f"LEAK: {len(bad)} row(s) outside {pool} [{lo}..{hi}]. "
            f"First offenders: {sorted(bad)[:5]}")
    return True


# ----------------------------------------------------------------------------
# Public loaders
# ----------------------------------------------------------------------------
def load_dev(datadir):
    df = _load_raw(datadir)
    d = pd.to_datetime(df.timestamp_utc, unit="s", utc=True).dt.date
    return df[(d >= DEV_START) & (d <= DEV_END)].reset_index(drop=True)


def load_burnin(quarantine_dir):
    """December 2015 only. STATE SEEDING ONLY — these rows are never evaluated.

    The quarantine is a MEASUREMENT-INVARIANCE boundary, not a data-quality one:
    pre-2016 came off a structurally different feed. Seeding an EMA across it is
    acceptable (it only warms state); scoring a trade in it is not.
    """
    files = sorted(glob.glob(os.path.join(quarantine_dir, "xauusd_m1_201512.csv")))
    if not files:
        sys.exit(f"no 201512 burn-in file in {quarantine_dir}")
    df = pd.read_csv(files[0], dtype=str)
    df["timestamp_utc"] = df["timestamp_utc"].astype("int64")
    df.attrs["evaluation_eligible"] = False
    return df


def load_holdout(datadir, unseal_token=None, reason=None):
    if unseal_token != UNSEAL_TOKEN or not reason:
        raise PermissionError(
            "HOLDOUT IS SEALED.\n"
            "It opens ONCE, only after a candidate clears the ENTIRE freeze battery\n"
            "on DEV. To open it, pass the unseal token and a written reason. The\n"
            "call is appended to UNSEAL_LOG.txt and cannot be unwritten.")
    with open(UNSEAL_LOG, "a") as f:
        f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()}  UNSEALED  {reason}\n")
    df = _load_raw(datadir)
    d = pd.to_datetime(df.timestamp_utc, unit="s", utc=True).dt.date
    return df[(d >= HOLDOUT_START) & (d <= HOLDOUT_END)].reset_index(drop=True)


# ----------------------------------------------------------------------------
# seal / verify
# ----------------------------------------------------------------------------
def _compute(datadir):
    df = _load_raw(datadir)
    ts = df["timestamp_utc"].values
    n_sess, n_main, tdays = _trading_days(ts)

    obs = dict(
        active_bars    = len(df),
        sessions       = n_sess,
        main_sessions  = n_main,
        tradeable_days = len(tdays),
        first_bar      = str(pd.to_datetime(ts[0], unit="s")),
        last_bar       = str(pd.to_datetime(ts[-1], unit="s")),
    )
    counts = {p: 0 for p in ("QUARANTINE", "DEV", "HOLDOUT", "EXCISED")}
    for d in tdays:
        counts[_pool_of(d)] += 1
    obs.update(dev_days=counts["DEV"], holdout_days=counts["HOLDOUT"],
               excised_days=counts["EXCISED"])

    bad = [k for k, v in INVARIANTS.items() if obs[k] != v]
    if bad:
        detail = "\n".join(f"    {k}: expected {INVARIANTS[k]}, got {obs[k]}" for k in bad)
        sys.exit(f"INVARIANT MISMATCH — ABORTING. The data is not what the seal "
                 f"was computed against:\n{detail}")

    d = pd.to_datetime(df.timestamp_utc, unit="s", utc=True).dt.date
    slices = {
        "DEV":     df[(d >= DEV_START)     & (d <= DEV_END)],
        "HOLDOUT": df[(d >= HOLDOUT_START) & (d <= HOLDOUT_END)],
        "EXCISED": df[(d >= EXCISED_START) & (d <= EXCISED_END)],
    }
    assert sum(len(v) for v in slices.values()) == len(df), \
        "partition is not exhaustive over the active set"
    obs["slices"] = {k: dict(bars=len(v), sha256=_canonical_hash(v))
                     for k, v in slices.items()}
    return obs


def cmd_seal(datadir):
    if os.path.exists(SEAL_FILE):
        sys.exit(f"A SEAL ALREADY EXISTS at {SEAL_FILE}. Sealing is irreversible; "
                 f"refusing to overwrite. Use 'verify'.")
    obs = _compute(datadir)
    obs["sealed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    obs["partition"] = dict(
        quarantine=[None, str(QUARANTINE_END)], dev=[str(DEV_START), str(DEV_END)],
        holdout=[str(HOLDOUT_START), str(HOLDOUT_END)],
        excised=[str(EXCISED_START), str(EXCISED_END)],
        window="02:00-16:00Z")
    with open(SEAL_FILE, "w") as f:
        json.dump(obs, f, indent=2)
    _report(obs); print(f"\nSEALED -> {SEAL_FILE}\nCommit it. This is irreversible.")


def cmd_verify(datadir):
    if not os.path.exists(SEAL_FILE):
        sys.exit("no seal found — run 'seal' first")
    rec = json.load(open(SEAL_FILE))
    obs = _compute(datadir)
    diffs = [k for k in rec["slices"]
             if rec["slices"][k]["sha256"] != obs["slices"][k]["sha256"]]
    _report(obs)
    if diffs:
        sys.exit(f"\nSEAL BROKEN — slice(s) altered since sealing: {diffs}")
    print(f"\nSEAL INTACT. Sealed {rec['sealed_utc']}.")
    if os.path.exists(UNSEAL_LOG):
        print(f"UNSEAL LOG IS NON-EMPTY:\n{open(UNSEAL_LOG).read().rstrip()}")


def _report(obs):
    print("PROJECT K — HOLDOUT SEAL")
    print(f"  active bars    {obs['active_bars']:,}   {obs['first_bar']} -> {obs['last_bar']}")
    print(f"  sessions       {obs['sessions']:,} ({obs['main_sessions']:,} main)")
    print(f"  tradeable days {obs['tradeable_days']:,} under 02:00-16:00Z\n")
    for k in ("DEV", "HOLDOUT", "EXCISED"):
        s = obs["slices"][k]
        days = obs[f"{k.lower()}_days"]
        print(f"  {k:<9} {days:>5} days  {s['bars']:>9,} bars  sha256 {s['sha256'][:16]}...")


def cmd_status():
    print(__doc__.split("Usage")[0].rstrip())
    print("SEALED" if os.path.exists(SEAL_FILE) else "NOT YET SEALED")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    c = sys.argv[1]
    if c == "status": cmd_status()
    elif c in ("seal", "verify"):
        if len(sys.argv) < 3: sys.exit(f"usage: {sys.argv[0]} {c} /path/to/ctrader_m1")
        (cmd_seal if c == "seal" else cmd_verify)(sys.argv[2])
    else: sys.exit(__doc__)
