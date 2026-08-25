# 15M RECONSTRUCTION — PROVENANCE & INVARIANTS

The 15M series is a **pure deterministic OHLC aggregate** of the SHA-verified 1M
bulk in `ctrader_m1/`. It is not estimated and carries no new data facts. Any
fresh environment can rebuild it in ~1 minute and verify it byte-for-behaviour
against the frozen invariants below.

## REBUILD (one command, self-verifying)

```
python3 m15/build_m15.py      # aborts unless every invariant below passes
```

Writes `m15/data/{xauusd_m15.pkl, xauusd_m15.csv, dev_candles.pkl}`.

## FROZEN INVARIANTS (Cycle-1, asserted by build_m15.py)

```
→ full 15M series      : 326,542 bars · 0 dup / 0 non-mono / 0 OHLC breach
→ span                 : 2012-10-14 22:00 .. 2026-07-31 20:45  (bars labelled at OPEN)
→ resample rule        : 15-min wall-clock bucket = floor(ts,900s);
                         open=first, high=max, low=min, close=last
→ dev_candles          : 2015-12-01 .. 2024-12-31  (214,571 bars)
                         = 2015-12 seed (EMA burn-in, 2,008 bars) + 2016-2024
                         EMAs 8/21 on 15M closes, SMA-seeded, rolled, never reseeded
→ HOLDOUT (SEALED)     : trading-day close-date in [2025-01-01, 2026-06-29]
                         = 35,176 bars / 384 days  (never read by build_m15 dev slice)
→ EXCISED              : 2026-06-30 .. 2026-07-31  (never train/test/holdout)
→ folds F1..F8         : test 2017..2024; 257-259 days & 23.5k-23.7k bars each;
                         max missing-rate 0.042% (<0.05%); no quarantine risk
→ 8/21 UNFILTERED BASE : 10,096 completed trades (2016-2024) @ +$0.0137/trade
                         GROSS (zero-cost) — a NON-survivor; benchmark anchor
→ oracle gate          : validate_c5 => CLEARED (A=PASS feature/cross, B=PASS engine)
```

## FILE CHECKSUMS (convenience only — environment-dependent)

`xauusd_m15.csv` sha256 (built on python 3.12.3 / pandas 3.0.2 / numpy 2.4.4):
`b27ccc235c4f420d4931e45fb76311ca0c7f853c89fd639bd0c5b8276af2734a`

NOTE: `.pkl` and even `.csv` byte-hashes can vary across pandas/numpy versions
(pickle protocol, float formatting). Treat the SEMANTIC invariants above as the
source of truth — build_m15.py enforces them and aborts on any mismatch. Do NOT
rely on the pkl hash for verification.

## CARRIED-FORWARD CAVEATS (do not lose)

```
→ STEP=60 hardcode lives in code/c4_core.py + code/fe_pipeline.py (gap layer)
  and code/build_dataset.py (60s expected-grid). Consistent across canonical &
  vectorized, so the oracle gate PASSES and gross P&L + the timestamp/hour cost
  engine are unaffected. But gaps_spanned / gap_unobserved / weekend_spanned are
  WRONG on 15M, and build_dataset's expected/missing/quarantine grid is 1M-scaled.
  If any gap-COUNT feature is used, fix STEP=900 and re-validate FIRST.
→ code/ is the FROZEN 1M library (MANIFEST.sha256, 12/12 verified). It is NEVER
  edited. All 15M work is additive under m15/.
→ SPREAD model is a 1M July-2026 extrapolation applied to 15M — flag on every
  net-of-cost result; re-log fresh spread before any demo.
```
