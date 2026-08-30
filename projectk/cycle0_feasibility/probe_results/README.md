# Cycle 0 — Task 2 probe results (SNAPSHOTS, not reproducible)

These outputs are committed because **they cannot be regenerated.** Unlike
`session_calendar_proof.py`, whose output re-derives identically from frozen
data, these probes observe a **moving target**: the broker's symbol offering
changes, and dated futures contracts expire and are delisted. Re-running the
bots in six months answers a different question.

Bots that produced these live in `ctrader_bots/`.

| file | produced by | run (UTC) |
|---|---|---|
| `symbol_census.txt` | `SymbolCensus.cs` | 2026-08-30 14:23 |
| `depth_probe_batch1_futures.*` | `SymbolDepthProbe.cs` (v1) | 2026-08-30 14:43 |
| `depth_probe_batch2_crossasset.*` | `SymbolDepthProbe_v2.cs` | 2026-08-30 15:06 |

## FINDINGS

```
→ 198 symbols on the account. XAUUSD appears EXACTLY ONCE — no suffixed
  variants, so ctrader_m1/ came from an unambiguous symbol.
→ NOT AVAILABLE (checked explicitly): bonds/treasuries/yields, any volatility
  index, any dollar-index symbol. The real-yield channel is therefore NOT
  obtainable from this feed. DXY is not a symbol but IS constructible: all six
  basket components (EURUSD, USDJPY, GBPUSD, USDCAD, USDSEK, USDCHF) are present.
→ GOLD FUTURES ARE DEAD AS A DATA SOURCE. Only the front contract is carried,
  with ~32 days of history (GCZ26_CFD: 33,719 bars, 2026-07-27 onward,
  SERVER_EXHAUSTED after ONE load call). Expired contracts are delisted, not
  archived (GCM26: "Symbol has no quotes"). A 2016-2024 futures-spot basis
  series cannot be built. Closed on data availability, not on mechanism.
→ FIVE CROSS-ASSET SYMBOLS CLEAR THE DATA-QUALITY GATE: XAGUSD, EURUSD, USDJPY,
  XAUEUR, US500 — all reach 2016, all 100% non-zero tick volume, all cover
  >=99.7% of XAUUSD's minutes. XTIUSD reaches only 2016-06-09 (SERVER_EXHAUSTED,
  a true depth limit) and cannot cover a full 2016 dev window.
→ EURUSD and USDJPY quote for ~39 minutes per day while XAUUSD does not, i.e.
  through part of gold's ~64-minute daily break. XAGUSD and XAUEUR are exactly
  session-identical to XAUUSD (0.00% extra).
```

## CAVEATS — READ BEFORE REUSING THESE NUMBERS

```
→ BATCH 1 FILES ARE RECONSTRUCTED, NOT ORIGINAL. Both probe runs wrote to a
  fixed filename, so Batch 2 overwrote Batch 1 on disk. The content here was
  recovered verbatim from the run output but was not re-observed. Treat as
  Observed-with-a-provenance-caveat.
→ BATCH 1'S `align_ref_covered_pct` IS INVALID. The v1 metric compared a
  symbol's bars against a FIXED 60-day window, so a symbol with 32 days of
  history scored ~58% purely because 32/60 = 53%. It measured history length
  while appearing to measure session alignment. GCZ26_CFD's 58.16% means
  nothing; its `align_sym_extra_pct` of 0.03% is the real signal and shows
  near-perfect session agreement. Fixed in v2, which intersects the window with
  each symbol's own range and reports `overlap_days`.
→ `reached_target = YES` WITH `stop_reason = REACHED_TARGET` IS A FLOOR, NOT A
  MEASUREMENT. The loader stops voluntarily at the target date, so those five
  symbols' true depth is UNMEASURED and may be far deeper. Only
  `SERVER_EXHAUSTED` reports a real limit.
→ TICK VOLUME COUNTS PRICE UPDATES, NOT CONTRACTS. A weak proxy whose level
  depends on broker feed throttling. Carry this flag on every result using it.
→ A PREDICTION THAT FAILED, RECORDED DELIBERATELY: US500 was expected to score
  well below 90% coverage on the reasoning that index CFDs follow equity cash
  hours. It scored 99.74%. This is a CFD on the S&P FUTURES, which track CME
  Globex (~23h/day), not the cash index. The session-mismatch concern was
  overstated for this broker's product set.
```

## NOT DONE

```
→ Batch 3 (GBPUSD, USDCAD, USDSEK, USDCHF) — deferred, not failed. Would
  complete an exact DXY. EURUSD + USDJPY alone are 71% of DXY by weight, and
  each extra series adds comparison-budget surface.
```
