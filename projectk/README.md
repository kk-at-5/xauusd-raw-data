# PROJECT K — XAUUSD monthly-consistency, information-source-open

Authority: PROJECT_K_RULES.md (rulebook) + HANDOVER_NEXT_PROJECT.md (inherited
facts and closure record). Nothing in this directory overrides either.

Organised by cycle. State, logs and derived data are NOT committed — builders
and verifiers are, and they regenerate their outputs on demand.

## cycle0_feasibility
Data-feasibility inventory. No strategy work.

- `session_calendar_proof.py` — TASK 1. Derives the daily session structure
  from the 1M bulk and establishes the largest fixed-UTC trading window that
  never straddles the swap rollover in any DST regime. Self-verifies against
  frozen invariants and reports MISMATCH rather than proceeding.

  Run: `python3 session_calendar_proof.py ../../ctrader_m1`
