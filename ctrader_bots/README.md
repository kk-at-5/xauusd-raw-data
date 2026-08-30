# cTrader extraction bots (cAlgo / C#)

Repo-level shared tooling. NOT project-specific — these produced the 1M bulk
in ctrader_m1/ and are the extraction path for any future series.

| bot | purpose |
|---|---|
| HistoryDepthProbe.cs | observes server history depth, bid-basis, per-day census. Run BEFORE any bulk pull. |
| HistoryDumper.cs     | bulk 1M extraction to monthly CSVs + manifest + summary. Produced ctrader_m1/. |
| SpreadLogger.cs      | samples live bid/ask every N seconds. Produced spread/. |

USAGE: paste into cTrader Automate editor -> Build -> attach to the target
symbol's chart -> Start. All three are READ-ONLY: no order API call appears
in any of them. AccessRights.FullAccess is a .NET file-write permission only.

NOTE: HistoryDumper aborts and writes nothing if the load cap is hit before
history is exhausted. That is deliberate — a partial pull must never be
mistaken for a complete one.
