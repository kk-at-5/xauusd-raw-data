// ============================================================================
// HistoryDepthProbe.cs — XAUUSD m1 history depth probe (IC Markets / cTrader)
// PROJECT REBIRTH — pre-migration probe, per ratified P2 scope (31 Jul 2026)
//
// PURPOSE (observe, never assume):
//   1. How far back XAUUSD m1 history actually goes on this server
//   2. Whether bars are bid-based (live tick comparison, not presumption)
//   3. Completeness: per-trading-day candle census, duplicates,
//      monotonicity, OHLC consistency
//
// OUTPUTS (to OutputFolder):
//   probe_summary.txt   — headline numbers, read this first
//   day_census.csv      — trading_day, bar_count, first_bar_utc, last_bar_utc
//   sample_20260701.csv — one POOL trading day's bars for manual TV
//                         spot-check (pool day only — holdout days untouched)
//   bid_check.csv       — per-tick barClose vs Bid vs Ask comparison
//
// USAGE: paste into cTrader Automate editor -> Build -> attach to a XAUUSD
//   chart (any timeframe) -> Start. Depth load may take several minutes for
//   multi-year m1 history; progress prints to the log every 100 load calls.
//   Bid check needs LIVE TICKS -> run during market hours. If the market is
//   closed it times out and reports the check as NOT COMPLETED (rerun at
//   next open for that part only; depth + census remain valid).
// ============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using cAlgo.API;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class HistoryDepthProbe : Robot
    {
        [Parameter("Output Folder", DefaultValue = @"C:\Users\kmykr\xauusd-project\ctrader_probe")]
        public string OutputFolder { get; set; }

        [Parameter("Max LoadMoreHistory Calls", DefaultValue = 20000)]
        public int MaxLoadCalls { get; set; }

        [Parameter("Bid Check Ticks", DefaultValue = 30)]
        public int BidCheckTicks { get; set; }

        [Parameter("Bid Check Timeout (sec)", DefaultValue = 180)]
        public int BidCheckTimeoutSec { get; set; }

        [Parameter("Sample Trading Day (yyyy-MM-dd)", DefaultValue = "2026-07-01")]
        public string SampleDay { get; set; }

        private Bars _m1;
        private bool _probeDone;
        private bool _finalized;
        private int _loadCalls;
        private bool _exhausted;
        private int _tickChecks;
        private int _tickMatches;
        private int _timerSeconds;
        private readonly StringBuilder _tickLog = new StringBuilder();
        private readonly StringBuilder _summary = new StringBuilder();
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        private class DayCensus
        {
            public int Count;
            public DateTime First;
            public DateTime Last;
        }

        protected override void OnStart()
        {
            try
            {
                Directory.CreateDirectory(OutputFolder);
                _m1 = MarketData.GetBars(TimeFrame.Minute);

                Print("PROBE START {0} | initial bars={1} | earliest={2:yyyy-MM-dd HH:mm}Z",
                    SymbolName, _m1.Count, _m1.OpenTimes[0]);

                // ------------------------------------------------------------
                // 1. DEPTH — load history until the server refuses more
                // ------------------------------------------------------------
                while (_loadCalls < MaxLoadCalls)
                {
                    int loaded = _m1.LoadMoreHistory();
                    _loadCalls++;
                    if (loaded == 0) { _exhausted = true; break; }
                    if (_loadCalls % 100 == 0)
                        Print("load call {0}: {1} bars, earliest {2:yyyy-MM-dd HH:mm}Z",
                            _loadCalls, _m1.Count, _m1.OpenTimes[0]);
                }

                int n = _m1.Count;
                // Exclude the last bar: it may still be forming if market open.
                int lastIdx = n - 2;
                DateTime earliest = _m1.OpenTimes[0];
                DateTime latest = _m1.OpenTimes[lastIdx];

                _summary.AppendLine("=== HISTORY DEPTH PROBE SUMMARY ===");
                _summary.AppendLine(string.Format(Inv, "symbol            : {0}", SymbolName));
                _summary.AppendLine(string.Format(Inv, "run time (UTC)    : {0:yyyy-MM-dd HH:mm:ss}", Server.Time));
                _summary.AppendLine(string.Format(Inv, "load calls used   : {0} (cap {1})", _loadCalls, MaxLoadCalls));
                _summary.AppendLine(string.Format(Inv, "history exhausted : {0}", _exhausted
                    ? "YES (LoadMoreHistory returned 0 - true server depth reached)"
                    : "NO  (call cap hit - TRUE DEPTH IS DEEPER, raise MaxLoadCalls)"));
                _summary.AppendLine(string.Format(Inv, "total m1 bars     : {0}", n));
                _summary.AppendLine(string.Format(Inv, "earliest bar open : {0:yyyy-MM-dd HH:mm}Z", earliest));
                _summary.AppendLine(string.Format(Inv, "latest closed bar : {0:yyyy-MM-dd HH:mm}Z", latest));
                _summary.AppendLine(string.Format(Inv, "span              : {0:0.1} calendar days",
                    (latest - earliest).TotalDays));
                _summary.AppendLine();

                // ------------------------------------------------------------
                // 2. STRUCTURAL SCAN + PER-TRADING-DAY CENSUS
                //    Trading-day label = calendar date of (open_time + 2h).
                //    This maps 22:02 opens (summer) AND 23:02 opens (winter)
                //    to the following date, and 20:5x / 21:5x closes to their
                //    own date, i.e. day labelled by its CLOSE date in both
                //    DST regimes. Structural fields only - no EMAs, no
                //    crosses, no direction statistics.
                // ------------------------------------------------------------
                int dup = 0, nonMono = 0, ohlcViol = 0;
                var days = new SortedDictionary<string, DayCensus>(StringComparer.Ordinal);

                for (int i = 0; i <= lastIdx; i++)
                {
                    DateTime t = _m1.OpenTimes[i];
                    if (i > 0)
                    {
                        if (t == _m1.OpenTimes[i - 1]) dup++;
                        else if (t < _m1.OpenTimes[i - 1]) nonMono++;
                    }

                    double o = _m1.OpenPrices[i], h = _m1.HighPrices[i];
                    double l = _m1.LowPrices[i], c = _m1.ClosePrices[i];
                    if (l > Math.Min(o, c) + 1e-9 || h < Math.Max(o, c) - 1e-9) ohlcViol++;

                    string day = t.AddHours(2).ToString("yyyy-MM-dd", Inv);
                    DayCensus dc;
                    if (!days.TryGetValue(day, out dc))
                    {
                        dc = new DayCensus { Count = 0, First = t, Last = t };
                        days[day] = dc;
                    }
                    dc.Count++;
                    if (t < dc.First) dc.First = t;
                    if (t > dc.Last) dc.Last = t;
                }

                var censusPath = Path.Combine(OutputFolder, "day_census.csv");
                using (var w = new StreamWriter(censusPath, false, Encoding.ASCII))
                {
                    w.WriteLine("trading_day,bar_count,first_bar_utc,last_bar_utc");
                    foreach (var kv in days)
                        w.WriteLine(string.Format(Inv, "{0},{1},{2:yyyy-MM-dd HH:mm:ss},{3:yyyy-MM-dd HH:mm:ss}",
                            kv.Key, kv.Value.Count, kv.Value.First, kv.Value.Last));
                }

                var counts = days.Values.Select(d => d.Count).OrderBy(x => x).ToList();
                int median = counts.Count > 0 ? counts[counts.Count / 2] : 0;
                int under1300 = counts.Count(x => x < 1300);
                int weekendLabelled = days.Keys.Count(k =>
                {
                    var d = DateTime.ParseExact(k, "yyyy-MM-dd", Inv);
                    return d.DayOfWeek == DayOfWeek.Saturday || d.DayOfWeek == DayOfWeek.Sunday;
                });

                _summary.AppendLine("=== STRUCTURAL SCAN (completed bars only) ===");
                _summary.AppendLine(string.Format(Inv, "duplicate timestamps      : {0}", dup));
                _summary.AppendLine(string.Format(Inv, "non-monotonic timestamps  : {0}", nonMono));
                _summary.AppendLine(string.Format(Inv, "OHLC consistency breaches : {0}", ohlcViol));
                _summary.AppendLine(string.Format(Inv, "trading days in census    : {0}", days.Count));
                _summary.AppendLine(string.Format(Inv, "bars/day min|median|max   : {0} | {1} | {2}",
                    counts.Count > 0 ? counts[0] : 0, median, counts.Count > 0 ? counts[counts.Count - 1] : 0));
                _summary.AppendLine(string.Format(Inv, "days with <1300 bars      : {0} (see day_census.csv - early closes, partial edge days, or gaps)", under1300));
                _summary.AppendLine(string.Format(Inv, "weekend-labelled days     : {0} (expect 0; nonzero = labelling or feed anomaly)", weekendLabelled));
                _summary.AppendLine();

                // ------------------------------------------------------------
                // 3. SAMPLE DUMP — one POOL trading day only (already
                //    unblinded exploration data; holdout days never touched).
                //    Schema: OHLC only. EMAs are NOT dumped - they are
                //    recomputed downstream with >=200-candle burn-in.
                // ------------------------------------------------------------
                int sampleRows = 0;
                var samplePath = Path.Combine(OutputFolder,
                    "sample_" + SampleDay.Replace("-", "") + ".csv");
                using (var w = new StreamWriter(samplePath, false, Encoding.ASCII))
                {
                    w.WriteLine("timestamp_utc,datetime_utc,open,high,low,close");
                    for (int i = 0; i <= lastIdx; i++)
                    {
                        DateTime t = _m1.OpenTimes[i];
                        if (t.AddHours(2).ToString("yyyy-MM-dd", Inv) != SampleDay) continue;
                        long epoch = (long)(DateTime.SpecifyKind(t, DateTimeKind.Utc)
                                     - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds;
                        w.WriteLine(string.Format(Inv, "{0},{1:yyyy-MM-dd HH:mm:ss},{2:0.00},{3:0.00},{4:0.00},{5:0.00}",
                            epoch, t, _m1.OpenPrices[i], _m1.HighPrices[i],
                            _m1.LowPrices[i], _m1.ClosePrices[i]));
                        sampleRows++;
                    }
                }
                _summary.AppendLine("=== SAMPLE DUMP ===");
                _summary.AppendLine(string.Format(Inv, "sample day {0}: {1} rows -> {2}",
                    SampleDay, sampleRows, samplePath));
                _summary.AppendLine();

                Print("Depth + census + sample done. Entering bid-check phase ({0} ticks or {1}s timeout)...",
                    BidCheckTicks, BidCheckTimeoutSec);

                _tickLog.AppendLine("tick_time_utc,forming_bar_close,bid,ask,close_eq_bid");
                _probeDone = true;
                Timer.Start(1);
            }
            catch (Exception ex)
            {
                Print("PROBE ERROR in OnStart: {0}", ex.Message);
                _summary.AppendLine("ERROR: " + ex);
                FinalizeProbe();
            }
        }

        // ----------------------------------------------------------------
        // 4. BID-BASIS CHECK — if the m1 feed is bid-based, the forming
        //    bar's close tracks Symbol.Bid tick-by-tick (and differs from
        //    Ask by the live spread). Observed, not presumed.
        // ----------------------------------------------------------------
        protected override void OnTick()
        {
            if (!_probeDone || _finalized) return;

            double barClose = _m1.ClosePrices.LastValue;
            double bid = Symbol.Bid;
            double ask = Symbol.Ask;
            bool match = Math.Abs(barClose - bid) < Symbol.TickSize / 2.0;

            _tickChecks++;
            if (match) _tickMatches++;
            _tickLog.AppendLine(string.Format(Inv, "{0:yyyy-MM-dd HH:mm:ss.fff},{1:0.00},{2:0.00},{3:0.00},{4}",
                Server.Time, barClose, bid, ask, match ? 1 : 0));

            if (_tickChecks >= BidCheckTicks) FinalizeProbe();
        }

        protected override void OnTimer()
        {
            if (_finalized) return;
            _timerSeconds++;
            if (_timerSeconds >= BidCheckTimeoutSec) FinalizeProbe();
        }

        private void FinalizeProbe()
        {
            if (_finalized) return;
            _finalized = true;
            try { Timer.Stop(); } catch { }

            try
            {
                _summary.AppendLine("=== BID-BASIS CHECK ===");
                if (_tickChecks == 0)
                {
                    _summary.AppendLine("NOT COMPLETED - no live ticks received (market closed?).");
                    _summary.AppendLine("Depth/census/sample above remain valid. Rerun during market hours for this check only.");
                }
                else
                {
                    _summary.AppendLine(string.Format(Inv, "ticks compared            : {0}", _tickChecks));
                    _summary.AppendLine(string.Format(Inv, "forming close == Bid      : {0} of {1}", _tickMatches, _tickChecks));
                    _summary.AppendLine(_tickMatches == _tickChecks
                        ? "VERDICT: consistent with BID-based bars (see bid_check.csv; Ask column should differ by live spread)"
                        : "VERDICT: NOT cleanly bid-matching - inspect bid_check.csv before any migration step");
                }

                System.IO.File.WriteAllText(Path.Combine(OutputFolder, "bid_check.csv"), _tickLog.ToString());
                System.IO.File.WriteAllText(Path.Combine(OutputFolder, "probe_summary.txt"), _summary.ToString());
                Print("PROBE COMPLETE. Read probe_summary.txt in {0}", OutputFolder);
            }
            catch (Exception ex)
            {
                Print("PROBE ERROR in FinalizeProbe: {0}", ex.Message);
            }
            Stop();
        }
    }
}
