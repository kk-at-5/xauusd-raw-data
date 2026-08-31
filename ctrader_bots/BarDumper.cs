// ============================================================================
// BarDumper.cs - generic OHLC extraction at any timeframe, any symbol
// PROJECT K - Cycle 1, step 5 (M1 pre-registration section 9)
//
// WHY A NEW BOT AND NOT AN EDIT TO TickVolumeDumper:
//   That bot writes a fixed schema (timestamp,close,tick_volume) at a fixed
//   timeframe for a fixed filename stem. M1 needs OHLC at H1 for three symbols.
//   Editing it in place would change a SHA-manifested file that Cycle 0's
//   validated tick-volume dataset was produced by, breaking reproducibility of
//   an already-closed task. Additive, not destructive.
//
// WHAT M1 NEEDS (pre-registration section 2.1): the OPEN price of the bar
//   timestamped 00:00Z and 02:00Z for XAUUSD, EURUSD, USDJPY, and 16:00Z for
//   XAUUSD. An H1 bar's Open IS the price at its OpenTime, so H1 is exactly
//   sufficient and is ~1/60 the size of M1 (a few MB, commits in full).
//
// THE CROSS-FEED GATE IS NOT OPTIONAL. Pull XAUUSD at H1 as well as the two FX
//   symbols. Gold is the only symbol for which both H1 and the existing M1
//   ground truth exist, so it is the ONLY available proof that cTrader's H1
//   aggregation is hour-aligned in UTC and price-identical to M1. If XAUUSD H1
//   opens do not match ctrader_m1/ exactly at 00:00Z / 02:00Z / 16:00Z, the FX
//   H1 pulls are DISCARDED and the extraction is redesigned. Run all three, then
//   gate; do not trust the FX files before the gold gate passes.
//
// THREE CYCLE-0 DEFECTS EXPLICITLY FIXED HERE:
//   1. A probe crashed the cBot host on a delisted symbol and cTrader restarted
//      it several hundred times. -> the symbol is resolved and null-checked
//      BEFORE any market-data call, and every path is inside try/catch.
//   2. A probe wrote to a fixed filename, so batch 2 overwrote batch 1.
//      -> every output filename carries SYMBOL and TIMEFRAME.
//   3. A personal Windows path was committed as a Parameter DefaultValue in six
//      .cs files. -> the default here is a relative folder name.
//
// OUTPUT
//   <OutputFolder>\<symbol>_<tf>.csv                       active (>= 2016-01-01)
//   <OutputFolder>\quarantine_pre2016\<symbol>_<tf>.csv    reference / burn-in only
//   <OutputFolder>\<symbol>_<tf>_manifest.csv
//   <OutputFolder>\<symbol>_<tf>_summary.txt
//
//   SCHEMA IS IDENTICAL TO ctrader_m1/ so the same validators apply:
//     timestamp_utc,datetime_utc,open,high,low,close
//   Prices are written at the symbol's own Digits (XAUUSD 2, EURUSD 5,
//   USDJPY 3), read from the API, never assumed.
//
// READ-ONLY BY CONSTRUCTION: no order API call appears in this file.
//
// USAGE: paste -> Build -> attach to ANY chart -> set Symbol + Timeframe -> Start.
//   Run three times: XAUUSD, EURUSD, USDJPY. Timeframe = Hour for all three.
//   Set "Market Closed" = true at a weekend so the final bar is complete.
// ============================================================================

using System;
using System.Globalization;
using System.IO;
using System.Text;
using cAlgo.API;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess, AddIndicators = false)]
    public class BarDumper : Robot
    {
        [Parameter("Output Folder", DefaultValue = "ctrader_h1")]
        public string OutputFolder { get; set; }

        [Parameter("Symbol", DefaultValue = "XAUUSD")]
        public string TargetSymbol { get; set; }

        [Parameter("Timeframe (Minute/Minute5/Minute15/Hour/Hour4/Daily)", DefaultValue = "Hour")]
        public string TimeframeName { get; set; }

        [Parameter("Quarantine Before (yyyy-MM-dd)", DefaultValue = "2016-01-01")]
        public string QuarantineBefore { get; set; }

        [Parameter("Max Load Calls", DefaultValue = 20000)]
        public int MaxLoadCalls { get; set; }

        [Parameter("Market Closed (include final bar)", DefaultValue = true)]
        public bool MarketClosed { get; set; }

        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
        private static readonly DateTime Epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        private static TimeFrame ResolveTimeframe(string name)
        {
            switch (name.Trim().ToLowerInvariant())
            {
                case "minute":   return TimeFrame.Minute;
                case "minute5":  return TimeFrame.Minute5;
                case "minute15": return TimeFrame.Minute15;
                case "hour":     return TimeFrame.Hour;
                case "hour4":    return TimeFrame.Hour4;
                case "daily":    return TimeFrame.Daily;
                default:         return null;
            }
        }

        protected override void OnStart()
        {
            try
            {
                // ---------- 0. RESOLVE AND VALIDATE BEFORE TOUCHING MARKET DATA ----------
                TimeFrame tf = ResolveTimeframe(TimeframeName);
                if (tf == null)
                {
                    Print("ABORT: unrecognised Timeframe '{0}'. Use Minute/Minute5/Minute15/Hour/Hour4/Daily. Nothing written.", TimeframeName);
                    Stop(); return;
                }

                Symbol sym = null;
                try { sym = Symbols.GetSymbol(TargetSymbol); }
                catch (Exception ex) { Print("ABORT: symbol '{0}' could not be resolved: {1}. Nothing written.", TargetSymbol, ex.Message); Stop(); return; }
                if (sym == null)
                {
                    Print("ABORT: symbol '{0}' is not on this account (delisted or misspelt). Nothing written.", TargetSymbol);
                    Stop(); return;
                }

                int digits = sym.Digits;
                string priceFmt = "0." + new string('0', digits);
                string tfTag = TimeframeName.Trim().ToLowerInvariant();
                string stem = TargetSymbol.ToLowerInvariant() + "_" + tfTag;

                string quarDir = Path.Combine(OutputFolder, "quarantine_pre2016");
                Directory.CreateDirectory(OutputFolder);
                Directory.CreateDirectory(quarDir);
                DateTime quarBefore = DateTime.ParseExact(QuarantineBefore, "yyyy-MM-dd", Inv);

                Bars bars = MarketData.GetBars(tf, TargetSymbol);
                if (bars == null || bars.Count == 0)
                {
                    Print("ABORT: no bars returned for {0} at {1}. Symbol may have no quotes. Nothing written.", TargetSymbol, TimeframeName);
                    Stop(); return;
                }

                Print("{0} {1}: digits={2}, starting depth load...", TargetSymbol, TimeframeName, digits);

                // ---------- 1. LOAD FULL DEPTH ----------
                int loadCalls = 0; bool exhausted = false;
                while (loadCalls < MaxLoadCalls)
                {
                    int loaded = bars.LoadMoreHistory();
                    loadCalls++;
                    if (loaded == 0) { exhausted = true; break; }
                    if (loadCalls % 50 == 0)
                        Print("load call {0}: {1} bars, earliest {2:yyyy-MM-dd HH:mm}Z",
                            loadCalls, bars.Count, bars.OpenTimes[0]);
                }
                if (!exhausted)
                {
                    Print("ABORT: load cap {0} hit before history exhausted - raise MaxLoadCalls. Nothing written.", MaxLoadCalls);
                    Stop(); return;
                }

                int lastIdx = MarketClosed ? bars.Count - 1 : bars.Count - 2;
                if (lastIdx < 0) { Print("ABORT: nothing to write after excluding the incomplete final bar."); Stop(); return; }

                Print("Loaded {0} bars ({1:yyyy-MM-dd HH:mm}Z -> {2:yyyy-MM-dd HH:mm}Z). Writing...",
                    lastIdx + 1, bars.OpenTimes[0], bars.OpenTimes[lastIdx]);

                // ---------- 2. WRITE: one active file, one quarantine file ----------
                string activePath = Path.Combine(OutputFolder, stem + ".csv");
                string quarPath   = Path.Combine(quarDir,      stem + ".csv");
                const string HEADER = "timestamp_utc,datetime_utc,open,high,low,close";

                var wA = new StreamWriter(activePath, false, Encoding.ASCII);
                var wQ = new StreamWriter(quarPath,   false, Encoding.ASCII);
                wA.WriteLine(HEADER); wQ.WriteLine(HEADER);

                int rowsA = 0, rowsQ = 0, dup = 0, nonMono = 0, breach = 0;
                DateTime firstA = DateTime.MinValue, lastA = DateTime.MinValue;
                DateTime firstQ = DateTime.MinValue, lastQ = DateTime.MinValue;

                for (int i = 0; i <= lastIdx; i++)
                {
                    DateTime t = bars.OpenTimes[i];
                    if (i > 0)
                    {
                        if (t == bars.OpenTimes[i - 1]) dup++;
                        else if (t < bars.OpenTimes[i - 1]) nonMono++;
                    }

                    double o = bars.OpenPrices[i], h = bars.HighPrices[i];
                    double l = bars.LowPrices[i],  c = bars.ClosePrices[i];
                    if (h < l || o > h || o < l || c > h || c < l) breach++;

                    long epoch = (long)(DateTime.SpecifyKind(t, DateTimeKind.Utc) - Epoch).TotalSeconds;
                    string line = string.Format(Inv, "{0},{1:yyyy-MM-dd HH:mm:ss},{2},{3},{4},{5}",
                        epoch, t,
                        o.ToString(priceFmt, Inv), h.ToString(priceFmt, Inv),
                        l.ToString(priceFmt, Inv), c.ToString(priceFmt, Inv));

                    if (t < quarBefore)
                    {
                        wQ.WriteLine(line); rowsQ++;
                        if (firstQ == DateTime.MinValue) firstQ = t;
                        lastQ = t;
                    }
                    else
                    {
                        wA.WriteLine(line); rowsA++;
                        if (firstA == DateTime.MinValue) firstA = t;
                        lastA = t;
                    }
                }
                wA.Flush(); wA.Dispose();
                wQ.Flush(); wQ.Dispose();

                var manifest = new StringBuilder();
                manifest.AppendLine("file,quarantined,rows,first_bar_utc,last_bar_utc");
                manifest.AppendLine(string.Format(Inv, "{0},0,{1},{2:yyyy-MM-dd HH:mm:ss},{3:yyyy-MM-dd HH:mm:ss}",
                    stem + ".csv", rowsA, firstA, lastA));
                manifest.AppendLine(string.Format(Inv, "quarantine_pre2016/{0},1,{1},{2:yyyy-MM-dd HH:mm:ss},{3:yyyy-MM-dd HH:mm:ss}",
                    stem + ".csv", rowsQ, firstQ, lastQ));
                System.IO.File.WriteAllText(Path.Combine(OutputFolder, stem + "_manifest.csv"),
                    manifest.ToString(), Encoding.ASCII);

                // ---------- 3. SUMMARY ----------
                var s = new StringBuilder();
                s.AppendLine("=== BAR DUMP SUMMARY ===");
                s.AppendLine(string.Format(Inv, "symbol             : {0}", TargetSymbol));
                s.AppendLine(string.Format(Inv, "timeframe          : {0}", TimeframeName));
                s.AppendLine(string.Format(Inv, "digits             : {0}", digits));
                s.AppendLine(string.Format(Inv, "run time (UTC)     : {0:yyyy-MM-dd HH:mm:ss}", Server.Time));
                s.AppendLine(string.Format(Inv, "history exhausted  : YES ({0} load calls)", loadCalls));
                s.AppendLine(string.Format(Inv, "bars written       : {0}", rowsA + rowsQ));
                s.AppendLine(string.Format(Inv, "  active (>= {0}) : {1}", QuarantineBefore, rowsA));
                s.AppendLine(string.Format(Inv, "  quarantined      : {0}", rowsQ));
                s.AppendLine(string.Format(Inv, "first bar          : {0:yyyy-MM-dd HH:mm}Z", bars.OpenTimes[0]));
                s.AppendLine(string.Format(Inv, "last bar written   : {0:yyyy-MM-dd HH:mm}Z", bars.OpenTimes[lastIdx]));
                s.AppendLine(string.Format(Inv, "duplicate ts       : {0}", dup));
                s.AppendLine(string.Format(Inv, "non-monotonic ts   : {0}", nonMono));
                s.AppendLine(string.Format(Inv, "OHLC breaches      : {0}", breach));
                s.AppendLine();
                s.AppendLine("NOT VALIDATED YET. This file is not a lever until it clears:");
                s.AppendLine("  1. the CROSS-FEED GATE (XAUUSD H1 opens vs ctrader_m1/ at 00:00/02:00/16:00Z), and");
                s.AppendLine("  2. the per-source DATA-QUALITY GATE (coverage, gaps, alignment to XAUUSD days).");
                s.AppendLine("Rows dated 2023-01-01 onward are HOLDOUT under the Cycle 1 seal and are not");
                s.AppendLine("to be examined. The seal is a date range binding EVERY source, not only XAUUSD.");
                System.IO.File.WriteAllText(Path.Combine(OutputFolder, stem + "_summary.txt"),
                    s.ToString(), Encoding.ASCII);

                Print("DUMP COMPLETE {0} {1}: active {2}, quarantine {3}, dup {4}, nonmono {5}, breach {6}",
                    TargetSymbol, TimeframeName, rowsA, rowsQ, dup, nonMono, breach);
            }
            catch (Exception ex)
            {
                Print("BAR DUMP ERROR: {0}", ex);
            }
            Stop();
        }
    }
}
