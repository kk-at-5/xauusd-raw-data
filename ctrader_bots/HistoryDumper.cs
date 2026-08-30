// ============================================================================
// HistoryDumper.cs - XAUUSD m1 bulk history extraction (IC Markets / cTrader)
// PROJECT REBIRTH - P2 step 3, ratified 31 Jul 2026
//
// SCOPE (ratified): full server depth (observed: Oct 2012). Active dataset =
//   2016-01-01 onward. Pre-2016 bars are written to a quarantine subfolder:
//   reference / EMA burn-in seeding ONLY, never analysis.
//
// OUTPUT:
//   <OutputFolder>\xauusd_m1_YYYYMM.csv                (active, monthly)
//   <OutputFolder>\quarantine_pre2016\xauusd_m1_YYYYMM.csv
//   <OutputFolder>\manifest.csv    (per file: rows, first/last bar UTC)
//   <OutputFolder>\dump_summary.txt
//
// SCHEMA: timestamp_utc,datetime_utc,open,high,low,close  (OHLC only -
//   EMAs are recomputed downstream with >=200-candle burn-in, identical
//   code path for both feeds: instrument consistency)
//
// USAGE: paste -> Build -> attach to XAUUSD chart -> Start. Runs to
//   completion in one pass and stops itself. Expect the history load to
//   take ~5 min (observed in probe) plus a short write phase; progress
//   prints per month written.
// ============================================================================

using System;
using System.Globalization;
using System.IO;
using System.Text;
using cAlgo.API;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class HistoryDumper : Robot
    {
        [Parameter("Output Folder", DefaultValue = @"C:\Users\kmykr\xauusd-project\ctrader_bulk")]
        public string OutputFolder { get; set; }

        [Parameter("Quarantine Before (yyyy-MM-dd)", DefaultValue = "2016-01-01")]
        public string QuarantineBefore { get; set; }

        [Parameter("Max LoadMoreHistory Calls", DefaultValue = 20000)]
        public int MaxLoadCalls { get; set; }

        [Parameter("Market Closed (include final bar)", DefaultValue = false)]
        public bool MarketClosed { get; set; }

        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
        private static readonly DateTime Epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        protected override void OnStart()
        {
            try
            {
                string quarDir = Path.Combine(OutputFolder, "quarantine_pre2016");
                Directory.CreateDirectory(OutputFolder);
                Directory.CreateDirectory(quarDir);

                DateTime quarBefore = DateTime.ParseExact(QuarantineBefore, "yyyy-MM-dd", Inv);
                Bars m1 = MarketData.GetBars(TimeFrame.Minute);

                // ---------- 1. LOAD FULL DEPTH ----------
                int loadCalls = 0;
                bool exhausted = false;
                while (loadCalls < MaxLoadCalls)
                {
                    int loaded = m1.LoadMoreHistory();
                    loadCalls++;
                    if (loaded == 0) { exhausted = true; break; }
                    if (loadCalls % 100 == 0)
                        Print("load call {0}: {1} bars, earliest {2:yyyy-MM-dd HH:mm}Z",
                            loadCalls, m1.Count, m1.OpenTimes[0]);
                }
                if (!exhausted)
                {
                    Print("ABORT: load cap {0} hit before history exhausted - raise MaxLoadCalls and rerun. Nothing written.", MaxLoadCalls);
                    Stop();
                    return;
                }

                // Exclude the final bar only if the market may still be forming it;
                // on a post-close run (MarketClosed=true) the final bar is complete.
                int lastIdx = MarketClosed ? m1.Count - 1 : m1.Count - 2;
                Print("Loaded {0} bars ({1:yyyy-MM-dd HH:mm}Z -> {2:yyyy-MM-dd HH:mm}Z). Writing monthly files...",
                    lastIdx + 1, m1.OpenTimes[0], m1.OpenTimes[lastIdx]);

                // ---------- 2. SINGLE-PASS MONTHLY WRITE ----------
                var manifest = new StringBuilder();
                manifest.AppendLine("file,quarantined,rows,first_bar_utc,last_bar_utc");

                StreamWriter w = null;
                string curMonth = null;
                string curFile = null;
                bool curQuar = false;
                int curRows = 0;
                DateTime curFirst = DateTime.MinValue, curLast = DateTime.MinValue;
                int totalRows = 0, quarRows = 0, fileCount = 0;
                int dup = 0, nonMono = 0, ohlcViol = 0;

                for (int i = 0; i <= lastIdx; i++)
                {
                    DateTime t = m1.OpenTimes[i];

                    if (i > 0)
                    {
                        if (t == m1.OpenTimes[i - 1]) dup++;
                        else if (t < m1.OpenTimes[i - 1]) nonMono++;
                    }
                    double o = m1.OpenPrices[i], h = m1.HighPrices[i];
                    double l = m1.LowPrices[i], c = m1.ClosePrices[i];
                    if (l > Math.Min(o, c) + 1e-9 || h < Math.Max(o, c) - 1e-9) ohlcViol++;

                    string month = t.ToString("yyyyMM", Inv);
                    if (month != curMonth)
                    {
                        if (w != null)
                        {
                            w.Flush(); w.Dispose();
                            manifest.AppendLine(string.Format(Inv, "{0},{1},{2},{3:yyyy-MM-dd HH:mm:ss},{4:yyyy-MM-dd HH:mm:ss}",
                                curFile, curQuar ? 1 : 0, curRows, curFirst, curLast));
                            Print("wrote {0}{1}: {2} rows", curQuar ? "quarantine\\" : "", curFile, curRows);
                        }
                        curMonth = month;
                        curQuar = t < quarBefore;
                        curFile = "xauusd_m1_" + month + ".csv";
                        string dir = curQuar ? quarDir : OutputFolder;
                        w = new StreamWriter(Path.Combine(dir, curFile), false, Encoding.ASCII);
                        w.WriteLine("timestamp_utc,datetime_utc,open,high,low,close");
                        curRows = 0;
                        curFirst = t;
                        fileCount++;
                    }

                    long epoch = (long)(DateTime.SpecifyKind(t, DateTimeKind.Utc) - Epoch).TotalSeconds;
                    w.WriteLine(string.Format(Inv, "{0},{1:yyyy-MM-dd HH:mm:ss},{2:0.00},{3:0.00},{4:0.00},{5:0.00}",
                        epoch, t, o, h, l, c));
                    curRows++;
                    curLast = t;
                    totalRows++;
                    if (curQuar) quarRows++;
                }
                if (w != null)
                {
                    w.Flush(); w.Dispose();
                    manifest.AppendLine(string.Format(Inv, "{0},{1},{2},{3:yyyy-MM-dd HH:mm:ss},{4:yyyy-MM-dd HH:mm:ss}",
                        curFile, curQuar ? 1 : 0, curRows, curFirst, curLast));
                    Print("wrote {0}{1}: {2} rows", curQuar ? "quarantine\\" : "", curFile, curRows);
                }

                System.IO.File.WriteAllText(Path.Combine(OutputFolder, "manifest.csv"), manifest.ToString());

                // ---------- 3. SUMMARY ----------
                var s = new StringBuilder();
                s.AppendLine("=== BULK DUMP SUMMARY ===");
                s.AppendLine(string.Format(Inv, "symbol             : {0}", SymbolName));
                s.AppendLine(string.Format(Inv, "run time (UTC)     : {0:yyyy-MM-dd HH:mm:ss}", Server.Time));
                s.AppendLine(string.Format(Inv, "history exhausted  : YES ({0} load calls)", loadCalls));
                s.AppendLine(string.Format(Inv, "bars written       : {0}", totalRows));
                s.AppendLine(string.Format(Inv, "  active (>= {0})  : {1}", QuarantineBefore, totalRows - quarRows));
                s.AppendLine(string.Format(Inv, "  quarantined      : {0}", quarRows));
                s.AppendLine(string.Format(Inv, "monthly files      : {0}", fileCount));
                s.AppendLine(string.Format(Inv, "first bar          : {0:yyyy-MM-dd HH:mm}Z", m1.OpenTimes[0]));
                s.AppendLine(string.Format(Inv, "last bar written   : {0:yyyy-MM-dd HH:mm}Z", m1.OpenTimes[lastIdx]));
                s.AppendLine(string.Format(Inv, "duplicate ts       : {0}", dup));
                s.AppendLine(string.Format(Inv, "non-monotonic ts   : {0}", nonMono));
                s.AppendLine(string.Format(Inv, "OHLC breaches      : {0}", ohlcViol));
                System.IO.File.WriteAllText(Path.Combine(OutputFolder, "dump_summary.txt"), s.ToString());

                Print("DUMP COMPLETE. {0} files, {1} rows. Read dump_summary.txt in {2}", fileCount, totalRows, OutputFolder);
            }
            catch (Exception ex)
            {
                Print("DUMP ERROR: {0}", ex);
            }
            Stop();
        }
    }
}
