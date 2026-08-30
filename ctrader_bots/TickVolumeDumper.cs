// ============================================================================
// TickVolumeDumper.cs - XAUUSD m1 TICK VOLUME companion extraction
// PROJECT K - Cycle 0 data-feasibility inventory, TASK 2B
//
// WHY THIS IS A COMPANION, NOT A REPLACEMENT:
//   ctrader_m1/ is SHA-manifested ground truth that two closed projects and
//   Project K's Task 1 assert against. Overwriting it with a new schema would
//   silently break those invariants. This bot writes a SEPARATE dataset that
//   joins to it on timestamp.
//
// SCHEMA: timestamp_utc,close,tick_volume
//   `close` is carried ONLY so the companion can be validated against
//   ctrader_m1/ - if the broker has revised any historical bar since the
//   2026-08-01 dump, the close comparison exposes it. It is not new data.
//
// TICK VOLUME CAVEAT (carry this on every result that uses it):
//   cTrader TickVolume counts PRICE UPDATES, not contracts traded. It is a
//   WEAK PROXY for true volume and its level depends on the broker's feed
//   throttling, which can change over time without notice. Treat any
//   multi-year level comparison with suspicion; period-over-period changes
//   within a stable regime are the more defensible use.
//
// OUTPUT (mirrors ctrader_m1/ layout exactly so validation is file-for-file):
//   <OutputFolder>\xauusd_tv_YYYYMM.csv
//   <OutputFolder>\quarantine_pre2016\xauusd_tv_YYYYMM.csv
//   <OutputFolder>\tv_manifest.csv
//   <OutputFolder>\tv_dump_summary.txt
//
// READ-ONLY BY CONSTRUCTION: no order API call appears in this file.
//
// USAGE: paste -> Build -> attach to a XAUUSD chart -> Start.
//   Set "Market Closed" = true when running at a weekend (the final bar is
//   then complete and safe to include). Expect ~5-10 min.
// ============================================================================

using System;
using System.Globalization;
using System.IO;
using System.Text;
using cAlgo.API;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess, AddIndicators = false)]
    public class TickVolumeDumper : Robot
    {
        [Parameter("Output Folder", DefaultValue = @"C:\Users\kmykr\xauusd-project\ctrader_m1_tickvol")]
        public string OutputFolder { get; set; }

        [Parameter("Symbol", DefaultValue = "XAUUSD")]
        public string TargetSymbol { get; set; }

        [Parameter("Quarantine Before (yyyy-MM-dd)", DefaultValue = "2016-01-01")]
        public string QuarantineBefore { get; set; }

        [Parameter("Max Load Calls", DefaultValue = 20000)]
        public int MaxLoadCalls { get; set; }

        [Parameter("Market Closed (include final bar)", DefaultValue = true)]
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
                Bars m1 = MarketData.GetBars(TimeFrame.Minute, TargetSymbol);

                // ---------- 1. LOAD FULL DEPTH ----------
                int loadCalls = 0; bool exhausted = false;
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
                    Print("ABORT: load cap {0} hit before history exhausted - raise MaxLoadCalls. Nothing written.", MaxLoadCalls);
                    Stop(); return;
                }

                int lastIdx = MarketClosed ? m1.Count - 1 : m1.Count - 2;
                Print("Loaded {0} bars ({1:yyyy-MM-dd HH:mm}Z -> {2:yyyy-MM-dd HH:mm}Z). Writing...",
                    lastIdx + 1, m1.OpenTimes[0], m1.OpenTimes[lastIdx]);

                // ---------- 2. SINGLE-PASS MONTHLY WRITE ----------
                var manifest = new StringBuilder();
                manifest.AppendLine("file,quarantined,rows,first_bar_utc,last_bar_utc");

                StreamWriter w = null;
                string curMonth = null, curFile = null;
                bool curQuar = false;
                int curRows = 0, totalRows = 0, quarRows = 0, fileCount = 0;
                int dup = 0, nonMono = 0, zeroVol = 0;
                long volSum = 0;
                DateTime curFirst = DateTime.MinValue, curLast = DateTime.MinValue;

                for (int i = 0; i <= lastIdx; i++)
                {
                    DateTime t = m1.OpenTimes[i];
                    if (i > 0)
                    {
                        if (t == m1.OpenTimes[i - 1]) dup++;
                        else if (t < m1.OpenTimes[i - 1]) nonMono++;
                    }

                    double tv = m1.TickVolumes[i];
                    if (tv <= 0) zeroVol++;
                    volSum += (long)tv;

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
                        curFile = "xauusd_tv_" + month + ".csv";
                        w = new StreamWriter(Path.Combine(curQuar ? quarDir : OutputFolder, curFile), false, Encoding.ASCII);
                        w.WriteLine("timestamp_utc,close,tick_volume");
                        curRows = 0; curFirst = t; fileCount++;
                    }

                    long epoch = (long)(DateTime.SpecifyKind(t, DateTimeKind.Utc) - Epoch).TotalSeconds;
                    w.WriteLine(string.Format(Inv, "{0},{1:0.00},{2:0}", epoch, m1.ClosePrices[i], tv));
                    curRows++; curLast = t; totalRows++;
                    if (curQuar) quarRows++;
                }
                if (w != null)
                {
                    w.Flush(); w.Dispose();
                    manifest.AppendLine(string.Format(Inv, "{0},{1},{2},{3:yyyy-MM-dd HH:mm:ss},{4:yyyy-MM-dd HH:mm:ss}",
                        curFile, curQuar ? 1 : 0, curRows, curFirst, curLast));
                    Print("wrote {0}{1}: {2} rows", curQuar ? "quarantine\\" : "", curFile, curRows);
                }

                System.IO.File.WriteAllText(Path.Combine(OutputFolder, "tv_manifest.csv"), manifest.ToString(), Encoding.ASCII);

                // ---------- 3. SUMMARY ----------
                var s = new StringBuilder();
                s.AppendLine("=== TICK VOLUME COMPANION DUMP SUMMARY ===");
                s.AppendLine(string.Format(Inv, "symbol             : {0}", TargetSymbol));
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
                s.AppendLine(string.Format(Inv, "ZERO tick volume   : {0} bars ({1:0.000}%)",
                    zeroVol, totalRows > 0 ? 100.0 * zeroVol / totalRows : 0));
                s.AppendLine(string.Format(Inv, "mean tick volume   : {0:0.00} ticks/bar",
                    totalRows > 0 ? (double)volSum / totalRows : 0));
                s.AppendLine();
                s.AppendLine("NOT VALIDATED YET. Run validate_tickvol.py against ctrader_m1/ before use:");
                s.AppendLine("it proves timestamp-for-timestamp and close-for-close agreement, i.e. that");
                s.AppendLine("this is the same feed and that no historical bar has been revised.");
                System.IO.File.WriteAllText(Path.Combine(OutputFolder, "tv_dump_summary.txt"), s.ToString(), Encoding.ASCII);

                Print("TV DUMP COMPLETE. {0} files, {1} rows. Read tv_dump_summary.txt", fileCount, totalRows);
            }
            catch (Exception ex)
            {
                Print("TV DUMP ERROR: {0}", ex);
            }
            Stop();
        }
    }
}
