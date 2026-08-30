// ============================================================================
// SymbolDepthProbe_v2.cs - depth / tick-volume / session-alignment probe
// PROJECT K - Cycle 0 data-feasibility inventory, TASK 2C  (v2)
//
// CHANGES FROM v1 (all three were bugs found in the Batch-1 run):
//   1. ALIGNMENT IS NOW INTERSECTED with each symbol's OWN date range.
//      v1 compared a symbol's bars against a fixed 60-day reference window.
//      A symbol with only 32 days of history scored 58% coverage purely
//      because 32/60 = 53% - an artifact, not a session mismatch. The metric
//      measured history length while appearing to measure alignment.
//      v2 reports overlap_days so short history is VISIBLE as short.
//   2. CSV IS WRITTEN AFTER EVERY SYMBOL, not once at the end. A crash now
//      costs one symbol's row, not the entire run.
//   3. Symbols.GetSymbol IS NO LONGER CALLED. On a delisted symbol it throws
//      a HOST-level dispatcher exception; catching it does not restore host
//      state, so the subsequent Stop() crashed the instance and cTrader
//      auto-restarted it in a loop. MarketData.GetBars is used directly.
//      Cost of the change: digits/pip/tick are no longer reported. They are
//      not needed for a symbol used as INFORMATION rather than traded.
//
// OUTPUT (to OutputFolder):
//   depth_probe.csv       - one row per symbol, rewritten after each symbol
//   depth_probe_log.txt   - readable summary, rewritten after each symbol
//
// READ-ONLY BY CONSTRUCTION: no order API call appears in this file.
//
// USAGE: paste -> Build -> attach to ANY chart -> set Symbols -> Start.
//   Run in BATCHES. If the instance ever restarts itself, STOP IT MANUALLY -
//   partial results are already on disk.
// ============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using cAlgo.API;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess, AddIndicators = false)]
    public class SymbolDepthProbeV2 : Robot
    {
        [Parameter("Symbols (comma separated)", DefaultValue = "XAGUSD,EURUSD,USDJPY,XAUEUR,US500,XTIUSD")]
        public string SymbolList { get; set; }

        [Parameter("Output Folder", DefaultValue = @"C:\Users\kmykr\xauusd-project\projectk_probe")]
        public string OutputFolder { get; set; }

        [Parameter("Target Date (yyyy-MM-dd)", DefaultValue = "2016-01-01")]
        public string TargetDate { get; set; }

        [Parameter("Max Load Calls Per Symbol", DefaultValue = 3000)]
        public int MaxLoadCalls { get; set; }

        [Parameter("Alignment Window (days)", DefaultValue = 60)]
        public int AlignmentDays { get; set; }

        [Parameter("Reference Symbol", DefaultValue = "XAUUSD")]
        public string RefSymbol { get; set; }

        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
        private static readonly DateTime Epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        private readonly StringBuilder _csv = new StringBuilder();
        private readonly StringBuilder _log = new StringBuilder();
        private HashSet<long> _refMin = new HashSet<long>();
        private List<long> _refSorted = new List<long>();
        private DateTime _alignFrom, _alignTo;

        protected override void OnStart()
        {
            try
            {
                Directory.CreateDirectory(OutputFolder);
                DateTime target = DateTime.ParseExact(TargetDate.Trim(), "yyyy-MM-dd", Inv);
                _alignTo = Server.Time;
                _alignFrom = _alignTo.AddDays(-AlignmentDays);

                _log.AppendLine("=== SYMBOL DEPTH / VOLUME / ALIGNMENT PROBE (v2) ===");
                _log.AppendLine(string.Format(Inv, "run time (UTC) : {0:yyyy-MM-dd HH:mm:ss}", Server.Time));
                _log.AppendLine(string.Format(Inv, "broker         : {0}", Account.BrokerName));
                _log.AppendLine(string.Format(Inv, "target date    : {0:yyyy-MM-dd}", target));
                _log.AppendLine(string.Format(Inv, "reference      : {0}", RefSymbol));
                _log.AppendLine(string.Format(Inv, "align window   : {0:yyyy-MM-dd} -> {1:yyyy-MM-dd} ({2} days)",
                    _alignFrom, _alignTo, AlignmentDays));
                _log.AppendLine("alignment is INTERSECTED with each symbol's own range (see overlap_days)");
                _log.AppendLine();

                _csv.AppendLine("symbol,bars,earliest_utc,latest_utc,span_days,reached_target,stop_reason," +
                                "load_calls,overlap_days,ref_min_in_overlap,sym_bars_in_overlap," +
                                "tickvol_nonzero_pct,align_covered_pct,align_extra_pct,note");

                LoadReference();
                Flush();

                foreach (string raw in SymbolList.Split(','))
                {
                    string name = raw.Trim();
                    if (name.Length == 0) continue;
                    Print("probing {0} ...", name);
                    try { ProbeOne(name, target); }
                    catch (Exception ex)
                    {
                        _csv.AppendLine(string.Format(Inv, "{0},0,,,,NO,EXCEPTION,0,,,,,,,{1}", name, Clean(ex.Message)));
                        _log.AppendLine(name + ": FAILED - " + ex.Message);
                        _log.AppendLine();
                        Print("{0}: FAILED - {1}", name, ex.Message);
                    }
                    Flush();               // write after EVERY symbol
                }
                Print("PROBE COMPLETE -> {0}", OutputFolder);
            }
            catch (Exception ex)
            {
                _log.AppendLine("PROBE ERROR: " + ex);
                Print("PROBE ERROR: {0}", ex.Message);
                Flush();
            }
            try { Stop(); } catch { }
        }

        private void LoadReference()
        {
            try
            {
                Bars rb = MarketData.GetBars(TimeFrame.Minute, RefSymbol);
                int guard = 0;
                while (rb.Count > 0 && rb.OpenTimes[0] > _alignFrom && guard < 400)
                {
                    if (rb.LoadMoreHistory() < 1) break;
                    guard++;
                }
                for (int i = 0; i < rb.Count; i++)
                {
                    DateTime t = rb.OpenTimes[i];
                    if (t >= _alignFrom && t <= _alignTo) { _refMin.Add(ToEpoch(t)); _refSorted.Add(ToEpoch(t)); }
                }
                _refSorted.Sort();
                _log.AppendLine(string.Format(Inv, "reference minutes in window: {0} ({1} load calls)",
                    _refMin.Count, guard));
            }
            catch (Exception ex)
            {
                _log.AppendLine("REFERENCE LOAD FAILED: " + ex.Message);
                _log.AppendLine("Alignment columns will be blank (not measured).");
            }
            _log.AppendLine();
        }

        private void ProbeOne(string name, DateTime target)
        {
            Bars b = MarketData.GetBars(TimeFrame.Minute, name);   // no GetSymbol - see header
            if (b == null) { WriteEmpty(name, "GetBars returned null"); return; }

            int calls = 0; string stop = "UNKNOWN";
            while (true)
            {
                if (b.Count > 0 && b.OpenTimes[0] <= target) { stop = "REACHED_TARGET"; break; }
                if (calls >= MaxLoadCalls) { stop = "CAP_HIT_DEPTH_UNKNOWN"; break; }
                int got = b.LoadMoreHistory();
                calls++;
                if (got < 1) { stop = "SERVER_EXHAUSTED"; break; }
                if (calls % 200 == 0)
                    Print("  {0}: call {1}, {2} bars, earliest {3:yyyy-MM-dd}", name, calls, b.Count, b.OpenTimes[0]);
            }

            int n = b.Count;
            if (n == 0) { WriteEmpty(name, "resolved but zero bars"); return; }
            DateTime first = b.OpenTimes[0], last = b.OpenTimes[n - 1];

            // --- FIX 1: intersect the alignment window with this symbol's own range ---
            DateTime lo = first > _alignFrom ? first : _alignFrom;
            DateTime hi = last < _alignTo ? last : _alignTo;
            double overlapDays = hi > lo ? (hi - lo).TotalDays : 0.0;

            int refInOverlap = 0;
            if (overlapDays > 0)
            {
                long l = ToEpoch(lo), h = ToEpoch(hi);
                for (int i = 0; i < _refSorted.Count; i++)
                    if (_refSorted[i] >= l && _refSorted[i] <= h) refInOverlap++;
            }

            int symInOverlap = 0, nonZeroVol = 0, matched = 0, extra = 0;
            for (int i = 0; i < n; i++)
            {
                DateTime t = b.OpenTimes[i];
                if (t < lo || t > hi) continue;
                symInOverlap++;
                if (b.TickVolumes[i] > 0) nonZeroVol++;
                if (_refMin.Count > 0)
                {
                    if (_refMin.Contains(ToEpoch(t))) matched++; else extra++;
                }
            }

            string volPct   = symInOverlap > 0 ? (100.0 * nonZeroVol / symInOverlap).ToString("0.00", Inv) : "";
            string covPct   = refInOverlap  > 0 ? (100.0 * matched    / refInOverlap ).ToString("0.00", Inv) : "";
            string extraPct = symInOverlap > 0 && _refMin.Count > 0
                              ? (100.0 * extra / symInOverlap).ToString("0.00", Inv) : "";

            _csv.AppendLine(string.Format(Inv,
                "{0},{1},{2:yyyy-MM-dd HH:mm},{3:yyyy-MM-dd HH:mm},{4:0.0},{5},{6},{7},{8:0.0},{9},{10},{11},{12},{13},",
                name, n, first, last, (last - first).TotalDays, first <= target ? "YES" : "NO",
                stop, calls, overlapDays, refInOverlap, symInOverlap, volPct, covPct, extraPct));

            _log.AppendLine(name);
            _log.AppendLine(string.Format(Inv, "   bars {0}   {1:yyyy-MM-dd} -> {2:yyyy-MM-dd}   span {3:0.0} days",
                n, first, last, (last - first).TotalDays));
            _log.AppendLine(string.Format(Inv, "   reached {0:yyyy-MM-dd}: {1}   stop: {2} ({3} load calls)",
                target, first <= target ? "YES" : "NO", stop, calls));
            _log.AppendLine(string.Format(Inv, "   overlap with reference window: {0:0.0} days " +
                "({1} ref minutes, {2} symbol bars)", overlapDays, refInOverlap, symInOverlap));
            _log.AppendLine(string.Format(Inv, "   tick volume non-zero : {0}%", volPct));
            _log.AppendLine(string.Format(Inv, "   covers {0}% of reference minutes IN THE OVERLAP", covPct));
            _log.AppendLine(string.Format(Inv, "   {0}% of its own bars fall outside reference minutes", extraPct));
            _log.AppendLine();
            Print("  {0}: {1} bars from {2:yyyy-MM-dd}, {3}, cover {4}%", name, n, first, stop, covPct);
        }

        private void WriteEmpty(string name, string note)
        {
            _csv.AppendLine(string.Format(Inv, "{0},0,,,,NO,NO_DATA,0,,,,,,,{1}", name, Clean(note)));
            _log.AppendLine(name + ": " + note);
            _log.AppendLine();
        }

        private void Flush()
        {
            try
            {
                System.IO.File.WriteAllText(Path.Combine(OutputFolder, "depth_probe.csv"), _csv.ToString(), Encoding.ASCII);
                System.IO.File.WriteAllText(Path.Combine(OutputFolder, "depth_probe_log.txt"), _log.ToString(), Encoding.ASCII);
            }
            catch (Exception ex) { Print("WRITE ERROR: {0}", ex.Message); }
        }

        private static long ToEpoch(DateTime t)
        {
            return (long)(DateTime.SpecifyKind(t, DateTimeKind.Utc) - Epoch).TotalSeconds;
        }

        private static string Clean(string s)
        {
            return s == null ? "" : s.Replace(',', ';').Replace('\r', ' ').Replace('\n', ' ');
        }
    }
}
