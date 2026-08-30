// ============================================================================
// SymbolCensus.cs - account symbol name census (IC Markets / cTrader)
// PROJECT K - Cycle 0 data-feasibility inventory, TASK 2A
//
// PURPOSE (observe, never assume):
//   List every symbol NAME exposed on this account. Nothing else.
//   This is the cheapest possible probe and it decides the shortlist for
//   the depth/alignment probe (Task 2C). No shortlist is guessed in advance.
//
// DELIBERATELY NOT DONE HERE:
//   Symbols.GetSymbol(name) is NOT called. It is slow and can fail outright
//   ("Symbol not found or has no quotes") even for names present in the list.
//   Resolving hundreds of symbols blindly would hang the run. Resolution
//   happens in Task 2C, on a shortlist, with failures caught per symbol.
//   No bars are loaded. No Symbol object is touched. Names only.
//
// OUTPUT (to OutputFolder):
//   symbol_census.txt   - header block + one symbol name per line
//
// READ-ONLY BY CONSTRUCTION: no order API call appears in this file.
//   AccessRights.FullAccess is a .NET file-write permission, not a trading one.
//
// USAGE: paste into cTrader Automate editor -> Build -> attach to ANY chart
//   -> Start. Completes in seconds and stops itself.
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
    public class SymbolCensus : Robot
    {
        [Parameter("Output Folder", DefaultValue = @"C:\Users\kmykr\xauusd-project\projectk_probe")]
        public string OutputFolder { get; set; }

        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        protected override void OnStart()
        {
            try
            {
                Directory.CreateDirectory(OutputFolder);

                var names = new List<string>();
                foreach (string s in Symbols) names.Add(s);
                names.Sort(StringComparer.Ordinal);

                var sb = new StringBuilder();
                sb.AppendLine("=== ACCOUNT SYMBOL CENSUS ===");
                sb.AppendLine(string.Format(Inv, "run time (UTC)   : {0:yyyy-MM-dd HH:mm:ss}", Server.Time));
                sb.AppendLine(string.Format(Inv, "broker           : {0}", Account.BrokerName));
                sb.AppendLine(string.Format(Inv, "attached chart   : {0}", SymbolName));
                sb.AppendLine(string.Format(Inv, "symbols exposed  : {0}", names.Count));
                sb.AppendLine("names only - no symbol resolved, no bars loaded");
                sb.AppendLine("=============================");
                foreach (string n in names) sb.AppendLine(n);

                string path = Path.Combine(OutputFolder, "symbol_census.txt");
                System.IO.File.WriteAllText(path, sb.ToString(), Encoding.ASCII);

                Print("SYMBOL CENSUS COMPLETE: {0} symbols -> {1}", names.Count, path);
            }
            catch (Exception ex)
            {
                Print("CENSUS ERROR: {0}", ex);
            }
            Stop();
        }
    }
}
