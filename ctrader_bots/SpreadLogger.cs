using System;
using System.IO;
using cAlgo.API;

namespace cAlgo.Robots
{
    // TimeZone = UTC: all Server-side times in this robot are UTC (project rule: UTC always).
    // AccessRights = FullAccess: required ONLY for writing the CSV to disk.
    //   This is a .NET sandbox permission, not a trading permission.
    //   The bot is read-only by construction: there is no order/trade API call
    //   anywhere in this file (no ExecuteMarketOrder, no PlaceLimitOrder, nothing).
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess, AddIndicators = false)]
    public class SpreadLogger : Robot
    {
        [Parameter("Sample interval (seconds)", DefaultValue = 5, MinValue = 1, MaxValue = 60)]
        public int SampleSeconds { get; set; }

        [Parameter("Output folder", DefaultValue = "C:\\Users\\kmykr\\xauusd-project\\spread")]
        public string OutputFolder { get; set; }

        private const string Header = "timestamp_utc,bid,ask,spread_usd_oz,spread_pips,market_open";

        protected override void OnStart()
        {
            Directory.CreateDirectory(OutputFolder);   // creates the folder if missing
            Timer.Start(TimeSpan.FromSeconds(SampleSeconds));
            Print("SpreadLogger started. Folder: {0} | every {1}s | Symbol: {2} | PipSize: {3}",
                  OutputFolder, SampleSeconds, SymbolName, Symbol.PipSize);
            WriteSample();  // log one row immediately so you can verify without waiting
        }

        protected override void OnTimer()
        {
            WriteSample();
        }

        private void WriteSample()
        {
            try
            {
                DateTime utc = DateTime.UtcNow;          // machine clock (NTP-synced), monotonic even when no ticks arrive
                double bid = Symbol.Bid;
                double ask = Symbol.Ask;
                double spreadUsd = ask - bid;            // $/oz — the ground-truth column
                double spreadPips = spreadUsd / Symbol.PipSize;   // derived; valid iff PipSize = 0.01
                bool open = Symbol.MarketHours.IsOpened();        // false during the ~63min dead gap & weekend (quotes are stale then)

                string path = Path.Combine(OutputFolder,
                    "spread_" + utc.ToString("yyyy-MM-dd") + ".csv");   // one file per UTC calendar day

                bool needHeader = !System.IO.File.Exists(path);

                // InvariantCulture: forces '.' decimal separator regardless of Windows locale
                // (a locale set to comma-decimals would silently corrupt the CSV otherwise)
                string row = string.Format(
                    System.Globalization.CultureInfo.InvariantCulture,
                    "{0:yyyy-MM-dd HH:mm:ss},{1:F2},{2:F2},{3:F3},{4:F1},{5}",
                    utc, bid, ask, spreadUsd, spreadPips, open ? 1 : 0);

                System.IO.File.AppendAllText(path,
                    (needHeader ? Header + Environment.NewLine : "") + row + Environment.NewLine);
            }
            catch (Exception ex)
            {
                Print("SpreadLogger write error: {0}", ex.Message);
            }
        }

        protected override void OnStop()
        {
            Print("SpreadLogger stopped.");
        }
    }
}
