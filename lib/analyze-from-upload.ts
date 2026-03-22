import type { Horizon, TrendSignal } from "./mock-signals";
import type { StockRow } from "./csv-parse";
import {
  annualizedVolatilityPct,
  horizonTradingDays,
  rsiFromCloses,
  simpleReturns,
} from "./indicators";

function fmtDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function sentimentFromMomentum(momentumPct: number): "bullish" | "bearish" | "neutral" {
  if (momentumPct > 5) return "bullish";
  if (momentumPct < -5) return "bearish";
  return "neutral";
}

function summaryForSentiment(s: "bullish" | "bearish" | "neutral"): string {
  switch (s) {
    case "bullish":
      return "Over your selected horizon on uploaded closes, price change is up — educational only, not a forecast.";
    case "bearish":
      return "Over your selected horizon on uploaded closes, price change is down — educational only, not a forecast.";
    default:
      return "Over your selected horizon on uploaded closes, price change is modest — educational only, not a forecast.";
  }
}

export function analyzeFromUpload(
  ticker: string,
  horizon: Horizon,
  rows: StockRow[],
): {
  ticker: string;
  horizon: Horizon;
  sentiment: "bullish" | "bearish" | "neutral";
  signals: TrendSignal[];
  mockSeedNote: string;
  summary: string;
} {
  const upper = ticker.trim().toUpperCase();
  const windowDays = horizonTradingDays(horizon);
  const windowRows = rows.slice(-Math.min(windowDays, rows.length));
  const closes = windowRows.map((r) => r.close);

  const first = closes[0];
  const last = closes[closes.length - 1];
  const momentumPct = first > 0 ? ((last - first) / first) * 100 : 0;

  const rets = simpleReturns(closes);
  const volPct = annualizedVolatilityPct(rets);
  const allCloses = rows.map((r) => r.close);
  const rsi = rsiFromCloses(allCloses, 14);

  const sentiment = sentimentFromMomentum(momentumPct);

  const volDisplay =
    volPct !== null ? `${volPct.toFixed(1)}%` : "n/a (need more rows in window)";

  const rsiDisplay = rsi !== null ? rsi.toFixed(0) : "n/a (need at least 15 daily closes)";

  const signals: TrendSignal[] = [
    {
      label: "Horizon momentum (your CSV)",
      value: `${momentumPct >= 0 ? "+" : ""}${momentumPct.toFixed(2)}%`,
      detail: `Change from first to last close in the last ${windowRows.length} trading rows used for this horizon.`,
    },
    {
      label: "Annualized vol (window)",
      value: volDisplay,
      detail: "From daily simple returns in the horizon window; illustrative.",
    },
    {
      label: "RSI(14) on full series",
      value: rsiDisplay,
      detail: "Classic RSI on your uploaded closes through the latest date — not a buy/sell signal.",
    },
  ];

  const start = rows[0].date;
  const end = rows[rows.length - 1].date;
  const mockSeedNote = `Uploaded data: ${rows.length} rows · ${fmtDate(start)} → ${fmtDate(end)} · horizon window uses last ${windowRows.length} rows.`;

  return {
    ticker: upper,
    horizon,
    sentiment,
    signals,
    mockSeedNote,
    summary: summaryForSentiment(sentiment),
  };
}
