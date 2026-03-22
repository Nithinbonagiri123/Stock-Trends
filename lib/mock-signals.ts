/**
 * Deterministic mock “signals” for demo / education only — not real market data.
 */
export type Horizon = "1w" | "1m" | "3m";

export type TrendSignal = {
  label: string;
  value: string;
  detail: string;
};

function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildMockAnalysis(ticker: string, horizon: Horizon) {
  const upper = ticker.trim().toUpperCase();
  const seed = hashString(upper + horizon);
  const rnd = mulberry32(seed);

  const momentum = (rnd() * 40 - 20).toFixed(1);
  const vol = (rnd() * 35 + 10).toFixed(0);
  const rsi = Math.round(30 + rnd() * 40);

  const sentiment =
    Number(momentum) > 5 ? "bullish" : Number(momentum) < -5 ? "bearish" : "neutral";

  const signals: TrendSignal[] = [
    {
      label: "Mock momentum (demo)",
      value: `${momentum}%`,
      detail: "Illustrative only — not calculated from live prices.",
    },
    {
      label: "Mock volatility index",
      value: vol,
      detail: "Higher = more swing risk in this toy model.",
    },
    {
      label: "Mock RSI-style read",
      value: String(rsi),
      detail: "Educational placeholder, not a trading signal.",
    },
  ];

  return {
    ticker: upper,
    horizon,
    sentiment: sentiment as "bullish" | "bearish" | "neutral",
    signals,
    mockSeedNote: `Demo seed: ${seed} (same ticker + horizon → same mock output).`,
  };
}
