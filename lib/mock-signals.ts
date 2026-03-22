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

function isCryptoUsdPair(symbol: string): boolean {
  return /^[A-Z0-9]{2,}-USD$/i.test(symbol.trim());
}

export function buildMockAnalysis(ticker: string, horizon: Horizon) {
  const upper = ticker.trim().toUpperCase();
  const seed = hashString(upper + horizon);
  const rnd = mulberry32(seed);
  const crypto = isCryptoUsdPair(upper);
  const btc = upper.startsWith("BTC");

  const momentum = (rnd() * 40 - 20).toFixed(1);
  const vol = (rnd() * 35 + 10).toFixed(0);
  const rsi = Math.round(30 + rnd() * 40);

  const sentiment =
    Number(momentum) > 5 ? "bullish" : Number(momentum) < -5 ? "bearish" : "neutral";

  const momLabel = btc
    ? "Mock momentum — Bitcoin / USD (demo)"
    : crypto
      ? "Mock momentum — crypto / USD (demo)"
      : "Mock momentum (demo)";
  const momDetail = crypto
    ? "Illustrative only — not spot or live crypto prices."
    : "Illustrative only — not calculated from live prices.";
  const volDetail = crypto
    ? "Toy volatility read for a crypto pair — not on-chain or exchange data."
    : "Higher = more swing risk in this toy model.";
  const rsiDetail = crypto
    ? "Educational placeholder for a USD-quoted pair — not a trade signal."
    : "Educational placeholder, not a trading signal.";

  const signals: TrendSignal[] = [
    {
      label: momLabel,
      value: `${momentum}%`,
      detail: momDetail,
    },
    {
      label: crypto ? "Mock volatility (crypto demo)" : "Mock volatility index",
      value: vol,
      detail: volDetail,
    },
    {
      label: crypto ? "Mock RSI-style (pair)" : "Mock RSI-style read",
      value: String(rsi),
      detail: rsiDetail,
    },
  ];

  const note = crypto
    ? `Demo seed: ${seed}. Yahoo-style symbol ${upper} (e.g. BTC-USD for Bitcoin spot vs USD) — still mock math.`
    : `Demo seed: ${seed} (same ticker + horizon → same mock output).`;

  const summary =
    btc
      ? sentiment === "bullish"
        ? "In this demo model, Bitcoin (USD pair) momentum tilts slightly positive — not live BTC; not financial advice."
        : sentiment === "bearish"
          ? "In this demo model, Bitcoin (USD pair) momentum tilts slightly negative — illustrative only."
          : "In this demo model, Bitcoin (USD pair) reads near the middle — no real forecast implied."
      : crypto
        ? sentiment === "bullish"
          ? "In this demo model, this crypto/USD pair tilts slightly positive — still meaningless for real trading."
          : sentiment === "bearish"
            ? "In this demo model, this crypto/USD pair tilts slightly negative — purely illustrative."
            : "In this demo model, this crypto/USD pair sits near the middle — no forecast implied."
        : sentiment === "bullish"
          ? "In this demo model, momentum tilts slightly positive — still meaningless for real trading."
          : sentiment === "bearish"
            ? "In this demo model, momentum tilts slightly negative — purely illustrative."
            : "In this demo model, readings sit near the middle — no real forecast implied.";

  return {
    ticker: upper,
    horizon,
    sentiment: sentiment as "bullish" | "bearish" | "neutral",
    signals,
    mockSeedNote: note,
    summary,
  };
}
