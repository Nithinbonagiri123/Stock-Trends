import type { Horizon } from "./mock-signals";

export function horizonTradingDays(h: Horizon): number {
  switch (h) {
    case "1w":
      return 5;
    case "1m":
      return 21;
    case "3m":
      return 63;
    default:
      return 21;
  }
}

/** Simple returns from closes (length n-1). */
export function simpleReturns(closes: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    const prev = closes[i - 1];
    if (prev === 0) continue;
    out.push((closes[i] - prev) / prev);
  }
  return out;
}

/** Sample stdev of returns; annualized vol as % (sqrt(252) * stdev * 100). */
export function annualizedVolatilityPct(returns: number[]): number | null {
  if (returns.length < 2) return null;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance =
    returns.reduce((s, r) => s + (r - mean) ** 2, 0) / (returns.length - 1);
  const stdev = Math.sqrt(Math.max(variance, 0));
  return stdev * Math.sqrt(252) * 100;
}

/** Wilder RSI; needs at least period+1 closes. */
export function rsiFromCloses(closes: number[], period = 14): number | null {
  if (closes.length < period + 1) return null;
  const changes: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    changes.push(closes[i] - closes[i - 1]);
  }
  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 0; i < period; i++) {
    const c = changes[i];
    if (c > 0) avgGain += c;
    else avgLoss -= c;
  }
  avgGain /= period;
  avgLoss /= period;

  for (let i = period; i < changes.length; i++) {
    const c = changes[i];
    const gain = c > 0 ? c : 0;
    const loss = c < 0 ? -c : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
  }

  if (avgLoss === 0) return avgGain === 0 ? 50 : 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}
