import { NextResponse } from "next/server";
import { buildMockAnalysis, type Horizon } from "@/lib/mock-signals";

export const runtime = "nodejs";

const HORIZONS: Horizon[] = ["1w", "1m", "3m"];

function isTicker(s: string): boolean {
  return /^[A-Z0-9.\-]{1,12}$/i.test(s.trim());
}

export async function POST(req: Request) {
  let body: { ticker?: string; horizon?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const ticker = (body.ticker || "").trim();
  const horizon = (body.horizon || "1m") as Horizon;

  if (!ticker || !isTicker(ticker)) {
    return NextResponse.json(
      { error: "Enter a valid ticker symbol (letters, numbers, dot or hyphen)." },
      { status: 400 },
    );
  }

  if (!HORIZONS.includes(horizon)) {
    return NextResponse.json({ error: "Invalid horizon." }, { status: 400 });
  }

  const mock = buildMockAnalysis(ticker, horizon);

  return NextResponse.json({
    disclaimer:
      "Educational demo only. Not financial advice. Mock data — not live market data. Past or simulated patterns do not predict future results.",
    ...mock,
  });
}
