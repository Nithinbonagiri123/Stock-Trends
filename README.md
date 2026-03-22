# Stock AI trend explorer

Educational **prototype**: enter a ticker and see **deterministic mock “signals”** (not live market data). Optionally set `OPENAI_API_KEY` so the API can add short, disclaimer-heavy narrative text — still based on those mocks until you integrate real quotes.

## Not financial advice

This project does **not** predict real stock prices. Extend it with licensed data and proper compliance before any production use.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Environment

Copy `.env.example` to `.env.local` and add `OPENAI_API_KEY` if you want AI-generated explanations.

## Next steps for a real product

- Market data: [Polygon](https://polygon.io/), [Alpha Vantage](https://www.alphavantage.co/), exchange APIs, etc.
- Backtesting and models: separate service; legal/compliance review for user-facing “predictions.”
- Auth, billing, rate limits, and clear disclosures.
# Stock-Trends
