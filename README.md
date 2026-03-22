# Stock AI trend explorer

Educational **prototype**: enter a ticker and see **deterministic mock “signals”** and a short **local** narrative (not live market data). No third-party API keys required.

## Not financial advice

This project does **not** predict real stock prices. Extend it with licensed data and proper compliance before any production use.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Next steps for a real product

- Market data: [Polygon](https://polygon.io/), [Alpha Vantage](https://www.alphavantage.co/), exchange APIs, etc.
- Backtesting and models: separate service; legal/compliance review for user-facing “predictions.”
- Auth, billing, rate limits, and clear disclosures.
