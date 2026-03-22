"use client";

import { useState } from "react";
import { analyzeFromUpload } from "@/lib/analyze-from-upload";
import { guessTickerFromFileName, parseStockCsv, type StockRow } from "@/lib/csv-parse";

type Horizon = "1w" | "1m" | "3m";

type AnalyzeResponse = {
  disclaimer: string;
  ticker: string;
  horizon: Horizon;
  sentiment: "bullish" | "bearish" | "neutral";
  signals: { label: string; value: string; detail: string }[];
  summary: string;
  mockSeedNote: string;
  dataSource?: "upload" | "mock";
  error?: string;
};

const UPLOAD_DISCLAIMER =
  "Educational demo only. Not financial advice. Metrics are computed from your uploaded CSV — verify dates and prices. Past data does not predict future results.";

const QUICK_TICKERS = [
  ["BTC-USD", "Bitcoin"],
  ["ETH-USD", "Ethereum"],
  ["AAPL", "Apple"],
] as const;

type UploadedStock = {
  id: string;
  ticker: string;
  fileName: string;
  rows: StockRow[];
};

export default function Home() {
  const [ticker, setTicker] = useState("BTC-USD");
  const [horizon, setHorizon] = useState<Horizon>("1m");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploads, setUploads] = useState<UploadedStock[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);

  function removeUpload(id: string) {
    setUploads((prev) => prev.filter((u) => u.id !== id));
  }

  function updateUploadTicker(id: string, next: string) {
    const t = next.toUpperCase().replace(/[^A-Z0-9.\-]/g, "").slice(0, 12);
    setUploads((prev) => prev.map((u) => (u.id === id ? { ...u, ticker: t } : u)));
  }

  async function onPickFiles(files: FileList | null) {
    if (!files?.length) return;
    setUploadError(null);
    for (const file of files) {
      const text = await file.text();
      const parsed = parseStockCsv(text);
      if (!parsed.ok) {
        setUploadError(`${file.name}: ${parsed.error}`);
        continue;
      }
      const guessed = guessTickerFromFileName(file.name);
      const initialTicker = (guessed || ticker.trim() || "TICKER").toUpperCase();
      setUploads((prev) => {
        const without = prev.filter(
          (u) => u.ticker.toUpperCase() !== initialTicker.toUpperCase(),
        );
        return [
          ...without,
          {
            id: crypto.randomUUID(),
            ticker: initialTicker,
            fileName: file.name,
            rows: parsed.rows,
          },
        ];
      });
    }
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    const key = ticker.trim().toUpperCase();
    const match = uploads.find((u) => u.ticker.toUpperCase() === key);

    if (match && match.rows.length >= 2) {
      const a = analyzeFromUpload(key, horizon, match.rows);
      setResult({
        disclaimer: UPLOAD_DISCLAIMER,
        ...a,
        dataSource: "upload",
      });
      setLoading(false);
      return;
    }

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, horizon }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Request failed");
        return;
      }
      setResult({ ...(data as AnalyzeResponse), dataSource: "mock" });
    } catch {
      setError("Network error");
    } finally {
      setLoading(false);
    }
  }

  const hasUploadForTicker = uploads.some((u) => u.ticker.toUpperCase() === ticker.trim().toUpperCase());

  return (
    <div className="min-h-full bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      <main className="mx-auto flex max-w-2xl flex-col gap-10 px-4 py-12">
        <header className="space-y-3">
          <p className="text-sm font-medium uppercase tracking-wide text-emerald-700 dark:text-emerald-400">
            Prototype
          </p>
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
            AI trend explorer
          </h1>
          <p className="text-zinc-600 dark:text-zinc-400">
            Run <strong>demo math</strong> on equities or <strong>crypto pairs</strong> (e.g.{" "}
            <code className="font-mono text-sm">BTC-USD</code> for Bitcoin vs USD), or{" "}
            <strong>upload CSV history</strong> — momentum / vol / RSI style readouts are
            illustrative unless you use your own file.
          </p>
        </header>

        <aside
          className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-100"
          role="note"
        >
          <strong className="font-semibold">Important:</strong> Nothing here is investment
          advice. Uploaded files stay in this tab (not sent to a server for analysis). Trading
          involves risk.
        </aside>

        <section className="space-y-4 rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div className="space-y-1">
            <h2 className="text-sm font-semibold">Historical CSV (optional)</h2>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Export daily prices from Yahoo Finance or similar: include <strong>Date</strong>{" "}
              and <strong>Adj Close</strong> or <strong>Close</strong>. Upload one file per
              stock; the ticker is guessed from the filename (e.g. <code className="font-mono">MSFT.csv</code>
              ) — you can fix it below. Use the same ticker in the form when you run analysis.
            </p>
          </div>

          <label
            className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-zinc-300 bg-zinc-50 px-4 py-8 text-center text-sm transition hover:border-emerald-500/50 hover:bg-zinc-100 dark:border-zinc-600 dark:bg-zinc-950 dark:hover:border-emerald-500/40 dark:hover:bg-zinc-900"
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            onDrop={(e) => {
              e.preventDefault();
              e.stopPropagation();
              void onPickFiles(e.dataTransfer.files);
            }}
          >
            <span className="font-medium text-emerald-700 dark:text-emerald-400">
              Drop CSV files here or click to browse
            </span>
            <span className="text-xs text-zinc-500">Multiple files allowed</span>
            <input
              type="file"
              accept=".csv,text/csv"
              multiple
              className="sr-only"
              onChange={(e) => {
                void onPickFiles(e.target.files);
                e.target.value = "";
              }}
            />
          </label>

          {uploadError && (
            <p className="text-sm text-red-600 dark:text-red-400" role="alert">
              {uploadError}
            </p>
          )}

          {uploads.length > 0 && (
            <ul className="space-y-2">
              {uploads.map((u) => {
                const first = u.rows[0]?.date;
                const last = u.rows[u.rows.length - 1]?.date;
                const range =
                  first && last
                    ? `${first.toISOString().slice(0, 10)} → ${last.toISOString().slice(0, 10)}`
                    : "";
                return (
                  <li
                    key={u.id}
                    className="flex flex-wrap items-center gap-3 rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-800/80"
                  >
                    <input
                      type="text"
                      value={u.ticker}
                      onChange={(e) => updateUploadTicker(u.id, e.target.value)}
                      className="w-24 rounded border border-zinc-300 bg-white px-2 py-1 font-mono text-xs uppercase dark:border-zinc-600 dark:bg-zinc-950"
                      aria-label="Ticker for this file"
                      maxLength={12}
                    />
                    <span className="min-w-0 flex-1 truncate text-zinc-600 dark:text-zinc-300">
                      {u.fileName}
                      <span className="text-zinc-400"> · {u.rows.length} rows</span>
                      {range ? (
                        <span className="block text-xs text-zinc-400">{range}</span>
                      ) : null}
                    </span>
                    <button
                      type="button"
                      onClick={() => removeUpload(u.id)}
                      className="shrink-0 rounded-lg border border-zinc-300 px-2 py-1 text-xs hover:bg-zinc-100 dark:border-zinc-600 dark:hover:bg-zinc-700"
                    >
                      Remove
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <form
          onSubmit={onSubmit}
          className="space-y-6 rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
        >
          <div className="space-y-2">
            <label htmlFor="ticker" className="text-sm font-medium">
              Ticker
            </label>
            <input
              id="ticker"
              name="ticker"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g. BTC-USD, AAPL"
              className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 font-mono text-sm outline-none ring-emerald-500/0 transition focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/30 dark:border-zinc-600 dark:bg-zinc-950"
              maxLength={12}
              autoComplete="off"
            />
            <div className="flex flex-wrap gap-2 pt-1">
              <span className="w-full text-xs text-zinc-500 dark:text-zinc-400">Quick pick</span>
              {QUICK_TICKERS.map(([sym, label]) => (
                <button
                  key={sym}
                  type="button"
                  onClick={() => setTicker(sym)}
                  className={`rounded-full border px-3 py-1 text-xs transition ${
                    ticker === sym
                      ? "border-orange-600 bg-orange-600 text-white dark:border-orange-500 dark:bg-orange-600"
                      : "border-zinc-300 bg-zinc-50 hover:border-zinc-400 dark:border-zinc-600 dark:bg-zinc-800"
                  }`}
                >
                  {label} ({sym})
                </button>
              ))}
            </div>
            {uploads.length > 0 && !hasUploadForTicker && (
              <p className="text-xs text-amber-800 dark:text-amber-200/90">
                No uploaded file uses this ticker yet — analysis will use the built-in demo, or
                change the ticker to match an upload.
              </p>
            )}
          </div>

          <div className="space-y-2">
            <span className="text-sm font-medium">Horizon (demo)</span>
            <div className="flex flex-wrap gap-2">
              {(
                [
                  ["1w", "1 week"],
                  ["1m", "1 month"],
                  ["3m", "3 months"],
                ] as const
              ).map(([v, label]) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => setHorizon(v)}
                  className={`rounded-full border px-4 py-1.5 text-sm transition ${
                    horizon === v
                      ? "border-emerald-600 bg-emerald-600 text-white dark:border-emerald-500 dark:bg-emerald-600"
                      : "border-zinc-300 bg-zinc-50 hover:border-zinc-400 dark:border-zinc-600 dark:bg-zinc-800"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <button
            type="submit"
            disabled={loading || !ticker.trim()}
            className="w-full rounded-xl bg-emerald-600 py-3 text-sm font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Analyzing…" : hasUploadForTicker ? "Analyze uploaded data" : "Run demo analysis"}
          </button>
        </form>

        {error && (
          <p className="text-sm text-red-600 dark:text-red-400" role="alert">
            {error}
          </p>
        )}

        {result && (
          <section className="space-y-6 rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-lg font-semibold">
                {result.ticker}{" "}
                <span className="text-zinc-500 dark:text-zinc-400">· {result.horizon}</span>
              </h2>
              <div className="flex flex-wrap items-center gap-2">
                {result.dataSource === "upload" && (
                  <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-medium text-sky-900 dark:bg-sky-900/50 dark:text-sky-100">
                    Your CSV
                  </span>
                )}
                <span
                  className={`rounded-full px-3 py-0.5 text-xs font-medium uppercase ${
                    result.sentiment === "bullish"
                      ? "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/50 dark:text-emerald-100"
                      : result.sentiment === "bearish"
                        ? "bg-rose-100 text-rose-900 dark:bg-rose-900/50 dark:text-rose-100"
                        : "bg-zinc-200 text-zinc-800 dark:bg-zinc-700 dark:text-zinc-100"
                  }`}
                >
                  {result.sentiment}
                </span>
              </div>
            </div>

            <p className="text-xs text-zinc-500 dark:text-zinc-400">{result.disclaimer}</p>

            <div className="rounded-lg bg-zinc-50 p-4 text-sm leading-relaxed dark:bg-zinc-800/80">
              <p className="mb-1 text-xs font-medium uppercase tracking-wide text-zinc-500">
                Narrative {result.dataSource === "upload" ? "(from your data)" : "(demo)"}
              </p>
              <p>{result.summary}</p>
            </div>

            <ul className="space-y-3">
              {result.signals.map((s) => (
                <li
                  key={s.label}
                  className="flex flex-col gap-1 border-b border-zinc-100 pb-3 last:border-0 dark:border-zinc-800"
                >
                  <div className="flex items-center justify-between gap-4">
                    <span className="text-sm font-medium">{s.label}</span>
                    <span className="font-mono text-sm text-emerald-700 dark:text-emerald-400">
                      {s.value}
                    </span>
                  </div>
                  <span className="text-xs text-zinc-500 dark:text-zinc-400">{s.detail}</span>
                </li>
              ))}
            </ul>

            <p className="text-xs text-zinc-400 dark:text-zinc-500">{result.mockSeedNote}</p>
          </section>
        )}

        <footer className="text-xs text-zinc-500 dark:text-zinc-500">
          Mock mode calls the local API route; uploaded CSVs are analyzed in your browser only.
        </footer>
      </main>
    </div>
  );
}
