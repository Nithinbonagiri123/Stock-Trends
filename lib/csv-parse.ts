/**
 * Parse OHLC-style CSV exports (e.g. Yahoo Finance: Date, Open, High, Low, Close, Adj Close, Volume).
 * Educational use only — validate data before relying on it.
 */

export type StockRow = { date: Date; close: number };

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      inQuotes = !inQuotes;
    } else if ((c === "," || c === ";") && !inQuotes) {
      out.push(cur.trim());
      cur = "";
    } else {
      cur += c;
    }
  }
  out.push(cur.trim());
  return out;
}

function normHeader(h: string): string {
  return h
    .toLowerCase()
    .replace(/\s+/g, "")
    .replace(/[_-]/g, "");
}

function parseNumber(raw: string): number | null {
  const s = raw.replace(/^"|"$/g, "").replace(/,/g, "").trim();
  if (s === "" || s === "null") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

/** First token like AAPL from "AAPL.csv", "aapl_daily.csv". */
export function guessTickerFromFileName(fileName: string): string {
  const base = fileName.replace(/\.[^.]+$/i, "");
  const m = base.match(/^([A-Za-z][A-Za-z0-9.\-]{0,11})/);
  return m ? m[1].toUpperCase() : "";
}

export function parseStockCsv(text: string):
  | { ok: true; rows: StockRow[] }
  | { ok: false; error: string } {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length < 2) {
    return { ok: false, error: "CSV needs a header row and at least one data row." };
  }

  const headerCells = splitCsvLine(lines[0]).map(normHeader);
  const dateIdx = headerCells.findIndex(
    (h) => h === "date" || h === "datetime" || h === "timestamp",
  );
  const adjIdx = headerCells.findIndex(
    (h) => h === "adjclose" || h === "adjustedclose",
  );
  const closeIdx = headerCells.findIndex((h) => h === "close");

  let priceIdx = -1;
  if (adjIdx >= 0) priceIdx = adjIdx;
  else if (closeIdx >= 0) priceIdx = closeIdx;
  else {
    return {
      ok: false,
      error:
        'Need a "Close" or "Adj Close" column (Yahoo-style exports work). Found: ' +
        lines[0].slice(0, 120),
    };
  }

  if (dateIdx < 0) {
    return {
      ok: false,
      error: 'Need a "Date" column. Found: ' + lines[0].slice(0, 120),
    };
  }

  const rows: StockRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = splitCsvLine(lines[i]);
    if (cells.length <= Math.max(dateIdx, priceIdx)) continue;

    const rawDate = cells[dateIdx]?.replace(/^"|"$/g, "").trim() ?? "";
    const t = Date.parse(rawDate);
    if (Number.isNaN(t)) continue;

    const close = parseNumber(cells[priceIdx] ?? "");
    if (close === null || close <= 0) continue;

    rows.push({ date: new Date(t), close });
  }

  if (rows.length < 2) {
    return {
      ok: false,
      error: "No valid rows with a parseable date and positive close price.",
    };
  }

  rows.sort((a, b) => a.date.getTime() - b.date.getTime());

  const deduped: StockRow[] = [];
  for (const r of rows) {
    const last = deduped[deduped.length - 1];
    if (last && last.date.getTime() === r.date.getTime()) {
      deduped[deduped.length - 1] = r;
    } else {
      deduped.push(r);
    }
  }

  return { ok: true, rows: deduped };
}
