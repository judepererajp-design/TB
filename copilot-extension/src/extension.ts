/**
 * @copyright 2024-2026 nirholas. All rights reserved.
 * @license SPDX-License-Identifier: SEE LICENSE IN LICENSE
 * @see https://github.com/nirholas/free-crypto-news
 *
 * This file is part of free-crypto-news.
 * Unauthorized copying, modification, or distribution is strictly prohibited.
 * For licensing inquiries: nirholas@users.noreply.github.com
 */

import * as vscode from "vscode";

const API_BASE = "https://cryptocurrency.cv";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NewsArticle {
  title: string;
  source: string;
  link: string;
  timeAgo: string;
  description?: string;
  sentiment?: string;
  category?: string;
  priority?: string;
}

interface PriceInfo {
  usd: number;
  change24h: number;
  marketCap?: number;
  volume24h?: number;
}

interface GasPrice {
  slow: number;
  standard: number;
  fast: number;
  usdSlow?: number;
  usdStandard?: number;
  usdFast?: number;
}

interface FearGreedCurrent {
  value: number;
  valueClassification: string;
  timestamp?: number;
  timeUntilUpdate?: string;
}

interface FearGreedResponse {
  current: FearGreedCurrent;
  trend?: {
    direction: string;
    change7d: number;
    change30d: number;
    averageValue7d?: number;
    averageValue30d?: number;
  };
  breakdown?: Record<string, { value: number; weight: number }>;
  lastUpdated?: string;
}

interface FearGreedLegacy {
  value: number;
  classification: string;
  timestamp?: string;
  previous?: { value: number; classification: string };
}

interface GlossaryTerm {
  term: string;
  definition: string;
  category?: string;
  relatedTerms?: string[];
}

interface SentimentArticle {
  title: string;
  link: string;
  source: string;
  sentiment: string;
  confidence: number;
  reasoning: string;
  impactLevel: string;
  timeHorizon: string;
  affectedAssets: string[];
}

interface MarketSentiment {
  overall: string;
  score: number;
  confidence: number;
  summary: string;
  keyDrivers: string[];
}

interface SentimentResponse {
  articles: SentimentArticle[];
  market: MarketSentiment;
  distribution?: Record<string, number>;
  highImpactNews?: SentimentArticle[];
}

interface ExplainResponse {
  success: boolean;
  explanation?: {
    summary: string;
    background: string;
    whyTrending: string;
    marketImplications: string;
    outlook: string;
  };
  articleCount?: number;
  recentHeadlines?: string[];
  message?: string;
}

interface ResearchReport {
  summary: string;
  sentiment: string;
  keyFindings: string[];
  risks: string[];
  opportunities: string[];
  outlook: string;
  priceData?: {
    price: number;
    change24h: number;
    change7d?: number;
  };
  marketCap?: number;
}

interface ResearchResponse {
  success: boolean;
  report?: ResearchReport;
  quickTake?: {
    take: string;
    sentiment: string;
    confidence: number;
  };
  articlesAnalyzed?: number;
  error?: string;
}

interface BreakingNewsResponse {
  articles: NewsArticle[];
  count?: number;
  updatedAt?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Default per-request timeout (ms). Kept conservative so the chat UI never
// hangs indefinitely on a flaky upstream.
const DEFAULT_TIMEOUT_MS = 15_000;

function getApiBase(): string {
  const config = vscode.workspace.getConfiguration("crypto");
  const raw = (config.get<string>("apiUrl") || API_BASE).trim();
  // Only accept http(s) schemes; fall back to the default otherwise to avoid
  // arbitrary-scheme fetches (file://, data://, javascript:, ...).
  let normalized: URL;
  try {
    normalized = new URL(raw);
  } catch {
    return API_BASE;
  }
  if (normalized.protocol !== "http:" && normalized.protocol !== "https:") {
    return API_BASE;
  }
  // Strip trailing slash so concatenation with endpoints like "/api/news"
  // never produces a double slash.
  return raw.replace(/\/+$/, "");
}

/**
 * Only accept http(s) URLs for rendering as markdown links. Anything else
 * (javascript:, data:, vscode:, relative strings, etc.) is reduced to "#" so a
 * malicious or malformed `link` field from the upstream API cannot be rendered
 * as an active link in the chat stream.
 */
function safeUrl(link: string | undefined | null): string {
  if (!link || typeof link !== "string") return "#";
  const trimmed = link.trim();
  try {
    const u = new URL(trimmed);
    if (u.protocol !== "http:" && u.protocol !== "https:") return "#";
    return u.toString();
  } catch {
    return "#";
  }
}

/**
 * Neutralize characters that would break a markdown table cell or let
 * upstream-supplied text inject markdown structure. We only do this for table
 * cells — regular prose keeps its formatting.
 */
function escapeTableCell(value: string | number | undefined | null): string {
  if (value === undefined || value === null) return "—";
  return String(value)
    // Escape backslashes first so subsequent escapes aren't undone by a
    // caller-supplied "\" turning our "\|" into a literal "\" + unescaped "|".
    .replace(/\\/g, "\\\\")
    .replace(/\|/g, "\\|")
    .replace(/\r?\n/g, " ");
}

/** Format a number safely; returns `fallback` for undefined/NaN/non-finite. */
function fmtNum(
  value: number | undefined | null,
  digits = 2,
  fallback = "—",
): string {
  if (value === undefined || value === null) return fallback;
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return n.toFixed(digits);
}

/** Format a USD amount with locale grouping; returns `fallback` when absent. */
function fmtUsd(value: number | undefined | null, fallback = "N/A"): string {
  if (value === undefined || value === null) return fallback;
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return n.toLocaleString(undefined, { maximumFractionDigits: 8 });
}

async function fetchAPI<T = unknown>(
  endpoint: string,
  token?: vscode.CancellationToken,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const baseUrl = getApiBase();
  const url = `${baseUrl}${endpoint}`;

  const controller = new AbortController();

  // Honor an already-cancelled token (onCancellationRequested does not fire
  // for tokens that were cancelled before the listener was attached).
  if (token?.isCancellationRequested) {
    controller.abort();
  } else {
    token?.onCancellationRequested(() => controller.abort());
  }

  const timeoutHandle = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(
        `API request failed (${response.status}): ${response.statusText}`,
      );
    }
    return (await response.json()) as T;
  } finally {
    clearTimeout(timeoutHandle);
  }
}

function formatArticles(articles: NewsArticle[]): string {
  if (articles.length === 0) return "*No articles found.*";
  return articles
    .map((a, i) => {
      const sentiment =
        a.sentiment === "bullish"
          ? "🟢"
          : a.sentiment === "bearish"
            ? "🔴"
            : "⚪";
      const desc = a.description
        ? `\n   > ${a.description.slice(0, 120)}…`
        : "";
      return `${i + 1}. ${sentiment} **${a.title}**${desc}\n   📰 ${a.source} • ${a.timeAgo}\n   🔗 [Read more](${safeUrl(a.link)})`;
    })
    .join("\n\n");
}

/** Clamp a Fear/Greed-style value into `[0,100]` and default NaN to 50. */
function clampFG(value: number | undefined | null): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return 50;
  if (n < 0) return 0;
  if (n > 100) return 100;
  return Math.round(n);
}

function fearGreedEmoji(value: number): string {
  const v = clampFG(value);
  if (v < 25) return "😱";
  if (v < 40) return "😨";
  if (v < 60) return "😐";
  if (v < 75) return "😀";
  return "🤑";
}

function fearGreedBar(value: number): string {
  const v = clampFG(value);
  const filled = Math.max(0, Math.min(20, Math.floor(v / 5)));
  const empty = 20 - filled;
  return `\`${"█".repeat(filled)}${"░".repeat(empty)}\` ${v}/100`;
}

function sentimentEmoji(sentiment: string): string {
  switch (sentiment) {
    case "very_bullish":
      return "🟢🟢";
    case "bullish":
      return "🟢";
    case "bearish":
      return "🔴";
    case "very_bearish":
      return "🔴🔴";
    default:
      return "⚪";
  }
}

function sentimentLabel(sentiment: string): string {
  return sentiment.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

async function handleNews(
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<vscode.ChatResult> {
  stream.markdown("📰 **Latest Crypto News**\n\n");
  stream.progress("Fetching latest news…");

  const data = await fetchAPI<{ articles: NewsArticle[] }>(
    "/api/news?limit=10",
    token,
  );
  const articles = data.articles || [];

  stream.markdown(formatArticles(articles));
  stream.markdown(
    "\n\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
  );
  return { metadata: { command: "news" } };
}

async function handleBreaking(
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<vscode.ChatResult> {
  stream.markdown("🚨 **Breaking Crypto News**\n\n");
  stream.progress("Fetching breaking news…");

  const data = await fetchAPI<BreakingNewsResponse>(
    "/api/breaking?limit=10",
    token,
  );
  const articles = data.articles || [];

  if (articles.length === 0) {
    stream.markdown("*No breaking news right now. Check back soon!*\n");
    stream.markdown(
      "\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
    );
    return { metadata: { command: "breaking" } };
  }

  for (let i = 0; i < articles.length; i++) {
    const a = articles[i]!;
    const priority =
      a.priority === "high"
        ? "🔴 HIGH"
        : a.priority === "medium"
          ? "🟡 MED"
          : "";
    const badge = priority ? ` \`${priority}\`` : "";
    const sentimentIcon =
      a.sentiment === "bullish"
        ? "🟢"
        : a.sentiment === "bearish"
          ? "🔴"
          : "⚪";
    const desc = a.description ? `\n   > ${a.description.slice(0, 140)}` : "";

    stream.markdown(
      `${i + 1}. ${sentimentIcon}${badge} **${a.title}**${desc}\n   📰 ${a.source} • ${a.timeAgo}\n   🔗 [Read more](${safeUrl(a.link)})\n\n`,
    );
  }

  if (data.updatedAt) {
    stream.markdown(`*Updated: ${data.updatedAt}*\n`);
  }

  stream.markdown(
    "\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
  );
  return { metadata: { command: "breaking" } };
}

async function handlePrice(
  query: string,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<vscode.ChatResult> {
  const coin = query.trim().toLowerCase() || "bitcoin";
  stream.markdown(
    `💰 **Price: ${coin.charAt(0).toUpperCase() + coin.slice(1)}**\n\n`,
  );
  stream.progress(`Looking up ${coin} price…`);

  const data = await fetchAPI<{ prices: Record<string, PriceInfo> }>(
    `/api/prices?coin=${encodeURIComponent(coin)}`,
    token,
  );
  const prices = data.prices || {};

  if (Object.keys(prices).length === 0) {
    stream.markdown(`*Could not find price data for "${coin}".*`);
    return { metadata: { command: "price" } };
  }

  stream.markdown(
    "| Coin | Price | 24h Change | Market Cap | Volume (24h) |\n",
  );
  stream.markdown("|------|-------|------------|------------|-------------|\n");

  for (const [symbol, info] of Object.entries(prices).slice(0, 10)) {
    const changeEmoji =
      (info.change24h ?? 0) > 0
        ? "📈"
        : (info.change24h ?? 0) < 0
          ? "📉"
          : "➡️";
    const cap =
      info.marketCap !== undefined && Number.isFinite(info.marketCap)
        ? `$${(info.marketCap / 1e9).toFixed(2)}B`
        : "—";
    const vol =
      info.volume24h !== undefined && Number.isFinite(info.volume24h)
        ? `$${(info.volume24h / 1e9).toFixed(2)}B`
        : "—";
    stream.markdown(
      `| ${escapeTableCell(symbol.toUpperCase())} | $${fmtUsd(info.usd)} | ${changeEmoji} ${fmtNum(info.change24h, 2, "0.00")}% | ${cap} | ${vol} |\n`,
    );
  }

  stream.markdown(
    "\n\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
  );
  return { metadata: { command: "price" } };
}

async function handleMarket(
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<vscode.ChatResult> {
  stream.markdown("📊 **Market Overview**\n\n");
  stream.progress("Loading market data…");

  const [sentimentData, priceData, fgData] = await Promise.all([
    fetchAPI<SentimentResponse>("/api/sentiment", token).catch(() => null),
    fetchAPI<{ prices: Record<string, PriceInfo> }>(
      "/api/prices?limit=5",
      token,
    ),
    fetchAPI<FearGreedResponse>("/api/fear-greed", token).catch(() => null),
  ]);

  // --- Fear & Greed section ---
  if (fgData?.current) {
    const fgVal = clampFG(fgData.current.value);
    const fgLabel = fgData.current.valueClassification || "Unknown";
    stream.markdown(`### Fear & Greed Index\n\n`);
    stream.markdown(`${fearGreedEmoji(fgVal)} **${fgVal}** — ${fgLabel}\n\n`);
    stream.markdown(`${fearGreedBar(fgVal)}\n\n`);
    if (fgData.trend) {
      const dir =
        fgData.trend.direction === "improving"
          ? "⬆️"
          : fgData.trend.direction === "worsening"
            ? "⬇️"
            : "➡️";
      const c7 = fgData.trend.change7d ?? 0;
      const c30 = fgData.trend.change30d ?? 0;
      stream.markdown(
        `7d change: ${dir} ${c7 > 0 ? "+" : ""}${c7} · 30d change: ${c30 > 0 ? "+" : ""}${c30}\n\n`,
      );
    }
  }

  // --- Sentiment section ---
  if (sentimentData?.market) {
    const market = sentimentData.market;
    const score = Number(market.score);
    const emoji = score > 20 ? "🟢" : score < -20 ? "🔴" : "🟡";

    stream.markdown(`### News Sentiment\n\n`);
    stream.markdown(
      `**Overall:** ${emoji} ${sentimentLabel(market.overall || "neutral")} (score: ${score > 0 ? "+" : ""}${Number.isFinite(score) ? score : 0}/100, confidence: ${market.confidence ?? 0}%)\n\n`,
    );
    if (market.summary) {
      stream.markdown(`> ${market.summary}\n\n`);
    }

    // Use distribution counts if available
    const dist = sentimentData.distribution;
    if (dist) {
      if (dist.very_bullish !== undefined)
        stream.markdown(`- 🟢🟢 Very Bullish: ${dist.very_bullish}\n`);
      if (dist.bullish !== undefined)
        stream.markdown(`- 🟢 Bullish: ${dist.bullish}\n`);
      if (dist.neutral !== undefined)
        stream.markdown(`- ⚪ Neutral: ${dist.neutral}\n`);
      if (dist.bearish !== undefined)
        stream.markdown(`- 🔴 Bearish: ${dist.bearish}\n`);
      if (dist.very_bearish !== undefined)
        stream.markdown(`- 🔴🔴 Very Bearish: ${dist.very_bearish}\n`);
      stream.markdown("\n");
    }
  }

  // --- Top coins section ---
  const prices = priceData.prices || {};
  if (Object.keys(prices).length > 0) {
    stream.markdown("### Top Coins\n\n");
    stream.markdown("| Coin | Price | 24h |\n|------|-------|-----|\n");
    for (const [symbol, info] of Object.entries(prices).slice(0, 5)) {
      const change = info.change24h ?? 0;
      const arrow = change > 0 ? "📈" : change < 0 ? "📉" : "➡️";
      stream.markdown(
        `| ${escapeTableCell(symbol.toUpperCase())} | $${fmtUsd(info.usd)} | ${arrow} ${fmtNum(info.change24h, 2, "0.00")}% |\n`,
      );
    }
  }

  stream.markdown(
    "\n\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
  );
  return { metadata: { command: "market" } };
}

async function handleSentiment(
  query: string,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<vscode.ChatResult> {
  const coin = query.trim().toUpperCase() || "";

  if (!coin) {
    stream.markdown(
      "⚠️ Please provide a coin, e.g. `/sentiment BTC` or `/sentiment ethereum`",
    );
    return { metadata: { command: "sentiment" } };
  }

  stream.markdown(`🧠 **Sentiment Analysis: ${coin}**\n\n`);
  stream.progress(`Analyzing sentiment for ${coin}…`);

  const data = await fetchAPI<SentimentResponse>(
    `/api/sentiment?asset=${encodeURIComponent(coin)}&limit=20`,
    token,
  );

  // Market-level summary
  if (data.market) {
    const m = data.market;
    const score = Number(m.score);
    stream.markdown(`### Overall Market Mood\n\n`);
    stream.markdown(
      `${sentimentEmoji(m.overall)} **${sentimentLabel(m.overall)}** (score: ${score > 0 ? "+" : ""}${Number.isFinite(score) ? score : 0}/100, confidence: ${m.confidence ?? 0}%)\n\n`,
    );
    if (m.summary) {
      stream.markdown(`> ${m.summary}\n\n`);
    }

    if (m.keyDrivers && m.keyDrivers.length > 0) {
      stream.markdown("**Key Drivers:**\n");
      for (const driver of m.keyDrivers) {
        stream.markdown(`- ${driver}\n`);
      }
      stream.markdown("\n");
    }
  }

  // Distribution breakdown
  if (data.distribution) {
    const d = data.distribution;
    stream.markdown("### Sentiment Distribution\n\n");
    stream.markdown("| Sentiment | Count |\n|-----------|-------|\n");
    if (d.very_bullish !== undefined)
      stream.markdown(`| 🟢🟢 Very Bullish | ${d.very_bullish} |\n`);
    if (d.bullish !== undefined)
      stream.markdown(`| 🟢 Bullish | ${d.bullish} |\n`);
    if (d.neutral !== undefined)
      stream.markdown(`| ⚪ Neutral | ${d.neutral} |\n`);
    if (d.bearish !== undefined)
      stream.markdown(`| 🔴 Bearish | ${d.bearish} |\n`);
    if (d.very_bearish !== undefined)
      stream.markdown(`| 🔴🔴 Very Bearish | ${d.very_bearish} |\n`);
    stream.markdown("\n");
  }

  // High-impact news
  const highImpact =
    data.highImpactNews ||
    data.articles?.filter((a) => a.impactLevel === "high") ||
    [];
  if (highImpact.length > 0) {
    stream.markdown("### High-Impact News\n\n");
    for (const article of highImpact.slice(0, 5)) {
      const assets = (article.affectedAssets || []).join(", ");
      stream.markdown(
        `- ${sentimentEmoji(article.sentiment)} **${article.title}**\n  ${article.reasoning}\n  ⏱ ${article.timeHorizon} · Affects: ${assets}\n  🔗 [Read](${safeUrl(article.link)})\n\n`,
      );
    }
  }

  // Remaining articles summary
  const remaining = (data.articles || []).filter(
    (a) => a.impactLevel !== "high",
  );
  if (remaining.length > 0) {
    stream.markdown(
      `### Other ${coin} News (${remaining.length} articles)\n\n`,
    );
    for (const article of remaining.slice(0, 5)) {
      stream.markdown(
        `- ${sentimentEmoji(article.sentiment)} **${article.title}** — ${article.reasoning}\n`,
      );
    }
    if (remaining.length > 5) {
      stream.markdown(
        `\n*…and ${remaining.length - 5} more articles analyzed.*\n`,
      );
    }
  }

  stream.markdown(
    "\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
  );
  return { metadata: { command: "sentiment" } };
}

async function handleSearch(
  query: string,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<vscode.ChatResult> {
  if (!query) {
    stream.markdown(
      "⚠️ Please provide a search term, e.g. `/search bitcoin ETF`",
    );
    return { metadata: { command: "search" } };
  }

  stream.markdown(`🔍 **Search: "${query}"**\n\n`);
  stream.progress(`Searching for "${query}"…`);

  const data = await fetchAPI<{ articles: NewsArticle[]; total?: number }>(
    `/api/news?search=${encodeURIComponent(query)}&limit=10`,
    token,
  );
  const articles = data.articles || [];

  if (articles.length === 0) {
    stream.markdown(`*No articles found for "${query}".*`);
  } else {
    stream.markdown(`Found **${data.total ?? articles.length}** results:\n\n`);
    stream.markdown(formatArticles(articles));
  }

  stream.markdown(
    "\n\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
  );
  return { metadata: { command: "search" } };
}

async function handleGas(
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<vscode.ChatResult> {
  stream.markdown("⛽ **Ethereum Gas Prices**\n\n");
  stream.progress("Fetching gas prices…");

  const data = await fetchAPI<{ gas: GasPrice }>("/api/gas", token);
  const gas = data.gas || ({} as GasPrice);

  stream.markdown("| Speed | Gwei | Est. USD |\n");
  stream.markdown("|-------|------|----------|\n");
  stream.markdown(
    `| 🐢 Slow | ${fmtNum(gas.slow, 0)} gwei | ${gas.usdSlow !== undefined ? "$" + fmtNum(gas.usdSlow, 2) : "—"} |\n`,
  );
  stream.markdown(
    `| 🚶 Standard | ${fmtNum(gas.standard, 0)} gwei | ${gas.usdStandard !== undefined ? "$" + fmtNum(gas.usdStandard, 2) : "—"} |\n`,
  );
  stream.markdown(
    `| 🚀 Fast | ${fmtNum(gas.fast, 0)} gwei | ${gas.usdFast !== undefined ? "$" + fmtNum(gas.usdFast, 2) : "—"} |\n`,
  );

  stream.markdown(
    "\n\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
  );
  return { metadata: { command: "gas" } };
}

async function handleFearGreed(
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<vscode.ChatResult> {
  stream.markdown("😱 **Fear & Greed Index**\n\n");
  stream.progress("Fetching index…");

  // Single request — the endpoint may return either the structured response
  // (with `current`) or the legacy flat shape (`value` + `classification`).
  // We branch off the parsed payload rather than retrying on any failure, so
  // a network error isn't silently duplicated into a second failing fetch.
  const raw = await fetchAPI<Partial<FearGreedResponse> & Partial<FearGreedLegacy>>(
    "/api/fear-greed",
    token,
  );

  if (raw && (raw as FearGreedResponse).current) {
    const data = raw as FearGreedResponse;
    const value = clampFG(data.current.value);
    const label = data.current.valueClassification || "Neutral";

    stream.markdown(
      `**Current:** ${fearGreedEmoji(value)} **${value}** — ${label}\n\n`,
    );
    stream.markdown(`${fearGreedBar(value)}\n\n`);

    if (data.trend) {
      const dir =
        data.trend.direction === "improving"
          ? "⬆️"
          : data.trend.direction === "worsening"
            ? "⬇️"
            : "➡️";
      const c7 = data.trend.change7d ?? 0;
      const c30 = data.trend.change30d ?? 0;
      stream.markdown(`**Trend:** ${dir} ${data.trend.direction}\n`);
      stream.markdown(`- 7-day change: ${c7 > 0 ? "+" : ""}${c7}\n`);
      stream.markdown(`- 30-day change: ${c30 > 0 ? "+" : ""}${c30}\n\n`);
    }

    if (data.breakdown) {
      stream.markdown("### Breakdown\n\n");
      stream.markdown(
        "| Factor | Value | Weight |\n|--------|-------|--------|\n",
      );
      for (const [factor, info] of Object.entries(data.breakdown)) {
        const name = factor
          .replace(/([A-Z])/g, " $1")
          .replace(/^./, (s) => s.toUpperCase());
        const weight = Number.isFinite(Number(info.weight))
          ? (Number(info.weight) * 100).toFixed(0)
          : "—";
        stream.markdown(
          `| ${escapeTableCell(name)} | ${info.value} | ${weight}% |\n`,
        );
      }
      stream.markdown("\n");
    }

    stream.markdown(`*Updated: ${data.lastUpdated || "Recently"}*\n`);
    stream.markdown(
      "\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
    );
    return { metadata: { command: "fear-greed" } };
  }

  // Legacy flat shape fallback
  const legacy = raw as FearGreedLegacy;
  const value = clampFG(legacy.value);
  const label = legacy.classification || "Neutral";

  stream.markdown(
    `**Current:** ${fearGreedEmoji(value)} **${value}** — ${label}\n\n`,
  );
  stream.markdown(`${fearGreedBar(value)}\n\n`);

  if (legacy.previous) {
    const prev = legacy.previous;
    const prevVal = clampFG(prev.value);
    const dir = prevVal < value ? "⬆️" : prevVal > value ? "⬇️" : "➡️";
    stream.markdown(
      `**Previous:** ${prevVal} — ${prev.classification} ${dir}\n\n`,
    );
  }

  stream.markdown(`*Updated: ${legacy.timestamp || "Recently"}*\n`);
  stream.markdown(
    "\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
  );
  return { metadata: { command: "fear-greed" } };
}

async function handleExplain(
  topic: string,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<vscode.ChatResult> {
  if (!topic) {
    stream.markdown(
      "⚠️ Please provide a topic to explain, e.g. `/explain Bitcoin` or `/explain DeFi`",
    );
    return { metadata: { command: "explain" } };
  }

  stream.markdown(`💡 **Why is "${topic}" Trending?**\n\n`);
  stream.progress(`Researching "${topic}"…`);

  // Try the AI trending-explainer endpoint first
  try {
    const data = await fetchAPI<ExplainResponse>(
      `/api/ai/explain?topic=${encodeURIComponent(topic)}&includePrice=true`,
      token,
    );

    if (data.success && data.explanation) {
      const ex = data.explanation;

      stream.markdown(`### Summary\n\n${ex.summary}\n\n`);

      if (ex.background) {
        stream.markdown(`### Background\n\n${ex.background}\n\n`);
      }

      if (ex.whyTrending) {
        stream.markdown(`### Why It's Trending\n\n${ex.whyTrending}\n\n`);
      }

      if (ex.marketImplications) {
        stream.markdown(
          `### Market Implications\n\n${ex.marketImplications}\n\n`,
        );
      }

      if (ex.outlook) {
        stream.markdown(`### Outlook\n\n${ex.outlook}\n\n`);
      }

      if (data.recentHeadlines && data.recentHeadlines.length > 0) {
        stream.markdown("### Recent Headlines\n\n");
        for (const hl of data.recentHeadlines.slice(0, 5)) {
          stream.markdown(`- ${hl}\n`);
        }
        stream.markdown("\n");
      }

      if (data.articleCount) {
        stream.markdown(`*Based on ${data.articleCount} recent articles.*\n`);
      }

      stream.markdown(
        "\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
      );
      return { metadata: { command: "explain" } };
    }

    // AI endpoint returned no results — try glossary fallback
    if (data.message) {
      stream.markdown(`*${data.message}*\n\n`);
    }
  } catch {
    // AI explain endpoint unavailable, fall through to glossary
  }

  // Glossary fallback
  stream.progress(`Looking up "${topic}" in glossary…`);
  try {
    const glossary = await fetchAPI<{ term: GlossaryTerm }>(
      `/api/glossary?term=${encodeURIComponent(topic)}`,
      token,
    );

    if (glossary.term) {
      stream.markdown(`### 📖 ${glossary.term.term}\n\n`);
      stream.markdown(`${glossary.term.definition}\n\n`);
      if (glossary.term.category) {
        stream.markdown(`**Category:** ${glossary.term.category}\n\n`);
      }
      if (glossary.term.relatedTerms && glossary.term.relatedTerms.length > 0) {
        stream.markdown(
          `**Related:** ${glossary.term.relatedTerms.join(", ")}\n`,
        );
      }
    } else {
      stream.markdown(
        `*No explanation or glossary entry found for "${topic}".*`,
      );
    }
  } catch {
    stream.markdown(
      `*Could not find an explanation for "${topic}". Try a different topic.*`,
    );
  }

  stream.markdown(
    "\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
  );
  return { metadata: { command: "explain" } };
}

async function handleResearch(
  topic: string,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<vscode.ChatResult> {
  if (!topic) {
    stream.markdown(
      "⚠️ Please provide a topic, e.g. `/research Bitcoin` or `/research Solana DeFi`",
    );
    return { metadata: { command: "research" } };
  }

  stream.markdown(`🔬 **Research Report: ${topic}**\n\n`);
  stream.progress(
    `Generating deep research on "${topic}"… (this may take a moment)`,
  );

  const data = await fetchAPI<ResearchResponse>(
    `/api/ai/research?topic=${encodeURIComponent(topic)}`,
    token,
  );

  if (!data.success) {
    stream.markdown(
      `*${data.error || `Could not generate research for "${topic}".`}*\n`,
    );
    stream.markdown(
      "\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
    );
    return { metadata: { command: "research" } };
  }

  const report = data.report;

  if (!report) {
    stream.markdown(
      `*No report data returned for "${topic}". Try a different query.*\n`,
    );
    stream.markdown(
      "\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
    );
    return { metadata: { command: "research" } };
  }

  // Sentiment badge
  const badge = sentimentEmoji(report.sentiment);
  stream.markdown(
    `**Sentiment:** ${badge} ${sentimentLabel(report.sentiment)}\n\n`,
  );

  // Price data if available
  if (report.priceData) {
    const p = report.priceData;
    const c24 = p.change24h ?? 0;
    const ch24 = c24 > 0 ? "📈" : c24 < 0 ? "📉" : "➡️";
    stream.markdown(
      `**Price:** $${fmtUsd(p.price)} ${ch24} ${fmtNum(p.change24h, 2, "0.00")}% (24h)`,
    );
    if (p.change7d !== undefined) {
      stream.markdown(` · ${fmtNum(p.change7d, 2, "0.00")}% (7d)`);
    }
    stream.markdown("\n");
    if (report.marketCap !== undefined && Number.isFinite(report.marketCap)) {
      stream.markdown(
        `**Market Cap:** $${(report.marketCap / 1e9).toFixed(2)}B\n`,
      );
    }
    stream.markdown("\n");
  }

  // Summary
  stream.markdown(`### Executive Summary\n\n${report.summary}\n\n`);

  // Key findings
  if (report.keyFindings && report.keyFindings.length > 0) {
    stream.markdown("### Key Findings\n\n");
    for (const finding of report.keyFindings) {
      stream.markdown(`- ${finding}\n`);
    }
    stream.markdown("\n");
  }

  // Opportunities
  if (report.opportunities && report.opportunities.length > 0) {
    stream.markdown("### Opportunities\n\n");
    for (const opp of report.opportunities) {
      stream.markdown(`- 🟢 ${opp}\n`);
    }
    stream.markdown("\n");
  }

  // Risks
  if (report.risks && report.risks.length > 0) {
    stream.markdown("### Risks\n\n");
    for (const risk of report.risks) {
      stream.markdown(`- ⚠️ ${risk}\n`);
    }
    stream.markdown("\n");
  }

  // Outlook
  if (report.outlook) {
    stream.markdown(`### Outlook\n\n${report.outlook}\n\n`);
  }

  if (data.articlesAnalyzed) {
    stream.markdown(
      `*Based on analysis of ${data.articlesAnalyzed} recent articles.*\n`,
    );
  }

  stream.markdown(
    "\n---\n*Source: [cryptocurrency.cv](https://cryptocurrency.cv)*",
  );
  return { metadata: { command: "research" } };
}

// ---------------------------------------------------------------------------
// Chat participant
// ---------------------------------------------------------------------------

const chatHandler: vscode.ChatRequestHandler = async (
  request: vscode.ChatRequest,
  _context: vscode.ChatContext,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<vscode.ChatResult> => {
  const command = request.command;
  const query = request.prompt.trim();

  try {
    switch (command) {
      case "breaking":
        return await handleBreaking(stream, token);
      case "news":
        return await handleNews(stream, token);
      case "price":
        return await handlePrice(query, stream, token);
      case "market":
        return await handleMarket(stream, token);
      case "sentiment":
        return await handleSentiment(query, stream, token);
      case "search":
        return await handleSearch(query, stream, token);
      case "gas":
        return await handleGas(stream, token);
      case "fear-greed":
        return await handleFearGreed(stream, token);
      case "explain":
        return await handleExplain(query, stream, token);
      case "research":
        return await handleResearch(query, stream, token);
      default:
        // No command — treat prompt as a search if text is present
        if (query) {
          return await handleSearch(query, stream, token);
        }
        // Show help
        stream.markdown("👋 **Welcome to @crypto!**\n\n");
        stream.markdown("Available commands:\n\n");
        stream.markdown("| Command | Description |\n");
        stream.markdown("|---------|-------------|\n");
        stream.markdown("| `/breaking` | Latest breaking crypto news |\n");
        stream.markdown("| `/news` | Latest crypto news headlines |\n");
        stream.markdown(
          "| `/price <coin>` | Current price (e.g. `/price bitcoin`) |\n",
        );
        stream.markdown(
          "| `/market` | Market overview with prices & Fear/Greed |\n",
        );
        stream.markdown(
          "| `/sentiment <coin>` | AI sentiment analysis (e.g. `/sentiment BTC`) |\n",
        );
        stream.markdown("| `/search <query>` | Search news articles |\n");
        stream.markdown("| `/gas` | Ethereum gas prices |\n");
        stream.markdown("| `/fear-greed` | Fear & Greed Index |\n");
        stream.markdown("| `/explain <topic>` | Why is a topic trending? |\n");
        stream.markdown("| `/research <topic>` | Deep AI research report |\n");
        stream.markdown(
          "\nOr just type a question and I'll search for relevant news.\n",
        );
        return { metadata: { command: "help" } };
    }
  } catch (error: unknown) {
    let message = "An unknown error occurred.";
    if (error instanceof Error) {
      message = error.name === "AbortError"
        ? "Request was cancelled or timed out."
        : error.message || message;
    }
    stream.markdown(`\n\n❌ **Error:** ${message}\n\nPlease try again later.`);
    return { metadata: { command: command ?? "unknown", error: true } };
  }
};

// ---------------------------------------------------------------------------
// Extension lifecycle
// ---------------------------------------------------------------------------

export function activate(context: vscode.ExtensionContext) {
  // Register @crypto chat participant
  const participant = vscode.chat.createChatParticipant(
    "crypto-news.crypto",
    chatHandler,
  );
  participant.iconPath = vscode.Uri.joinPath(
    context.extensionUri,
    "media",
    "icon.png",
  );
  context.subscriptions.push(participant);

  // Refresh command
  context.subscriptions.push(
    vscode.commands.registerCommand("crypto.refresh", async () => {
      vscode.window.showInformationMessage("Crypto data refreshed!");
    }),
  );

  // Dashboard command
  context.subscriptions.push(
    vscode.commands.registerCommand("crypto.openDashboard", async () => {
      const panel = vscode.window.createWebviewPanel(
        "cryptoDashboard",
        "Crypto Dashboard",
        vscode.ViewColumn.One,
        {
          // Panel is purely static informational content — no scripts needed.
          enableScripts: false,
          // Restrict the webview to this extension's own resources.
          localResourceRoots: [context.extensionUri],
        },
      );
      panel.webview.html = getDashboardHTML();
    }),
  );

  console.log("Crypto News Copilot extension activated!");
}

function getDashboardHTML(): string {
  // Strict CSP: inline styles only (the page ships with an inline <style>
  // block), no scripts, no remote loads. Webview is also created with
  // enableScripts: false and a restricted localResourceRoots.
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';" />
  <title>Crypto Dashboard</title>
  <style>
    body { font-family: system-ui; padding: 20px; background: #1e1e1e; color: #fff; }
    h1 { color: #ffffff; }
    .card { background: #2d2d2d; border-radius: 8px; padding: 16px; margin: 12px 0; }
    .bullish { color: #00ff88; }
    .bearish { color: #ff4444; }
    a { color: #58a6ff; }
    code { background: #3d3d3d; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>📰 Crypto Dashboard</h1>
  <div class="card">
    <p>Use <code>@crypto</code> in Copilot Chat to get started!</p>
    <p>Available commands:</p>
    <ul>
      <li><code>/breaking</code> — Breaking crypto news</li>
      <li><code>/news</code> — Latest crypto news</li>
      <li><code>/price &lt;coin&gt;</code> — Current price for a coin</li>
      <li><code>/market</code> — Market overview with prices &amp; Fear/Greed</li>
      <li><code>/sentiment &lt;coin&gt;</code> — AI sentiment analysis for a coin</li>
      <li><code>/search &lt;query&gt;</code> — Search news articles</li>
      <li><code>/gas</code> — Ethereum gas prices</li>
      <li><code>/fear-greed</code> — Fear &amp; Greed Index</li>
      <li><code>/explain &lt;topic&gt;</code> — Why is a topic trending?</li>
      <li><code>/research &lt;topic&gt;</code> — Deep AI research report</li>
    </ul>
  </div>
  <div class="card">
    <p>Powered by <a href="https://cryptocurrency.cv">Free Crypto News API</a></p>
  </div>
</body>
</html>`;
}

export function deactivate() {}
