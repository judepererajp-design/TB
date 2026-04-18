# Crypto News for GitHub Copilot

Query real-time crypto news, prices, gas fees, Fear & Greed Index, and more — directly in GitHub Copilot Chat.

Powered by the free [cryptocurrency.cv](https://cryptocurrency.cv) API. No API key required.

## Features

- 🚨 **Breaking News** — Breaking crypto headlines with priority & sentiment indicators
- 📰 **Latest News** — Top crypto headlines from 200+ sources
- 💰 **Live Prices** — Real-time coin prices with 24h change, market cap & volume
- 📊 **Market Overview** — Prices, Fear & Greed Index, and news sentiment in one view
- 🧠 **Sentiment Analysis** — AI-powered per-coin sentiment with key drivers & impact levels
- 🔍 **Search** — Full-text search across news articles
- ⛽ **Gas Prices** — Current Ethereum gas (slow / standard / fast) with USD estimates
- 😱 **Fear & Greed** — Market emotion gauge with trend, 7d/30d changes & factor breakdown
- 💡 **Trending Explainer** — AI-generated explanation of why a topic is trending
- 🔬 **Deep Research** — AI research reports with key findings, risks, opportunities & outlook

## Installation

### From VS Code Marketplace

1. Open **Extensions** in VS Code (`Ctrl+Shift+X`)
2. Search for **"Crypto News for Copilot"**
3. Click **Install**
4. Ensure GitHub Copilot Chat is enabled

### Manual / Dev Install

```bash
cd copilot-extension
pnpm install
bun run compile
# Press F5 in VS Code to launch Extension Development Host
```

## Usage

Open **Copilot Chat** and type `@crypto` followed by a command:

```
@crypto /breaking
@crypto /news
@crypto /price bitcoin
@crypto /market
@crypto /sentiment BTC
@crypto /search ethereum ETF
@crypto /gas
@crypto /fear-greed
@crypto /explain DeFi
@crypto /research Solana
```

Or type a free-form question:

```
@crypto what's happening with Solana?
@crypto latest Ethereum news
```

<!-- screenshot placeholder -->
<!-- ![screenshot](media/screenshot.png) -->

## Commands

| Command             | Description                                    | Example                    |
| ------------------- | ---------------------------------------------- | -------------------------- |
| `/breaking`         | Latest breaking crypto news with priority      | `@crypto /breaking`        |
| `/news`             | Latest crypto news headlines                   | `@crypto /news`            |
| `/price <coin>`     | Current price, 24h change, market cap & volume | `@crypto /price bitcoin`   |
| `/market`           | Market overview with prices, F&G & sentiment   | `@crypto /market`          |
| `/sentiment <coin>` | AI sentiment analysis for a coin               | `@crypto /sentiment BTC`   |
| `/search <query>`   | Search news articles                           | `@crypto /search SEC`      |
| `/gas`              | Ethereum gas prices (slow/standard/fast)       | `@crypto /gas`             |
| `/fear-greed`       | Fear & Greed Index with trend & breakdown      | `@crypto /fear-greed`      |
| `/explain <topic>`  | Why is a topic trending? AI explainer          | `@crypto /explain staking` |
| `/research <topic>` | Deep AI research report                        | `@crypto /research Solana` |

## Configuration

Access via **Settings → Extensions → Crypto News**:

| Setting                | Default                     | Description               |
| ---------------------- | --------------------------- | ------------------------- |
| `crypto.apiUrl`        | `https://cryptocurrency.cv` | API base URL              |
| `crypto.defaultLimit`  | `10`                        | Items per request         |
| `crypto.showSentiment` | `true`                      | Show sentiment indicators |

## API

This extension uses the [Free Crypto News API](https://cryptocurrency.cv):

- 200+ news sources aggregated in real time
- No API key required for basic usage
- JSON REST endpoints, RSS/Atom feeds
- [Full API docs](https://cryptocurrency.cv/developers)

## Development

```bash
cd copilot-extension
pnpm install
bun run watch          # compile in watch mode
# Press F5 to launch Extension Development Host
bun run compile        # one-time build
bun run lint           # lint sources
```

## License

See [LICENSE](../LICENSE) file.

## Links

- [cryptocurrency.cv](https://cryptocurrency.cv)
- [API Documentation](https://cryptocurrency.cv/developers)
- [GitHub Repository](https://github.com/nirholas/free-crypto-news)
