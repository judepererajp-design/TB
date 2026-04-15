"""
TitanBot Pro — Narrative Tracker
==================================
Project Inheritance Feature 8.

Groups headlines into keyword-based thematic clusters (no LLM).
Tracks each narrative's velocity (articles/24h vs 7d avg).
Classifies momentum as: rising, peaked, fading.

Rules:
  - Narrative requires ≥3 articles to be recognised.
  - Rising narrative can adjust confidence (bonus in ensemble voter).
  - Must NOT trigger trades independently.
  - Total contextual adjustments (narrative + F&G + pump/dump) capped at ±20%.

Fallback: if no narratives detected, returns empty — no effect on pipeline.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from config.constants import NewsIntelligence
from config.feature_flags import ff
from config.shadow_mode import shadow_log

logger = logging.getLogger(__name__)


# ── Narrative keyword clusters ─────────────────────────────────
# Each entry: (narrative_name, keywords_set)
# A headline matches a narrative if ≥1 keyword is found in the title.
_NARRATIVE_KEYWORDS: List[tuple] = [
    ("Bitcoin ETF", {"etf", "blackrock etf", "spot etf", "etf approval",
                     "etf inflow", "etf outflow", "gbtc", "ibit"}),
    ("Ethereum Upgrade", {"ethereum upgrade", "dencun", "pectra", "eip",
                          "proto-danksharding", "blob", "eth upgrade"}),
    ("Regulatory Action", {"sec", "cftc", "regulation", "regulatory",
                           "lawsuit", "enforcement", "compliance"}),
    ("DeFi Growth", {"defi", "tvl", "lending protocol", "yield",
                     "liquidity pool", "dex volume", "aave", "uniswap"}),
    ("Stablecoin Flow", {"stablecoin", "usdt", "usdc", "tether",
                         "circle", "depeg", "stablecoin mint",
                         "stablecoin burn"}),
    ("AI Crypto", {"ai crypto", "ai token", "artificial intelligence",
                   "machine learning", "gpt", "llm", "fetch.ai",
                   "singularitynet", "render", "bittensor"}),
    ("Layer 2", {"layer 2", "l2", "rollup", "optimism", "arbitrum",
                 "base", "zk-rollup", "zksync", "scroll", "linea"}),
    ("Bitcoin Halving", {"halving", "halvening", "block reward",
                         "miner reward", "bitcoin halving"}),
    ("Exchange Crisis", {"exchange hack", "insolvency", "withdrawal freeze",
                         "bankrupt", "ftx", "rug pull", "exit scam"}),
    ("Institutional Adoption", {"institutional", "wall street",
                                 "pension fund", "sovereign fund",
                                 "corporate treasury", "microstrategy",
                                 "tesla bitcoin"}),
    ("Macro Risk", {"fed", "interest rate", "inflation", "cpi",
                    "recession", "gdp", "unemployment", "fomc",
                    "hawkish", "dovish", "rate cut", "rate hike"}),
    ("Geopolitical", {"war", "conflict", "sanction", "tariff",
                      "trade war", "geopolitical", "military",
                      "escalation", "nuclear"}),
    ("Memecoin", {"memecoin", "meme coin", "doge", "shib",
                  "pepe", "bonk", "wif", "floki"}),
    ("NFT / Gaming", {"nft", "gaming", "metaverse", "play-to-earn",
                      "axie", "sandbox", "immutable"}),
]


@dataclass
class NarrativeState:
    """Tracks a single narrative's state over time."""
    name: str
    articles_24h: int = 0
    unique_titles_24h: int = 0
    articles_7d_avg: float = 0.0
    velocity: float = 0.0  # articles_24h / articles_7d_avg
    momentum: Literal["rising", "peaked", "fading", "dormant"] = "dormant"
    last_headline_ts: float = 0.0
    headline_timestamps: List[float] = field(default_factory=list)


class NarrativeTracker:
    """
    Groups headlines into thematic narratives and tracks momentum.
    Call process_headline() for each new headline, then query
    get_active_narratives() for current narrative states.
    """

    def __init__(self):
        # narrative_name -> list of (timestamp, title)
        self._history: Dict[str, List[tuple]] = defaultdict(list)
        # Cached narrative states
        self._states: Dict[str, NarrativeState] = {}

    def process_headline(self, title: str, published_at: float):
        """
        Classify a headline into narratives and record it.
        A headline can match multiple narratives.
        """
        if not ff.is_active("NARRATIVE_TRACKER"):
            return

        low = title.lower()
        for narrative_name, keywords in _NARRATIVE_KEYWORDS:
            if any(kw in low for kw in keywords):
                self._history[narrative_name].append((published_at, title))

        # Prune entries older than 7 days
        cutoff = time.time() - (7 * 86400)
        for name in list(self._history):
            self._history[name] = [
                (ts, t) for ts, t in self._history[name] if ts > cutoff
            ]
            if not self._history[name]:
                del self._history[name]

        self._recompute_states()

    def _recompute_states(self):
        """Recompute narrative states from history."""
        now = time.time()
        cutoff_24h = now - 86400
        cutoff_7d = now - (7 * 86400)

        self._states.clear()

        for name, entries in self._history.items():
            articles_24h = sum(1 for ts, _ in entries if ts > cutoff_24h)
            unique_titles_24h = len({
                title.lower().strip()
                for ts, title in entries
                if ts > cutoff_24h
            })
            articles_7d = len([ts for ts, _ in entries if ts > cutoff_7d])
            articles_7d_avg = articles_7d / 7.0 if articles_7d > 0 else 0.0

            # Skip narratives below minimum article threshold or where
            # one syndicated headline is being replayed everywhere.
            if (
                articles_24h < NewsIntelligence.NARRATIVE_MIN_ARTICLES
                or unique_titles_24h < NewsIntelligence.NARRATIVE_MIN_ARTICLES
            ):
                continue

            velocity = (
                articles_24h / articles_7d_avg
                if articles_7d_avg > 0 else float(articles_24h)
            )

            if velocity >= NewsIntelligence.NARRATIVE_VELOCITY_RISING_THRESHOLD:
                momentum = "rising"
            elif velocity <= NewsIntelligence.NARRATIVE_VELOCITY_FADING_THRESHOLD:
                momentum = "fading"
            else:
                momentum = "peaked"

            last_ts = max(ts for ts, _ in entries) if entries else 0.0

            self._states[name] = NarrativeState(
                name=name,
                articles_24h=articles_24h,
                unique_titles_24h=unique_titles_24h,
                articles_7d_avg=round(articles_7d_avg, 1),
                velocity=round(velocity, 2),
                momentum=momentum,
                last_headline_ts=last_ts,
                headline_timestamps=[ts for ts, _ in entries if ts > cutoff_24h],
            )

    def get_active_narratives(self) -> List[NarrativeState]:
        """
        Get all active narratives (≥3 articles in 24h).
        Sorted by velocity descending.
        """
        return sorted(
            self._states.values(),
            key=lambda n: n.velocity,
            reverse=True,
        )

    def get_narrative_confidence_adjustment(self, symbol: str, direction: str) -> float:
        """
        Returns a confidence multiplier adjustment from active narratives.
        Capped at ±CONTEXTUAL_CONFIDENCE_CAP (±20%).

        A rising bullish narrative → small boost for LONGs.
        A rising bearish narrative → small boost for SHORTs.
        This is a contextual overlay, NOT a signal generator.

        Returns:
            Float multiplier (e.g. 1.05 for +5% boost, 0.95 for -5% penalty).
            Returns 1.0 if feature is off or no relevant narratives.
        """
        if not ff.is_active("NARRATIVE_TRACKER"):
            return 1.0

        adjustment = 0.0
        cap = NewsIntelligence.CONTEXTUAL_CONFIDENCE_CAP

        _BULLISH_NARRATIVES = {
            "Bitcoin ETF", "Institutional Adoption", "DeFi Growth",
            "AI Crypto", "Layer 2", "Bitcoin Halving",
        }
        _BEARISH_NARRATIVES = {
            "Regulatory Action", "Exchange Crisis", "Macro Risk",
            "Geopolitical",
        }

        for state in self._states.values():
            if state.momentum != "rising":
                continue

            # Per-narrative adjustment: ±2-5% based on velocity
            per_narrative_pct = min(0.05, state.velocity * 0.01)

            if state.name in _BULLISH_NARRATIVES:
                if direction == "LONG":
                    adjustment += per_narrative_pct
                else:
                    adjustment -= per_narrative_pct * 0.5
            elif state.name in _BEARISH_NARRATIVES:
                if direction == "SHORT":
                    adjustment += per_narrative_pct
                else:
                    adjustment -= per_narrative_pct * 0.5

        # Clamp total adjustment to ±cap
        adjustment = max(-cap, min(cap, adjustment))
        multiplier = 1.0 + adjustment

        if ff.is_shadow("NARRATIVE_TRACKER"):
            shadow_log("NARRATIVE_TRACKER", {
                "symbol": symbol,
                "direction": direction,
                "active_narratives": [
                    {
                        "name": s.name,
                        "momentum": s.momentum,
                        "velocity": s.velocity,
                        "unique_titles_24h": s.unique_titles_24h,
                    }
                    for s in self._states.values()
                ],
                "adjustment": adjustment,
                "multiplier": multiplier,
            })
            from config.fcn_logger import fcn_log
            _active = [s.name for s in self._states.values() if s.momentum == "rising"]
            fcn_log("NARRATIVE_TRACKER", f"shadow | {symbol} dir={direction} adj={adjustment:+.3f} narratives={_active}")
            return 1.0  # Shadow: no effect

        if ff.is_enabled("NARRATIVE_TRACKER"):
            from config.fcn_logger import fcn_log
            _active = [s.name for s in self._states.values() if s.momentum == "rising"]
            fcn_log("NARRATIVE_TRACKER", f"live | {symbol} dir={direction} adj={adjustment:+.3f} mult={multiplier:.3f} narratives={_active}")
        return multiplier if ff.is_enabled("NARRATIVE_TRACKER") else 1.0

    def get_raw_adjustment(self, symbol: str, direction: str) -> float:
        """Return the raw narrative adjustment for global contextual cap wiring."""
        if not ff.is_active("NARRATIVE_TRACKER"):
            return 0.0
        if ff.is_shadow("NARRATIVE_TRACKER"):
            # Compute and log for visibility, but return 0.0 (no live effect)
            _adj = self.get_narrative_confidence_adjustment(symbol, direction) - 1.0
            from config.fcn_logger import fcn_log
            _active = [s.name for s in self._states.values() if s.momentum == "rising"]
            fcn_log("NARRATIVE_TRACKER", f"shadow | {symbol} dir={direction} raw_adj={_adj:+.3f} narratives={_active} (no live effect)")
            return 0.0
        return self.get_narrative_confidence_adjustment(symbol, direction) - 1.0

    def get_summary(self) -> Dict:
        """Return a summary for dashboards / Telegram commands."""
        return {
            "active_count": len(self._states),
            "narratives": [
                {
                    "name": s.name,
                    "articles_24h": s.articles_24h,
                    "unique_titles_24h": s.unique_titles_24h,
                    "velocity": s.velocity,
                    "momentum": s.momentum,
                }
                for s in sorted(
                    self._states.values(),
                    key=lambda n: n.velocity,
                    reverse=True,
                )
            ],
        }


# Singleton
narrative_tracker = NarrativeTracker()


def clamp_contextual_adjustments(
    fear_greed_adj: float = 0.0,
    narrative_adj: float = 0.0,
    pump_dump_adj: float = 0.0,
) -> float:
    """
    Enforce the GLOBAL contextual confidence cap (±20%) across ALL
    contextual signals combined (Fear & Greed + Narrative + Pump/Dump).

    Each argument is a raw adjustment amount (e.g. -0.07 for 7% tighter).
    Returns the clamped combined multiplier (e.g. 0.93).

    This prevents correlated contextual signals from compounding
    beyond the intended ±20% limit.
    """
    cap = NewsIntelligence.CONTEXTUAL_CONFIDENCE_CAP
    total = fear_greed_adj + narrative_adj + pump_dump_adj
    clamped = max(-cap, min(cap, total))

    if abs(total) > cap:
        logger.debug(
            f"🔒 Contextual cap hit: raw_total={total:+.3f} "
            f"(F&G={fear_greed_adj:+.3f}, narrative={narrative_adj:+.3f}, "
            f"pump_dump={pump_dump_adj:+.3f}) → clamped to {clamped:+.3f}"
        )

    return 1.0 + clamped
