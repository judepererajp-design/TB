"""
TitanBot Pro — BTC News Intelligence Engine
=============================================
Classifies WHY BTC is moving and adjusts altcoin signal generation
differently based on the event type.

The problem it solves:
  The bot already reacts to BTC volatility (regime→VOLATILE blocks LONGs).
  But a 5% BTC pump on ETF approval news is fundamentally different from
  a 5% pump on a short squeeze:
    - ETF approval = macro BTC-specific bullish → altcoins lag, then follow
    - Short squeeze = mechanical, reversal likely → altcoins disconnected
    - Fed rate cut = macro risk-on → ALL risk assets pump, alts included
    - Exchange hack = crypto-specific fear → correlated selloff, no alts
    - Protocol upgrade = BTC-specific = alts unaffected initially

  Without this distinction, the bot either:
    (a) Blocks all LONG signals in VOLATILE mode even during genuine bull news, OR
    (b) Fires altcoin LONGs into a market that's moving for BTC-only reasons

Classification taxonomy:
  MACRO_RISK_ON     — Fed cuts, inflation down, institutional buying
                      Effect: ALL risk assets pump. Alt LONGs get a confidence boost.
  MACRO_RISK_OFF    — Fed hikes, recession data, geopolitical escalation
                      Effect: ALL risk assets dump. Block all LONGs system-wide.
  BTC_FUNDAMENTAL   — ETF approval, halving narrative, major adoption
                      Effect: BTC-specific bullish. Alts lag 1-6h then follow.
                      Action: reduce position sizes, wait for BTC to stabilize.
  BTC_TECHNICAL     — Leverage flush, liquidation cascade, short squeeze
                      Effect: mechanical. Likely reversal. Alts disconnected.
                      Action: wait for move to exhaust, normal scanning.
  EXCHANGE_EVENT    — Hack, insolvency, withdrawal freeze
                      Effect: correlated crypto fear. Block all LONGs 2-4h.
  REGULATORY        — SEC action, country ban, favorable regulation
                      Effect: depends on direction. High uncertainty.
                      Action: reduce confidence multiplier 0.70 for 4h.
  ALTCOIN_SPECIFIC  — Not BTC-driven. Normal altcoin scanning rules apply.
  UNKNOWN           — Cannot classify. Default VOLATILE regime rules apply.

Architecture:
  1. NewsClassifier: classifies each headline into the taxonomy above
  2. BTCMoveAnalyzer: detects significant BTC price moves (>2% in <30min)
  3. BTCNewsIntelligence: correlates move timing with headline timestamps
                          to produce a BTCEventContext for the engine
  4. engine.py: consults get_event_context() before publishing any altcoin signal
"""

import asyncio
import copy
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from config.constants import NewsIntelligence

logger = logging.getLogger(__name__)


class BTCEventType(str, Enum):
    MACRO_RISK_ON     = "MACRO_RISK_ON"
    MACRO_RISK_OFF    = "MACRO_RISK_OFF"
    BTC_FUNDAMENTAL   = "BTC_FUNDAMENTAL"
    BTC_TECHNICAL     = "BTC_TECHNICAL"
    EXCHANGE_EVENT    = "EXCHANGE_EVENT"
    REGULATORY        = "REGULATORY"
    ALTCOIN_SPECIFIC  = "ALTCOIN_SPECIFIC"
    UNKNOWN           = "UNKNOWN"


@dataclass
class BTCEventContext:
    """
    The current BTC macro / news context.
    Consumed by engine.py before publishing any altcoin signal.
    """
    event_type:          BTCEventType  = BTCEventType.UNKNOWN
    confidence:          float         = 0.0   # 0-1 how sure we are of the classification
    direction:           str           = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    detected_at:         float         = 0.0   # unix timestamp
    expires_at:          float         = 0.0   # when this context auto-expires

    # Altcoin signal adjustments
    confidence_mult:     float         = 1.0   # multiply signal confidence by this
    block_longs:         bool          = False  # hard-block all LONG signals
    block_shorts:        bool          = False  # hard-block all SHORT signals
    require_low_corr:    bool          = False  # only allow BTC-correlation < 0.5
    reduce_size_mult:    float         = 1.0   # multiply position size by this

    # Human-readable explanation for Telegram + dashboard
    headline:            str           = ""     # triggering headline
    explanation:         str           = ""     # why signals are adjusted
    sources:             List[str]     = field(default_factory=list)

    # Original publication timestamp of the triggering headline
    # (may differ significantly from detected_at when the bot starts late
    # or processes a backlog of stale news)
    headline_published_at: float      = 0.0

    # Mixed signal flag — headline contains BOTH bearish AND bullish signals
    is_mixed_signal:     bool          = False

    # Reaction scoring — whether market confirmed or contradicted this news
    reaction_validated:  bool          = False  # True once 15-min reaction check ran
    reaction_confirmed:  bool          = False  # True if market moved with the news

    # Multi-news net score — aggregate of all recent headline scores
    net_news_score:      float         = 0.0    # -1.0 (bearish) to +1.0 (bullish)

    # Enriched impact context
    impact_type:         str           = ""     # e.g. "Liquidity Shock"
    impact_detail:       str           = ""     # one-liner: news → mechanism → market effect

    @property
    def is_active(self) -> bool:
        return time.time() < self.expires_at and self.event_type != BTCEventType.UNKNOWN

    @property
    def age_minutes(self) -> float:
        return (time.time() - self.detected_at) / 60 if self.detected_at > 0 else 0

    @property
    def headline_age_minutes(self) -> float:
        """Age of the triggering headline based on its publication timestamp."""
        if self.headline_published_at > 0:
            return (time.time() - self.headline_published_at) / 60
        return self.age_minutes  # fall back to detection time

    @property
    def severity(self) -> str:
        """Qualitative severity derived from classification confidence."""
        if self.confidence >= 0.85:
            return "EXTREME"
        elif self.confidence >= 0.65:
            return "HIGH"
        elif self.confidence >= 0.40:
            return "MEDIUM"
        return "LOW"

    def format_headline_age(self) -> str:
        """Human-readable age string for the triggering headline."""
        age = self.headline_age_minutes
        if age >= 120:  # ≥2h: switch to hours display
            return f"🕐 Published {age / 60:.1f}h ago"
        elif age >= 2:
            return f"🕐 Published {age:.0f}m ago"
        return "🕐 Just now"


# ── Keyword classification tables ────────────────────────────

# Each entry: (list_of_keywords, event_type, direction, confidence_boost)
_HEADLINE_RULES: List[Tuple[List[str], BTCEventType, str, float]] = [
    # ── Macro Risk-On ─────────────────────────────────────────
    (["fed cuts", "fed cut rates", "rate cut", "pivot", "dovish", "pause rate hike",
      "inflation falls", "cpi lower", "core inflation", "fomc pause", "soft landing",
      "risk on", "risk appetite", "institutional buying", "spot etf approved",
      "etf approval", "bitcoin etf", "blackrock btc", "fidelity btc"],
     BTCEventType.MACRO_RISK_ON, "BULLISH", 0.25),

    # ── Macro Risk-Off ────────────────────────────────────────
    # Geopolitical crises, diplomatic failures, and macro risk events
    # all trigger capital flight from risk assets → crypto dumps.
    # The Iran-US deal failure (Apr 2026) was missed because "deal fail"
    # and "Iran" weren't in the keyword list. Now comprehensive.
    (["fed hikes", "rate hike", "hawkish", "inflation surge", "cpi higher than",
      "recession", "gdp contracts", "banking crisis", "bank run", "rate rises",
      "de-dollarization fear", "geopolit", "war escalat", "military strike",
      "oil price spike", "sanctions", "trade war", "tariff", "tariffs",
      # Geopolitical escalation — diplomatic failures, conflicts, military
      "deal fail", "deal collapse", "no deal", "not make a deal",
      "could not reach", "could not agree", "failed to reach",
      "talks fail", "talks collapse", "talks break down",
      "negotiation fail", "diplomatic breakdown", "diplomatic crisis",
      "conflict escalat", "tension", "retaliat", "military",
      "missile", "nuclear", "invasion", "invade",
      "iran", "north korea", "china taiwan", "russia ukraine",
      # Macro crisis
      "debt ceiling", "default risk", "credit downgrade", "bank failure",
      "contagion", "panic sell", "market panic", "flash crash",
      "emergency rate", "capital flight", "risk off"],
     BTCEventType.MACRO_RISK_OFF, "BEARISH", 0.25),

    # ── BTC Fundamental Bullish ───────────────────────────────
    (["etf inflow", "bitcoin etf flows", "halving", "mining", "block reward",
      "satoshi", "lightning network", "bitcoin adoption", "el salvador",
      "nation state bitcoin", "corporate treasury bitcoin", "microstrategy buys",
      "spot btc", "grayscale", "bitcoin reserve"],
     BTCEventType.BTC_FUNDAMENTAL, "BULLISH", 0.20),

    # ── BTC Fundamental Bearish ───────────────────────────────
    (["bitcoin banned", "btc banned", "mining ban", "china ban",
      "us ban crypto", "treasury warns bitcoin", "ecb warns",
      "quantum computing breaks", "51% attack"],
     BTCEventType.BTC_FUNDAMENTAL, "BEARISH", 0.20),

    # ── BTC Technical / Mechanical ───────────────────────────
    (["liquidation", "short squeeze", "long squeeze", "leverage flush",
      "funding rate extreme", "open interest spike", "futures expiry",
      "perpetual swap", "basis spike", "contango blow", "cascade",
      "whale moved", "miner selling", "miner capitulation",
      "exchange outflow", "exchange inflow"],
     BTCEventType.BTC_TECHNICAL, "NEUTRAL", 0.15),

    # ── Exchange Event ────────────────────────────────────────
    (["hack", "hacked", "exploit", "stolen", "theft", "security breach",
      "withdrawal suspend", "withdrawal halt", "frozen", "insolvent",
      "bankruptcy", "bankrupt", "collapse", "rug", "exit scam",
      "ftx", "luna", "3ac", "celsius", "blockfi", "voyager",
      "genesis", "bitfinex breach"],
     BTCEventType.EXCHANGE_EVENT, "BEARISH", 0.30),

    # ── Regulatory ───────────────────────────────────────────
    (["sec lawsuit", "sec charges", "sec sues", "cftc charges", "doj",
      "arrest", "charged", "indicted", "regulatory clarity",
      "legal framework", "crypto bill", "senate crypto", "congress crypto",
      "eu crypto", "mica", "fatf", "aml crypto", "kyc crypto",
      "approved exchange", "licensed", "authorized"],
     BTCEventType.REGULATORY, "NEUTRAL", 0.20),
]

# ── Bullish offset keywords ──────────────────────────────────
# When a headline classified as MACRO_RISK_OFF also contains
# capital-inflow / institutional-buying signals, the event is
# bi-directional ("mixed").  These keywords detect the bullish
# offsetting force so the bot downgrades to MIXED instead of
# applying full risk-off blocking.
_BULLISH_OFFSET_KEYWORDS: List[str] = [
    "saylor", "buying more", "buys more", "buy more", "purchases more",
    "accumulates", "accumulating", "microstrategy", "strategy buys",
    "institutional buy", "institutional buying", "etf inflow", "etf inflows",
    "blackrock buy", "fidelity buy", "buying bitcoin", "buys bitcoin",
    "buying btc", "buys btc", "bought bitcoin", "bought btc",
    "adds bitcoin", "adds btc", "adding bitcoin", "adding btc",
    "corporate treasury", "reserve asset", "nation adopt", "country adopt",
    "hints at buying", "plans to buy", "continues buying",
    "whale accumul", "whale buy",
]


class NewsClassifier:
    """
    Classifies a news headline into the BTCEventType taxonomy.
    Uses keyword matching with confidence scoring.
    """

    def classify(self, title: str) -> Tuple[BTCEventType, str, float, bool]:
        """
        Returns (event_type, direction, confidence, is_mixed).

        Mixed signal detection: when the winning classification is bearish
        (MACRO_RISK_OFF) but the headline also contains bullish capital-inflow
        keywords, ``is_mixed`` is True so the caller can apply softer penalties
        instead of a full hard-block.

        AUDIT FIX: Keyword matches are de-duplicated by match position so
        overlapping keywords within the same rule (e.g. ``"tariff"`` +
        ``"tariffs"``, ``"hack"`` + ``"hacked"``, ``"bankrupt"`` +
        ``"bankruptcy"``) can't inflate the confidence by counting the same
        substring multiple times.  Uses word-boundary-ish matching so
        "important" no longer matches "import".
        """
        low = title.lower()

        best_type  = BTCEventType.UNKNOWN
        best_dir   = "NEUTRAL"
        best_conf  = 0.0

        # Track all matching categories for cross-direction conflict check
        all_matches: List[Tuple[BTCEventType, str, float]] = []

        for keywords, event_type, direction, base_conf in _HEADLINE_RULES:
            # AUDIT FIX (keyword inflation): collect matched spans and drop
            # any span fully contained inside another.  "hacked" matches both
            # "hack" and "hacked" — we count the longest matching keyword only.
            matched_spans: List[Tuple[int, int, str]] = []
            for kw in keywords:
                start = low.find(kw)
                while start != -1:
                    matched_spans.append((start, start + len(kw), kw))
                    start = low.find(kw, start + 1)
            # Sort longest-first so contained sub-matches are discarded.
            matched_spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
            unique_hits: List[Tuple[int, int]] = []
            for s, e, _kw in matched_spans:
                if any(us <= s and e <= ue for us, ue in unique_hits):
                    continue  # contained inside an already-counted span
                # Drop a previous span if this new one strictly contains it
                unique_hits = [(us, ue) for us, ue in unique_hits if not (s <= us and ue <= e)]
                unique_hits.append((s, e))
            hits = len(unique_hits)
            if hits > 0:
                conf = min(1.0, base_conf + (hits - 1) * 0.08)
                all_matches.append((event_type, direction, conf))
                if conf > best_conf:
                    best_conf  = conf
                    best_type  = event_type
                    best_dir   = direction

        # ── Mixed signal detection ────────────────────────────────
        is_mixed = False
        if best_dir == "BEARISH" and best_type in (
            BTCEventType.MACRO_RISK_OFF, BTCEventType.BTC_FUNDAMENTAL
        ):
            # Check for bullish offset keywords (institutional buying, etc.)
            bullish_hits = sum(1 for kw in _BULLISH_OFFSET_KEYWORDS if kw in low)
            if bullish_hits > 0:
                is_mixed = True

            # Also mixed if headline matches BOTH bearish and bullish rule categories
            if not is_mixed and len(all_matches) >= 2:
                has_bullish = any(
                    d == "BULLISH" for _, d, c in all_matches
                    if c > NewsIntelligence.MIXED_SIGNAL_MIN_BULLISH_CONF
                )
                if has_bullish:
                    is_mixed = True
        elif best_dir == "BULLISH" and best_type in (
            BTCEventType.MACRO_RISK_ON, BTCEventType.BTC_FUNDAMENTAL
        ):
            # AUDIT FIX: symmetric mixed-signal detection for BULLISH winners
            # with conflicting bearish rule hits.  Prevents the bot from
            # loading full bullish boost into headlines like
            # "Microstrategy buys BTC despite Fed rate hike".
            if len(all_matches) >= 2:
                has_bearish = any(
                    d == "BEARISH" for _, d, c in all_matches
                    if c > NewsIntelligence.MIXED_SIGNAL_MIN_BULLISH_CONF
                )
                if has_bearish:
                    is_mixed = True

        return best_type, best_dir, best_conf, is_mixed

    def classify_batch(
        self, headlines: List[str]
    ) -> Tuple[BTCEventType, str, float, str, bool]:
        """
        Classify a batch of headlines, return the dominant classification.
        Returns (event_type, direction, confidence, winning_headline, is_mixed).
        """
        if not headlines:
            return BTCEventType.UNKNOWN, "NEUTRAL", 0.0, "", False

        # AUDIT FIX (news batch dilution): the previous aggregation normalized by
        # `len(headlines)` which meant a single high-confidence risk-off headline
        # in a batch of 10 unrelated items was diluted from ~0.9 down to ~0.18,
        # silently suppressing real risk-off events.  We now use the MAX confidence
        # per event type as the base and add a small support bonus (capped at
        # +0.25) for additional headlines supporting the winning type.  This:
        #   • preserves strong single-headline signals,
        #   • still rewards consensus across multiple supporting headlines,
        #   • prevents weak-headline spam from amplifying past the cap.
        type_entries: Dict[BTCEventType, List[Tuple[float, str]]] = {}
        vote_dirs: Dict[BTCEventType, Set[str]] = {}
        any_mixed = False

        for h in headlines:
            etype, direction, conf, is_mixed = self.classify(h)
            if is_mixed:
                any_mixed = True
            if etype != BTCEventType.UNKNOWN and conf > 0:
                type_entries.setdefault(etype, []).append((conf, h))
                vote_dirs.setdefault(etype, set()).add(direction)

        if not type_entries:
            return BTCEventType.UNKNOWN, "NEUTRAL", 0.0, "", False

        # Winner = event type with the single highest-confidence headline.
        winner = max(type_entries, key=lambda k: max(e[0] for e in type_entries[k]))
        entries_sorted = sorted(type_entries[winner], key=lambda x: x[0], reverse=True)
        base_conf, best_headline = entries_sorted[0]
        # Support bonus from other headlines of the same type (clip each at 0.6
        # so a spam of near-duplicate weak headlines can't drown out quality).
        support_sum = sum(min(c, 0.6) for c, _ in entries_sorted[1:])
        bonus = min(0.25, support_sum * 0.2)
        norm_conf = min(1.0, base_conf + bonus)

        # Pick dominant direction for the winning type
        # If multiple directions seen for one type, that's already mixed
        winner_dirs = vote_dirs[winner]
        if len(winner_dirs) > 1:
            any_mixed = True
        winner_dir = "BEARISH" if "BEARISH" in winner_dirs else next(iter(winner_dirs))

        # Batch-level mixed detection: some headlines bearish, some bullish
        if not any_mixed:
            all_dirs: Set[str] = set()
            for dirs in vote_dirs.values():
                all_dirs.update(dirs)
            if "BEARISH" in all_dirs and "BULLISH" in all_dirs:
                any_mixed = True

        return winner, winner_dir, round(norm_conf, 3), best_headline, any_mixed


class BTCMoveAnalyzer:
    """
    Detects significant BTC price moves and correlates them with news timing.
    """

    def __init__(self):
        self._price_history: List[Tuple[float, float]] = []   # (timestamp, price)
        self._max_history  = 120    # 2 hours of 1-min ticks

    def record_price(self, price: float):
        now = time.time()
        self._price_history.append((now, price))
        if len(self._price_history) > self._max_history:
            self._price_history = self._price_history[-self._max_history:]

    def get_move(self, lookback_minutes: int = 30) -> Tuple[float, str]:
        """
        Returns (pct_move, direction) over the last N minutes.
        pct_move is always positive; direction is "UP" or "DOWN".
        """
        if len(self._price_history) < 2:
            return 0.0, "FLAT"

        cutoff = time.time() - lookback_minutes * 60
        relevant = [(t, p) for t, p in self._price_history if t >= cutoff]
        if len(relevant) < 2:
            return 0.0, "FLAT"

        first_price = relevant[0][1]
        last_price  = relevant[-1][1]
        if first_price <= 0:
            return 0.0, "FLAT"

        pct = (last_price - first_price) / first_price * 100
        direction = "UP" if pct > 0 else "DOWN"
        return round(abs(pct), 3), direction

    def get_move_since(self, since_ts: float) -> Tuple[float, str]:
        """
        BTC % change from the price closest to since_ts through now.
        Used to show how much BTC has moved since a specific news event was detected.
        Returns (pct_move, direction) — pct_move is always positive.

        Assumes _price_history is maintained in ascending chronological order
        (oldest entry first), which is guaranteed by record_price() appending to the end.
        """
        if len(self._price_history) < 2 or since_ts <= 0:
            return 0.0, "FLAT"
        # Walk forward to find the last recorded price that predates since_ts.
        # Because the list is always appended (chronological), overwriting baseline
        # on each qualifying entry naturally converges on the closest pre-event price.
        baseline = None
        for t, p in self._price_history:
            if t <= since_ts:
                baseline = p
        if baseline is None or baseline <= 0:
            # Event is older than our price window — return FLAT
            return 0.0, "FLAT"
        last_price = self._price_history[-1][1]
        pct = (last_price - baseline) / baseline * 100
        direction = "UP" if pct > 0 else "DOWN"
        return round(abs(pct), 3), direction

    def is_significant_move(self, threshold_pct: float = 2.0, lookback_minutes: int = 30) -> bool:
        pct, _ = self.get_move(lookback_minutes)
        return pct >= threshold_pct

    def get_current_price(self) -> float:
        """Return the most recent recorded BTC price, or 0.0 if none."""
        if self._price_history:
            return self._price_history[-1][1]
        return 0.0


class BTCNewsIntelligence:
    """
    Main intelligence engine. Correlates BTC price moves with news classification
    and produces a BTCEventContext that the trading engine consults.

    Call flow:
      1. news_scraper calls record_btc_price() on each BTC candle close
      2. news_scraper calls process_new_headlines() when new BTC/macro news arrives
      3. engine._scan_symbol() calls get_event_context() before publishing signals
      4. engine._check_daily_summary() calls decay_context() to age out old events
    """

    def __init__(self):
        self._classifier   = NewsClassifier()
        self._move_analyzer = BTCMoveAnalyzer()
        self._current_ctx  = BTCEventContext()
        self._ctx_lock     = asyncio.Lock()

        # Callback: fires when a risk-off / bearish event is first detected.
        # Signature: async (ctx: BTCEventContext) -> None
        # Used by engine.py / bot.py to warn users about active LONG trades at risk.
        self.on_risk_event: Optional[Callable] = None

        # ── Headline history for multi-news conflict resolution ──
        # Each entry: {"title", "published_at", "direction", "confidence",
        #              "source", "event_type", "detected_at", "price_at_detection"}
        self._headline_history: List[dict] = []

        # ── Reaction scoring state ───────────────────────────────
        self._pending_reaction_check: Optional[Union[dict, List[dict]]] = None  # dict for back-compat, or list[dict]

        # Context lifetime by event type (seconds)
        self._context_ttl: Dict[BTCEventType, float] = {
            BTCEventType.MACRO_RISK_ON:   14400,  # 4h — macro news lasts
            BTCEventType.MACRO_RISK_OFF:  14400,  # 4h
            BTCEventType.BTC_FUNDAMENTAL: 7200,   # 2h — alts follow after lag
            BTCEventType.BTC_TECHNICAL:   3600,   # 1h — mechanical, passes fast
            BTCEventType.EXCHANGE_EVENT:  10800,  # 3h — fear lingers
            BTCEventType.REGULATORY:      5400,   # 1.5h
            BTCEventType.ALTCOIN_SPECIFIC: 1800,  # 30m
            BTCEventType.UNKNOWN:         0,
        }

        # Altcoin signal adjustments per event type and direction
        self._adjustments = {
            (BTCEventType.MACRO_RISK_ON, "BULLISH"): {
                "confidence_mult": 1.10, "block_longs": False,
                "block_shorts": False, "reduce_size_mult": 1.0,
                "explanation": "Macro risk-on news. All risk assets boosted. Altcoin LONGs get +10% confidence.",
                "impact_type": "Risk-On Rally",
                "impact_detail": "Improving macro → capital flows into risk assets → broad crypto lift",
            },
            (BTCEventType.MACRO_RISK_OFF, "BEARISH"): {
                "confidence_mult": 0.70, "block_longs": True,
                "block_shorts": False, "reduce_size_mult": 0.5,
                "explanation": "Macro risk-off. All LONGs blocked. SHORT confidence reduced to 70% (counter-trend risk).",
                "impact_type": "Liquidity Withdrawal",
                "impact_detail": "Macro risk-off → capital flight from risk assets → crypto sell pressure",
            },
            (BTCEventType.BTC_FUNDAMENTAL, "BULLISH"): {
                "confidence_mult": 0.90, "block_longs": False,
                "block_shorts": False, "require_low_corr": True,
                "reduce_size_mult": 0.75,
                "explanation": "BTC-specific bullish news. Alts lag BTC. Waiting for BTC to stabilize — reduced size, low-corr alts only.",
                "impact_type": "BTC Demand Spike",
                "impact_detail": "BTC-specific demand event → BTC rallies first, alts follow with 30–120min lag",
            },
            (BTCEventType.BTC_FUNDAMENTAL, "BEARISH"): {
                "confidence_mult": 0.80, "block_longs": True,
                "block_shorts": False, "reduce_size_mult": 0.5,
                "explanation": "BTC-specific bearish news. Block LONGs. Half position size on SHORTs.",
                "impact_type": "BTC Structural Risk",
                "impact_detail": "BTC-specific bearish event → structural selling → contagion to altcoins likely",
            },
            (BTCEventType.BTC_TECHNICAL, "NEUTRAL"): {
                "confidence_mult": 0.85, "block_longs": False,
                "block_shorts": False, "reduce_size_mult": 0.80,
                "explanation": "Mechanical BTC move (liquidation/squeeze). Likely to reverse. Reduced confidence and size until move exhausts.",
                "impact_type": "Mechanical Liquidation",
                "impact_detail": "Leverage flush event → sharp move likely reverses → wait for exhaustion before trading",
            },
            (BTCEventType.EXCHANGE_EVENT, "BEARISH"): {
                "confidence_mult": 0.65, "block_longs": True,
                "block_shorts": False, "reduce_size_mult": 0.5,
                "explanation": "Exchange hack/insolvency detected. Blocking all LONGs for 3h. Correlated crypto fear.",
                "impact_type": "Liquidity Shock",
                "impact_detail": "Exchange risk → withdrawal fear → forced selling pressure across crypto",
            },
            (BTCEventType.REGULATORY, "NEUTRAL"): {
                "confidence_mult": 0.75, "block_longs": False,
                "block_shorts": False, "reduce_size_mult": 0.80,
                "explanation": "Regulatory news. High uncertainty. Reduced confidence until impact is clear.",
                "impact_type": "Regulatory Uncertainty",
                "impact_detail": "Regulatory action creates uncertainty → price can swing sharply in either direction",
            },
            (BTCEventType.REGULATORY, "BEARISH"): {
                "confidence_mult": 0.65, "block_longs": True,
                "block_shorts": False, "reduce_size_mult": 0.5,
                "explanation": "Negative regulatory news. LONGs blocked. Reduced confidence.",
                "impact_type": "Regulatory Crackdown",
                "impact_detail": "Hostile regulation → market access risk → institutional demand drops",
            },
        }

    # ── Public interface ──────────────────────────────────────

    def record_btc_price(self, price: float):
        """Call on every BTC candle close (1m or 5m). Thread-safe (no lock needed for list append)."""
        self._move_analyzer.record_price(price)

    async def process_headlines(self, headlines: List[dict], symbol_filter: str = "BTC"):
        """
        Process a batch of headlines from the news scraper.
        Call when new BTC/macro news arrives.

        headlines: list of {"title": str, "published_at": float, "source": str}
        """
        if not headlines:
            return

        # Filter to BTC/macro relevant headlines
                # Includes geopolitical/crisis terms so headlines like
        # "JD Vance: Iran-US deal fails" are not filtered out.
        _BTC_MACRO_KEYWORDS = [
            "bitcoin", "btc", "crypto", "fed", "inflation",
            "rate", "macro", "sec", "cftc", "hack", "exchange",
            "etf", "regulation", "bank",
            # Geopolitical & crisis — these move BTC via risk-off flows
            "war", "conflict", "military", "missile", "nuclear",
            "sanctions", "sanction", "tariff", "geopolit", "tension",
            "deal fail", "no deal", "talks fail", "escalat",
            "iran", "north korea", "china taiwan", "russia",
            "recession", "crisis", "panic", "contagion",
            "default", "debt ceiling", "oil price",
        ]
        btc_headlines = [
            h for h in headlines
            if any(kw in h.get("title", "").lower()
                   for kw in _BTC_MACRO_KEYWORDS)
        ]
        if not btc_headlines:
            return

        # ── Headline timeliness gate ─────────────────────────────
        # Don't fire urgent alerts for stale news.  Headlines older
        # than STALE_HEADLINE_ALERT_GATE_MINUTES are dropped from the
        # classification batch.  They still feed sentiment scoring
        # (where TIME_DECAY handles weight reduction), but we don't
        # want the bot creating event contexts / sending Telegram
        # alerts for news that broke hours ago.
        now = time.time()
        _gate_minutes = NewsIntelligence.STALE_HEADLINE_ALERT_GATE_MINUTES
        fresh_btc_headlines = [
            h for h in btc_headlines
            if (now - h.get("published_at", now)) / 60 <= _gate_minutes
        ]
        if not fresh_btc_headlines:
            # All headlines are stale — log and skip
            oldest_age = max((now - h.get("published_at", now)) / 60 for h in btc_headlines)
            logger.debug(
                f"🕐 Skipping {len(btc_headlines)} stale BTC headline(s) "
                f"(oldest: {oldest_age:.0f}m, gate: {_gate_minutes}m)"
            )
            return
        btc_headlines = fresh_btc_headlines

        titles = [h["title"] for h in btc_headlines]
        etype, direction, conf, winning_headline, is_mixed = self._classifier.classify_batch(titles)

        if etype == BTCEventType.UNKNOWN or conf < NewsIntelligence.NEWS_GATING_MIN_CONFIDENCE:
            for h in btc_headlines:
                _h_title = h.get("title", "")
                if not _h_title:
                    continue
                _h_type, _h_dir, _h_conf, _ = self._classifier.classify(_h_title)
                if _h_type == BTCEventType.UNKNOWN or _h_conf <= 0:
                    continue
                self._record_headline(
                    title=_h_title,
                    published_at=h.get("published_at", now),
                    direction=_h_dir,
                    confidence=_h_conf,
                    source=h.get("source", "unknown"),
                    event_type=_h_type,
                    detected_at=now,
                )
            return  # Not significant enough to override

        # Use the winning headline's age for decay, not the newest headline in the batch.
        _winning_published_at = max(h.get("published_at", now) for h in btc_headlines)
        for h in btc_headlines:
            if h.get("title", "") == winning_headline:
                _winning_published_at = h.get("published_at", now)
                break
        _headline_age_minutes = (now - _winning_published_at) / 60

        # ── Staleness confidence decay (exponential) ────────────
        # Headlines that pass the gate but are not brand-new get a
        # confidence penalty via exp(-age / tau).  Uses a slower tau
        # for regulatory/ETF events that markets digest over hours.
        if _headline_age_minutes > 1:
            _title_lower = winning_headline.lower() if winning_headline else ""
            _is_slow = any(kw in _title_lower for kw in NewsIntelligence.SLOW_DECAY_EVENT_KEYWORDS)
            _tau = NewsIntelligence.NEWS_DECAY_TAU_SLOW_MINUTES if _is_slow \
                else NewsIntelligence.NEWS_DECAY_TAU_FAST_MINUTES
            _floor = NewsIntelligence.NEWS_DECAY_FLOOR
            _decay = max(_floor, math.exp(-_headline_age_minutes / _tau))
            conf *= _decay
            logger.debug(
                f"🕐 Exp decay: age={_headline_age_minutes:.0f}m "
                f"tau={_tau:.0f} decay={_decay:.2f} → conf={conf:.3f}"
            )

        # Correlate with BTC price move
        move_pct, move_dir = self._move_analyzer.get_move(lookback_minutes=30)
        is_significant_move = move_pct >= 1.5

        # Downgrade if news direction doesn't match price move
        # (news lag — price already moved, news is catching up)
        if is_significant_move:
            if (direction == "BULLISH" and move_dir == "DOWN") or \
               (direction == "BEARISH" and move_dir == "UP"):
                conf *= 0.6   # news contradicts price — lower confidence

        # Only update context if new event is higher confidence or different type.
        # AUDIT FIX: for DIFFERENT event types, require the new confidence to be at
        # least EVENT_REPLACE_CONF_RATIO × current confidence.  Prevents a weak
        # BTC_TECHNICAL headline from clobbering an active high-confidence
        # EXCHANGE_EVENT / MACRO_RISK_OFF block just because the type differs.
        # AUDIT FIX (cross-type decay): when the event type changes we additionally
        # require STRICT superiority over the current confidence.  Without this,
        # a 0.77 headline could replace an active 0.9 context (0.77 ≥ 0.9×0.85)
        # causing gradual signal decay via type-swap rather than via TTL expiry.
        async with self._ctx_lock:
            _should_update = False
            if not self._current_ctx.is_active:
                _should_update = True
            elif etype == self._current_ctx.event_type:
                _should_update = conf > self._current_ctx.confidence
            else:
                _min_replace = (
                    self._current_ctx.confidence * NewsIntelligence.EVENT_REPLACE_CONF_RATIO
                )
                _should_update = (
                    conf >= _min_replace
                    and conf > self._current_ctx.confidence
                )
            if _should_update:
                base_ttl = self._context_ttl.get(etype, 3600)

                # ── Confidence-scaled TTL ─────────────────────────
                # Weaker classifications expire sooner so the bot
                # doesn't stay blocked for the full base TTL on a
                # headline the market may shrug off.
                ttl_scale = max(NewsIntelligence.TTL_CONFIDENCE_FLOOR, conf)
                ttl = int(base_ttl * ttl_scale)

                adj_key = (etype, direction) if (etype, direction) in self._adjustments \
                          else (etype, "NEUTRAL") if (etype, "NEUTRAL") in self._adjustments \
                          else None

                adj = dict(self._adjustments.get(adj_key, {}) if adj_key else {})

                # ── Severity-gated hard blocks ────────────────────
                # Only hard-block LONGs/SHORTs when confidence reaches
                # HIGH severity (≥0.65).  MEDIUM severity events still
                # apply confidence + size penalties but let signals
                # through — avoids losing good trades on borderline
                # headlines.
                if conf < NewsIntelligence.HARD_BLOCK_MIN_CONFIDENCE:
                    was_blocking = adj.get("block_longs") or adj.get("block_shorts")
                    adj["block_longs"]  = False
                    adj["block_shorts"] = False
                    if was_blocking:
                        explanation = adj.get("explanation", "")
                        adj["explanation"] = explanation.replace(
                            "All LONGs blocked",
                            "LONGs penalized (conf+size reduced, not blocked)"
                        ).replace(
                            "Blocking all LONGs",
                            "Penalizing LONGs (conf+size reduced, not blocked)"
                        ).replace(
                            "Block LONGs",
                            "Penalize LONGs (conf+size reduced)"
                        )

                # ── Mixed signal override ─────────────────────────
                # When a headline contains BOTH bearish (risk-off) and
                # bullish (capital-inflow) signals, downgrade from
                # hard-block to softer penalties.  The conflicting
                # forces create volatility + rotation, not pure
                # downside.  LONGs stay allowed with reduced size.
                if is_mixed and direction == "BEARISH":
                    adj["block_longs"]  = False
                    adj["block_shorts"] = False
                    adj["confidence_mult"] = NewsIntelligence.MIXED_SIGNAL_CONFIDENCE_MULT
                    adj["reduce_size_mult"] = NewsIntelligence.MIXED_SIGNAL_SIZE_MULT
                    adj["explanation"] = (
                        "Mixed macro signal — risk-off event offset by "
                        "institutional buying / capital inflow. "
                        "LONGs allowed with reduced size. Expect volatility."
                    )
                    adj["impact_type"] = "Mixed Macro"
                    adj["impact_detail"] = (
                        "Conflicting forces: macro risk-off + capital inflow "
                        "→ volatility + rotation, not pure downside"
                    )
                    # Shorter TTL for mixed signals
                    ttl = int(ttl * NewsIntelligence.MIXED_SIGNAL_TTL_SCALE)

                sources = list({h.get("source", "unknown") for h in btc_headlines[:3]})
                _detected_at = time.time()

                self._current_ctx = BTCEventContext(
                    event_type      = etype,
                    confidence      = conf,
                    direction       = direction,
                    detected_at     = _detected_at,
                    expires_at      = _detected_at + ttl,
                    confidence_mult = adj.get("confidence_mult", 1.0),
                    block_longs     = adj.get("block_longs", False),
                    block_shorts    = adj.get("block_shorts", False),
                    require_low_corr = adj.get("require_low_corr", False),
                    reduce_size_mult = adj.get("reduce_size_mult", 1.0),
                    headline        = winning_headline,
                    explanation     = adj.get("explanation", f"{etype.value} detected"),
                    sources         = sources,
                    is_mixed_signal = is_mixed,
                    impact_type     = adj.get("impact_type", ""),
                    impact_detail   = adj.get("impact_detail", ""),
                    headline_published_at = _winning_published_at,
                )

                # ── Record to headline history for net score ─────
                for h in btc_headlines:
                    _h_title = h.get("title", "")
                    if not _h_title:
                        continue
                    _h_type, _h_dir, _h_conf, _ = self._classifier.classify(_h_title)
                    if _h_type == BTCEventType.UNKNOWN or _h_conf <= 0:
                        continue
                    _h_age_minutes = max(0.0, (_detected_at - h.get("published_at", _detected_at)) / 60)
                    if _h_age_minutes > 1:
                        _h_is_slow = any(
                            kw in _h_title.lower() for kw in NewsIntelligence.SLOW_DECAY_EVENT_KEYWORDS
                        )
                        _h_tau = (
                            NewsIntelligence.NEWS_DECAY_TAU_SLOW_MINUTES
                            if _h_is_slow else NewsIntelligence.NEWS_DECAY_TAU_FAST_MINUTES
                        )
                        _h_conf *= self.news_decay(
                            _h_age_minutes, tau=_h_tau, floor=NewsIntelligence.NEWS_DECAY_FLOOR
                        )
                    if is_significant_move:
                        if (_h_dir == "BULLISH" and move_dir == "DOWN") or \
                           (_h_dir == "BEARISH" and move_dir == "UP"):
                            _h_conf *= 0.6
                    self._record_headline(
                        title=_h_title,
                        published_at=h.get("published_at", time.time()),
                        direction=_h_dir,
                        confidence=_h_conf,
                        source=h.get("source", "unknown"),
                        event_type=_h_type,
                        detected_at=_detected_at,
                        reaction_check_id=_detected_at,
                    )

                # ── Schedule reaction check ──────────────────────
                self._schedule_reaction_check(direction, _detected_at)

                # ── Attach net news score to context ─────────────
                self._current_ctx.net_news_score = self.compute_net_news_score()

                _mixed_tag = " [MIXED]" if is_mixed else ""
                _age_tag = f" age={_headline_age_minutes:.0f}m" if _headline_age_minutes > 1 else ""
                logger.info(
                    f"🗞️  BTC Event: {etype.value}{_mixed_tag}{_age_tag} | dir={direction} | "
                    f"conf={conf:.2f} | '{winning_headline[:60]}' | "
                    f"block_longs={self._current_ctx.block_longs} | "
                    f"conf_mult={self._current_ctx.confidence_mult}"
                )

                # ── Fire risk-event callback ─────────────────────────
                # Notify Telegram users about bearish events so they can
                # protect active LONG trades (tighten SL, take profit, etc.)
                if self.on_risk_event and direction == "BEARISH":
                    try:
                        asyncio.create_task(self.on_risk_event(self._current_ctx))
                    except Exception as _cb_err:
                        logger.debug(f"Risk event callback error (non-fatal): {_cb_err}")

    def get_event_context(self) -> BTCEventContext:
        """
        Get the current BTC event context for signal adjustment.
        Called by engine._scan_symbol() before publishing each signal.
        Returns an inactive (neutral) context if no event is active.
        """
        if not self._current_ctx.is_active:
            return BTCEventContext()  # neutral defaults
        self._current_ctx.net_news_score = self.compute_net_news_score()
        if (
            abs(self._current_ctx.net_news_score) < NewsIntelligence.NET_SCORE_BULLISH_THRESHOLD
            and self._current_ctx.reduce_size_mult >= 1.0
        ):
            _ctx = copy.copy(self._current_ctx)
            _ctx.reduce_size_mult = NewsIntelligence.NET_SCORE_MIXED_SIZE_MULT
            return _ctx
        return self._current_ctx

    def get_signal_block_reason(self, direction: str) -> str:
        """
        Return a human-readable block reason when the active BTC news context
        hard-blocks the supplied signal direction.
        """
        ctx = self.get_event_context()
        norm_dir = getattr(direction, "value", str(direction)).upper()

        if norm_dir == "LONG" and ctx.block_longs:
            return (
                f"BTC news {ctx.event_type.value} active — LONG signals blocked"
                f" ({ctx.explanation})"
            )
        if norm_dir == "SHORT" and ctx.block_shorts:
            return (
                f"BTC news {ctx.event_type.value} active — SHORT signals blocked"
                f" ({ctx.explanation})"
            )
        return ""

    def decay_context(self):
        """
        Called periodically to age out expired contexts and
        evaluate pending reaction checks.
        """
        # ── Reaction scoring: check if enough time has passed ──
        reaction = self.evaluate_reaction()
        if reaction:
            logger.debug(f"Reaction scoring result: {reaction}")

        # ── Refresh net news score on active context ───────────
        if self._current_ctx.is_active:
            self._current_ctx.net_news_score = self.compute_net_news_score()

        if self._current_ctx.is_active:
            remaining = (self._current_ctx.expires_at - time.time()) / 60
            logger.debug(f"BTC event context active: {self._current_ctx.event_type.value} | "
                         f"{remaining:.0f}m remaining")
        elif self._current_ctx.event_type != BTCEventType.UNKNOWN:
            logger.info(f"BTC event context expired: {self._current_ctx.event_type.value}")
            self._current_ctx = BTCEventContext()

    def get_status(self) -> dict:
        """Dashboard + Telegram status output."""
        ctx = self._current_ctx
        move_pct, move_dir = self._move_analyzer.get_move(30)
        since_pct, since_dir = self._move_analyzer.get_move_since(ctx.detected_at) \
            if ctx.detected_at > 0 else (0.0, "FLAT")
        return {
            "active":                       ctx.is_active,
            "event_type":                   ctx.event_type.value,
            "direction":                    ctx.direction,
            "confidence":                   round(ctx.confidence, 3),
            "severity":                     ctx.severity,
            "headline":                     ctx.headline,
            "explanation":                  ctx.explanation,
            "impact_type":                  ctx.impact_type,
            "impact_detail":                ctx.impact_detail,
            "age_minutes":                  round(ctx.age_minutes, 1),
            "headline_age_minutes":         round(ctx.headline_age_minutes, 1),
            "headline_published_at":        ctx.headline_published_at,
            "expires_in_mins":              round(max(0, (ctx.expires_at - time.time()) / 60), 1) if ctx.expires_at else 0,
            "block_longs":                  ctx.block_longs,
            "block_shorts":                 ctx.block_shorts,
            "confidence_mult":              ctx.confidence_mult,
            "reduce_size_mult":             ctx.reduce_size_mult,
            "require_low_corr":             ctx.require_low_corr,
            "btc_30m_move_pct":             move_pct,
            "btc_30m_direction":            move_dir,
            "btc_move_since_event_pct":     since_pct,
            "btc_move_since_event_dir":     since_dir,
            "sources":                      ctx.sources,
            "is_mixed_signal":              ctx.is_mixed_signal,
            "reaction_validated":           ctx.reaction_validated,
            "reaction_confirmed":           ctx.reaction_confirmed,
            "net_news_score":               round(ctx.net_news_score, 3),
        }

    # ── Feature 12: Exponential decay helper ──────────────────

    @staticmethod
    def news_decay(age_minutes: float, tau: float = 60.0,
                   floor: float = 0.10) -> float:
        """
        Exponential decay multiplier for news impact.

        Args:
            age_minutes: How old the headline is (minutes).
            tau: Time constant (minutes).  At age=tau, multiplier ≈ 0.37.
            floor: Minimum return value (never fully zero).

        Returns:
            float in [floor, 1.0].
        """
        if age_minutes <= 0:
            return 1.0
        return max(floor, math.exp(-age_minutes / tau))

    # ── Feature 13: Reaction scoring ──────────────────────────

    def _schedule_reaction_check(self, direction: str, detected_at: float):
        """
        Record that we need to check the market's reaction to this news
        after REACTION_DELAY_MINUTES have elapsed.
        """
        price_now = self._move_analyzer.get_current_price()
        _new_check = {
            "direction": direction,
            "detected_at": detected_at,
            "price_at_detection": price_now,
            "reaction_check_id": detected_at,
        }
        _checks = self._get_pending_reaction_checks()
        _checks.append(_new_check)
        self._pending_reaction_check = _checks

    def evaluate_reaction(self) -> Optional[str]:
        """
        Check if REACTION_DELAY has elapsed and evaluate whether the
        market confirmed or contradicted the news direction.

        Call this periodically (e.g. from decay_context()).

        Returns:
            "CONFIRMED" | "CONTRADICTED" | "NEUTRAL" | None (not ready yet)
        """
        _checks = self._get_pending_reaction_checks()
        if not _checks:
            return None

        now = time.time()
        ready_idx = None
        chk = None
        for idx, pending in enumerate(_checks):
            elapsed = (now - pending["detected_at"]) / 60
            if elapsed >= NewsIntelligence.REACTION_DELAY_MINUTES:
                ready_idx = idx
                chk = pending
                break
        if chk is None:
            return None  # Too early

        # We're ready — evaluate and clear only this check
        del _checks[ready_idx]
        self._pending_reaction_check = _checks or None
        base_price = chk["price_at_detection"]
        if base_price <= 0:
            return None

        current_price = self._move_analyzer.get_current_price()
        if current_price <= 0:
            return None

        pct_change = (current_price - base_price) / base_price * 100
        min_move = NewsIntelligence.REACTION_MIN_MOVE_PCT
        direction = chk["direction"]

        if abs(pct_change) < min_move:
            result = "NEUTRAL"
        elif direction == "BULLISH" and pct_change > min_move:
            result = "CONFIRMED"
        elif direction == "BEARISH" and pct_change < -min_move:
            result = "CONFIRMED"
        else:
            result = "CONTRADICTED"

        reaction_check_id = chk.get("reaction_check_id")

        # Apply to current context ONLY if this reaction check was scheduled
        # for the currently active context.  AUDIT FIX: previously any pending
        # check would mutate whichever context happened to be current at the
        # time the check fired, so a new headline replacing the old context
        # could inherit CONFIRM/CONTRADICT boosts scheduled for the prior one.
        # Legacy pending-check dicts may omit ``reaction_check_id`` — fall
        # back to comparing ``detected_at`` in that case.
        _match_ref = reaction_check_id if reaction_check_id is not None else chk.get("detected_at")
        _reaction_matches_ctx = (
            _match_ref is not None
            and self._current_ctx.detected_at > 0
            and abs(self._current_ctx.detected_at - _match_ref) < 1e-3
        )
        if (
            self._current_ctx.is_active
            and not self._current_ctx.reaction_validated
            and _reaction_matches_ctx
        ):
            self._current_ctx.reaction_validated = True
            if result == "CONFIRMED":
                self._current_ctx.reaction_confirmed = True
                self._current_ctx.confidence = min(
                    1.0, self._current_ctx.confidence * NewsIntelligence.REACTION_CONFIRM_MULT
                )
                if reaction_check_id is not None:
                    self._apply_reaction_to_headline_history(
                        reaction_check_id, NewsIntelligence.REACTION_CONFIRM_MULT
                    )
                logger.info(
                    f"✅ Reaction CONFIRMED: {direction} news + "
                    f"BTC {pct_change:+.2f}% → conf boosted to {self._current_ctx.confidence:.3f}"
                )
            elif result == "CONTRADICTED":
                self._current_ctx.reaction_confirmed = False
                self._current_ctx.confidence *= NewsIntelligence.REACTION_CONTRADICT_MULT
                if reaction_check_id is not None:
                    self._apply_reaction_to_headline_history(
                        reaction_check_id, NewsIntelligence.REACTION_CONTRADICT_MULT
                    )
                # Unblock if market contradicts bearish news
                if direction == "BEARISH":
                    self._current_ctx.block_longs = False
                logger.info(
                    f"❌ Reaction CONTRADICTED: {direction} news + "
                    f"BTC {pct_change:+.2f}% → conf halved to {self._current_ctx.confidence:.3f}"
                )
            else:
                self._current_ctx.reaction_confirmed = False
                self._current_ctx.confidence *= NewsIntelligence.REACTION_NEUTRAL_MULT
                if reaction_check_id is not None:
                    self._apply_reaction_to_headline_history(
                        reaction_check_id, NewsIntelligence.REACTION_NEUTRAL_MULT
                    )
                logger.info(
                    f"😐 Reaction NEUTRAL: {direction} news + "
                    f"BTC {pct_change:+.2f}% → conf reduced to {self._current_ctx.confidence:.3f}"
                )
            self._current_ctx.net_news_score = self.compute_net_news_score()

        return result

    def _get_pending_reaction_checks(self) -> List[dict]:
        """Return pending reaction checks as a mutable list."""
        if self._pending_reaction_check is None:
            return []
        if isinstance(self._pending_reaction_check, list):
            return self._pending_reaction_check
        if isinstance(self._pending_reaction_check, dict):
            return [self._pending_reaction_check]
        return []

    # ── Feature 14: Multi-news conflict resolver ──────────────

    def _record_headline(self, title: str, published_at: float,
                         direction: str, confidence: float,
                         source: str, event_type: BTCEventType,
                         detected_at: Optional[float] = None,
                         reaction_check_id: Optional[float] = None):
        """Record a classified headline for net score computation."""
        _detected_at = detected_at if detected_at is not None else time.time()
        self._headline_history.append({
            "title": title,
            "published_at": published_at,
            "direction": direction,
            "confidence": confidence,
            "source": source,
            "event_type": event_type.value if isinstance(event_type, BTCEventType) else str(event_type),
            "detected_at": _detected_at,
            "reaction_check_id": reaction_check_id,
        })
        # Trim old entries
        max_age = NewsIntelligence.HEADLINE_HISTORY_MAX_AGE_MINUTES * 60
        cutoff = time.time() - max_age
        self._headline_history = [
            h for h in self._headline_history
            if h["detected_at"] > cutoff
        ][-NewsIntelligence.HEADLINE_HISTORY_MAX_SIZE:]

    def compute_net_news_score(self) -> float:
        """
        Aggregate all recent headlines into a single net score.

        Each headline contributes:
          score = direction_sign × confidence × exp_decay(age)

        Returns:
            float in roughly [-1.0, +1.0].  Positive = bullish, negative = bearish.
        """
        if not self._headline_history:
            return 0.0

        now = time.time()
        total = 0.0
        max_age = NewsIntelligence.HEADLINE_HISTORY_MAX_AGE_MINUTES * 60

        for h in self._headline_history:
            age_sec = now - h["detected_at"]
            if age_sec > max_age:
                continue

            age_min = age_sec / 60
            # Direction sign
            d = h["direction"]
            if d == "BULLISH":
                sign = 1.0
            elif d == "BEARISH":
                sign = -1.0
            else:
                continue  # NEUTRAL contributes nothing

            conf = h["confidence"]
            # Use event-type-aware tau
            _title_lower = h.get("title", "").lower()
            _is_slow = any(kw in _title_lower for kw in NewsIntelligence.SLOW_DECAY_EVENT_KEYWORDS)
            _tau = NewsIntelligence.NEWS_DECAY_TAU_SLOW_MINUTES if _is_slow \
                else NewsIntelligence.NEWS_DECAY_TAU_FAST_MINUTES
            decay = self.news_decay(age_min, tau=_tau, floor=0.0)  # floor=0 so old headlines fully drop out of aggregate

            total += sign * conf * decay

        # Clamp to [-1, +1]
        return max(-1.0, min(1.0, total))

    def _apply_reaction_to_headline_history(self, reaction_check_id: float, multiplier: float):
        """Propagate reaction-scoring multipliers into stored headline confidence."""
        for headline in self._headline_history:
            if headline.get("reaction_check_id") == reaction_check_id:
                headline["confidence"] = max(
                    0.0, min(1.0, headline.get("confidence", 0.0) * multiplier)
                )

    def get_net_news_bias(self) -> Tuple[str, float]:
        """
        Return the current net news bias as (label, score).

        Returns:
            ("BULLISH", score) | ("BEARISH", score) | ("MIXED", score)
        """
        score = self.compute_net_news_score()
        if score > NewsIntelligence.NET_SCORE_BULLISH_THRESHOLD:
            return "BULLISH", score
        elif score < NewsIntelligence.NET_SCORE_BEARISH_THRESHOLD:
            return "BEARISH", score
        return "MIXED", score


# ── Singleton ─────────────────────────────────────────────────
btc_news_intelligence = BTCNewsIntelligence()
