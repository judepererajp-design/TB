"""
TitanBot Pro — Rules Ensemble Voter
=====================================
A deterministic multi-source voting system that replaces AI for
pre-publish signal decisions.

Philosophy (from the design discussion):
  "Rules ensemble: sources vote with clear weights.
   If 3+ of 5 sources oppose direction, suppress.
   AI should EXPLAIN what the bot is doing, not DECIDE what it does."

HOW IT WORKS
─────────────
Each data source casts a VOTE:
  +1  = SUPPORTS the signal direction
   0  = NEUTRAL / no data
  -1  = OPPOSES the signal direction

Sources:
  1. OI Trend          — is money flowing INTO this direction?
  2. Smart Money       — are top traders positioned this way?
  3. CVD               — are buyers/sellers the aggressor right now?
  4. Crowd Sentiment   — is the trade overcrowded or well-positioned?
  5. Whale Flow        — did large orders just align with this?

DECISION RULES (strict, deterministic):
  ┌─────────────────────────────────────────────────────────┐
  │  SUPPRESS (hard block, regardless of confidence):       │
  │    • 4 or 5 sources oppose (overwhelming opposition)    │
  │    • CVD + Smart Money BOTH strongly oppose (fakeout)   │
  │                                                         │
  │  REDUCE SIZE (soft penalty, -10 confidence):           │
  │    • 3 sources oppose                                   │
  │    • Crowd is overcrowded in signal direction           │
  │                                                         │
  │  BOOST (add +5 confidence, A-grade candidate):         │
  │    • 4 or 5 sources align                               │
  │    • CVD + Smart Money BOTH strongly align              │
  │                                                         │
  │  PASS THROUGH (no adjustment):                          │
  │    • 2 or fewer oppose AND 2 or fewer support           │
  └─────────────────────────────────────────────────────────┘

Why not AI for this decision?
  - AI adds 3-8s latency per signal (unacceptable)
  - AI hallucinations can suppress good signals or publish bad ones
  - Rules are auditable — you can see exactly why a signal was killed
  - AI should explain and analyse, not gatekeep execution

The veto system (3-AI debate) remains intact for STRATEGY-LEVEL
decisions (should we suppress IchimokuCloud entirely?) — it runs
every 2 hours and makes structural improvements. That is the right
use of AI: slow, deliberate, retrospective.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
SUPPRESS_STREAK_WARNING_THRESHOLD = 20


class VoteValue(int, Enum):
    STRONG_SUPPORT =  2
    SUPPORT        =  1
    NEUTRAL        =  0
    OPPOSE         = -1
    STRONG_OPPOSE  = -2


@dataclass
class Vote:
    source:    str        # human-readable source name
    value:     VoteValue  # the vote
    reason:    str        # one-line reason
    weight:    float = 1.0  # some sources are more reliable than others


@dataclass
class EnsembleVerdict:
    """Result of the ensemble vote for one signal."""

    # Decision
    action:       str    # "PUBLISH" | "SUPPRESS" | "REDUCE"
    confidence_adj: int  # adjustment to apply (+5, -10, 0, etc.)
    reason:       str    # one-sentence synthesis for logs/Telegram

    # Vote breakdown
    votes:        List[Vote] = field(default_factory=list)
    support_count:   int = 0
    oppose_count:    int = 0
    neutral_count:   int = 0
    weighted_score:  float = 0.0

    # Flags
    hard_suppress:   bool = False  # true = skip confidence threshold check
    cvd_sm_fakeout:  bool = False  # CVD + SmartMoney both strongly oppose
    cvd_sm_confirm:  bool = False  # CVD + SmartMoney both strongly support
    overcrowded:     bool = False  # crowd sentiment is extreme

    def as_log_line(self) -> str:
        vote_str = " ".join(
            f"{v.source}={'✅' if v.value > 0 else '❌' if v.value < 0 else '➖'}"
            for v in self.votes
        )
        return (
            f"Ensemble {self.action} "
            f"(+{self.support_count}/-{self.oppose_count}) "
            f"adj={self.confidence_adj:+d} | {vote_str}"
        )


class EnsembleVoter:
    """
    Collects votes from all data sources and produces a deterministic
    publish/suppress/reduce decision for each signal.
    """

    # ── Regime-conditional vote weights ──────────────────────────────
    # BUG-NEW-3 FIX: duplicate 'VOLATILE' key silently discarded first entry.
    # Added NEUTRAL regime weights (previously fell back to flat _WEIGHTS).
    # Added VOLATILE_PANIC as separate state for extreme moves.
    _WEIGHTS_BY_REGIME = {
        "BULL_TREND":    {"cvd":1.8,"smart_money":2.0,"oi_trend":1.8,"crowd":1.0,"whale_flow":1.2,"basis":1.0,"onchain":1.2,"stablecoin":1.0,"whale_intent":1.3,"mining_health":0.9,"network_demand":0.9,"vol_regime":0.8},
        "BEAR_TREND":    {"cvd":1.8,"smart_money":2.0,"oi_trend":1.8,"crowd":1.0,"whale_flow":1.2,"basis":1.0,"onchain":1.2,"stablecoin":1.0,"whale_intent":1.3,"mining_health":0.9,"network_demand":0.9,"vol_regime":0.8},
        "CHOPPY":        {"cvd":2.2,"smart_money":1.6,"oi_trend":1.0,"crowd":1.6,"whale_flow":1.0,"basis":1.2,"onchain":1.0,"stablecoin":0.9,"whale_intent":1.1,"mining_health":0.8,"network_demand":0.8,"vol_regime":1.0},
        "VOLATILE":      {"cvd":1.5,"smart_money":2.2,"oi_trend":1.2,"crowd":0.8,"whale_flow":2.0,"basis":0.8,"onchain":1.0,"stablecoin":1.1,"whale_intent":1.2,"mining_health":0.8,"network_demand":0.8,"vol_regime":1.1},
        "VOLATILE_PANIC":{"cvd":1.4,"smart_money":2.0,"oi_trend":1.0,"crowd":0.8,"whale_flow":2.2,"basis":0.6,"onchain":0.9,"stablecoin":1.1,"whale_intent":1.2,"mining_health":0.7,"network_demand":0.7,"vol_regime":1.2},
        "NEUTRAL":       {"cvd":1.6,"smart_money":1.8,"oi_trend":1.2,"crowd":1.3,"whale_flow":1.0,"basis":1.1,"onchain":1.1,"stablecoin":1.0,"whale_intent":1.2,"mining_health":0.8,"network_demand":0.8,"vol_regime":0.9},
    }
    _WEIGHTS = {"cvd":2.0,"smart_money":1.8,"oi_trend":1.4,"crowd":1.2,"whale_flow":1.0,"basis":1.0,"onchain":1.1,"stablecoin":1.0,"whale_intent":1.2,"mining_health":0.8,"network_demand":0.8,"vol_regime":0.9}

    @classmethod
    def _get_weights(cls) -> dict:
        try:
            from analyzers.regime import regime_analyzer
            regime_val = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        except Exception:
            regime_val = 'UNKNOWN'
            logger.warning("EnsembleVoter: regime lookup failed, using UNKNOWN")

        # Tier 1: Try adaptive weights first (self-learning from outcomes)
        try:
            from signals.adaptive_weights import adaptive_weight_manager
            adaptive = adaptive_weight_manager.get_weights(regime_val)
            if adaptive:
                # Merge: adaptive overrides static, but keep any sources
                # not tracked by adaptive (fallback to static)
                base = dict(cls._WEIGHTS_BY_REGIME.get(regime_val, cls._WEIGHTS))
                base.update(adaptive)
                logger.debug("EnsembleVoter: using adaptive weights for regime=%s", regime_val)
                return base
        except Exception:
            logger.info("EnsembleVoter: adaptive weights unavailable, falling back to static")

        # Tier 2: Static regime-conditional weights
        return cls._WEIGHTS_BY_REGIME.get(regime_val, cls._WEIGHTS)


    def evaluate(
        self,
        symbol:         str,
        direction:      str,
        entry_price:    float,
        # Data from all sources (all optional — degrades gracefully)
        cvd_signal:     Optional[str]   = None,   # "STRONG_BUY"|"BUY"|"NEUTRAL"|"SELL"|"STRONG_SELL"
        sm_direction:   Optional[str]   = None,   # "STRONG_LONG"|"LEAN_LONG"|"NEUTRAL"|"LEAN_SHORT"|"STRONG_SHORT"
        oi_trend:       Optional[str]   = None,   # "RISING_FAST"|"RISING"|"FLAT"|"FALLING"|"FALLING_FAST"
        crowd_sentiment: Optional[str]  = None,   # "OVERCROWDED_LONG"|"LONG_HEAVY"|"BALANCED"|"SHORT_HEAVY"|"OVERCROWDED_SHORT"
        whale_aligned:  Optional[bool]  = None,   # True/False/None
        basis_pct:      Optional[float] = None,   # float, e.g. 0.45 or -0.3
        max_pain_delta: Optional[int]   = None,   # confidence_delta from options
        netflow_delta:  Optional[int]   = None,   # confidence_delta from netflow
        # A+ Upgrade: new data sources
        onchain_zone:   Optional[str]   = None,   # "OVERVALUED"|"FAIR_VALUE"|"UNDERVALUED"
        stablecoin_trend: Optional[str] = None,   # "STRONG_INFLOW"|"INFLOW"|"NEUTRAL"|"OUTFLOW"|"STRONG_OUTFLOW"
        mining_health:  Optional[str]   = None,   # "STRONG"|"NEUTRAL"|"WEAK"
        network_demand: Optional[str]   = None,   # "STRONG"|"NEUTRAL"|"WEAK"
        whale_intent:   Optional[str]   = None,   # "BULLISH"|"NEUTRAL"|"BEARISH"
        vol_regime:     Optional[str]   = None,   # "COMPRESSED"|"LOW"|"NEUTRAL"|"HIGH"|"EXTREME"
        # Signal quality info for high-confidence override
        signal_confidence: Optional[float] = None,  # raw confidence score (0-100)
        signal_rr:         Optional[float] = None,   # reward/risk ratio
        # FIX Q17: short-window price momentum (pct change, e.g. last-hour %).
        # Required to disambiguate "rising OI into rally" (validated longs) from
        # "rising OI into short (longs getting trapped)".  When omitted, the
        # ambiguous "rising OI into SHORT" vote is neutralised — fading rising
        # OI without price confirmation is a low-edge trade that historically
        # hurt win-rate.  Callers can pass the 1h price change pct when known.
        price_change_pct:  Optional[float] = None,
    ) -> EnsembleVerdict:
        """
        Collect all votes and produce a verdict.
        """
        from config.constants import EnsembleVoter as _EVC

        votes: List[Vote] = []
        is_long = direction == "LONG"
        weights = self._get_weights()

        # ── 1. CVD vote ────────────────────────────────────────────────────────
        cvd_vote = VoteValue.NEUTRAL
        if cvd_signal and cvd_signal != "NEUTRAL":
            buy_signal = cvd_signal in ("BUY", "STRONG_BUY")
            strong     = "STRONG" in cvd_signal

            if is_long and buy_signal:
                cvd_vote = VoteValue.STRONG_SUPPORT if strong else VoteValue.SUPPORT
            elif is_long and not buy_signal:
                cvd_vote = VoteValue.STRONG_OPPOSE  if strong else VoteValue.OPPOSE
            elif not is_long and not buy_signal:
                cvd_vote = VoteValue.STRONG_SUPPORT if strong else VoteValue.SUPPORT
            elif not is_long and buy_signal:
                cvd_vote = VoteValue.STRONG_OPPOSE  if strong else VoteValue.OPPOSE

        votes.append(Vote(
            source = "cvd",
            value  = cvd_vote,
            reason = f"CVD={cvd_signal or 'N/A'}",
            weight = weights.get("cvd", 1.0),
        ))

        # ── 2. Smart Money vote ────────────────────────────────────────────────
        sm_vote = VoteValue.NEUTRAL
        if sm_direction and sm_direction != "NEUTRAL":
            sm_long  = sm_direction in ("STRONG_LONG",  "LEAN_LONG")
            sm_strong = "STRONG" in sm_direction

            if is_long and sm_long:
                sm_vote = VoteValue.STRONG_SUPPORT if sm_strong else VoteValue.SUPPORT
            elif is_long and not sm_long:
                sm_vote = VoteValue.STRONG_OPPOSE  if sm_strong else VoteValue.OPPOSE
            elif not is_long and not sm_long:
                sm_vote = VoteValue.STRONG_SUPPORT if sm_strong else VoteValue.SUPPORT
            elif not is_long and sm_long:
                sm_vote = VoteValue.STRONG_OPPOSE  if sm_strong else VoteValue.OPPOSE

        votes.append(Vote(
            source = "smart_money",
            value  = sm_vote,
            reason = f"SM={sm_direction or 'N/A'}",
            weight = weights.get("smart_money", 1.0),
        ))

        # ── 3. OI Trend vote ───────────────────────────────────────────────────
        oi_vote = VoteValue.NEUTRAL
        if oi_trend and oi_trend != "FLAT":
            oi_rising  = oi_trend in ("RISING", "RISING_FAST")
            oi_strong  = "FAST" in oi_trend
            oi_falling = oi_trend in ("FALLING", "FALLING_FAST")

            # Rising OI = new money entering market = supports breakouts
            # Falling OI = positions closing = warns of reversals
            if is_long and oi_rising:
                oi_vote = VoteValue.STRONG_SUPPORT if oi_strong else VoteValue.SUPPORT
            elif is_long and oi_falling:
                oi_vote = VoteValue.STRONG_OPPOSE  if oi_strong else VoteValue.OPPOSE
            elif not is_long and oi_falling:
                oi_vote = VoteValue.STRONG_SUPPORT if oi_strong else VoteValue.SUPPORT
            elif not is_long and oi_rising:
                # FIX Q17: "Rising OI into a short" only supports the short when
                # the market is actually going down (rally exhausting into new
                # longs).  If price is still rising with OI, that's validated
                # buying — fading it is a low-edge trade.  Neutralise when we
                # lack price-direction evidence; only credit on clearly negative
                # price momentum (>0.2% drop in the reference window).
                if price_change_pct is not None and price_change_pct < -0.002:
                    # Longs piling in while price is falling → fade them
                    oi_vote = VoteValue.SUPPORT
                else:
                    oi_vote = VoteValue.NEUTRAL

        votes.append(Vote(
            source = "oi_trend",
            value  = oi_vote,
            reason = f"OI={oi_trend or 'N/A'}",
            weight = weights.get("oi_trend", 1.0),
        ))

        # ── 4. Crowd Sentiment vote ────────────────────────────────────────────
        crowd_vote = VoteValue.NEUTRAL
        overcrowded = False

        if crowd_sentiment and crowd_sentiment != "BALANCED":
            longs_crowded  = crowd_sentiment in ("OVERCROWDED_LONG",  "LONG_HEAVY")
            shorts_crowded = crowd_sentiment in ("OVERCROWDED_SHORT", "SHORT_HEAVY")
            extreme        = "OVERCROWDED" in crowd_sentiment

            if is_long and shorts_crowded:
                # Shorts crowded → short squeeze setup → supports long
                crowd_vote = VoteValue.STRONG_SUPPORT if extreme else VoteValue.SUPPORT
            elif is_long and longs_crowded:
                # Longs crowded → no more buyers → opposes long
                crowd_vote = VoteValue.STRONG_OPPOSE if extreme else VoteValue.OPPOSE
                overcrowded = extreme
            elif not is_long and longs_crowded:
                # Longs crowded → fade longs → supports short
                crowd_vote = VoteValue.STRONG_SUPPORT if extreme else VoteValue.SUPPORT
            elif not is_long and shorts_crowded:
                # Shorts crowded → squeeze risk → opposes short
                crowd_vote = VoteValue.STRONG_OPPOSE if extreme else VoteValue.OPPOSE
                overcrowded = extreme

        votes.append(Vote(
            source = "crowd",
            value  = crowd_vote,
            reason = f"Crowd={crowd_sentiment or 'N/A'}",
            weight = weights.get("crowd", 1.0),
        ))

        # ── 5. Whale Flow vote ─────────────────────────────────────────────────
        whale_vote = VoteValue.NEUTRAL
        if whale_aligned is True:
            whale_vote = VoteValue.SUPPORT
        elif whale_aligned is False:
            whale_vote = VoteValue.OPPOSE

        votes.append(Vote(
            source = "whale_flow",
            value  = whale_vote,
            reason = f"Whale={'aligned' if whale_aligned else 'opposed' if whale_aligned is False else 'N/A'}",
            weight = weights.get("whale_flow", 1.0),
        ))

        # ── 6. Basis vote (optional supplementary) ────────────────────────────
        if basis_pct is not None:
            # FIX Q12: threshold scaled per-asset.  Majors trade with tight
            # basis in normal regimes, so 0.3% is a meaningful extreme.  Mid-
            # and small-cap alts routinely sit at ±0.5%+ basis with no trading
            # signal.  Scale the trigger up for non-major symbols so the vote
            # isn't constantly firing on naturally-wide-basis tokens.
            _sym_upper = (symbol or "").upper()
            _is_major = any(
                _sym_upper.startswith(m)
                for m in ("BTC", "ETH", "SOL", "BNB")
            )
            basis_trigger = 0.3 if _is_major else 0.6
            basis_vote = VoteValue.NEUTRAL
            if is_long and basis_pct > basis_trigger:
                basis_vote = VoteValue.OPPOSE  # longs crowded in perp
            elif is_long and basis_pct < -basis_trigger:
                basis_vote = VoteValue.SUPPORT  # shorts crowded in perp
            elif not is_long and basis_pct < -basis_trigger:
                basis_vote = VoteValue.OPPOSE   # shorts crowded
            elif not is_long and basis_pct > basis_trigger:
                basis_vote = VoteValue.SUPPORT  # longs crowded, short them

            votes.append(Vote(
                source = "basis",
                value  = basis_vote,
                reason = f"Basis={basis_pct:+.2f}% vs ±{basis_trigger:.1f}%",
                weight = weights.get("basis", 1.0),
            ))

        # ── 7. On-Chain Valuation vote (A+ Upgrade) ───────────────────────────
        if onchain_zone and onchain_zone != "FAIR_VALUE":
            oc_vote = VoteValue.NEUTRAL
            if is_long and onchain_zone == "UNDERVALUED":
                oc_vote = VoteValue.SUPPORT
            elif is_long and onchain_zone == "OVERVALUED":
                oc_vote = VoteValue.OPPOSE
            elif not is_long and onchain_zone == "OVERVALUED":
                oc_vote = VoteValue.SUPPORT
            elif not is_long and onchain_zone == "UNDERVALUED":
                oc_vote = VoteValue.OPPOSE

            votes.append(Vote(
                source = "onchain",
                value  = oc_vote,
                reason = f"OnChain={onchain_zone}",
                weight = weights.get("onchain", 1.0),
            ))

        # ── 8. Stablecoin Liquidity vote (A+ Upgrade) ─────────────────────────
        stablecoin_signal = {
            "AMPLE": "INFLOW",
            "DRAINING": "OUTFLOW",
        }.get(stablecoin_trend, stablecoin_trend)
        if stablecoin_signal and stablecoin_signal not in ("NEUTRAL", "UNKNOWN"):
            sc_vote = VoteValue.NEUTRAL
            inflow = stablecoin_signal in ("STRONG_INFLOW", "INFLOW")
            strong = "STRONG" in stablecoin_signal

            if is_long and inflow:
                sc_vote = VoteValue.STRONG_SUPPORT if strong else VoteValue.SUPPORT
            elif is_long and not inflow:
                sc_vote = VoteValue.STRONG_OPPOSE if strong else VoteValue.OPPOSE
            elif not is_long and not inflow:
                sc_vote = VoteValue.SUPPORT  # Capital flight supports shorts
            elif not is_long and inflow:
                sc_vote = VoteValue.OPPOSE   # Fresh capital opposes shorts

            votes.append(Vote(
                source = "stablecoin",
                value  = sc_vote,
                reason = f"Stablecoin={stablecoin_signal}",
                weight = weights.get("stablecoin", 1.0),
            ))

        # ── 9. Whale Intent vote (A+ Upgrade) ─────────────────────────────────
        if whale_intent and whale_intent != "NEUTRAL":
            wi_vote = VoteValue.NEUTRAL
            if is_long and whale_intent == "BULLISH":
                wi_vote = VoteValue.SUPPORT
            elif is_long and whale_intent == "BEARISH":
                wi_vote = VoteValue.OPPOSE
            elif not is_long and whale_intent == "BEARISH":
                wi_vote = VoteValue.SUPPORT
            elif not is_long and whale_intent == "BULLISH":
                wi_vote = VoteValue.OPPOSE

            votes.append(Vote(
                source = "whale_intent",
                value  = wi_vote,
                reason = f"WhaleIntent={whale_intent}",
                weight = weights.get("whale_intent", 1.0),
            ))

        # ── 10. Mining health vote (Phase 4) ──────────────────────────────────
        if mining_health and mining_health != "NEUTRAL":
            mh_vote = VoteValue.NEUTRAL
            if is_long and mining_health == "STRONG":
                mh_vote = VoteValue.SUPPORT
            elif is_long and mining_health == "WEAK":
                mh_vote = VoteValue.OPPOSE
            elif not is_long and mining_health == "WEAK":
                mh_vote = VoteValue.SUPPORT
            elif not is_long and mining_health == "STRONG":
                mh_vote = VoteValue.OPPOSE

            votes.append(Vote(
                source = "mining_health",
                value  = mh_vote,
                reason = f"Mining={mining_health}",
                weight = weights.get("mining_health", 1.0),
            ))

        # ── 11. Network demand vote (Phase 4) ─────────────────────────────────
        if network_demand and network_demand != "NEUTRAL":
            nd_vote = VoteValue.NEUTRAL
            if is_long and network_demand == "STRONG":
                nd_vote = VoteValue.SUPPORT
            elif is_long and network_demand == "WEAK":
                nd_vote = VoteValue.OPPOSE
            elif not is_long and network_demand == "WEAK":
                nd_vote = VoteValue.SUPPORT
            elif not is_long and network_demand == "STRONG":
                nd_vote = VoteValue.OPPOSE

            votes.append(Vote(
                source = "network_demand",
                value  = nd_vote,
                reason = f"Network={network_demand}",
                weight = weights.get("network_demand", 1.0),
            ))

        # ── 12. Volatility regime vote (Phase 4) ──────────────────────────────
        if vol_regime and vol_regime != "NEUTRAL":
            vr_vote = VoteValue.NEUTRAL
            # FIX P4-2: treat LOW the same as COMPRESSED — consistent with
            # VOL_COMPRESS_REGIMES = ("COMPRESSED", "LOW") in constants.py.
            if vol_regime in ("COMPRESSED", "LOW"):
                vr_vote = VoteValue.SUPPORT
            elif vol_regime in ("HIGH", "EXTREME"):
                vr_vote = VoteValue.OPPOSE

            votes.append(Vote(
                source = "vol_regime",
                value  = vr_vote,
                reason = f"VolRegime={vol_regime}",
                weight = weights.get("vol_regime", 1.0),
            ))

        # ── Tally votes ────────────────────────────────────────────────────────
        support_count  = sum(1 for v in votes if v.value > 0)
        oppose_count   = sum(1 for v in votes if v.value < 0)
        neutral_count  = sum(1 for v in votes if v.value == 0)

        # Weighted score: sum of (vote_value × weight)
        weighted_score = sum(v.value * v.weight for v in votes)

        # ── Special condition flags ────────────────────────────────────────────
        cvd_vote_val = next((v.value for v in votes if v.source == "cvd"), VoteValue.NEUTRAL)
        sm_vote_val  = next((v.value for v in votes if v.source == "smart_money"), VoteValue.NEUTRAL)

        # Both CVD and Smart Money strongly oppose = high-confidence fakeout
        cvd_sm_fakeout = (
            cvd_vote_val <= VoteValue.STRONG_OPPOSE and
            sm_vote_val  <= VoteValue.OPPOSE
        ) or (
            cvd_vote_val <= VoteValue.OPPOSE and
            sm_vote_val  <= VoteValue.STRONG_OPPOSE
        )

        # Both CVD and Smart Money strongly support = high-confidence setup
        cvd_sm_confirm = (
            cvd_vote_val >= VoteValue.STRONG_SUPPORT and
            sm_vote_val  >= VoteValue.SUPPORT
        ) or (
            cvd_vote_val >= VoteValue.SUPPORT and
            sm_vote_val  >= VoteValue.STRONG_SUPPORT
        )

        # ── Decision logic (R8-F5: weighted score thresholds) ─────────────────
        # Previously used raw oppose_count/support_count which treated all sources
        # equally. Now weighted_score drives thresholds so high-weight sources
        # (Smart Money at 2.0, CVD at 1.8) have proportional influence.
        # Count-based rules kept as fallback for overwhelming opposition.
        total_sources = len([v for v in votes if v.value != VoteValue.NEUTRAL])

        # SUPPRESS conditions
        hard_suppress = False
        if oppose_count >= 4:
            # Overwhelming count-based opposition — always suppress
            hard_suppress = True
            action        = "SUPPRESS"
            conf_adj      = 0   # irrelevant, signal dies
            reason        = (f"Ensemble SUPPRESS: {oppose_count} sources oppose "
                             f"({', '.join(v.source for v in votes if v.value < 0)})")

        elif cvd_sm_fakeout:
            hard_suppress = True
            action        = "SUPPRESS"
            conf_adj      = 0
            reason        = ("Ensemble SUPPRESS: CVD + Smart Money both oppose "
                             "— high-confidence fakeout signal")

        elif weighted_score <= max(-3.0, _EVC.SUPPRESS_WEIGHTED_THRESHOLD):
            # R8-F5: Weighted threshold — strong negative conviction.
            # FIX B8: outer gate was hardcoded -3.0; if an operator tunes
            # SUPPRESS_WEIGHTED_THRESHOLD to a less-negative value (e.g. -2.5),
            # scores in (-3.0, -2.5] would have fallen through to the REDUCE
            # branch instead of being suppressed.  Use max(-3.0, SUPPRESS)
            # so the outer gate is always at least as permissive as the
            # configured SUPPRESS threshold.
            if weighted_score <= _EVC.SUPPRESS_WEIGHTED_THRESHOLD:
                hard_suppress = True
                action        = "SUPPRESS"
                conf_adj      = 0
                reason        = (f"Ensemble SUPPRESS: weighted score {weighted_score:.1f} "
                                 f"below {_EVC.SUPPRESS_WEIGHTED_THRESHOLD} threshold "
                                 f"({', '.join(f'{v.source}({v.value*v.weight:+.1f})' for v in votes if v.value < 0)})")
            else:
                # Score is between SUPPRESS_WEIGHTED_THRESHOLD and -3.0 —
                # downgrade to REDUCE instead of hard kill.
                action   = "REDUCE"
                conf_adj = max(-15, int(weighted_score * 3))
                conf_adj = min(-5, conf_adj)
                reason   = (f"Ensemble REDUCE: weighted score {weighted_score:.1f} "
                            f"(softened from old -3.0 suppress) "
                            f"({', '.join(v.source for v in votes if v.value < 0)})")

        # REDUCE conditions
        elif weighted_score <= _EVC.REDUCE_WEIGHTED_THRESHOLD or overcrowded:
            # R8-F5: Use weighted threshold instead of oppose_count >= 3
            action   = "REDUCE"
            # Scale penalty by how negative the weighted score is
            conf_adj = max(-15, int(weighted_score * 3))  # -1.5 → -4, -2.5 → -7
            conf_adj = min(-5, conf_adj)  # At least -5 penalty
            if overcrowded:
                reason = (f"Ensemble REDUCE: overcrowded "
                          f"({crowd_sentiment}) — reducing confidence")
            else:
                reason = (f"Ensemble REDUCE: weighted score {weighted_score:.1f} "
                          f"({', '.join(v.source for v in votes if v.value < 0)})")

        # BOOST conditions
        elif weighted_score >= _EVC.BOOST_WEIGHTED_THRESHOLD or cvd_sm_confirm:
            # R8-F5: Use weighted threshold instead of support_count >= 4
            action   = "BOOST"
            # Scale boost by conviction strength
            conf_adj = min(+10, max(+3, int(weighted_score * 1.5)))
            if cvd_sm_confirm:
                reason = ("Ensemble BOOST: CVD + Smart Money both confirm "
                          "— high-confidence setup")
            else:
                reason = (f"Ensemble BOOST: weighted score +{weighted_score:.1f} "
                          f"({', '.join(v.source for v in votes if v.value > 0)})")

        # PASS — balanced or insufficient data
        else:
            action   = "PASS"
            # R8-F5: Apply minor adjustment based on weighted score direction
            if weighted_score >= 1.0:
                conf_adj = +2  # Slight lean positive
                reason = (f"Ensemble PASS: weighted +{weighted_score:.1f} "
                          f"— slight positive lean, +2 conf")
            elif weighted_score <= -1.0:
                conf_adj = -3  # Slight lean negative
                reason = (f"Ensemble PASS: weighted {weighted_score:.1f} "
                          f"— slight negative lean, -3 conf")
            else:
                conf_adj = 0
                if support_count > oppose_count:
                    reason = (f"Ensemble PASS: {support_count}v{oppose_count} "
                              f"weak support — no boost")
                elif oppose_count > support_count:
                    reason = (f"Ensemble PASS: {oppose_count}v{support_count} "
                              f"weak opposition — no suppress")
                else:
                    reason = f"Ensemble PASS: {neutral_count} neutral sources — insufficient data"

        # ── High-confidence override: downgrade SUPPRESS → REDUCE ──────
        # A genuinely strong signal (high conf + decent RR) shouldn't be
        # hard-killed by mild ensemble disagreement.  Downgrade to REDUCE
        # so the confidence penalty still applies but the signal survives.
        # FIX AUDIT-2: Guard against signal_rr being None — previously
        # `None >= float` raised TypeError and crashed the signal pipeline.
        if hard_suppress and signal_confidence is not None and signal_rr is not None:
            if (signal_confidence >= _EVC.SUPPRESS_OVERRIDE_MIN_CONF
                    and signal_rr >= _EVC.SUPPRESS_OVERRIDE_MIN_RR):
                hard_suppress = False
                action   = "REDUCE"
                conf_adj = max(-15, int(weighted_score * 3))
                conf_adj = min(-5, conf_adj)
                reason   = (f"Ensemble REDUCE (override): score {weighted_score:.1f} "
                            f"but conf={signal_confidence} rr={signal_rr:.1f} "
                            f"→ downgraded from SUPPRESS")

        if not hasattr(self, "_suppress_streaks"):
            self._suppress_streaks = {}
        if action == "SUPPRESS":
            _suppress_streak = self._suppress_streaks.get(symbol, 0) + 1
            self._suppress_streaks[symbol] = _suppress_streak
        else:
            _suppress_streak = 0
            self._suppress_streaks.pop(symbol, None)

        # Log at appropriate levels: normal SUPPRESS=info, anomalous streaks=warning,
        # BOOST/REDUCE=info, PASS=debug.
        # Streak warning boundary: only warn at {N=threshold, 2N, 3N, 5N, 10N}
        # boundaries instead of every increment past N. Prevents 20+ identical
        # "Ensemble ZEC SUPPRESS streak=20..23..24" warnings when the veto is
        # stable across many scan cycles.
        _streak_boundaries = {
            SUPPRESS_STREAK_WARNING_THRESHOLD,
            SUPPRESS_STREAK_WARNING_THRESHOLD * 2,
            SUPPRESS_STREAK_WARNING_THRESHOLD * 3,
            SUPPRESS_STREAK_WARNING_THRESHOLD * 5,
            SUPPRESS_STREAK_WARNING_THRESHOLD * 10,
        }
        _warn_streak = (
            action == "SUPPRESS"
            and _suppress_streak in _streak_boundaries
        )
        _log_fn = (
            logger.warning if _warn_streak
            else logger.info if action in ("BOOST", "REDUCE")
            else logger.info if action == "SUPPRESS" and _suppress_streak < SUPPRESS_STREAK_WARNING_THRESHOLD
            else logger.debug
        )
        if action == "SUPPRESS":
            _log_fn(
                "Ensemble %s %s: score=%.1f sup=%d opp=%d adj=%+d streak=%d | %s",
                symbol, action, weighted_score, support_count, oppose_count,
                conf_adj, _suppress_streak, reason,
            )
        else:
            _log_fn(
                "Ensemble %s %s: score=%.1f sup=%d opp=%d adj=%+d | %s",
                symbol, action, weighted_score, support_count, oppose_count,
                conf_adj, reason,
            )

        return EnsembleVerdict(
            action         = action,
            confidence_adj = conf_adj,
            reason         = reason,
            votes          = votes,
            support_count  = support_count,
            oppose_count   = oppose_count,
            neutral_count  = neutral_count,
            weighted_score = weighted_score,
            hard_suppress  = hard_suppress,
            cvd_sm_fakeout = cvd_sm_fakeout,
            cvd_sm_confirm = cvd_sm_confirm,
            overcrowded    = overcrowded,
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
ensemble_voter = EnsembleVoter()
