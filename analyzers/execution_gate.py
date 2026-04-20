"""
TitanBot Pro — Execution Quality Gate
=======================================
Separates "good setup" from "good trade."

A great setup (strong confluence, valid R:R, trend-aligned) can still be a
**bad trade** if execution conditions are poor.  This module evaluates six
execution-specific factors as a group and hard-blocks the signal when too
many execution killers stack up — regardless of setup confidence.

Factors evaluated:
  1. Session quality     — killzone vs dead zone vs weekend
  2. Trigger quality     — signal trigger cleanness from TriggerQualityAnalyzer
  3. Spread / liquidity  — bid-ask spread as execution cost proxy
  4. Whale alignment     — whether whale intent conflicts with signal direction
  5. Entry position      — premium/discount/EQ relative to direction
  6. Volume environment  — volume context (breakout, normal, low, climactic)

Output:
  - execution_score (0-100): composite execution quality
  - should_block (bool): True when execution conditions are too poor
  - factors (dict): per-factor scores for transparency
  - reason (str): human-readable explanation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config.constants import ExecutionGate as EG
from signals.signal_pipeline import has_reversal_confirmation

logger = logging.getLogger(__name__)

def _log_gate_decision(result: "ExecutionAssessment", symbol: str = "", direction: str = "", grade: str = "B"):
    """Log every execution gate evaluation to the dedicated log file."""
    try:
        from config.exec_gate_logger import exec_gate_log
        if result.should_block:
            action = "BLOCK"
        elif result.should_penalize:
            action = "PENALIZE"
        else:
            action = "PASS"
        extra = {}
        if result.is_near_miss:
            extra["near_miss"] = True
        exec_gate_log(
            symbol, direction,
            score=result.execution_score,
            action=action,
            grade=grade,
            factors=result.factors,
            bad_factors=result.bad_factors,
            kill_combo=result.kill_combo,
            reason=result.reason,
            penalty_mult=result.penalty_mult,
            extra=extra if extra else None,
        )
    except Exception as _e:
        # AUDIT FIX (PR-2): previously a silent `except: pass` that would
        # also hide genuine logger misconfiguration.  Log at DEBUG so the
        # main execution path isn't noisy on a detached file-handler, but
        # the exception is no longer invisible on investigation.
        logger.debug("exec_gate_log dispatch failed: %s", _e, exc_info=True)


@dataclass
class ExecutionAssessment:
    """Result of execution quality evaluation."""
    execution_score: float = 50.0          # 0-100 composite score
    should_block: bool = False             # True → hard-block (NO TRADE)
    should_penalize: bool = False          # True → apply soft confidence penalty
    penalty_mult: float = 1.0             # Confidence multiplier if penalized
    factors: Dict[str, float] = field(default_factory=dict)  # Per-factor scores
    bad_factors: List[str] = field(default_factory=list)      # Factors below threshold
    kill_combo: str = ""                   # Non-empty if a kill combo fired
    reason: str = ""                       # Human-readable summary
    notes: List[str] = field(default_factory=list)            # Detailed factor notes
    is_near_miss: bool = False             # True → blocked but close to threshold
    context_snapshot: Dict[str, object] = field(default_factory=dict)
    block_threshold_used: float = 35.0


class ExecutionQualityGate:
    """
    Evaluates execution conditions and decides whether a signal should be
    blocked, penalized, or passed through.

    This is NOT a replacement for the existing soft penalties — it's an
    additional layer that catches cases where multiple execution factors
    are simultaneously bad (the stacking problem).
    """

    POSITIONING_NEUTRAL_SCORE = 60.0
    POSITIONING_ADJUSTMENT_FACTOR = 0.20
    POSITIONING_ADJUSTMENT_MIN = -8.0
    POSITIONING_ADJUSTMENT_MAX = 8.0

    def evaluate(
        self,
        *,
        context: Optional[Dict[str, object]] = None,
        setup_context: Optional[Dict[str, object]] = None,
        direction: str,                            # "LONG" or "SHORT"
        grade: str = "B",                          # Alpha model grade
        confidence: float = 70.0,                  # Current final confidence
        symbol: str = "",                          # Trading pair (for logging)
        # Session context
        session_name: str = "",                     # "Dead Zone", "London Open", etc.
        is_killzone: bool = False,
        is_dead_zone: bool = False,
        is_weekend: bool = False,
        # Trigger quality (from aggregator raw_data)
        trigger_quality_score: float = 0.5,        # 0.0-1.0
        trigger_quality_label: str = "MEDIUM",
        # Spread / liquidity
        spread_bps: float = 0.0,                   # Bid-ask spread in basis points
        # Whale context
        whale_aligned: Optional[bool] = None,      # True=aligned, False=opposing, None=no data
        whale_buy_ratio: float = 0.5,              # 0.0-1.0 whale buy ratio
        # Entry position
        eq_zone: str = "",                          # 'premium', 'discount', 'eq_dead_zone', 'trending', 'no_range'
        eq_zone_depth: float = 0.0,                # 0.0-1.0
        # Volume environment
        volume_context: str = "NORMAL",            # BREAKOUT, ABOVE_AVG, NORMAL, LOW_VOL, CLIMACTIC
        volume_score: float = 50.0,                # 0-100 from aggregator
        # Trade type (from regime_levels)
        trade_type: str = "",                      # e.g. "PULLBACK_SHORT", "LOCAL_CONTINUATION_LONG"
        trade_type_strength: int = 0,              # 0-4 local structure strength
    ) -> ExecutionAssessment:
        """
        Evaluate execution quality for a signal.

        Returns ExecutionAssessment with block/penalize decision and full breakdown.
        """
        result = ExecutionAssessment()
        # AUDIT FIX (execution_gate locals antipattern): these positioning
        # fields used to be assigned only inside the `if context:` block,
        # which forced the call below to `locals().get(...)` as a safety
        # net.  Initialize them to neutral defaults up-front so callers can
        # be referenced directly and it's clear what defaults apply when
        # no context is supplied.
        derivatives_score = 50.0
        sentiment_score = 50.0
        funding_rate = 0.0
        oi_change_24h = 0.0
        # AUDIT FIX (PR-2 / slippage-missing → reject): only the
        # context-driven call path knows whether the upstream truly had
        # a spread reading or just didn't provide one.  Legacy
        # positional callers keep the old behaviour (treated as
        # "known" so we do not retroactively penalise them).
        _spread_known = True
        if context:
            session_ctx = dict((context or {}).get("session") or {})
            trigger_ctx = dict((context or {}).get("trigger") or {})
            liquidity_ctx = dict((context or {}).get("liquidity") or {})
            whale_ctx = dict((context or {}).get("whales") or {})
            positioning_ctx = dict((context or {}).get("positioning") or {})
            location_ctx = dict((context or {}).get("location") or {})
            market_ctx = dict((context or {}).get("market") or {})
            trade_ctx = dict((context or {}).get("trade") or {})

            session_name = session_ctx.get("name", session_name)
            is_killzone = bool(session_ctx.get("is_killzone", is_killzone))
            is_dead_zone = bool(session_ctx.get("is_dead_zone", is_dead_zone))
            is_weekend = bool(session_ctx.get("is_weekend", is_weekend))

            _tqs = trigger_ctx.get("score")
            trigger_quality_score = float(_tqs) if _tqs is not None else trigger_quality_score
            _tql = trigger_ctx.get("label")
            trigger_quality_label = str(_tql) if _tql is not None else trigger_quality_label

            _sbps = liquidity_ctx.get("spread_bps")
            # AUDIT FIX (PR-2): track whether the caller explicitly
            # provided a spread measurement.  "No spread data" used to
            # silently score as moderate (60) — effectively a free pass.
            # We now flag it below so the composite can penalise
            # slippage-blind entries.
            _spread_known = _sbps is not None
            spread_bps = float(_sbps) if _sbps is not None else spread_bps
            _vc = liquidity_ctx.get("volume_context")
            volume_context = str(_vc) if _vc is not None else volume_context
            _vs = liquidity_ctx.get("volume_score")
            volume_score = float(_vs) if _vs is not None else volume_score

            whale_aligned = whale_ctx.get("aligned", whale_aligned)
            _wbr = whale_ctx.get("buy_ratio")
            whale_buy_ratio = float(_wbr) if _wbr is not None else whale_buy_ratio

            _derivatives_score = positioning_ctx.get("derivatives_score")
            derivatives_score = float(_derivatives_score) if _derivatives_score is not None else 50.0
            _sentiment_score = positioning_ctx.get("sentiment_score")
            sentiment_score = float(_sentiment_score) if _sentiment_score is not None else 50.0
            _funding_rate = positioning_ctx.get("funding_rate")
            funding_rate = float(_funding_rate) if _funding_rate is not None else 0.0
            _oi_change = positioning_ctx.get("oi_change_24h")
            oi_change_24h = float(_oi_change) if _oi_change is not None else 0.0

            _eqz = location_ctx.get("eq_zone")
            eq_zone = str(_eqz) if _eqz is not None else eq_zone
            _eqd = location_ctx.get("eq_distance")
            eq_zone_depth = float(_eqd) if _eqd is not None else eq_zone_depth

            _tt = trade_ctx.get("type")
            trade_type = str(_tt) if _tt is not None else trade_type
            _tts = trade_ctx.get("strength")
            trade_type_strength = int(_tts) if _tts is not None else trade_type_strength

            result.context_snapshot = {
                "session": session_ctx,
                "trigger": trigger_ctx,
                "liquidity": liquidity_ctx,
                "whales": whale_ctx,
                "positioning": positioning_ctx,
                "location": location_ctx,
                "market": market_ctx,
                "trade": trade_ctx,
            }
        if setup_context:
            result.context_snapshot["setup"] = dict(setup_context or {})

        # ── 1. Session score ──────────────────────────────────────
        session_score = self._score_session(
            session_name=session_name,
            is_killzone=is_killzone,
            is_dead_zone=is_dead_zone,
            is_weekend=is_weekend,
        )
        result.factors['session'] = session_score
        if session_score <= EG.BAD_FACTOR_THRESHOLD:
            result.bad_factors.append('session')

        # ── 2. Trigger quality score ──────────────────────────────
        tq_score = min(100.0, trigger_quality_score * EG.TRIGGER_SCORE_MULT)
        result.factors['trigger_quality'] = tq_score
        if tq_score <= EG.BAD_FACTOR_THRESHOLD:
            result.bad_factors.append('trigger_quality')

        # ── 3. Spread score ───────────────────────────────────────
        spread_score = self._score_spread(spread_bps)
        result.factors['spread'] = spread_score
        if spread_score <= EG.BAD_FACTOR_THRESHOLD:
            result.bad_factors.append('spread')
        elif not _spread_known:
            # AUDIT FIX (PR-2): when the upstream context explicitly
            # did not provide a spread reading, surface that as
            # ``spread_unknown`` in bad_factors so the composite
            # treats a slippage-blind entry as degraded rather than
            # silently scoring "moderate".
            result.bad_factors.append('spread_unknown')
            result.factors['spread_unknown'] = 1.0

        # ── 4. Whale alignment score ──────────────────────────────
        whale_score = self._score_whale_alignment(
            direction=direction,
            whale_aligned=whale_aligned,
            whale_buy_ratio=whale_buy_ratio,
        )
        result.factors['whale_alignment'] = whale_score
        if whale_score <= EG.BAD_FACTOR_THRESHOLD:
            result.bad_factors.append('whale_alignment')

        # ── 5. Entry position score ───────────────────────────────
        entry_score = self._score_entry_position(
            direction=direction,
            eq_zone=eq_zone,
            eq_zone_depth=eq_zone_depth,
        )
        result.factors['entry_position'] = entry_score
        if entry_score <= EG.BAD_FACTOR_THRESHOLD:
            result.bad_factors.append('entry_position')

        # ── 6. Volume environment score ───────────────────────────
        vol_score = self._score_volume_env(
            volume_context=volume_context,
            volume_score=volume_score,
        )
        result.factors['volume_env'] = vol_score
        if vol_score <= EG.BAD_FACTOR_THRESHOLD:
            result.bad_factors.append('volume_env')

        # ── 7. Positioning / derivatives score ────────────────────
        positioning_score = self._score_positioning(
            direction=direction,
            derivatives_score=derivatives_score,
            sentiment_score=sentiment_score,
            funding_rate=funding_rate,
            oi_change_24h=oi_change_24h,
        )
        result.factors['positioning'] = positioning_score
        if positioning_score <= EG.BAD_FACTOR_THRESHOLD:
            result.bad_factors.append('positioning')

        # ── Kill Combos — check BEFORE scoring ───────────────────
        # These are non-negotiable hard blocks.  Specific toxic
        # combinations always kill the signal, regardless of how
        # good other factors are.  This prevents the averaging
        # problem where one great factor masks several terrible ones.
        kill_combo = self._check_kill_combos(
            direction=direction,
            grade=grade,
            trigger_quality_score=trigger_quality_score,
            volume_context=volume_context,
            volume_score=volume_score,
            spread_bps=spread_bps,
            whale_aligned=whale_aligned,
            whale_buy_ratio=whale_buy_ratio,
            eq_zone=eq_zone,
            is_dead_zone=is_dead_zone,
            session_name=session_name,
            setup_context=setup_context,
            execution_context=context,
        )
        if kill_combo:
            result.kill_combo = kill_combo
            result.should_block = True
            # Still compute composite for transparency
            composite = (
                session_score   * EG.W_SESSION +
                tq_score        * EG.W_TRIGGER_QUALITY +
                spread_score    * EG.W_SPREAD +
                whale_score     * EG.W_WHALE_ALIGNMENT +
                entry_score     * EG.W_ENTRY_POSITION +
                vol_score       * EG.W_VOLUME_ENV
            )
            composite += self._positioning_adjustment(positioning_score)
            result.execution_score = max(0.0, min(100.0, composite))
            # Flag near-miss: kill combo fired but underlying score was close
            if EG.NEAR_MISS_LOWER <= result.execution_score < EG.NEAR_MISS_UPPER:
                result.is_near_miss = True
                result.notes.append(
                    f"📍 NEAR MISS: score {result.execution_score:.0f} "
                    f"(in {EG.NEAR_MISS_LOWER:.0f}-{EG.NEAR_MISS_UPPER:.0f} band) "
                    f"but kill combo forced block"
                )
            result.reason = (
                f"⛔ KILL COMBO: {kill_combo} | "
                f"exec_score={result.execution_score:.0f} (irrelevant — combo fires)"
            )
            result.notes.append(f"🔥 Kill combo fired: {kill_combo}")
            # Still build factor notes for transparency
            self._build_factor_notes(result, session_name, is_killzone,
                                     trigger_quality_label, spread_bps,
                                     whale_aligned, eq_zone, volume_context)
            _log_gate_decision(result, symbol=symbol, direction=direction, grade=grade)
            return result

        # ── Weighted composite ────────────────────────────────────
        composite = (
            session_score   * EG.W_SESSION +
            tq_score        * EG.W_TRIGGER_QUALITY +
            spread_score    * EG.W_SPREAD +
            whale_score     * EG.W_WHALE_ALIGNMENT +
            entry_score     * EG.W_ENTRY_POSITION +
            vol_score       * EG.W_VOLUME_ENV
        )
        composite += self._positioning_adjustment(positioning_score)

        # ── Stacking penalty ─────────────────────────────────────
        # When 3+ factors are bad simultaneously, apply extra penalty
        n_bad = len(result.bad_factors)
        stacking_penalty = 0.0
        if n_bad > 2:
            stacking_penalty = min(
                EG.MAX_STACKING_PENALTY,
                (n_bad - 2) * EG.STACKING_PENALTY_PER,
            )
            composite -= stacking_penalty
            result.notes.append(
                f"⚠️ {n_bad} execution factors below threshold — "
                f"stacking penalty -{stacking_penalty:.0f}"
            )

        composite = max(0.0, min(100.0, composite))
        result.execution_score = composite

        # ── Decision ──────────────────────────────────────────────
        # Determine block threshold (A+ gets relaxed but not exempt)
        block_threshold = self._resolve_block_threshold(
            grade=grade,
            confidence=confidence,
            context_snapshot=result.context_snapshot,
            session_name=session_name,
            is_killzone=is_killzone,
            is_dead_zone=is_dead_zone,
            is_weekend=is_weekend,
        )
        result.block_threshold_used = block_threshold

        if composite < block_threshold:
            result.should_block = True
            # Flag near-miss: blocked but score is in the near-miss band
            if composite >= EG.NEAR_MISS_LOWER:
                result.is_near_miss = True
                result.notes.append(
                    f"📍 NEAR MISS: score {composite:.0f} "
                    f"(threshold {block_threshold:.0f}, "
                    f"band {EG.NEAR_MISS_LOWER:.0f}-{EG.NEAR_MISS_UPPER:.0f})"
                )
            result.reason = (
                f"⛔ EXECUTION GATE: NO TRADE | score={composite:.0f}/{block_threshold:.0f} | "
                f"bad=[{', '.join(result.bad_factors)}]"
            )
        elif composite < EG.SOFT_PENALTY_THRESHOLD:
            result.should_penalize = True
            result.penalty_mult = EG.SOFT_PENALTY_MULT
            result.reason = (
                f"⚠️ EXECUTION GATE: weak execution ({composite:.0f}/100) — "
                f"conf ×{EG.SOFT_PENALTY_MULT:.2f} | "
                f"weak=[{', '.join(result.bad_factors)}]"
            )
        else:
            result.reason = f"✅ Execution quality: {composite:.0f}/100"

        # ── Build detailed notes ──────────────────────────────────
        self._build_factor_notes(result, session_name, is_killzone,
                                 trigger_quality_label, spread_bps,
                                 whale_aligned, eq_zone, volume_context)
        if result.context_snapshot:
            _market = result.context_snapshot.get("market") or {}
            _positioning = result.context_snapshot.get("positioning") or {}
            _vol_regime = _market.get("volatility_regime")
            _transition_type = _market.get("transition_type")
            _transition_risk = _market.get("transition_risk")
            _derivatives_score = _positioning.get("derivatives_score")
            _sentiment_score = _positioning.get("sentiment_score")
            _funding_rate = _positioning.get("funding_rate")
            _oi_change_24h = _positioning.get("oi_change_24h")
            result.notes.append(
                "  ℹ️ context: "
                f"deriv={_derivatives_score if _derivatives_score is not None else 'n/a'} "
                f"sent={_sentiment_score if _sentiment_score is not None else 'n/a'} "
                f"funding={_funding_rate if _funding_rate is not None else 'n/a'} "
                f"oi={_oi_change_24h if _oi_change_24h is not None else 'n/a'} "
                f"vol={_vol_regime or 'UNKNOWN'} "
                f"transition={_transition_type or 'stable'}"
                + (
                    f" ({float(_transition_risk):.2f})"
                    if _transition_risk is not None else ""
                )
            )

        _log_gate_decision(result, symbol=symbol, direction=direction, grade=grade)
        return result

    # ── Kill Combo Engine ─────────────────────────────────────────

    @staticmethod
    def _check_kill_combos(
        *,
        direction: str,
        grade: str,
        trigger_quality_score: float,
        volume_context: str,
        volume_score: float,
        spread_bps: float,
        whale_aligned: Optional[bool],
        whale_buy_ratio: float,
        eq_zone: str,
        is_dead_zone: bool,
        session_name: str,
        setup_context: Optional[Dict[str, object]] = None,
        execution_context: Optional[Dict[str, object]] = None,
    ) -> str:
        """
        Check for non-negotiable kill combinations.

        Returns empty string if no combo fires, or a human-readable
        combo name if one does.  First match wins (most severe first).
        """
        is_low_trigger = trigger_quality_score < EG.KC_TRIGGER_QUALITY_FLOOR
        is_low_volume = volume_context == EG.KC_VOLUME_LOW_LABEL or volume_score < 30.0
        is_wide_spread = spread_bps > EG.KC_SPREAD_WIDE_BPS
        is_dead = is_dead_zone or ('dead' in (session_name or '').lower())
        is_whale_opposing = False
        if whale_aligned is False:
            if direction == "SHORT":
                is_whale_opposing = whale_buy_ratio >= EG.KC_WHALE_OPPOSITION_RATIO
            else:  # LONG
                is_whale_opposing = (1.0 - whale_buy_ratio) >= EG.KC_WHALE_OPPOSITION_RATIO
        has_countertrend_override = ExecutionQualityGate._has_countertrend_override(
            direction=direction,
            setup_context=setup_context,
            execution_context=execution_context,
        )
        is_wrong_zone = (
            (direction == "LONG" and eq_zone == "premium") or
            (direction == "SHORT" and eq_zone == "discount")
        )

        semantic_kill = ExecutionQualityGate._check_semantic_kills(
            direction=direction,
            setup_context=setup_context,
            execution_context=execution_context,
        )
        if semantic_kill:
            return semantic_kill

        # KC5: Dead zone + low trigger + low volume = triple liquidity death
        # (Most severe — check first)
        if is_dead and is_low_trigger and is_low_volume:
            return "DEAD_ZONE + LOW_TRIGGER + LOW_VOLUME"

        # KC2: Extreme whale opposition
        # A+ signals exempt (Wyckoff UTAD = whales distributing)
        if (
            is_whale_opposing
            and not has_countertrend_override
            and not (EG.KC_WHALE_OPPOSITION_APLUS_EXEMPT and grade == "A+")
        ):
            return f"EXTREME_WHALE_OPPOSITION ({whale_buy_ratio:.0%} vs {direction})"

        # KC1: Terrible trigger + low volume = no edge
        if is_low_trigger and is_low_volume:
            return "LOW_TRIGGER + LOW_VOLUME"

        # KC3: Dead zone + wide spread = zero liquidity
        if is_dead and is_wide_spread:
            return f"DEAD_ZONE + WIDE_SPREAD ({spread_bps:.0f} bps)"

        # KC4: Wrong-zone entry + low trigger = worst timing
        if is_wrong_zone and is_low_trigger:
            return f"WRONG_ZONE ({eq_zone}) + LOW_TRIGGER"

        return ""

    @staticmethod
    def _check_semantic_kills(
        *,
        direction: str,
        setup_context: Optional[Dict[str, object]],
        execution_context: Optional[Dict[str, object]] = None,
    ) -> str:
        if not setup_context:
            return ""

        structure = dict((setup_context or {}).get("structure") or {})
        pattern = dict((setup_context or {}).get("pattern") or {})
        has_countertrend_override = ExecutionQualityGate._has_countertrend_override(
            direction=direction,
            setup_context=setup_context,
            execution_context=execution_context,
        )

        choch_direction = str(structure.get("choch_direction") or "").upper()
        if structure.get("choch") and choch_direction:
            if direction == "SHORT" and choch_direction == "BULLISH":
                return "SEMANTIC_KILL: BULLISH_CHOCH vs SHORT"
            if direction == "LONG" and choch_direction == "BEARISH":
                return "SEMANTIC_KILL: BEARISH_CHOCH vs LONG"

        phase = str(pattern.get("wyckoff_phase") or "").lower()
        bullish_phases = {"accumulation", "markup", "reaccumulation"}
        bearish_phases = {"distribution", "markdown", "redistribution"}
        if direction == "SHORT" and (phase in bullish_phases or pattern.get("spring_detected")):
            return f"SEMANTIC_KILL: BULLISH_WYCKOFF ({phase or 'spring'}) vs SHORT"
        if direction == "LONG" and (phase in bearish_phases or pattern.get("utad_detected")):
            return f"SEMANTIC_KILL: BEARISH_WYCKOFF ({phase or 'utad'}) vs LONG"

        # ── HTF Trend-Direction Guard ──────────────────────────────
        # Block signals that contradict the higher-timeframe trend.
        if EG.SEMANTIC_TREND_GUARD:
            trend = str(structure.get("trend") or "").upper()
            # Strong bullish regime → block naked shorts (no breakdown)
            if direction == "SHORT" and "BULL" in trend:
                # _has_countertrend_override() centralizes the allowed reversal/
                # continuation evidence (CHoCH, Wyckoff, local structure, crowded positioning).
                has_bearish_reversal = (
                    has_countertrend_override
                )
                if not has_bearish_reversal:
                    return f"SEMANTIC_KILL: HTF_TREND BULL vs SHORT (no reversal signal)"
            # Strong bearish regime → block naked longs (no reversal)
            if direction == "LONG" and "BEAR" in trend:
                # See _has_countertrend_override() for the accepted override evidence set.
                has_bullish_reversal = (
                    has_countertrend_override
                )
                if not has_bullish_reversal:
                    return f"SEMANTIC_KILL: HTF_TREND BEAR vs LONG (no reversal signal)"

        # ── Volatility Regime Guard ────────────────────────────────
        # In extreme volatility, block tight-stop scalp setups.
        if EG.SEMANTIC_VOLATILITY_GUARD and execution_context:
            market_ctx = dict((execution_context or {}).get("market") or {})
            vol_regime = str(market_ctx.get("volatility_regime") or "").upper()
            if vol_regime in {v.upper() for v in EG.SEMANTIC_EXTREME_VOL_LABELS}:
                location = dict((setup_context or {}).get("location") or {})
                setup_class = str(location.get("setup_class") or "").lower()
                if setup_class == "scalp":
                    return f"SEMANTIC_KILL: SCALP in {vol_regime} volatility"

        return ""

    @staticmethod
    def _has_countertrend_override(
        *,
        direction: str,
        setup_context: Optional[Dict[str, object]],
        execution_context: Optional[Dict[str, object]] = None,
    ) -> bool:
        if has_reversal_confirmation(
            direction,
            setup_context=setup_context,
            execution_context=execution_context,
        ):
            return True

        if not execution_context:
            return False

        positioning = dict((execution_context or {}).get("positioning") or {})
        derivatives_score = float(positioning.get("derivatives_score") or 0.0)
        sentiment_score = float(positioning.get("sentiment_score") or 0.0)
        funding_rate = float(positioning.get("funding_rate") or 0.0)
        oi_change_24h = float(positioning.get("oi_change_24h") or 0.0)

        positioning_score = max(derivatives_score, sentiment_score)
        crowded_countertrend = False
        if direction == "SHORT":
            crowded_countertrend = (
                funding_rate >= EG.COUNTERTREND_FUNDING_POSITIVE_EXTREME
                and oi_change_24h >= EG.COUNTERTREND_OI_CONFIRM_MIN
                and positioning_score >= EG.COUNTERTREND_POSITIONING_SCORE_MIN
            )
        elif direction == "LONG":
            crowded_countertrend = (
                funding_rate <= EG.COUNTERTREND_FUNDING_NEGATIVE_EXTREME
                and oi_change_24h >= EG.COUNTERTREND_OI_CONFIRM_MIN
                and positioning_score >= EG.COUNTERTREND_POSITIONING_SCORE_MIN
            )

        return crowded_countertrend

    @staticmethod
    def _resolve_block_threshold(
        *,
        grade: str,
        confidence: float,
        context_snapshot: Optional[Dict[str, object]],
        session_name: str,
        is_killzone: bool,
        is_dead_zone: bool,
        is_weekend: bool,
    ) -> float:
        threshold = EG.APLUS_BLOCK_THRESHOLD if (grade == "A+" and confidence >= 90) else EG.HARD_BLOCK_THRESHOLD
        if not context_snapshot:
            return threshold

        session_ctx = dict((context_snapshot or {}).get("session") or {})
        market_ctx = dict((context_snapshot or {}).get("market") or {})

        _session_name = str(session_ctx.get("name") or session_name or "").lower()
        _is_killzone = bool(session_ctx.get("is_killzone", is_killzone))
        _is_dead_zone = bool(session_ctx.get("is_dead_zone", is_dead_zone))
        _is_weekend = bool(session_ctx.get("is_weekend", is_weekend))

        if _is_dead_zone or "dead" in _session_name:
            threshold += EG.ADAPTIVE_DEAD_ZONE_ADJ
        elif _is_killzone:
            threshold += EG.ADAPTIVE_KILLZONE_ADJ
        elif _is_weekend or "weekend" in _session_name:
            threshold += EG.ADAPTIVE_WEEKEND_ADJ

        vol_regime = str(market_ctx.get("volatility_regime") or "").upper()
        if any(token in vol_regime for token in ("HIGH", "VOLATILE", "EXPANSION", "BREAKOUT")):
            threshold += EG.ADAPTIVE_HIGH_VOL_ADJ
        elif any(token in vol_regime for token in ("LOW", "QUIET", "COMPRESS", "SQUEEZE")):
            threshold += EG.ADAPTIVE_LOW_VOL_ADJ

        try:
            from analyzers.near_miss_tracker import near_miss_tracker
            threshold += float(near_miss_tracker.compute_threshold_adjustment() or 0.0)
        except Exception as _e:
            # AUDIT FIX (PR-2): was silent.  Threshold adjustment is a
            # real governance input; a failure here should surface in
            # the logs so we can tell adaptive gating from frozen
            # defaults.  WARN (not ERROR) because the fallback is safe.
            logger.warning(
                "ExecutionGate: near_miss_tracker threshold adjust failed: %s",
                _e, exc_info=True,
            )

        return max(EG.ADAPTIVE_THRESHOLD_MIN, min(EG.ADAPTIVE_THRESHOLD_MAX, threshold))

    # ── Factor notes helper ───────────────────────────────────────

    @staticmethod
    def _build_factor_notes(
        result: ExecutionAssessment,
        session_name: str,
        is_killzone: bool,
        trigger_quality_label: str,
        spread_bps: float,
        whale_aligned: Optional[bool],
        eq_zone: str,
        volume_context: str,
    ) -> None:
        """Append human-readable per-factor notes to result."""
        labels = {
            'session': session_name or ('Killzone' if is_killzone else 'Normal'),
            'trigger_quality': trigger_quality_label,
            'spread': f"{spread_bps:.0f} bps" if spread_bps > 0 else "unknown",
            'whale_alignment': (
                'aligned' if whale_aligned is True
                else 'opposing' if whale_aligned is False
                else 'neutral'
            ),
            'entry_position': eq_zone or 'unknown',
            'volume_env': volume_context,
            'positioning': 'derivatives / sentiment / oi',
        }
        for factor, score in result.factors.items():
            marker = "❌" if score <= EG.BAD_FACTOR_THRESHOLD else "✅"
            result.notes.append(
                f"  {marker} {factor}: {score:.0f}/100 ({labels.get(factor, '')})"
            )

    # ── Factor scoring methods ────────────────────────────────────

    @staticmethod
    def _score_session(
        *,
        session_name: str,
        is_killzone: bool,
        is_dead_zone: bool,
        is_weekend: bool,
    ) -> float:
        """Score session quality for execution."""
        if is_killzone:
            return EG.SESSION_SCORE_KILLZONE
        if is_dead_zone:
            return EG.SESSION_SCORE_DEAD_ZONE
        if is_weekend:
            return EG.SESSION_SCORE_WEEKEND

        # Map session names
        name_lower = session_name.lower() if session_name else ""
        if 'dead' in name_lower:
            return EG.SESSION_SCORE_DEAD_ZONE
        if 'asia' in name_lower:
            return EG.SESSION_SCORE_ASIA
        if any(kz in name_lower for kz in ('london', 'ny', 'new york', 'killzone')):
            return EG.SESSION_SCORE_KILLZONE
        if 'weekend' in name_lower or 'saturday' in name_lower or 'sunday' in name_lower:
            return EG.SESSION_SCORE_WEEKEND

        return EG.SESSION_SCORE_NORMAL

    @staticmethod
    def _score_spread(spread_bps: float) -> float:
        """Score spread quality — tighter is better."""
        if spread_bps <= 0:
            # No spread data — assume moderate (don't penalize or reward)
            return 60.0
        if spread_bps <= EG.SPREAD_TIGHT_BPS:
            return EG.SPREAD_SCORE_TIGHT
        if spread_bps <= EG.SPREAD_NORMAL_BPS:
            # Linear interpolation between tight and normal
            ratio = (spread_bps - EG.SPREAD_TIGHT_BPS) / (EG.SPREAD_NORMAL_BPS - EG.SPREAD_TIGHT_BPS)
            return EG.SPREAD_SCORE_TIGHT - ratio * (EG.SPREAD_SCORE_TIGHT - EG.SPREAD_SCORE_NORMAL)
        if spread_bps <= EG.SPREAD_WIDE_BPS:
            ratio = (spread_bps - EG.SPREAD_NORMAL_BPS) / (EG.SPREAD_WIDE_BPS - EG.SPREAD_NORMAL_BPS)
            return EG.SPREAD_SCORE_NORMAL - ratio * (EG.SPREAD_SCORE_NORMAL - EG.SPREAD_SCORE_WIDE)
        return EG.SPREAD_SCORE_EXTREME

    @staticmethod
    def _score_whale_alignment(
        *,
        direction: str,
        whale_aligned: Optional[bool],
        whale_buy_ratio: float,
    ) -> float:
        """Score whale alignment with signal direction."""
        if whale_aligned is None:
            return EG.WHALE_NEUTRAL_SCORE

        if whale_aligned:
            # Whales confirm direction — scale by buy ratio strength
            return EG.WHALE_ALIGNED_SCORE
        else:
            # Whales oppose direction — severity depends on buy ratio
            if direction == "SHORT":
                # SHORT while whales buy heavily — very bad
                opposition_strength = whale_buy_ratio  # higher = worse for shorts
            else:
                # LONG while whales sell heavily — very bad
                opposition_strength = 1.0 - whale_buy_ratio  # lower buy = more selling = worse for longs

            # Interpolate: strong opposition → WHALE_OPPOSING_SCORE,
            # mild opposition → midpoint
            if opposition_strength >= 0.8:
                return EG.WHALE_OPPOSING_SCORE
            elif opposition_strength >= 0.6:
                ratio = (opposition_strength - 0.6) / 0.2
                return EG.WHALE_NEUTRAL_SCORE - ratio * (EG.WHALE_NEUTRAL_SCORE - EG.WHALE_OPPOSING_SCORE)
            else:
                return EG.WHALE_NEUTRAL_SCORE

    @staticmethod
    def _score_entry_position(
        *,
        direction: str,
        eq_zone: str,
        eq_zone_depth: float,
    ) -> float:
        """Score entry position quality relative to direction."""
        if not eq_zone or eq_zone in ('no_range', 'trending'):
            return EG.ENTRY_TREND_BYPASS

        if eq_zone == 'eq_dead_zone':
            return EG.ENTRY_EQ_ZONE

        if eq_zone == 'discount':
            if direction == "LONG":
                return EG.ENTRY_DISCOUNT_LONG
            else:
                return EG.ENTRY_WRONG_ZONE
        elif eq_zone == 'premium':
            if direction == "SHORT":
                return EG.ENTRY_PREMIUM_SHORT
            else:
                return EG.ENTRY_WRONG_ZONE

        return EG.ENTRY_EQ_ZONE

    @staticmethod
    def _score_volume_env(
        *,
        volume_context: str,
        volume_score: float,
    ) -> float:
        """Score volume environment quality."""
        context_scores = {
            "BREAKOUT": EG.VOLUME_BREAKOUT_SCORE,
            "ABOVE_AVG": EG.VOLUME_ABOVE_AVG_SCORE,
            "HIGH": EG.VOLUME_ABOVE_AVG_SCORE,
            "NORMAL": EG.VOLUME_NORMAL_SCORE,
            "LOW_VOL": EG.VOLUME_LOW_SCORE,
            "CLIMACTIC": EG.VOLUME_CLIMACTIC_SCORE,
        }
        context_score = context_scores.get(volume_context, EG.VOLUME_NORMAL_SCORE)

        # Blend with the aggregator's volume score for nuance
        # 60% context label + 40% raw volume score
        blended = context_score * 0.60 + volume_score * 0.40
        return max(0.0, min(100.0, blended))

    # One-time warn flag: the contract with analyzers/derivatives.py is
    # that funding_rate is in **percentage points** (0.05 → 0.05%).  If
    # a caller passes the raw fractional form (0.0005) the thresholds
    # below will never fire; if they pass a wildly out-of-range value
    # (± > 5 pp) it's almost certainly a unit error on the upstream.
    # AUDIT FIX: additionally flag suspected fractional form — a
    # non-zero funding whose magnitude is below this threshold is
    # vanishingly rare in pp (would be <0.0001 %); it is the
    # fingerprint of a fractional (×100 missing) upstream.
    _FUNDING_SUSPECT_ABS_PP = 5.0
    _FUNDING_FRACTIONAL_SUSPECT_ABS = 1e-4   # |f| below this but > 0 ⇒ likely fractional
    _funding_unit_warned: bool = False

    @staticmethod
    def _score_positioning(
        *,
        direction: str,
        derivatives_score: float,
        sentiment_score: float,
        funding_rate: float,
        oi_change_24h: float,
    ) -> float:
        """Score direction-sensitive derivatives / positioning support."""
        base = derivatives_score * 0.65 + sentiment_score * 0.35
        oi_mag = abs(float(oi_change_24h or 0.0))
        if oi_mag >= 8.0:
            if base >= 60.0:
                base += 5.0
            elif base <= 40.0:
                base -= 5.0

        funding = float(funding_rate or 0.0)
        # AUDIT FIX (PR-2): assert funding unit against the
        # derivatives.py contract (percentage points).  We don't raise —
        # a wrong-unit caller shouldn't hard-stop gating — but we warn
        # exactly once per process so the issue is visible.
        # Two failure modes are both flagged:
        #   1. |f| > 5 pp   — almost certainly a percentage mistake upstream
        #   2. 0 < |f| < 1e-4 pp — almost certainly a fractional fundingRate
        #      that never got the ×100 scale applied (derivatives.py
        #      guarantees pp, so a legitimately microscopic funding is
        #      exceedingly rare).
        _abs_funding = abs(funding)
        _suspect_high = _abs_funding > ExecutionQualityGate._FUNDING_SUSPECT_ABS_PP
        _suspect_low = (
            0.0 < _abs_funding < ExecutionQualityGate._FUNDING_FRACTIONAL_SUSPECT_ABS
        )
        if (_suspect_high or _suspect_low) and not ExecutionQualityGate._funding_unit_warned:
            ExecutionQualityGate._funding_unit_warned = True
            _mode = "pp-out-of-range" if _suspect_high else "fractional-not-scaled"
            logger.warning(
                "ExecutionGate: funding_rate=%s appears to violate the "
                "percentage-point contract expected by derivatives.py "
                "(0.05 == 0.05%%) — suspected mode: %s",
                funding, _mode,
            )
        if direction == "LONG":
            if funding <= -0.01:
                base += 4.0
            elif funding >= 0.03:
                base -= 6.0
        else:
            if funding >= 0.03:
                base += 4.0
            elif funding <= -0.01:
                base -= 6.0

        return max(0.0, min(100.0, base))

    @staticmethod
    def _positioning_adjustment(positioning_score: float) -> float:
        """Translate positioning score into a bounded composite adjustment."""
        return max(
            ExecutionQualityGate.POSITIONING_ADJUSTMENT_MIN,
            min(
                ExecutionQualityGate.POSITIONING_ADJUSTMENT_MAX,
                (positioning_score - ExecutionQualityGate.POSITIONING_NEUTRAL_SCORE)
                * ExecutionQualityGate.POSITIONING_ADJUSTMENT_FACTOR,
            ),
        )


# ── Singleton ──────────────────────────────────────────────
execution_gate = ExecutionQualityGate()
