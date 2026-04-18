"""
Cross-symbol signal ranking.

Accepts a batch of ``ScoredSignal`` objects from a single scan cycle,
computes a composite rank score, filters correlated duplicates, and
returns the top-N signals for execution.
"""

import logging
from typing import Dict, List, Tuple

from config.constants import SignalRanking
from config.loader import cfg
from signals.aggregator import ScoredSignal
from strategies.base import SignalDirection

logger = logging.getLogger(__name__)


# ── Hardcoded correlation map (symmetrical) ─────────────────────
# Covers the top-4 majors where instantaneous pair correlation can be
# assumed >0.5 without computing it. The broader "BTC-proxy" case for
# alts is handled via correlation_analyzer.get_cached_correlation()
# inside _correlation_filter below.

_CORRELATION_MAP: Dict[Tuple[str, str], float] = {
    ("BTCUSDT", "ETHUSDT"): 0.85,
    ("BTCUSDT", "SOLUSDT"): 0.65,
    ("BTCUSDT", "BNBUSDT"): 0.72,
    ("ETHUSDT", "SOLUSDT"): 0.60,
    ("ETHUSDT", "BNBUSDT"): 0.68,
    ("SOLUSDT", "BNBUSDT"): 0.55,
}


def _pair_correlation(sym_a: str, sym_b: str) -> float:
    """Return the known correlation between two symbols, or 0.0.

    Lookup order:
      1. Hardcoded major-pair map (instant, deterministic).
      2. Cached BTC-correlation from correlation_analyzer for any pair
         that involves BTC but isn't in the hardcoded map — this lets
         long-tail alts fall under the same correlation gate as the
         majors once their BTC-beta has been computed at least once.
    """
    key = (sym_a, sym_b) if sym_a <= sym_b else (sym_b, sym_a)
    mapped = _CORRELATION_MAP.get(key, 0.0)
    if mapped > 0.0:
        return mapped
    # Fallback: if exactly one side is BTC, use the cached alt→BTC correlation.
    if "BTCUSDT" in key:
        other = key[1] if key[0] == "BTCUSDT" else key[0]
        try:
            from analyzers.correlation import correlation_analyzer
            return float(correlation_analyzer.get_cached_correlation(other))
        except Exception:
            return 0.0
    return 0.0


def _btc_correlation(symbol: str) -> float:
    """Return |ρ| of this symbol vs. BTC from the correlation analyzer cache."""
    if symbol == "BTCUSDT":
        return 1.0
    try:
        from analyzers.correlation import correlation_analyzer
        return abs(float(correlation_analyzer.get_cached_correlation(symbol)))
    except Exception:
        return 0.0


# ── Core class ───────────────────────────────────────────────────

class SignalRanker:
    """Ranks a batch of scored signals and filters correlated duplicates."""

    # ── public API ───────────────────────────────────────────────

    def rank(
        self,
        signals: List[ScoredSignal],
        regime: str,
        max_signals: int = SignalRanking.MAX_SIGNALS_PER_CYCLE,
    ) -> List[ScoredSignal]:
        """Score, de-correlate, and return the top *max_signals*.

        Parameters
        ----------
        signals:     Batch of signals from the current scan cycle.
        regime:      Current market regime string (e.g. ``"BULL_TREND"``).
        max_signals: Maximum number of signals to return.
        """
        if not signals:
            return []

        scored = [
            (sig, self._composite_score(sig, regime)) for sig in signals
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        for sig, score in scored:
            symbol = sig.base_signal.symbol
            direction = (
                "LONG"
                if sig.base_signal.direction == SignalDirection.LONG
                else "SHORT"
            )
            logger.info(
                f"📊 Rank | {symbol} {direction} "
                f"| score={score:.4f} conf={sig.final_confidence:.1f} "
                f"rr={sig.base_signal.rr_ratio:.2f} vol={sig.volume_score:.1f}"
            )

        ranked_signals = [s for s, _ in scored]
        filtered = self._correlation_filter(ranked_signals)

        if len(filtered) < len(ranked_signals):
            logger.info(
                f"🔗 Correlation filter removed "
                f"{len(ranked_signals) - len(filtered)} duplicate(s)"
            )

        result = filtered[:max_signals]
        logger.info(f"✅ Returning {len(result)}/{len(signals)} signals after ranking")
        return result

    # ── scoring ──────────────────────────────────────────────────

    @staticmethod
    def _composite_score(signal: ScoredSignal, regime: str) -> float:
        """Compute the weighted composite rank score for *signal*."""
        confidence_norm = signal.final_confidence / 100.0

        rr_capped = min(
            signal.base_signal.rr_ratio,
            SignalRanking.RR_NORMALIZATION_CAP,
        )
        rr_norm = rr_capped / SignalRanking.RR_NORMALIZATION_CAP

        volume_norm = signal.volume_score / 100.0

        regime_alignment = _regime_alignment(signal, regime)

        return (
            confidence_norm * SignalRanking.RANK_WEIGHT_CONFIDENCE
            + rr_norm * SignalRanking.RANK_WEIGHT_RR
            + volume_norm * SignalRanking.RANK_WEIGHT_VOLUME
            + regime_alignment * SignalRanking.RANK_WEIGHT_REGIME
        )

    # ── correlation filter ───────────────────────────────────────

    @staticmethod
    def _correlation_filter(
        signals: List[ScoredSignal],
    ) -> List[ScoredSignal]:
        """Remove lower-ranked signals that are correlated with a
        higher-ranked signal in the same direction.

        Runs two passes (both direction-aware):
          1. Pairwise correlation — kills a lower-ranked signal if its
             correlation with any already-kept same-direction signal
             exceeds ``CORRELATION_THRESHOLD`` (hardcoded majors +
             cached BTC-correlation fallback via _pair_correlation).
          2. BTC-concentration cap — once more than
             ``BTC_CONCENTRATION_MAX_SAME_SIDE`` kept signals on the
             same side are each individually high-BTC-correlated
             (≥ ``BTC_CONCENTRATION_CORR_THRESHOLD``), any further
             same-side high-BTC-corr signal is treated as a redundant
             "BTC proxy" and dropped. This catches the long-tail case
             where the pairwise map misses (e.g. 3 altcoins each with
             ρ≈0.75 to BTC but no pair is in the hardcoded table).

        The input list must already be sorted by rank (highest first).
        """
        kept: List[ScoredSignal] = []
        # direction -> count of kept signals that are BTC-proxies on that side
        btc_proxy_count: Dict[SignalDirection, int] = {}

        for sig in signals:
            sym = sig.base_signal.symbol
            direction = sig.base_signal.direction

            # ── Pass 1: pairwise correlation ─────────────────
            is_correlated = False
            for accepted in kept:
                if accepted.base_signal.direction != direction:
                    continue
                corr = _pair_correlation(sym, accepted.base_signal.symbol)
                if corr >= SignalRanking.CORRELATION_THRESHOLD:
                    logger.info(
                        f"🔗 Filtered {sym} (corr={corr:.2f} with "
                        f"{accepted.base_signal.symbol}, same direction)"
                    )
                    is_correlated = True
                    break
            if is_correlated:
                continue

            # ── Pass 2: BTC-concentration cap ─────────────────
            # Count this signal as a BTC-proxy if its |ρ_BTC| clears
            # the concentration threshold. BTCUSDT itself always
            # counts. When more than MAX_SAME_SIDE proxies on the
            # same direction are already kept, drop this one.
            sym_btc_corr = _btc_correlation(sym)
            is_btc_proxy = sym_btc_corr >= SignalRanking.BTC_CONCENTRATION_CORR_THRESHOLD
            if is_btc_proxy and btc_proxy_count.get(direction, 0) >= \
                    SignalRanking.BTC_CONCENTRATION_MAX_SAME_SIDE:
                logger.info(
                    f"🔗 Filtered {sym} (|ρ_BTC|={sym_btc_corr:.2f} — "
                    f"BTC-concentration cap reached for "
                    f"{getattr(direction, 'value', str(direction))})"
                )
                continue

            kept.append(sig)
            if is_btc_proxy:
                btc_proxy_count[direction] = btc_proxy_count.get(direction, 0) + 1

        return kept


# ── helpers ──────────────────────────────────────────────────────

def _regime_alignment(signal: ScoredSignal, regime: str) -> float:
    """Return 1.0 if the signal direction aligns with the regime, else 0.0."""
    regime_upper = regime.upper()
    is_long = signal.base_signal.direction == SignalDirection.LONG
    if is_long and "BULL" in regime_upper:
        return 1.0
    if not is_long and "BEAR" in regime_upper:
        return 1.0
    return 0.0


def rank_publish_candidates(candidates: List, regime: str):
    """Rank objects that expose a ``scored`` attribute and split selected/skipped."""
    if len(candidates) <= 1:
        return list(candidates), []

    ranker = SignalRanker()
    ranked_scored = ranker.rank(
        [candidate.scored for candidate in candidates],
        regime=regime,
    )
    candidate_by_scored_id = {
        id(candidate.scored): candidate for candidate in candidates
    }
    ranked = [
        candidate_by_scored_id[id(scored)]
        for scored in ranked_scored
        if id(scored) in candidate_by_scored_id
    ]
    ranked_ids = {id(candidate.scored) for candidate in ranked}
    skipped = [
        candidate for candidate in candidates
        if id(candidate.scored) not in ranked_ids
    ]
    return ranked, skipped
