"""
TitanBot Pro — Signal Clarity Scorer
======================================
Sometimes signals are technically valid but MESSY.
A messy signal = poor manual execution = avoidable loss.

Measures:
  1. Entry zone width (tight = clear, wide = ambiguous)
  2. RR clarity (integer-like ratio = clean, fractional mess = ambiguous)
  3. Stop placement precision (near structure = clean, floating = ambiguous)
  4. Conflicting indicator count (contradictions in confluence)
  5. ATR-relative entry width (wide relative to vol = dangerous)

Reject ambiguous setups. Cleaner trades → better manual execution.

Usage:
  from signals.signal_clarity import signal_clarity_scorer
  clarity = signal_clarity_scorer.score(signal, atr)
  if clarity.score < 40:
      → reject or heavy penalty
"""

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ClarityResult:
    score: int              # 0–100 (higher = cleaner signal)
    grade: str              # "A+", "A", "B", "C", "REJECT"
    is_clean: bool          # True if score >= 50
    reasons: List[str]      # Human-readable breakdown
    confidence_adjustment: float  # Multiplier to apply (0.7 to 1.1)

    @staticmethod
    def make_grade(score: int) -> str:
        if score >= 85: return "A+"
        if score >= 70: return "A"
        if score >= 55: return "B"
        if score >= 40: return "C"
        return "REJECT"


class SignalClarityScorer:
    """
    Scores how clean and actionable a signal is.
    High score = precise entry, clear stop, unambiguous RR.
    Low score = wide entry, floating stop, mixed indicators.
    """

    def score(self, signal, atr: float) -> ClarityResult:
        """
        Score signal clarity. 'signal' should have:
            entry_low, entry_high, stop_loss, take_profit_1, take_profit_2,
            rr_ratio, confluence (list of str)
        """
        total_score = 100
        reasons: List[str] = []

        entry_mid = (signal.entry_low + signal.entry_high) / 2.0
        entry_width = signal.entry_high - signal.entry_low

        # ── 1. Entry zone width (relative to ATR) ──────────────────────
        if atr > 0 and entry_width > 0:
            width_pct = entry_width / atr
            if width_pct < 0.3:
                reasons.append("✅ Entry zone tight (<0.3 ATR)")
                total_score += 5
            elif width_pct < 0.6:
                reasons.append(f"✅ Entry zone acceptable ({width_pct:.1f}x ATR)")
            elif width_pct < 1.0:
                deduct = 12
                total_score -= deduct
                reasons.append(f"⚠️ Entry zone wide ({width_pct:.1f}x ATR) −{deduct}pts")
            else:
                deduct = 25
                total_score -= deduct
                reasons.append(f"🚫 Entry zone very wide ({width_pct:.1f}x ATR) −{deduct}pts")

        # ── 2. RR ratio clarity ─────────────────────────────────────────
        rr = getattr(signal, 'rr_ratio', 0)
        if rr >= 2.5:
            reasons.append(f"✅ Strong RR ({rr:.1f}R)")
            total_score += 5
        elif rr >= 1.8:
            reasons.append(f"✅ Good RR ({rr:.1f}R)")
        elif rr >= 1.2:
            deduct = 8
            total_score -= deduct
            reasons.append(f"⚠️ Weak RR ({rr:.1f}R) −{deduct}pts")
        else:
            deduct = 20
            total_score -= deduct
            reasons.append(f"🚫 Poor RR ({rr:.1f}R) −{deduct}pts")

        # ── 3. Stop placement precision ─────────────────────────────────
        # Stop near a structural level = clean. Floating stop = ambiguous.
        stop = getattr(signal, 'stop_loss', 0.0)
        if stop > 0 and atr > 0 and entry_mid > 0:
            # Distance from entry to stop (in ATR)
            stop_dist_atr = abs(entry_mid - stop) / atr
            confluence_notes = getattr(signal, 'confluence', [])
            stop_near_structure = any(
                any(kw in note.lower() for kw in ['ob', 'fvg', 'structure', 'low', 'demand', 'supply'])
                for note in confluence_notes
                if isinstance(note, str)
            )
            if stop_near_structure:
                reasons.append("✅ Stop near structural level")
                total_score += 5
            elif stop_dist_atr < 0.5:
                deduct = 15
                total_score -= deduct
                reasons.append(f"⚠️ Stop too tight ({stop_dist_atr:.1f}x ATR) −{deduct}pts")
            elif stop_dist_atr > 3.0:
                deduct = 12
                total_score -= deduct
                reasons.append(f"⚠️ Stop too loose ({stop_dist_atr:.1f}x ATR) −{deduct}pts")
            else:
                reasons.append(f"✅ Stop distance reasonable ({stop_dist_atr:.1f}x ATR)")

        # ── 4. Conflicting indicators ────────────────────────────────────
        confluence = getattr(signal, 'confluence', [])
        # Conflict detection uses direction-specific keywords.
        # 'sweep' is intentionally excluded — a sweep note is always directionally aligned
        # (swept_low = bullish, swept_high = bearish) so it never indicates conflict.
        conflict_keywords_long  = ['bearish', 'short', 'resistance', 'rejection', 'bear']
        conflict_keywords_short = ['bullish', 'long', 'support', 'bounce', 'bull']
        direction = getattr(signal, 'direction', None)
        direction_val = direction.value if hasattr(direction, 'value') else str(direction)

        conflict_kw = conflict_keywords_long if direction_val == "LONG" else conflict_keywords_short
        conflicts = [
            note for note in confluence
            if isinstance(note, str)
               and any(kw in note.lower() for kw in conflict_kw)
               and not note.startswith('✅')  # Don't flag positive confluence
        ]
        if conflicts:
            deduct = min(25, len(conflicts) * 8)
            total_score -= deduct
            reasons.append(f"⚠️ {len(conflicts)} conflicting indicator(s) −{deduct}pts")
        else:
            reasons.append("✅ No contradictory signals")

        # ── 5. Confluence richness ───────────────────────────────────────
        positive_confluence = [
            note for note in confluence
            if isinstance(note, str)
               and any(kw in note.lower() for kw in ['✅', 'aligned', 'confirmed', 'sweep', 'ob', 'fvg'])
        ]
        if len(positive_confluence) >= 4:
            total_score += 8
            reasons.append(f"✅ Rich confluence ({len(positive_confluence)} factors)")
        elif len(positive_confluence) >= 2:
            reasons.append(f"✅ Adequate confluence ({len(positive_confluence)} factors)")
        elif len(positive_confluence) == 1:
            total_score -= 10
            reasons.append(f"⚠️ Thin confluence (only {len(positive_confluence)} factor) −10pts")
        else:
            total_score -= 18
            reasons.append("🚫 No positive confluence factors −18pts")

        # ── Clamp and grade ──────────────────────────────────────────────
        total_score = max(0, min(100, total_score))
        grade = ClarityResult.make_grade(total_score)
        is_clean = total_score >= 50

        # Confidence adjustment: clean signals get a small boost, messy a cut
        if total_score >= 80:
            conf_adj = 1.08
        elif total_score >= 60:
            conf_adj = 1.0
        elif total_score >= 45:
            conf_adj = 0.90
        else:
            conf_adj = 0.75

        logger.debug(
            f"Signal clarity: score={total_score} grade={grade} "
            f"conf_adj={conf_adj:.2f} | {'; '.join(reasons)}"
        )

        return ClarityResult(
            score=total_score,
            grade=grade,
            is_clean=is_clean,
            reasons=reasons,
            confidence_adjustment=conf_adj,
        )


# ── Singleton ─────────────────────────────────────────────────────────────
signal_clarity_scorer = SignalClarityScorer()
