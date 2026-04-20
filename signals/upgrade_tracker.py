"""
TitanBot Pro — Upgrade Tracker (Phase 1 + Phase 2 unified)
============================================================
Decides whether a C-grade signal has enough upgrade potential
to justify interrupting the trader (interrupt worthiness filtering).

Architecture:
  Phase 1 (rule-based):  Always active. Fast. No data required.
  Phase 2 (Bayesian):    Activates automatically per context key
                         once MIN_SAMPLES observations exist.
                         Runs alongside Phase 1, not instead of it.

The two phases are blended:
  final_up = phase1_weight * phase1_up + phase2_weight * learned_up

Phase 2 weight grows from 0.0 → 0.6 as data accumulates:
  - 20 samples  → 20% Phase 2 (just activated)
  - 40 samples  → 40% Phase 2
  - 60 samples  → 60% Phase 2 (full weight, ~5-6 weeks)
  - Never:   100% Phase 2 (Phase 1 always contributes 40%)

The ramp uses min(0.6, sample_count / 100) — deliberately slow.
Early posteriors are noisy. Phase 2 earns influence gradually.

Context keys for learning:
  strategy + regime + direction
  e.g. "IchimokuCloud|CHOPPY|SHORT"

Stored in DB table: c_signal_log
  signal_id, symbol, strategy, direction, regime,
  confidence, up_score, sent_to_user, upgraded,
  upgrade_grade, upgrade_latency_min, created_at

UP thresholds:
  UP >= 0.45  → CONTEXT message to signals channel
  UP <  0.45  → admin channel only

Minimum TP2: 1.5% (hard gate — below this, not worth attention)
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from analyzers.regime import regime_analyzer
from strategies.base import direction_str

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────
UP_SEND_THRESHOLD   = 0.45    # Minimum UP to send to signals channel
MIN_TP2_PCT         = 1.5     # Hard minimum floor — never below this regardless of regime
MIN_TP2_PCT_CHOPPY  = 0.9     # Choppy ranges are tighter; scale down toward this at chop=1.0
MIN_SAMPLES         = 20      # Minimum observations before Phase 2 activates
MAX_PHASE2_WEIGHT   = 0.60    # Phase 2 never fully replaces Phase 1
FULL_WEIGHT_SAMPLES = 100     # Samples needed to reach MAX_PHASE2_WEIGHT
DECAY_START_RATIO   = 0.60    # Decay begins after 60% of validity window
DECAY_LAMBDA        = 0.35    # Exponential decay speed after threshold
SCAN_SNAPSHOT_TTL   = 3 * 3600  # 3h — ignore stale prior scans for momentum velocity
MAX_CONFLUENCE_DELTA = 3
MAX_CONFIDENCE_DELTA = 20.0
MAX_ORDERFLOW_DELTA  = 25.0
CONFLUENCE_GROWTH_DENOM = MAX_CONFLUENCE_DELTA * 2.0
CONFIDENCE_GROWTH_DENOM = MAX_CONFIDENCE_DELTA * 2.0
ORDERFLOW_GROWTH_DENOM  = MAX_ORDERFLOW_DELTA * 2.0
BASE_MOMENTUM_WEIGHT    = 0.55
GROWTH_MOMENTUM_WEIGHT  = 0.45


# ── Beta posterior (same pattern as probability_engine.py) ────────────

@dataclass
class BetaPosterior:
    """
    Beta distribution for upgrade probability.
    alpha = upgrades observed + prior
    beta  = non-upgrades observed + prior

    Starts at alpha=2, beta=2 → mean=0.5 (neutral prior).
    Uncertainty is high until MIN_SAMPLES reached.
    """
    alpha: float = 2.0
    beta:  float = 2.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """Standard deviation of the posterior."""
        a, b = self.alpha, self.beta
        return math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))

    @property
    def sample_count(self) -> int:
        """Actual observations (excluding the 2+2 prior)."""
        return max(0, int(self.alpha + self.beta - 4))

    @property
    def is_mature(self) -> bool:
        return self.sample_count >= MIN_SAMPLES

    @property
    def phase2_weight(self) -> float:
        """
        How much Phase 2 contributes to final UP.

        Formula: min(MAX_PHASE2_WEIGHT, sample_count / FULL_WEIGHT_SAMPLES)

        This means:
          20 samples  → 20% weight  (just activated, still cautious)
          40 samples  → 40% weight  (~2-3 weeks of data)
          60 samples  → 60% weight  (full weight — ~5-6 weeks)
          100+ samples → 60% weight (capped, Phase 1 always contributes 40%)

        Deliberately slow. Early posteriors move fast — this keeps
        Phase 2 advisory until it has genuinely earned trust.
        """
        if self.sample_count < MIN_SAMPLES:
            return 0.0
        return round(min(MAX_PHASE2_WEIGHT, self.sample_count / FULL_WEIGHT_SAMPLES), 3)

    def update(self, upgraded: bool):
        if upgraded:
            self.alpha += 1.0
        else:
            self.beta  += 1.0

    def to_dict(self) -> dict:
        return {
            'alpha':         self.alpha,
            'beta':          self.beta,
            'mean':          round(self.mean, 4),
            'uncertainty':   round(self.uncertainty, 4),
            'sample_count':  self.sample_count,
            'phase2_weight': self.phase2_weight,
            'is_mature':     self.is_mature,
        }


# ── Upgrade velocity tracker ──────────────────────────────────────────

@dataclass
class VelocityTracker:
    """
    Tracks how quickly C signals upgrade in a given context.
    Fast upgrades = high-quality market environment.
    Slow/no upgrades = noise regime.

    Used as a meta-indicator for Phase 2.
    """
    upgrade_times_min: List[float] = field(default_factory=list)

    @property
    def avg_minutes(self) -> Optional[float]:
        if not self.upgrade_times_min:
            return None
        return sum(self.upgrade_times_min) / len(self.upgrade_times_min)

    @property
    def velocity_score(self) -> float:
        """
        0.0–1.0. Higher = faster upgrades = better environment.
        Based on: fast upgrade (<30 min) = 1.0, slow (>120 min) = 0.2
        """
        avg = self.avg_minutes
        if avg is None:
            return 0.5  # Unknown — neutral
        if avg < 30:
            return 1.0
        if avg > 120:
            return 0.2
        # Linear interpolation between 30 and 120 minutes
        return 1.0 - (avg - 30) / 90 * 0.8

    def record(self, minutes: float):
        self.upgrade_times_min.append(minutes)
        # Keep last 50 observations
        if len(self.upgrade_times_min) > 50:
            self.upgrade_times_min = self.upgrade_times_min[-50:]


# ── Recent scan snapshot for Phase 2 momentum velocity ─────────────────────

@dataclass
class ScanSnapshot:
    """
    Recent scored-signal snapshot for scan-to-scan momentum comparison.
    Used to replace the old static proxy with real growth/decay velocity.
    """
    confluence_count: int
    confidence: float
    orderflow_score: float
    seen_at: float = field(default_factory=time.time)


# ── Per-signal tracking record ────────────────────────────────────────

@dataclass
class UpgradeRecord:
    """Tracks one C signal from detection through resolution."""
    signal_id: int
    symbol: str
    strategy: str
    direction: str
    regime: str
    confidence_at_detection: float
    up_score: float
    phase1_up: float
    phase2_up: float
    phase2_weight: float
    sent_to_user: bool
    validity_candles: int
    created_at: float = field(default_factory=time.time)

    # Resolved fields
    upgraded: Optional[bool] = None
    upgrade_grade: Optional[str] = None
    upgrade_latency_min: Optional[float] = None
    candles_elapsed: int = 0
    expired: bool = False

    @property
    def context_key(self) -> str:
        return f"{self.strategy}|{self.regime}|{self.direction}"

    @property
    def age_minutes(self) -> float:
        return (time.time() - self.created_at) / 60


# ── Main upgrade tracker ──────────────────────────────────────────────

class UpgradeTracker:
    """
    Unified Phase 1 + Phase 2 upgrade probability engine.

    Phase 1: Rule-based, always running.
    Phase 2: Bayesian, activates per context key once MIN_SAMPLES reached.

    Both phases run concurrently. Final UP is a weighted blend.
    """

    def __init__(self):
        # Active signal tracking
        self._tracking: Dict[int, UpgradeRecord] = {}

        # Phase 2: Beta posteriors per context key
        # Key: "strategy|regime|direction"
        self._posteriors: Dict[str, BetaPosterior] = {}

        # Phase 2: Velocity tracker per regime
        self._velocity: Dict[str, VelocityTracker] = {}

        # Recent scored snapshots keyed by symbol|strategy|direction so Phase 1
        # momentum can compare confluence growth vs the previous scan instead of
        # relying only on a static orderflow/confluence proxy.
        self._recent_scans: Dict[str, ScanSnapshot] = {}

        # Accumulated stats (visible via /status and logs)
        self._stats = {
            'total_c_signals':  0,
            'sent_to_user':     0,
            'suppressed':       0,
            'upgraded':         0,
            'expired':          0,
            'phase2_active_keys': 0,
        }

        # DB reference injected after initialization
        self._db = None

    def set_db(self, db):
        """Inject DB for persistence. Called during engine startup."""
        self._db = db

    # ── Main entry point ──────────────────────────────────────────────

    def evaluate(self, scored, signal_id: int,
                 validity_candles: int = 6) -> Tuple[bool, float]:
        """
        Evaluate C signal interrupt worthiness.

        Returns:
            (send_to_user: bool, up_score: float)
        """
        sig = scored.base_signal
        direction = (direction_str(sig))
        regime = scored.regime or getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        strategy = sig.strategy or "Unknown"
        context_key = f"{strategy}|{regime}|{direction}"

        # ── Phase 1 ───────────────────────────────────────────────────
        phase1_up = self._compute_phase1(scored)

        # ── TP2 size gate ─────────────────────────────────────────────
        # Gap 5 fix: TP2 floor scales with chop_strength (tighter ranges = smaller TP2 ok)
        # but never drops below the hard execution-cost floor (MIN_TP2_PCT_CHOPPY).
        # Formula: floor = MIN_TP2_PCT - (MIN_TP2_PCT - MIN_TP2_PCT_CHOPPY) * chop_strength
        chop = regime_analyzer.chop_strength
        effective_tp2_floor = max(
            MIN_TP2_PCT_CHOPPY,
            MIN_TP2_PCT - (MIN_TP2_PCT - MIN_TP2_PCT_CHOPPY) * chop
        )
        entry_mid = (sig.entry_low + sig.entry_high) / 2
        if entry_mid > 0:
            tp2_pct = abs(sig.tp2 - entry_mid) / entry_mid * 100
        else:
            tp2_pct = 0.0

        if tp2_pct < effective_tp2_floor:
            self._log_suppressed(scored, signal_id, 0.0, 0.0, 0.0, reason="tp2_too_small")
            self._stats['total_c_signals'] += 1
            self._stats['suppressed'] += 1
            return False, 0.0

        # ── Phase 2 ───────────────────────────────────────────────────
        posterior = self._get_posterior(context_key)
        phase2_up = posterior.mean
        p2_weight = posterior.phase2_weight
        p1_weight = 1.0 - p2_weight

        # Velocity adjustment (meta-indicator)
        vel_tracker = self._velocity.get(regime)
        velocity_adj = vel_tracker.velocity_score if vel_tracker else 0.5
        # Velocity adjusts final UP slightly (±10%)
        velocity_factor = 0.9 + (velocity_adj * 0.2)

        # ── Blend ─────────────────────────────────────────────────────
        blended_up = (p1_weight * phase1_up + p2_weight * phase2_up) * velocity_factor
        final_up = round(min(blended_up, 1.0), 3)

        send = final_up >= UP_SEND_THRESHOLD

        # ── Store record ──────────────────────────────────────────────
        rec = UpgradeRecord(
            signal_id=signal_id,
            symbol=sig.symbol,
            strategy=strategy,
            direction=direction,
            regime=regime,
            confidence_at_detection=scored.final_confidence,
            up_score=final_up,
            phase1_up=phase1_up,
            phase2_up=phase2_up,
            phase2_weight=p2_weight,
            sent_to_user=send,
            validity_candles=validity_candles,
        )
        self._tracking[signal_id] = rec

        # ── Stats ─────────────────────────────────────────────────────
        self._stats['total_c_signals'] += 1
        if send:
            self._stats['sent_to_user'] += 1
        else:
            self._stats['suppressed'] += 1

        self._stats['phase2_active_keys'] = sum(
            1 for p in self._posteriors.values() if p.is_mature
        )

        # ── Log to DB for Phase 2 training ───────────────────────────
        self._log_to_db_async(rec)

        log_phase = f"P1={phase1_up:.2f} P2={phase2_up:.2f}(w={p2_weight:.2f}) vel={velocity_adj:.2f}"
        if send:
            logger.info(
                f"C→USER: {sig.symbol} {direction} UP={final_up:.2f} "
                f"[{log_phase}]"
            )
        else:
            logger.debug(
                f"C→ADMIN: {sig.symbol} {direction} UP={final_up:.2f} "
                f"[{log_phase}]"
            )

        return send, final_up

    # ── Outcome recording ─────────────────────────────────────────────

    def on_upgrade(self, signal_id: int, new_grade: str):
        """
        Called when a C signal upgrades to B/A/A+.
        Updates the Bayesian posterior for that context.
        """
        rec = self._tracking.get(signal_id)
        if not rec:
            return

        now = time.time()
        rec.upgraded = True
        rec.upgrade_grade = new_grade
        rec.upgrade_latency_min = (now - rec.created_at) / 60

        # Update Phase 2 posterior
        posterior = self._get_posterior(rec.context_key)
        posterior.update(upgraded=True)

        # Update velocity
        vel = self._velocity.setdefault(rec.regime, VelocityTracker())
        vel.record(rec.upgrade_latency_min)

        self._stats['upgraded'] += 1

        logger.info(
            f"Upgrade learned: #{signal_id} {rec.symbol} C→{new_grade} "
            f"in {rec.upgrade_latency_min:.1f}min "
            f"context={rec.context_key} "
            f"posterior={posterior.mean:.3f}(n={posterior.sample_count})"
        )

        # Persist updated posterior
        self._save_posterior_async(rec.context_key, posterior)
        self._update_db_outcome_async(signal_id, True, rec.upgrade_latency_min)

    def on_expired(self, signal_id: int):
        """Called when C signal expires without upgrading."""
        rec = self._tracking.get(signal_id)
        if not rec:
            return

        rec.expired = True
        rec.upgraded = False

        # Update Phase 2 posterior — expired = non-upgrade
        posterior = self._get_posterior(rec.context_key)
        posterior.update(upgraded=False)

        self._stats['expired'] += 1

        logger.debug(
            f"C expired: #{signal_id} {rec.symbol} "
            f"context={rec.context_key} "
            f"posterior now={posterior.mean:.3f}(n={posterior.sample_count})"
        )

        self._save_posterior_async(rec.context_key, posterior)
        self._update_db_outcome_async(signal_id, False, None)

    def tick_candle(self, signal_id: int):
        """
        Called each candle close for tracked signals.
        Applies time decay to UP score.
        Not used to resend messages — just for internal tracking.
        """
        rec = self._tracking.get(signal_id)
        if not rec:
            return
        rec.candles_elapsed += 1

        # Check if decay threshold passed
        if rec.candles_elapsed >= rec.validity_candles * DECAY_START_RATIO:
            candles_into_decay = rec.candles_elapsed - (rec.validity_candles * DECAY_START_RATIO)
            decay = math.exp(-DECAY_LAMBDA * candles_into_decay)
            rec.up_score = round(rec.up_score * decay, 3)

    # ── Phase 1: Rule-based UP ────────────────────────────────────────

    def _compute_phase1(self, scored) -> float:
        """
        Rule-based upgrade probability. Always runs.
        Returns 0.0–1.0.

        Components:
          30% — Regime compatibility with direction
          25% — Confluence momentum proxy
          25% — Confidence proximity to B threshold
          20% — Volume presence
        """
        sig = scored.base_signal
        direction = (direction_str(sig))
        regime = scored.regime or getattr(regime_analyzer.regime, 'value', 'UNKNOWN')

        regime_score  = self._score_regime(regime, direction)
        momentum      = self._score_momentum(scored)
        proximity     = self._score_proximity(scored.final_confidence)
        vol_score     = min(scored.volume_score / 100.0, 1.0)

        return round(
            0.30 * regime_score +
            0.25 * momentum     +
            0.25 * proximity    +
            0.20 * vol_score,
            3
        )

    def _score_regime(self, regime: str, direction: str) -> float:
        table = {
            ("BULL_TREND",  "LONG"):  0.90,
            ("BULL_TREND",  "SHORT"): 0.20,
            ("BEAR_TREND",  "SHORT"): 0.90,
            ("BEAR_TREND",  "LONG"):  0.20,
            ("CHOPPY",      "LONG"):  0.45,
            ("CHOPPY",      "SHORT"): 0.45,
            ("VOLATILE",       "LONG"):  0.35,
            ("VOLATILE",       "SHORT"): 0.35,
            ("VOLATILE_PANIC", "LONG"):  0.15,
            ("VOLATILE_PANIC", "SHORT"): 0.60,
        }
        return table.get((regime, direction), 0.40)

    def _score_momentum(self, scored) -> float:
        """
        Real scan-to-scan momentum velocity.

        Base signal quality still matters (orderflow + confluence density), but
        when we have a recent prior scan for the same symbol/strategy/direction
        we also score whether confluence, confidence, and orderflow are growing.
        """
        sig = scored.base_signal
        direction = (direction_str(sig))
        scan_key = f"{sig.symbol}|{sig.strategy}|{direction}"
        now = time.time()

        of_score = max(0.0, (scored.orderflow_score - 50) / 50)
        conf_count = len(getattr(scored, 'all_confluence', []))
        conf_density = min(conf_count / 5.0, 1.0)
        base_score = 0.5 * of_score + 0.5 * conf_density

        prev = self._recent_scans.get(scan_key)
        score = base_score

        if prev and now - prev.seen_at <= SCAN_SNAPSHOT_TTL:
            conf_delta = max(-MAX_CONFLUENCE_DELTA, min(MAX_CONFLUENCE_DELTA, conf_count - prev.confluence_count))
            conf_growth = 0.5 + (conf_delta / CONFLUENCE_GROWTH_DENOM)

            confidence_delta = max(
                -MAX_CONFIDENCE_DELTA,
                min(MAX_CONFIDENCE_DELTA, scored.final_confidence - prev.confidence)
            )
            confidence_growth = 0.5 + (confidence_delta / CONFIDENCE_GROWTH_DENOM)

            orderflow_delta = max(
                -MAX_ORDERFLOW_DELTA,
                min(MAX_ORDERFLOW_DELTA, scored.orderflow_score - prev.orderflow_score)
            )
            orderflow_growth = 0.5 + (orderflow_delta / ORDERFLOW_GROWTH_DENOM)

            growth_score = max(
                0.0,
                min(
                    1.0,
                    0.5 * conf_growth +
                    0.3 * confidence_growth +
                    0.2 * orderflow_growth,
                )
            )
            score = BASE_MOMENTUM_WEIGHT * base_score + GROWTH_MOMENTUM_WEIGHT * growth_score

        self._recent_scans[scan_key] = ScanSnapshot(
            confluence_count=conf_count,
            confidence=float(scored.final_confidence),
            orderflow_score=float(scored.orderflow_score),
            seen_at=now,
        )
        return round(score, 3)

    def _score_proximity(self, confidence: float) -> float:
        """How close is signal to B threshold (72)? Closer = more likely to cross."""
        b_threshold = 72.0
        return round(max(0.0, 1.0 - (b_threshold - confidence) / b_threshold), 3)

    # ── Phase 2: Bayesian posterior management ────────────────────────

    def _get_posterior(self, context_key: str) -> BetaPosterior:
        """
        Get or create Beta posterior for a context key.

        New contexts are initialized with regime-aware priors (Method 2)
        instead of a neutral 0.5 prior. This gives Phase 2 a head start
        based on known market structure. Bayesian updating corrects any
        wrong priors automatically as live data accumulates.
        """
        if context_key not in self._posteriors:
            self._posteriors[context_key] = self._make_prior(context_key)
            # Try to load from DB (overwrites prior if DB has data)
            self._load_posterior_async(context_key)
        return self._posteriors[context_key]

    # Regime-aware prior table
    # Format: (alpha, beta) → mean = alpha / (alpha + beta)
    # Using small pseudo-counts (sum ~4) so priors are weak and
    # quickly overridden by real observations.
    #
    # Rationale:
    #   Trend regimes: setups have clear direction → upgrade often (65%)
    #   Range/Chop:    setups lack momentum → upgrade rarely (35%)
    #   Volatile:      setups unstable → low upgrade rate (30%)
    #   Risk-Off:      longs suppressed, shorts more likely to confirm (25%/55%)
    #
    # These are informed estimates, not certainties.
    # With alpha+beta=4, just 5-6 real observations will start
    # shifting the posterior meaningfully.
    _REGIME_PRIORS = {
        # (regime, direction): (alpha, beta)
        ("BULL_TREND",  "LONG"):  (2.60, 1.40),  # ≈ 65%
        ("BULL_TREND",  "SHORT"): (1.20, 2.80),  # ≈ 30%
        ("BEAR_TREND",  "SHORT"): (2.60, 1.40),  # ≈ 65%
        ("BEAR_TREND",  "LONG"):  (1.20, 2.80),  # ≈ 30%
        ("CHOPPY",      "LONG"):  (1.40, 2.60),  # ≈ 35%
        ("CHOPPY",      "SHORT"): (1.40, 2.60),  # ≈ 35%
        ("VOLATILE",       "LONG"):  (1.20, 2.80),  # ≈ 30%
        ("VOLATILE",       "SHORT"): (1.20, 2.80),  # ≈ 30%
        ("VOLATILE_PANIC", "LONG"):  (1.00, 3.00),  # ≈ 25%
        ("VOLATILE_PANIC", "SHORT"): (2.20, 1.80),  # ≈ 55%
    }
    _DEFAULT_PRIOR = (2.00, 2.00)  # ≈ 50% — neutral for unknown regimes

    def _make_prior(self, context_key: str) -> BetaPosterior:
        """
        Build an informed Beta prior from context key.
        Extracts regime + direction from "strategy|regime|direction".
        """
        parts = context_key.split("|")
        regime    = parts[1] if len(parts) > 1 else ""
        direction = parts[2] if len(parts) > 2 else ""

        alpha, beta = self._REGIME_PRIORS.get(
            (regime, direction),
            self._DEFAULT_PRIOR
        )
        prior = BetaPosterior(alpha=alpha, beta=beta)

        logger.debug(
            f"New posterior initialized: {context_key} "
            f"prior={prior.mean:.2f} (alpha={alpha}, beta={beta})"
        )
        return prior

    def load_posteriors_from_db(self, rows: list):
        """
        Called at startup to restore learned posteriors from DB.
        rows: list of dicts with keys context_key, alpha, beta
        """
        for row in rows:
            key = row.get('context_key', '')
            if key:
                self._posteriors[key] = BetaPosterior(
                    alpha=row.get('alpha', 2.0),
                    beta=row.get('beta',  2.0),
                )
        mature = sum(1 for p in self._posteriors.values() if p.is_mature)
        logger.info(
            f"Upgrade posteriors loaded: {len(self._posteriors)} contexts, "
            f"{mature} mature (Phase 2 active)"
        )

    # ── Async DB helpers (fire-and-forget) ────────────────────────────

    def _safe_ensure_future(self, coro):
        """Schedule a coroutine as a fire-and-forget task with done-callback error logging."""
        import asyncio
        def _on_done(task):
            try:
                task.result()
            except Exception as e:
                logger.warning(f"Background DB task failed: {e}")
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(coro)
            task.add_done_callback(_on_done)
        except RuntimeError:
            pass  # No running loop — skip async DB write

    def _log_to_db_async(self, rec: UpgradeRecord):
        """Log new C signal detection to DB for Phase 2 training."""
        if not self._db:
            return
        async def _write():
            await self._db.log_c_signal(rec)
        self._safe_ensure_future(_write())

    def _update_db_outcome_async(self, signal_id: int,
                                  upgraded: bool,
                                  latency_min: Optional[float]):
        """Update outcome for a C signal in DB."""
        if not self._db:
            return
        async def _write():
            await self._db.update_c_signal_outcome(signal_id, upgraded, latency_min)
        self._safe_ensure_future(_write())

    def _save_posterior_async(self, context_key: str, posterior: BetaPosterior):
        """Persist posterior to DB after update."""
        if not self._db:
            return
        async def _write():
            await self._db.save_upgrade_posterior(context_key, posterior)
        self._safe_ensure_future(_write())

    def _load_posterior_async(self, context_key: str):
        """Load posterior from DB on first access."""
        if not self._db:
            return
        async def _read():
            row = await self._db.load_upgrade_posterior(context_key)
            if row:
                self._posteriors[context_key] = BetaPosterior(
                    alpha=row.get('alpha', 2.0),
                    beta=row.get('beta',  2.0),
                )
        self._safe_ensure_future(_read())

    def _log_suppressed(self, scored, signal_id, p1, p2, w, reason=""):
        """Persist a suppressed signal to DB for completeness of Phase 2 training data."""
        if not self._db:
            return
        sig = scored.base_signal
        direction = (direction_str(sig))
        regime = scored.regime or getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        rec = UpgradeRecord(
            signal_id=signal_id,
            symbol=sig.symbol,
            strategy=sig.strategy or "Unknown",
            direction=direction,
            regime=regime,
            confidence_at_detection=scored.final_confidence,
            up_score=0.0,
            phase1_up=p1,
            phase2_up=p2,
            phase2_weight=w,
            sent_to_user=False,
            validity_candles=0,
        )
        logger.debug(f"Suppressed C signal: {sig.symbol} {direction} reason={reason}")
        self._log_to_db_async(rec)

    # ── Status + diagnostics ──────────────────────────────────────────

    def get_stats(self) -> dict:
        total = self._stats['total_c_signals']
        upgraded = self._stats['upgraded']

        # Collect velocity data across all regimes
        all_times = []
        for vt in self._velocity.values():
            all_times.extend(vt.upgrade_times_min)
        avg_vel = sum(all_times) / len(all_times) if all_times else None

        return {
            'total_c_signals':     total,
            'sent_to_user':        self._stats['sent_to_user'],
            'suppressed':          self._stats['suppressed'],
            'upgraded':            upgraded,
            'expired':             self._stats['expired'],
            'upgrade_rate':        f"{upgraded/total*100:.1f}%" if total > 0 else "—",
            'suppress_rate':       f"{self._stats['suppressed']/total*100:.1f}%" if total > 0 else "—",
            'avg_upgrade_min':     f"{avg_vel:.1f}min" if avg_vel else "—",
            'phase2_active_keys':  self._stats['phase2_active_keys'],
            'total_context_keys':  len(self._posteriors),
        }

    def get_posterior_report(self) -> str:
        """Human-readable report of all learned posteriors."""
        if not self._posteriors:
            return "No Phase 2 data yet. Accumulating observations."

        lines = ["<b>📊 UPGRADE POSTERIORS (Phase 2)</b>\n"]
        mature = [(k, p) for k, p in self._posteriors.items() if p.is_mature]
        learning = [(k, p) for k, p in self._posteriors.items() if not p.is_mature]

        if mature:
            lines.append("<b>Active (learned):</b>")
            for key, p in sorted(mature, key=lambda x: x[1].mean, reverse=True):
                parts = key.split("|")
                strategy = parts[0] if len(parts) > 0 else key
                regime   = parts[1] if len(parts) > 1 else "?"
                direction = parts[2] if len(parts) > 2 else "?"
                bar = "█" * int(p.mean * 10) + "░" * (10 - int(p.mean * 10))
                # Show current credibility weight so it's clear how trusted this posterior is
                weight_pct = int(p.phase2_weight * 100)
                lines.append(
                    f"  {strategy[:12]} {direction} {regime[:8]}\n"
                    f"  UP={p.mean:.2f} {bar} n={p.sample_count} "
                    f"σ={p.uncertainty:.2f} P2={weight_pct}%"
                )

        if learning:
            lines.append(f"\n<b>Learning ({len(learning)} contexts):</b>")
            lines.append(f"<i>Need {MIN_SAMPLES} samples each before Phase 2 activates</i>")
            for key, p in learning[:5]:
                parts = key.split("|")
                strategy = parts[0] if len(parts) > 0 else key
                regime   = parts[1] if len(parts) > 1 else "?"
                direction = parts[2] if len(parts) > 2 else "?"
                lines.append(
                    f"  {strategy[:12]} {direction} {regime[:8]} "
                    f"n={p.sample_count}/{MIN_SAMPLES}"
                )
            if len(learning) > 5:
                lines.append(f"  ... and {len(learning) - 5} more")

        lines.append(
            f"\n<i>Phase 2 ramp: 20 samples=20%, 60 samples=60% (max). "
            f"Slow by design.</i>"
        )
        return "\n".join(lines)

    def get_velocity_report(self) -> str:
        """Upgrade velocity by regime — meta market indicator."""
        if not self._velocity:
            return "No velocity data yet."
        lines = ["<b>⚡ UPGRADE VELOCITY (meta-indicator)</b>\n"]
        for regime, vt in self._velocity.items():
            avg = vt.avg_minutes
            score = vt.velocity_score
            quality = "🟢 Fast" if score > 0.7 else "🟡 Moderate" if score > 0.4 else "🔴 Slow"
            lines.append(
                f"  {regime}: {quality}  "
                f"avg={avg:.0f}min  score={score:.2f}"
            )
        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────
upgrade_tracker = UpgradeTracker()
