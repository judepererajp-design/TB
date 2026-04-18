"""
TitanBot Pro — Main Engine
============================
The central async orchestrator. Runs the full scanning pipeline.

Loop structure:
  1. Regime update (every 5 min)
  2. Rotation update (every 10 min)
  3. Universe refresh (every 1 hour)
  4. For each due symbol:
     a. Fetch OHLCV for all required timeframes
     b. Run all enabled strategies
     c. Check for whale activity
     d. Run stalker scan
     e. Aggregate and score signals
     f. Validate against risk manager
     g. Publish approved signals to Telegram

Everything is async — the Mac's 4 cores handle concurrent
API fetches without blocking the main loop.
"""

import asyncio
import logging
import time
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config.loader import cfg
from config.constants import Penalties, Timing, Grading, STRATEGY_KEY_MAP, ExecutionGate as EG
from data.api_client import api
from data.database import db
from analyzers.regime import regime_analyzer
from analyzers.altcoin_rotation import rotation_tracker
from scanner.scanner import scanner, Tier
from strategies.smc import SMCStrategy
from strategies.breakout import BreakoutStrategy
from strategies.reversal import ReversalStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.price_action import PriceActionStrategy
from strategies.momentum import MomentumStrategy
from strategies.ichimoku import IchimokuStrategy
from strategies.elliott_wave import ElliottWaveStrategy
from strategies.funding_arb import FundingArbStrategy
from strategies.range_scalper import RangeScalperStrategy
from strategies.wyckoff import WyckoffStrategy
from strategies.base import SignalDirection, normalize_confidence  # FIX #3 + live-log fix
from patterns.harmonic import HarmonicDetector
from patterns.geometric import GeometricPatterns
from signals.aggregator import signal_aggregator, ScoredSignal
from signals.entry_refiner import EntryRefiner
from signals.invalidation_monitor import invalidation_monitor
from signals.signal_pipeline import evaluate_opposite_direction_conflict
from risk.manager import risk_manager
from governance.performance_tracker import performance_tracker
from tg.bot import telegram_bot
from tg.formatter import formatter
from tg.keyboards import keyboards

# ── Quant Pillars ─────────────────────────────────────────────
from core.probability_engine import probability_engine
from core.alpha_model import alpha_model
from core.portfolio_engine import portfolio_engine
from core.learning_loop import learning_loop
from core.feature_store import feature_store
from utils.formatting import fmt_price

# ── AI + Diagnostic modules ───────────────────────────────────
from analyzers.ai_analyst import ai_analyst
from analyzers.news_scraper import news_scraper
from analyzers.btc_news_intelligence import btc_news_intelligence  # NEW: BTC event classification
from analyzers.coingecko_client import coingecko
from analyzers.hypertracker_client import hypertracker
from analyzers.liquidation_analyzer import liquidation_analyzer
from analyzers.smart_money_client import smart_money
from analyzers.market_microstructure import microstructure
from signals.ensemble_voter import ensemble_voter
from core.diagnostic_engine import diagnostic_engine

# ── A+ Upgrade: New Institutional-Grade Analyzers ─────────────
from analyzers.onchain_analytics import onchain_analytics
from analyzers.stablecoin_flows import stablecoin_analyzer
from analyzers.mining_validator import mining_analyzer
from analyzers.network_activity import network_activity
from analyzers.volatility_structure import volatility_analyzer
from analyzers.wallet_behavior import wallet_profiler
from analyzers.leverage_mapper import leverage_mapper

logger = logging.getLogger(__name__)


@dataclass
class PreparedPublishCandidate:
    """A fully-vetted signal awaiting batch-level ranking/publish."""

    symbol: str
    signal: object
    scored: ScoredSignal
    sig_data: dict
    alpha_score: object
    prob_estimate: object
    sizing: object
    confluence: object
    regime_name: str

    @property
    def direction(self) -> str:
        return getattr(self.signal.direction, 'value', str(self.signal.direction))


async def _bootstrap_oi_history():
    """Background task: bootstrap historical OI data on startup."""
    try:
        from scripts.bootstrap_oi_history import bootstrap_and_load
        logger.info("📊 OI history bootstrap starting (background)...")
        ok = await bootstrap_and_load()
        if ok:
            logger.info("📊 OI history bootstrap complete — deep liquidation clusters now active")
        else:
            logger.warning("📊 OI history bootstrap returned no data")
    except Exception as e:
        logger.warning(f"OI bootstrap error (non-fatal): {e}")


async def _bootstrap_whale_wallets():
    """
    Background task: auto-fetch top Hyperliquid whale wallets on startup.

    Runs automatically so you never need to run fetch_hl_wallets.py manually.
    Logic:
      - If wallets are already configured in settings.yaml → skip (already seeded)
      - If wallets list is empty or missing → fetch from HL leaderboard now
      - Refreshes once per week (7 days) so list stays current
      - Writes results to settings.yaml and reloads the wallet watcher
    """
    await asyncio.sleep(45)  # let bot fully warm up first

    try:
        import re
        from pathlib import Path

        settings_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        if not settings_path.exists():
            return

        content = settings_path.read_text()

        # Check if wallets already seeded and recent (within 7 days)
        last_fetch_match = re.search(
            r'# Auto-fetched (\d{4}-\d{2}-\d{2})', content
        )
        if last_fetch_match:
            from datetime import datetime, timedelta
            try:
                last_date = datetime.strptime(last_fetch_match.group(1), "%Y-%m-%d")
                if datetime.now() - last_date < timedelta(days=7):
                    # Count existing wallets
                    existing = re.findall(r'hypertracker_watched_wallets:.*?(?=\n\S|\Z)',
                                         content, re.DOTALL)
                    addr_count = len(re.findall(r'- "0x', content))
                    if addr_count > 0:
                        logger.info(
                            f"👁 Whale wallets: {addr_count} already configured "
                            f"(fetched {last_fetch_match.group(1)}) — skipping refresh"
                        )
                        return
            except Exception:
                pass

        # Check if empty/unconfigured
        current_wallets = re.findall(r'- "0x[a-fA-F0-9]{40}"', content)
        if current_wallets:
            logger.info(
                f"👁 Whale wallets: {len(current_wallets)} configured — "
                f"will auto-refresh in 7 days"
            )
            return

        # Wallets are empty — fetch now
        logger.info("👁 Whale wallets: none configured, auto-fetching from HL leaderboard...")

        import aiohttp
        import time as _time

        wallets = []
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20),
            headers={"User-Agent": "TitanBot/3.0"},
        ) as session:
            # Fetch from Hyperliquid stats leaderboard (free, no auth)
            try:
                async with session.get(
                    "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
                ) as r:
                    if r.status == 200:
                        data = await r.json()
                        rows = data.get("leaderboardRows", [])
                        for row in rows[:50]:
                            addr = row.get("ethAddress", "")
                            if addr and len(addr) >= 20:
                                # Extract all-time PnL for sorting
                                pnl = 0.0
                                for perf in row.get("windowPerformances", []):
                                    if isinstance(perf, list) and len(perf) >= 2 and perf[0] == "allTime":
                                        try:
                                            pnl = float(perf[1].get("pnl", 0) or 0)
                                        except (ValueError, TypeError):
                                            pass
                                wallets.append((addr.lower(), pnl))
                        logger.info(f"👁 Fetched {len(wallets)} whale wallets from HL leaderboard")
            except Exception as e:
                logger.warning(f"👁 HL leaderboard fetch error: {e}")

        if not wallets:
            logger.warning("👁 No whale wallets fetched — will retry next restart")
            return

        # Sort by PnL, take top 50
        wallets.sort(key=lambda x: x[1], reverse=True)
        addresses = [w[0] for w in wallets[:50]]

        # Write to settings.yaml
        date_str = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')
        items = "\n".join(f'    - "{addr}"' for addr in addresses)
        wallet_yaml = f"    # Auto-fetched {date_str} — {len(addresses)} wallets\n{items}"

        # Replace existing wallet list
        new_content = re.sub(
            r'hypertracker_watched_wallets:\s*\n(?:    [^\n]*\n)*(?:    [^\n]*)?',
            f"hypertracker_watched_wallets:\n{wallet_yaml}",
            content,
            count=1,
        )
        if new_content == content:
            # Fallback line-by-line replacement
            lines = content.split("\n")
            out = []
            skip = False
            for line in lines:
                if "hypertracker_watched_wallets:" in line:
                    out.append("hypertracker_watched_wallets:")
                    out.append(wallet_yaml)
                    skip = True
                    continue
                if skip:
                    stripped = line.strip()
                    if stripped.startswith("-") or stripped.startswith("#") or stripped == "[]":
                        continue
                    else:
                        skip = False
                out.append(line)
            new_content = "\n".join(out)

        settings_path.write_text(new_content)
        # Validate the written YAML is parseable before declaring success.
        # A regex mishap (e.g. unmatched group) could corrupt the file;
        # detect early so the user can restore from backup.
        try:
            import yaml as _yaml
            _yaml.safe_load(new_content)
        except Exception as _yv_err:
            logger.error(
                f"👁 YAML validation failed after whale-wallet write: {_yv_err} — "
                f"settings.yaml may be corrupted, restoring original content"
            )
            settings_path.write_text(content)  # Restore original on validation failure
            return
        logger.info(
            f"👁 Auto-saved {len(addresses)} whale wallets to settings.yaml — "
            f"wallet watcher will pick up on next poll"
        )

    except Exception as e:
        logger.warning(f"Whale wallet auto-bootstrap error (non-fatal): {e}")

# Module-level entry refiner instance
_entry_refiner = EntryRefiner()


class Engine:
    """
    Main async engine. Single instance, runs forever until stopped.
    """

    def __init__(self):
        self._running = False
        self._strategies = []
        self._paused = False
        self._scan_semaphore = asyncio.Semaphore(
            cfg.system.get('async_workers', 3)
        )

        # Strategy class name → config key mapping (shared constant)
        self._strat_key_map = STRATEGY_KEY_MAP

        # Timing
        self._last_regime_update    = 0
        self._last_rotation_update  = 0
        self._last_daily_summary    = 0
        self._last_heartbeat        = 0
        self._last_console_status   = 0
        self._start_time            = time.time()

        # I1: Startup warmup — observation-only for first 2 minutes.
        # Gives regime, price cache, and F&G time to stabilize before firing signals.
        # FIX: was 900s (15 min) which meant zero signals for 15 minutes after startup.
        # 120s is sufficient — regime_analyzer.force_update() and htf_guardrail.warm_up()
        # already run synchronously during startup, so the data is ready.
        self._warmup_secs    = Timing.WARMUP_SECS    # 1 minute — regime data ready from startup
        self._warmup_active  = True   # Cleared once warmup_secs have elapsed
        self._warmup_end_time = 0.0   # Set when warmup completes; used by Gate 5

        # Stats
        self._scan_count   = 0
        self._signal_count = 0
        # Batch expiry queue: accumulate expired signals, flush as one summary per heartbeat
        self._pending_expired_batch: list = []   # [(symbol, direction, signal_id)]
        self._last_expiry_flush: float = 0       # timestamp of last batch flush
        self._cycle_count  = 0

        # Fix A4: strategy health monitoring — track scans since last signal per strategy
        # FIX #8: default to start_time so first health check doesn't show 492475h
        self._strategy_last_signal: Dict[str, float] = {}
        self._strategy_health_alert_interval = Timing.STRATEGY_HEALTH_ALERT_SECS  # alert if silent for 6h
        self._last_health_check = 0.0

        # FIX #1: Re-entry cooldown state (was missing — caused AttributeError on first LOSS)
        self._loss_cooldown: Dict[tuple, float] = {}
        self._reentry_cooldown_mins: int = cfg.system.get('reentry_cooldown_mins', 30)
        self._recent_symbol_direction: Dict[str, dict] = {}
        # Periodic pruning of cooldown state — these dicts are written on every
        # signal/loss but never trimmed, so on a long-running bot they grow with
        # the cumulative symbol set. Bound them with a TTL well past their
        # functional lifetime so live behaviour is unchanged.
        self._last_cooldown_prune: float = 0.0
        # Async lock guarding _loss_cooldown writes from outcome callbacks
        # (which can land on different tasks than the scan loop). Lazily
        # created on first use so __init__ does not require a running event loop.
        self._loss_cooldown_lock: Optional[asyncio.Lock] = None

    def _get_loss_cooldown_lock(self) -> asyncio.Lock:
        if self._loss_cooldown_lock is None:
            self._loss_cooldown_lock = asyncio.Lock()
        return self._loss_cooldown_lock

    def _prune_cooldown_state(self) -> None:
        """Drop _loss_cooldown / _recent_symbol_direction entries past TTL.

        TTL is the configured cooldown horizon × 4, which is far beyond any
        functional use of the entry; we keep some slack to avoid log spam if
        a stragler signal lands moments after expiry.
        """
        now = time.time()
        loss_ttl = max(60, self._reentry_cooldown_mins * 60) * 4
        recent_ttl = max(
            Timing.OPPOSITE_SIGNAL_COOLDOWN_BY_SETUP.get("intraday", 1800),
            Timing.OPPOSITE_SIGNAL_COOLDOWN_BY_SETUP.get("swing", 1800),
        ) * 4

        if self._loss_cooldown:
            stale = [k for k, ts in self._loss_cooldown.items() if now - ts > loss_ttl]
            for k in stale:
                self._loss_cooldown.pop(k, None)

        if self._recent_symbol_direction:
            stale_sym = [
                s for s, d in self._recent_symbol_direction.items()
                if now - float(d.get("ts", 0.0)) > recent_ttl
            ]
            for s in stale_sym:
                self._recent_symbol_direction.pop(s, None)

    @staticmethod
    def _rank_publish_candidates(
        candidates: List[PreparedPublishCandidate],
        regime_name: str,
    ) -> tuple[List[PreparedPublishCandidate], List[PreparedPublishCandidate]]:
        """Rank prepared publish candidates and split selected vs skipped."""
        from signals.signal_ranker import rank_publish_candidates

        return rank_publish_candidates(candidates, regime=regime_name)

    @staticmethod
    def _get_opposite_signal_cooldown_secs(setup_class: str) -> int:
        return Timing.OPPOSITE_SIGNAL_COOLDOWN_BY_SETUP.get(
            setup_class or "intraday",
            Timing.OPPOSITE_SIGNAL_COOLDOWN_BY_SETUP["intraday"],
        )

    async def _directional_stability_block_reason(self, symbol: str, signal: object) -> str:
        direction = getattr(signal.direction, 'value', str(signal.direction))
        setup_context = getattr(signal, 'setup_context', None)
        execution_context = getattr(signal, 'execution_context', None)

        for pos in getattr(portfolio_engine, "_positions", {}).values():
            if getattr(pos, "symbol", "") != symbol:
                continue
            pos_direction = getattr(pos, "direction", "")
            result = evaluate_opposite_direction_conflict(
                new_direction=direction,
                conflicting_direction=pos_direction,
                setup_context=setup_context,
                execution_context=execution_context,
                conflict_is_active=True,
            )
            if result.blocked:
                return f"{result.reason} | open position {pos_direction}"

        try:
            recent_active = await db.get_recent_signal_for_symbol(symbol)
        except Exception:
            recent_active = None
        if recent_active:
            recent_direction = str(recent_active.get("direction") or "")
            result = evaluate_opposite_direction_conflict(
                new_direction=direction,
                conflicting_direction=recent_direction,
                setup_context=setup_context,
                execution_context=execution_context,
                conflict_is_active=True,
            )
            if result.blocked:
                recent_id = recent_active.get("signal_id") or recent_active.get("id") or "?"
                return f"{result.reason} | unresolved signal #{recent_id} {recent_direction}"

        recent = self._recent_symbol_direction.get(symbol)
        if recent:
            result = evaluate_opposite_direction_conflict(
                new_direction=direction,
                conflicting_direction=recent.get("direction", ""),
                setup_context=setup_context,
                execution_context=execution_context,
                conflict_age_secs=max(0.0, time.time() - float(recent.get("ts", 0.0))),
                cooldown_secs=self._get_opposite_signal_cooldown_secs(
                    getattr(signal, "setup_class", "intraday"),
                ),
            )
            if result.blocked:
                return (
                    f"{result.reason} | recent {recent.get('direction', '')} "
                    f"[{recent.get('strategy', 'unknown')}]"
                )

        return ""

    async def _publish_prepared_candidate(
        self,
        candidate: PreparedPublishCandidate,
    ):
        """Publish a signal that already passed the full per-symbol pipeline."""
        symbol = candidate.symbol
        signal = candidate.signal
        scored = candidate.scored
        sig_data = candidate.sig_data
        alpha_score = candidate.alpha_score
        prob_estimate = candidate.prob_estimate
        sizing = candidate.sizing
        confluence = candidate.confluence
        regime_name = candidate.regime_name

        try:
            from signals.whale_aggregator import whale_aggregator
            from signals.signal_publisher import signal_publisher

            signal_id = await db.save_signal(sig_data)

            try:
                _whale_ctx = None
                _recent_w = whale_aggregator.get_recent_events(symbol=symbol, max_age_secs=300)
                if _recent_w:
                    _aligned = [w for w in _recent_w if
                        (w.side == 'buy' and getattr(signal.direction, 'value', str(signal.direction)) == 'LONG') or
                        (w.side == 'sell' and getattr(signal.direction, 'value', str(signal.direction)) == 'SHORT')]
                    if _aligned:
                        _total_usd = sum(w.order_usd for w in _aligned)
                        _whale_ctx = f"{len(_aligned)} whale orders aligned (${_total_usd/1000:.0f}k)"

                    if alpha_score.grade in ("A", "A+"):
                        try:
                            _ht_mp, _ht_sm = await hypertracker.get_primary_cohort_bias()
                            _ht_oi = await hypertracker.get_coin_intelligence(symbol)
                            _ht_intel = hypertracker.format_signal_intel(
                                _ht_mp, _ht_sm, _ht_oi, getattr(signal.direction, 'value', str(signal.direction))
                            )
                            if _ht_intel:
                                signal.raw_data['ht_intel'] = _ht_intel
                            if _ht_mp:
                                signal.raw_data['ht_mp_bias'] = _ht_mp.bias
                                signal.raw_data['ht_mp_bias_score'] = _ht_mp.bias_score
                                signal.raw_data['ht_mp_in_pos'] = _ht_mp.in_positions_pct
                            if _ht_sm:
                                signal.raw_data['ht_sm_bias'] = _ht_sm.bias
                                signal.raw_data['ht_sm_bias_score'] = _ht_sm.bias_score
                                signal.raw_data['ht_sm_in_pos'] = _ht_sm.in_positions_pct
                            if _ht_oi:
                                signal.raw_data['ht_long_pct'] = _ht_oi.long_pct
                                signal.raw_data['ht_short_pct'] = _ht_oi.short_pct
                                signal.raw_data['ht_is_crowded'] = _ht_oi.is_crowded
                                signal.raw_data['ht_liq_risk_pct'] = _ht_oi.liq_risk_pct
                        except Exception as _ht_err:
                            logger.debug(f"HyperTracker signal intel skipped: {_ht_err}")

                    pass
            except Exception as _narr_err:
                logger.debug(f"AI narrative generation skipped: {_narr_err}")

            msg_id = await signal_publisher.publish(
                scored=scored,
                signal=signal,
                signal_id=signal_id,
                sig_data=sig_data,
                alpha_score=alpha_score,
                prob_estimate=prob_estimate,
                sizing=sizing,
                confluence=confluence,
            )
            if msg_id:
                scanner.mark_signal(symbol)
                self._signal_count += 1
                self._cycle_signals_found += 1
                self._strategy_last_signal[signal.strategy.lower()] = time.time()
                self._recent_symbol_direction[symbol] = {
                    "direction": getattr(signal.direction, 'value', str(signal.direction)),
                    "strategy": signal.strategy,
                    "ts": time.time(),
                    "grade": getattr(alpha_score, "grade", ""),
                }
                if alpha_score.grade != "C":
                    risk_manager.commit_signal()
                self._last_signal_publish_time = time.time()

                try:
                    diagnostic_engine.record_signal_published()
                    diagnostic_engine.record_signal_card(
                        symbol=symbol,
                        grade=scored.grade,
                        char_count=len(
                            formatter.format_signal(scored) if hasattr(formatter, 'format_signal') else ""
                        ),
                        had_narrative=bool(signal.raw_data.get('ai_narrative')),
                        confluence_count=len(scored.all_confluence),
                        had_tp3=signal.tp3 is not None,
                        had_btc_context=True,
                    )
                except Exception:
                    pass

                try:
                    _narr = signal.raw_data.get('ai_narrative')
                    if _narr:
                        _quality = await ai_analyst.review_narrative(
                            narrative=_narr,
                            symbol=symbol,
                            direction=getattr(signal.direction, 'value', str(signal.direction)),
                            grade=scored.grade,
                        )
                        if _quality:
                            diagnostic_engine.record_narrative_quality(
                                symbol=symbol,
                                grade=scored.grade,
                                quality=_quality,
                            )
                except Exception:
                    pass

                try:
                    from signals.upgrade_tracker import upgrade_tracker as _ut_live
                    if signal_id in _ut_live._tracking:
                        _ut_live.on_upgrade(signal_id, alpha_score.grade)
                except Exception:
                    pass

                if alpha_score.grade == "C":
                    pass
                elif alpha_score.grade == "A+":
                    _volatile_regimes = ("VOLATILE", "VOLATILE_PANIC")
                    _is_risky_regime = regime_name in _volatile_regimes
                    if _is_risky_regime:
                        logger.info(
                            f"⚠️ A+ signal #{signal_id} in {regime_name} — "
                            f"downgrading to A-style execution (1 trigger required, not instant bypass)"
                        )
                        from core.execution_engine import execution_engine
                        execution_engine.track(
                            signal_id=signal_id, symbol=symbol,
                            direction=getattr(signal.direction, 'value', str(signal.direction)), strategy=signal.strategy,
                            entry_low=signal.entry_low, entry_high=signal.entry_high,
                            stop_loss=signal.stop_loss,
                            confidence=scored.final_confidence,
                            tp1=signal.tp1, tp2=signal.tp2, tp3=signal.tp3,
                            rr_ratio=signal.rr_ratio, message_id=msg_id, grade="A",
                            setup_class=getattr(signal, 'setup_class', 'intraday'),
                        )
                    else:
                        logger.info(f"⚡ A+ signal #{signal_id} — bypassing execution engine (immediate entry)")
                        try:
                            from signals.outcome_monitor import outcome_monitor
                            outcome_monitor.track_signal(
                                signal_id=signal_id, symbol=symbol,
                                direction=getattr(signal.direction, 'value', str(signal.direction)), strategy=signal.strategy,
                                entry_low=signal.entry_low, entry_high=signal.entry_high,
                                stop_loss=signal.stop_loss,
                                tp1=signal.tp1, tp2=signal.tp2, tp3=signal.tp3,
                                confidence=scored.final_confidence, message_id=msg_id,
                                raw_data=getattr(signal, 'raw_data', None),
                                trail_pct=getattr(signal, '_regime_trail_pct', 0.40),
                                regime=getattr(regime_analyzer.regime, 'value', 'UNKNOWN'),
                                setup_class=getattr(signal, 'setup_class', 'intraday'),
                            )
                        except Exception as _a_plus_err:
                            logger.error(f"A+ outcome registration failed: {_a_plus_err}")
                else:
                    if alpha_score.grade == "C":
                        logger.debug(
                            f"Signal #{signal_id} grade=C — skipping execution engine "
                            f"(informational only, no trade tracking)"
                        )
                    else:
                        from core.execution_engine import execution_engine
                        execution_engine.track(
                            signal_id=signal_id,
                            symbol=symbol,
                            direction=getattr(signal.direction, 'value', str(signal.direction)),
                            strategy=signal.strategy,
                            entry_low=signal.entry_low,
                            entry_high=signal.entry_high,
                            stop_loss=signal.stop_loss,
                            confidence=scored.final_confidence,
                            tp1=signal.tp1,
                            tp2=signal.tp2,
                            tp3=signal.tp3,
                            rr_ratio=signal.rr_ratio,
                            message_id=msg_id,
                            grade=alpha_score.grade,
                            setup_class=getattr(signal, 'setup_class', 'intraday'),
                        )
                    pass
        except Exception as e:
            logger.error(f"Signal processing error ({symbol}): {e}")

    @staticmethod
    def _log_task_exception(task: asyncio.Task):
        """Done-callback: log exceptions from fire-and-forget background tasks.

        Without this, background tasks created via create_task() that crash
        have their exception silently stored on the task object. The asyncio
        exception handler only fires on GC — which may never happen if we
        hold a reference.  This callback fires **immediately** when the task
        finishes, ensuring the error reaches errors.log.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                f"🚨 BACKGROUND TASK DIED | {task.get_name()} | "
                f"{type(exc).__name__}: {exc}",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def start(self):
        """Start everything up and begin the main loop"""
        try:
            await self._start_impl()
        except asyncio.CancelledError:
            raise  # let cancellation propagate normally
        except Exception as e:
            logger.critical(
                f"💀 ENGINE FATAL — {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise  # still propagate so main.py can handle it

    async def _start_impl(self):
        """Internal startup + main loop (wrapped by start() for crash logging)"""
        logger.info("=" * 60)
        logger.info("  TitanBot Pro Starting...")
        logger.info("=" * 60)

        # ── 1. Initialize exchange connection ─────────────────
        logger.info("Connecting to Binance Futures...")
        await api.initialize()

        # ── 2. Initialize database ────────────────────────────
        logger.info("Initializing database...")
        await db.initialize()

        # DEDUP-FIX (PERSISTENCE): Load unexpired dedup entries after DB ready.
        # Prevents re-sending signals that were published before the last restart.
        try:
            await signal_aggregator._deduplicator.load_from_db()
        except Exception as _dedup_load_err:
            logger.debug(f"Dedup DB load skipped (non-fatal): {_dedup_load_err}")

        # L3-FIX: Restore hourly signal counter from DB to prevent 2× burst on restart.
        try:
            await signal_aggregator.load_hourly_counter()
        except Exception as _hourly_load_err:
            logger.debug(f"Hourly counter load skipped (non-fatal): {_hourly_load_err}")

        # ── 3. Register strategies ────────────────────────────
        self._register_strategies()

        # ── 4. Initialize Telegram bot ────────────────────────
        logger.info("Starting Telegram bot...")
        telegram_bot.on_force_scan = self._force_scan
        telegram_bot.on_pause  = self._pause
        telegram_bot.on_resume = self._resume
        await telegram_bot.start()  # Handles initialize + _app.init + polling
        logger.info("✅ Telegram bot polling started")

        # ── 5. Performance tracker ────────────────────────────
        logger.debug("[5/12 Performance tracker DB query]")
        logger.info("⏳ [STARTUP 5/12] Performance tracker initializing...")
        await performance_tracker.initialize()
        logger.debug("[5/12 DONE]")
        logger.info("✅ [STARTUP 5/12] Performance tracker ready")

        # ── 5b-pre. Circuit breaker state restore (C3 persistence fix) ───
        # Must run after db.initialize() and before any trading activity so
        # a hard-kill or active cooldown from before the restart is reinstated.
        try:
            await risk_manager.circuit_breaker.restore()
            logger.info("✅ Circuit breaker state restored from DB")
        except Exception as _cb_restore_err:
            logger.warning(f"⚠️  Circuit breaker restore failed (non-fatal): {_cb_restore_err}")

        # ── 5a. Start shared price cache ─────────────────────
        from core.price_cache import price_cache
        price_cache.start()
        logger.info("📡 Price cache active")

        # V10: Warm up price cache before monitors start
        try:
            _all_syms = scanner.get_all_symbols() if hasattr(scanner, 'get_all_symbols') else []
            if _all_syms:
                logger.info(f"⏳ [STARTUP 5a/12] Price cache warming up {min(50, len(_all_syms))} symbols...")
                await price_cache.warm_up(_all_syms[:50])  # Top 50 symbols
                logger.info("✅ [STARTUP 5a/12] Price cache warm-up done")
            else:
                logger.info("⏳ [STARTUP 5a/12] Price cache warm-up skipped (no symbols yet)")
        except Exception as _wu_err:
            logger.debug(f"Price cache warm-up failed (non-fatal): {_wu_err}")

        # ── 5b. Start outcome monitor ──────────────────────
        from signals.outcome_monitor import outcome_monitor
        outcome_monitor.on_outcome = self._handle_signal_outcome
        outcome_monitor.on_state_change = self._handle_signal_state_change   # #10: TP1 follow-ups
        outcome_monitor.on_checkin = self._handle_trade_checkin               # hourly progress
        # Restore active trades from tracked_signals_v1 BEFORE start() so that
        # SL/TP monitoring resumes immediately with full be_stop/trail_stop/max_r.
        try:
            await outcome_monitor.restore()
        except Exception as _om_restore_err:
            logger.warning(f"OutcomeMonitor restore failed (non-fatal): {_om_restore_err}")
        outcome_monitor.start()
        logger.debug("[5b Outcome monitor]")
        logger.info("📡 Outcome monitor active")

        # ── 5c. Start invalidation monitor ─────────────────
        invalidation_monitor.on_invalidated = self._handle_signal_invalidated
        invalidation_monitor.on_entry_reached = self._handle_entry_reached
        invalidation_monitor.on_conflict = self._handle_signal_conflict
        invalidation_monitor.on_expired = self._handle_signal_expired
        invalidation_monitor.on_approaching = self._handle_signal_approaching
        invalidation_monitor.on_aging = self._handle_signal_aging
        invalidation_monitor.start()
        logger.debug("[5c Invalidation monitor]")
        logger.info("🔍 Invalidation monitor active")

        # ── 5d. Start quant pillars ────────────────────────────
        logger.debug("[5d Learning loop state]")
        logger.info("⏳ [STARTUP 5d/12] Loading learning loop state...")
        await learning_loop.load_state()
        learning_loop.start()

        # Load adaptive parameter store — must run after DB is ready
        from governance.adaptive_params import adaptive_params as _ap_store
        await _ap_store.load()
        logger.info("⚙️  Adaptive param store loaded")

        try:
            from signals.adaptive_weights import adaptive_weight_manager
            await adaptive_weight_manager.start()
            logger.info("📊 Adaptive weight manager active")
        except Exception as _awm_err:
            logger.warning(f"AdaptiveWeightManager startup failed (non-fatal): {_awm_err}")

        logger.info("🧠 Probability engine + Alpha model + Learning loop active")

        # ── 5d2. Load upgrade posteriors (Phase 2) ─────────────
        from signals.upgrade_tracker import upgrade_tracker as _ut
        from signals.signal_publisher import signal_publisher as _pub
        _ut.set_db(db)
        _pub.set_bot(telegram_bot)

        # Wire BTC news risk-event callback → Telegram alert
        # When a bearish macro/geopolitical event is detected, the user
        # gets an immediate alert listing active LONG trades at risk.
        btc_news_intelligence.on_risk_event = telegram_bot.send_news_risk_alert

        logger.debug("[5d2 Upgrade posteriors]")
        logger.info("⏳ [STARTUP 5d2/12] Loading upgrade posteriors...")
        posteriors = await db.load_all_upgrade_posteriors()
        _ut.load_posteriors_from_db(posteriors)
        mature_count = sum(1 for p in posteriors if p.get('alpha', 2) + p.get('beta', 2) > Penalties.ALPHA_BETA_MATURITY_THRESHOLD)
        logger.info(
            f"⬆️  Upgrade tracker loaded: {len(posteriors)} contexts, "
            f"{mature_count} Phase 2 active"
        )

        # ── 5e. Start network monitor ─────────────────────────
        from core.network_monitor import network_monitor
        network_monitor.on_offline = self._on_network_offline
        network_monitor.on_reconnect = self._on_network_reconnect
        network_monitor.start()

        # ── 5f. Start execution engine ────────────────────────
        from core.execution_engine import execution_engine
        execution_engine.on_stage_change = self._on_execution_stage_change
        # Restore pre-fill signals (WATCHING/ALMOST) from tracked_signals_v1 BEFORE
        # start() so trigger flags accumulated before the restart are preserved.
        try:
            await execution_engine.restore()
        except Exception as _ee_restore_err:
            logger.warning(f"ExecutionEngine restore failed (non-fatal): {_ee_restore_err}")
        execution_engine.start()
        logger.info("✅ Execution engine started")

        # ── 5i. Start AI Analyst + News Scraper + Diagnostic Engine ────
        try:
            ai_analyst.initialize()   # no key needed — uses Pollinations.ai
            _ai_mode = cfg.global_cfg.get('ai_mode', 'full')
            ai_analyst.set_mode(_ai_mode)
            logger.info(f"🤖 AI Analyst started — Pollinations.ai (no key) | mode: {_ai_mode}")
        except Exception as _ai_err:
            logger.warning(f"AI Analyst startup failed (non-fatal): {_ai_err}")

        try:
            news_scraper.start()
            logger.info("📰 News scraper started (15 RSS feeds + CoinPaprika + DeFiLlama)")
        except Exception as _ns_err:
            logger.warning(f"News scraper startup failed (non-fatal): {_ns_err}")

        # news_scraper covers all real-time RSS sources (no paid API needed)

        # R8-F2: Whale deposit monitor (exchange inflow detection)
        try:
            from analyzers.whale_deposit_monitor import whale_deposit_monitor
            await whale_deposit_monitor.start()
            logger.info("🐋 Whale deposit monitor started (blockchain.com + CryptoQuant)")
        except Exception as _wdm_err:
            logger.warning(f"Whale deposit monitor startup failed (non-fatal): {_wdm_err}")

        # R8-F9: BTC dominance tracker (altcoin rotation dynamics)
        try:
            from analyzers.btc_dominance import btc_dominance_tracker
            await btc_dominance_tracker.start()
            logger.info("📊 BTC dominance tracker started (CoinGecko + CoinPaprika)")
        except Exception as _btcd_err:
            logger.warning(f"BTC dominance tracker startup failed (non-fatal): {_btcd_err}")

        try:
            coingecko.start()
            logger.info("📈 CoinGecko trending client started (top 7 trending, 10min poll)")
        except Exception as _cg_err:
            logger.warning(f"CoinGecko startup failed (non-fatal): {_cg_err}")

        try:
            liquidation_analyzer.start()
            logger.info("📊 LiquidationAnalyzer started (7 exchanges: Binance·Bybit·OKX·Bitget·Gate·BitMEX·Hyperliquid)")
            # Bootstrap historical OI data in background (non-blocking)
            # FIX: store reference so the task is visible to the GC and can be
            # cancelled on clean shutdown. Unrooted fire-and-forget tasks are
            # silently dropped in some Python/asyncio versions under memory pressure.
            self._bootstrap_oi_task = asyncio.create_task(
                _bootstrap_oi_history(), name="bootstrap_oi_history"
            )
            self._bootstrap_oi_task.add_done_callback(self._log_task_exception)
        except Exception as _la_err:
            logger.warning(f"LiquidationAnalyzer startup failed (non-fatal): {_la_err}")

        try:
            smart_money.start()
            logger.info("🧠 SmartMoney client started (Hyperliquid top traders — free)")
        except Exception as _sm_err:
            logger.warning(f"SmartMoney startup failed (non-fatal): {_sm_err}")

        # ── Institutional Flow Engine ──────────────────────────
        try:
            from analyzers.institutional_flow import institutional_flow
            await institutional_flow.start()
            logger.info("🏦 InstitutionalFlow started (CME+Coinbase+Binance+Bybit+Macro)")
        except Exception as _if_err:
            logger.warning(f"InstitutionalFlow startup failed (non-fatal): {_if_err}")

        # ── Proactive Alert Engine ──────────────────────────────
        try:
            from signals.proactive_alerts import proactive_alerts as _pa
            _pa.start()
            logger.info("⚡ ProactiveAlerts started (regime/funding/BTC/opposing signals)")
        except Exception as _pa_err:
            logger.warning(f"ProactiveAlerts startup failed (non-fatal): {_pa_err}")

        try:
            microstructure.start()
            logger.info("🔬 MarketMicrostructure started (CVD · MaxPain · Netflow · Basis — free)")
        except Exception as _ms_err:
            logger.warning(f"MarketMicrostructure startup failed (non-fatal): {_ms_err}")

        # ── A+ Upgrade: Start new institutional-grade analyzers ───
        try:
            await onchain_analytics.start()
            logger.info("📊 OnChainAnalytics started (MVRV·SOPR·NUPL·HODL·Puell — free)")
        except Exception as _oca_err:
            logger.warning(f"OnChainAnalytics startup failed (non-fatal): {_oca_err}")

        try:
            await stablecoin_analyzer.start()
            logger.info("💵 StablecoinFlows started (supply·dominance·DefiLlama — free)")
        except Exception as _sf_err:
            logger.warning(f"StablecoinFlows startup failed (non-fatal): {_sf_err}")

        try:
            await mining_analyzer.start()
            logger.info("⛏️ MiningValidator started (HashRate·Ribbons·Fees·Revenue — free)")
        except Exception as _mv_err:
            logger.warning(f"MiningValidator startup failed (non-fatal): {_mv_err}")

        try:
            await network_activity.start()
            logger.info("🌐 NetworkActivity started (Addresses·Transactions·NVT — free)")
        except Exception as _na_err:
            logger.warning(f"NetworkActivity startup failed (non-fatal): {_na_err}")

        try:
            await volatility_analyzer.start()
            logger.info("📈 VolatilityStructure started (RV·IV·VolRegime·TermStructure — free)")
        except Exception as _vs_err:
            logger.warning(f"VolatilityStructure startup failed (non-fatal): {_vs_err}")

        # WalletBehaviorProfiler and LeverageMapper are event-driven (no background loop)
        logger.info("🐋 WalletBehaviorProfiler initialized (event-driven, fed via whale_aggregator)")
        logger.info("⚡ LeverageMapper initialized (event-driven, fed via liquidation_analyzer)")

        # Auto-bootstrap whale wallets if none are configured yet
        # FIX: store reference so task survives GC and can be cancelled on shutdown
        self._bootstrap_whale_task = asyncio.create_task(
            _bootstrap_whale_wallets(), name="bootstrap_whale_wallets"
        )
        self._bootstrap_whale_task.add_done_callback(self._log_task_exception)

        try:
            if hypertracker.initialize():
                # Wire whale position change alerts
                async def _on_whale_position_change(address, symbol, direction, size_usd, action):
                    try:
                        addr_short = f"{address[:6]}...{address[-4:]}"
                        d_emoji = "🟢" if direction == "LONG" else "🔴"
                        act_emoji = "📈" if action == "opened" else "📤"
                        size_str = f"${size_usd/1e6:.2f}M" if size_usd >= 1e6 else f"${size_usd:,.0f}"
                        text = (
                            f"🐳 <b>Tracked Whale — Position {action.title()}</b>\n\n"
                            f"{d_emoji} {symbol} {direction}  ·  {size_str}\n"
                            f"Wallet: <code>{addr_short}</code>\n"
                            f"{act_emoji} via HyperTracker"
                        )
                        await telegram_bot.send_signals_text(text=text)
                    except Exception:
                        pass
                hypertracker.on_whale_position_change = _on_whale_position_change
                hypertracker.start()
        except Exception as _ht_err:
            logger.warning(f"HyperTracker startup failed (non-fatal): {_ht_err}")

        # FIX-5: Load today's signal count so daily limit survives restart
        try:
            from risk.manager import risk_manager as _rm
            await _rm.load_daily_count()
        except Exception as _rc_err:
            logger.debug(f"Risk manager daily count load skipped: {_rc_err}")

        # D1-FIX: Restore last saved capital so sizing survives restarts.
        # Without this, _capital resets to config account_balance (e.g. $5 000)
        # on every restart, making position sizes wrong for the rest of the session.
        try:
            await portfolio_engine.load_capital()
        except Exception as _cap_err:
            logger.debug(f"Capital restore skipped (non-fatal): {_cap_err}")

        # R5-S3 FIX: Initialize circuit breaker peak capital at startup so the
        # peak-equity drawdown kill-switch (T3-FIX) is active from session start.
        # Previously _peak_capital stayed 0.0 until the first LOSS trade, meaning
        # the intraday peak protection was completely disabled during winning streaks.
        try:
            risk_manager.circuit_breaker.update_peak_capital(portfolio_engine._capital)
            logger.info(
                f"🛡️  Circuit breaker peak capital initialized: "
                f"${portfolio_engine._capital:,.2f}"
            )
        except Exception as _pk_err:
            logger.debug(f"Peak capital init skipped: {_pk_err}")

        # V2.10: Initialize veto system
        try:
            from analyzers.veto_system import veto_system as _vs
            _vs.initialize()   # no key needed — uses Pollinations.ai
            logger.info("⚖️  Veto system started (Pollinations.ai, no key)")
        except Exception as _vs_err:
            logger.warning(f"Veto system startup failed (non-fatal): {_vs_err}")

        try:
            from web.app import dashboard
            logger.info("⏳ [STARTUP 8a/12] Starting web dashboard on port 8080...")
            await dashboard.start(port=8080)
            logger.info("✅ [STARTUP 8a/12] Web dashboard started")
        except Exception as _web_err:
            logger.warning(f"Web dashboard startup failed (non-fatal): {_web_err}")

        try:
            # Wire diagnostic engine callbacks
            async def _diag_send_report(text: str):
                try:
                    await telegram_bot.send_admin_text(text=text)
                except Exception:
                    pass

            async def _diag_send_approval(approval):
                try:
                    await telegram_bot.send_approval_request(approval)
                except Exception:
                    pass

            diagnostic_engine.on_send_report = _diag_send_report
            diagnostic_engine.on_send_approval = _diag_send_approval
            diagnostic_engine.start()
            logger.info("🔬 Diagnostic engine started (self-healing observer)")

            # ── Health monitor + asyncio exception handler ─────
            from core.health_monitor import health_monitor, setup_asyncio_exception_handler
            setup_asyncio_exception_handler()
            health_monitor.start()
            logger.info("🔍 Health monitor started — silent failures now logged")
        except Exception as _de_err:
            logger.warning(f"Diagnostic engine startup failed (non-fatal): {_de_err}")

        # ── 5g. PHASE 1 FIX (P1-A): HTF guardrail warm-up ──────────────────────────
        # Must run AFTER exchange connection but BEFORE first scan.
        # Populates the weekly ADX cache so the hard block path works correctly
        # from scan #1. Without this, ADX=0.0 at cold-start bypasses the hard block.
        try:
            logger.info("⏳ [STARTUP 9/12] HTF guardrail warming up (BTC weekly + 4h OHLCV)...")
            from analyzers.htf_guardrail import htf_guardrail as _htf_warmup
            await _htf_warmup.warm_up()
            logger.info("✅ [STARTUP 9/12] HTF guardrail warm-up complete")
        except Exception as _htf_err:
            logger.warning(f"HTF guardrail warm-up failed (non-fatal): {_htf_err}")

        # ── 5h. PHASE 1 FIX (NO-RESTART): Recover open positions from exchange ──
        # If the bot crashed while a trade was open, re-register it with
        # portfolio_engine and outcome_monitor so SL/TP monitoring resumes.
        # NOTE: outcome_monitor.restore() (step 5b) already loads signals from
        # tracked_signals_v1 with full be_stop/trail_stop/max_r.  This block is
        # kept as a fallback for legacy open_positions rows that predate v6 and
        # are therefore not yet in tracked_signals_v1.
        try:
            from data.database import db as _db_ref
            _open_db_positions = await _db_ref.load_open_positions()
            if _open_db_positions:
                logger.info(f"🔄 Recovering {len(_open_db_positions)} open position(s) from DB...")
                from signals.outcome_monitor import outcome_monitor as _om_ref
                for _p in _open_db_positions:
                    try:
                        await portfolio_engine.open_position(
                            symbol=_p['symbol'], direction=_p['direction'],
                            strategy=_p['strategy'], entry_price=_p['entry_price'],
                            size_usdt=_p['size_usdt'], risk_usdt=_p['risk_usdt'],
                            stop_loss=_p['stop_loss'], sector=_p.get('sector',''),
                            signal_id=_p['signal_id'],
                            funding_rate_8h=float(_p.get('funding_rate', 0) or 0),
                        )
                        # Only register with outcome_monitor if not already restored
                        # by outcome_monitor.restore() from tracked_signals_v1.
                        if _p['signal_id'] not in _om_ref._active:
                            _om_ref.track_signal(
                                signal_id=_p['signal_id'], symbol=_p['symbol'],
                                direction=_p['direction'], strategy=_p['strategy'],
                                entry_low=_p['entry_price'] * 0.999,
                                entry_high=_p['entry_price'] * 1.001,
                                stop_loss=_p['stop_loss'],
                                tp1=_p.get('tp1', _p['entry_price']),
                                tp2=_p.get('tp2', _p['entry_price']),
                                tp3=_p.get('tp3'),
                                confidence=80,
                                message_id=_p.get('message_id'),
                                created_at=_p.get('opened_at'),
                            )
                        # FIX: Also restore execution_engine tracking so exec_state
                        # is correctly set to EXECUTE for recovered active positions
                        try:
                            from core.execution_engine import execution_engine as _ee_ref
                            if _p['signal_id'] not in _ee_ref._tracked:
                                _ee_ref.track(
                                    signal_id=_p['signal_id'],
                                    symbol=_p['symbol'],
                                    direction=_p['direction'],
                                    strategy=_p['strategy'],
                                    entry_low=_p['entry_price'] * 0.999,
                                    entry_high=_p['entry_price'] * 1.001,
                                    stop_loss=_p['stop_loss'],
                                    tp1=_p.get('tp1', _p['entry_price']),
                                    tp2=_p.get('tp2', _p['entry_price']),
                                    rr_ratio=_p.get('rr_ratio', 1.3),
                                    confidence=80,
                                    grade='B',
                                    message_id=_p.get('message_id'),
                                )
                        except Exception as _ee_err:
                            logger.debug(f"execution_engine re-track failed for #{_p['signal_id']}: {_ee_err}")
                        logger.info(f"  ✅ Recovered: {_p['symbol']} {_p['direction']} (signal #{_p['signal_id']})")
                    except Exception as _rpe:
                        logger.warning(f"  ⚠️ Failed to recover {_p['symbol']}: {_rpe}")
            else:
                logger.info("  No open positions to recover from DB")
        except Exception as _rec_err:
            logger.warning(f"Position recovery failed (non-fatal): {_rec_err}")

        # ── 6. Initial regime + rotation ─────────────────────
        logger.debug("[10/12 Market regime]")
        logger.info("⏳ [STARTUP 10/12] Analyzing market regime (BTC OHLCV fetch)...")
        await regime_analyzer.force_update()
        await rotation_tracker.force_update()
        logger.info(f"✅ [STARTUP 10/12] Regime: {getattr(regime_analyzer.regime, 'value', 'UNKNOWN')}")

        # ── 7. Build universe ─────────────────────────────────
        logger.debug("[11/12 Building universe]")
        logger.info("⏳ [STARTUP 11/12] Building symbol universe (fetching all tickers)...")
        await scanner.build_universe()
        logger.info(f"✅ [STARTUP 11/12] Universe built: {len(scanner.get_all_symbols())} symbols")

        # ── 8. Log startup to DB ──────────────────────────────
        all_symbols = scanner.get_all_symbols()
        regime_name = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
        try:
            await db.log_event("STARTUP", "TitanBot Pro started", {
                'version': cfg.global_cfg.get('version', '1.0.0'),
                'regime': regime_name,
                'symbols': len(all_symbols),
            })
        except Exception as e:
            logger.warning(f"DB startup log failed: {e}")

        # ── 9. Startup summary (rich console output) ──────────
        # FIX: Use self._strat_key_map (already complete) instead of a local dict.
        # The old local dict was missing 'HarmonicPattern' -> 'harmonic' and
        # 'GeometricPattern' -> 'geometric', causing both to fall back to
        # .lower() which produces keys not in settings.yaml → showed as ⛔ disabled.
        try:
            enabled_strats = []
            disabled_strats = []
            for s in self._strategies:
                config_key = self._strat_key_map.get(s.name, s.name.lower())
                if cfg.is_strategy_enabled(config_key):
                    enabled_strats.append(s.name)
                else:
                    disabled_strats.append(s.name)

            logger.info("")
            logger.info("┌─────────────────────────────────────────────────┐")
            logger.info("│         ⚡ TITANBOT PRO — ALL SYSTEMS GO         │")
            logger.info("├─────────────────────────────────────────────────┤")
            logger.info(f"│  Regime     : {regime_name:<35}│")
            logger.info(f"│  Chop       : {regime_analyzer.chop_strength:<35}│")
            logger.info(f"│  Symbols    : {len(all_symbols):<35}│")
            logger.info(f"│  Strategies : {len(enabled_strats):<35}│")
            base_min = cfg.aggregator.get('min_confidence', 78)
            adaptive_min = regime_analyzer.get_adaptive_min_confidence(base_min)
            logger.info(f"│  Min Conf.  : {adaptive_min} (base {base_min}, adaptive){'':<11}│")
            from strategies.base import cfg_min_rr
            current_min_rr = cfg_min_rr()
            logger.info(f"│  Min R:R    : {current_min_rr} (regime-adaptive){'':<17}│")
            logger.info(f"│  Max Sig/Hr : {signal_aggregator._max_per_hour if hasattr(signal_aggregator, '_max_per_hour') else cfg.aggregator.get('max_signals_per_hour', 12):<35}│")
            logger.info("├─────────────────────────────────────────────────┤")
            for s in enabled_strats:
                logger.info(f"│  ✅ {s:<44}│")
            for s in disabled_strats:
                logger.info(f"│  ⛔ {s:<44}│")
            logger.info("└─────────────────────────────────────────────────┘")
            logger.info("")
            logger.info(f"🔍 Scanning {len(all_symbols)} symbols across 3 tiers...")
            logger.info(f"   T1: {cfg.system.get('tier1_interval', 120)}s | "
                         f"T2: {cfg.system.get('tier2_interval', 300)}s | "
                         f"T3: {cfg.system.get('tier3_interval', 900)}s")
            logger.info("")
        except Exception as e:
            logger.error(f"Startup summary error: {e}", exc_info=True)
            enabled_strats = [s.name for s in self._strategies]

        # ── 10. Telegram startup messages ─────────────────────
        try:
            # User-facing startup card — uses real universe count from scanner
            if cfg.telegram.get("send_startup_message", True):
                await telegram_bot._send_startup_message(num_symbols=len(all_symbols))
            # Engine notice → admin channel only (suppressed if no separate admin channel)
            await self._send_welcome_telegram(regime_name, len(all_symbols), enabled_strats)
        except Exception as e:
            logger.warning(f"Telegram startup messages failed: {e}")

        self._running = True
        self._last_heartbeat = time.time()
        self._last_console_status = time.time()
        self._cycle_count = 0
        self._cycle_signals_found = 0
        self._cycle_symbols_scanned = 0

        # ── 11. Startup Health Diagnostics ─────────────────────
        try:
            logger.info("┌─────────────── SYSTEM HEALTH ────────────────┐")
            _checks = {
                'Exchange':      api._exchange_impl is not None,
                'Database':      db._conn is not None,
                'Telegram':      telegram_bot.has_signal_channel(),
                'Regime':        regime_analyzer.regime is not None,
                'Rotation':      len(rotation_tracker._sectors) > 0,
                'Scanner':       len(scanner.get_all_symbols()) > 0,
                'Strategies':    len(self._strategies) > 0,
                'Learning Loop': learning_loop._running,
            }
            all_ok = True
            for name, ok in _checks.items():
                status = "✅" if ok else "❌"
                if not ok:
                    all_ok = False
                logger.info(f"│  {status} {name:<20}│")
            logger.info(f"│  {'🟢 ALL SYSTEMS GO' if all_ok else '🔴 DEGRADED MODE':<27}│")
            logger.info("└──────────────────────────────────────────────┘")
        except Exception as e:
            logger.error(f"Health check failed, continuing startup: {e}", exc_info=True)
        logger.info("")
        logger.info("━" * 60)
        logger.info("  🚀 ENTERING MAIN SCAN LOOP")
        logger.info("  Live status prints every 10s.")
        logger.info("  Queue status prints every 30s.")
        logger.info("  Heartbeat prints every 60s.")
        logger.info("  Per-symbol output: 🔍 scanning → ✅ data → 🎯 signals")
        logger.info("━" * 60)
        logger.info("")

        # ── 12. Start main loop ────────────────────────────────
        await self._main_loop()

    async def _live_console_status(self):
        """Verbose live status — prints every 10s so you always see activity"""
        last_scan_count = 0
        while self._running:
            try:
                await asyncio.sleep(10)
                uptime = time.time() - self._start_time
                mins = int(uptime // 60)
                secs = int(uptime % 60)

                regime_name = regime_analyzer.regime.value if regime_analyzer.regime else "UNKNOWN"
                chop = regime_analyzer.chop_strength
                new_scans = self._scan_count - last_scan_count
                last_scan_count = self._scan_count

                from signals.outcome_monitor import outcome_monitor
                tracking = outcome_monitor.get_active_count()
                pending = invalidation_monitor.pending_count

                # Scan rate
                rate = new_scans / 10.0

                # Status icon
                if self._paused:
                    icon = "⏸"
                elif new_scans == 0:
                    icon = "⏳"
                else:
                    icon = "🔍"

                logger.info(
                    f"{icon} [{mins}m{secs:02d}s] "
                    f"regime={regime_name}(chop={chop:.2f}) | "
                    f"scans={self._scan_count}(+{new_scans} @{rate:.1f}/s) | "
                    f"signals={self._signal_count} | "
                    f"tracking={tracking} | pending={pending}"
                )

            except Exception as e:
                logger.error(f"Live status error: {e}")
                await asyncio.sleep(10)

    async def _dashboard_loop(self):
        """Periodic scan queue status — shows what's queued up next"""
        while self._running:
            try:
                await asyncio.sleep(30)

                due = scanner.get_due_symbols()
                all_syms = scanner.get_all_symbols()

                regime_name = regime_analyzer.regime.value if regime_analyzer.regime else "UNKNOWN"

                from core.execution_engine import execution_engine
                exec_summary = execution_engine.get_status_summary()

                from signals.whale_aggregator import whale_aggregator
                w = whale_aggregator.get_session_stats()

                logger.info(
                    f"📋 QUEUE | {len(due)} symbols due now / {len(all_syms)} total | "
                    f"regime={regime_name} | exec=[{exec_summary}] | "
                    f"whales={w['total_events']}(buy ${w['total_buy_volume']/1e6:.1f}M "
                    f"sell ${w['total_sell_volume']/1e6:.1f}M)"
                )

                if due:
                    preview = ', '.join(due[:8])
                    more = f' +{len(due)-8} more' if len(due) > 8 else ''
                    logger.info(f"   ↳ Next up: {preview}{more}")

            except Exception as e:
                logger.error(f"Dashboard loop error: {e}")
                await asyncio.sleep(30)


    async def stop(self):
        """Clean shutdown"""
        logger.info("Shutting down TitanBot Pro...")
        self._running = False

        # FIX: cancel stored background tasks before stopping monitors.
        # Previously these infinite-loop tasks were just abandoned — they'd keep
        # running and logging until the process died, sometimes raising after
        # other resources were already closed.
        for _task_attr in ('_console_status_task', '_dashboard_task',
                           '_bootstrap_oi_task', '_bootstrap_whale_task'):
            _t = getattr(self, _task_attr, None)
            if _t and not _t.done():
                _t.cancel()
                try:
                    await _t
                except (asyncio.CancelledError, Exception):
                    pass

        # Stop shared price cache first
        try:
            from core.price_cache import price_cache
            await price_cache.stop()
        except Exception as _e:
            logger.warning(f"Shutdown: price_cache.stop() failed: {_e}")
        # Stop outcome monitor
        try:
            from signals.outcome_monitor import outcome_monitor
            await outcome_monitor.stop()
        except Exception as _e:
            logger.warning(f"Shutdown: outcome_monitor.stop() failed: {_e}")
        # Stop invalidation monitor
        try:
            await invalidation_monitor.stop()
        except Exception as _e:
            logger.warning(f"Shutdown: invalidation_monitor.stop() failed: {_e}")
        # Stop learning loop
        try:
            await learning_loop.stop()
        except Exception as _e:
            logger.warning(f"Shutdown: learning_loop.stop() failed: {_e}")
        # Stop adaptive weight manager
        try:
            from signals.adaptive_weights import adaptive_weight_manager
            await adaptive_weight_manager.stop()
        except Exception as _e:
            logger.warning(f"Shutdown: adaptive_weight_manager.stop() failed: {_e}")
        # Stop execution engine
        try:
            from core.execution_engine import execution_engine
            await execution_engine.stop()
        except Exception as _e:
            logger.warning(f"Shutdown: execution_engine.stop() failed: {_e}")
        # Stop network monitor
        try:
            from core.network_monitor import network_monitor
            await network_monitor.stop()
        except Exception as _e:
            logger.warning(f"Shutdown: network_monitor.stop() failed: {_e}")
        try:
            await telegram_bot.stop()
        except Exception as _e:
            logger.warning(f"Shutdown: telegram_bot.stop() failed: {_e}")
        try:
            await api.close()
        except Exception as _e:
            logger.warning(f"Shutdown: api.close() failed: {_e}")
        try:
            await db.close()
        except Exception as _e:
            logger.warning(f"Shutdown: db.close() failed: {_e}")
        logger.info("Shutdown complete.")

    # ── Main loop ─────────────────────────────────────────────

    async def _main_loop(self):
        """The main scanning loop — runs forever with rich observability"""

        # Intervals
        HEARTBEAT_INTERVAL = 300       # Console heartbeat every 5 min
        TG_HEARTBEAT_INTERVAL = 3600   # Telegram heartbeat every 1 hour
        CONSOLE_STATUS_INTERVAL = 60   # Quick console status every 60s

        self._last_tg_heartbeat = time.time()

        logger.info("🟢 Scan loop started — watching markets...")
        # FIX: store task references so they can be cancelled on clean shutdown
        # and exceptions are visible. Unrooted tasks with infinite loops are
        # particularly dangerous — if they raise, the exception is silently dropped.
        self._console_status_task = asyncio.create_task(
            self._live_console_status(), name="live_console_status"
        )
        self._console_status_task.add_done_callback(self._log_task_exception)
        self._dashboard_task = asyncio.create_task(
            self._dashboard_loop(), name="dashboard_loop"
        )
        self._dashboard_task.add_done_callback(self._log_task_exception)



        while self._running:
            try:
                now = time.time()
                self._cycle_count += 1

                # ── Periodic regime refresh ────────────────────
                # FIX: store task reference so we never spawn a second update
                # while the previous one is still awaiting the exchange.
                regime_interval = cfg.system.get('regime_interval', 300)
                if now - self._last_regime_update >= regime_interval:
                    if not getattr(self, '_regime_task', None) or self._regime_task.done():
                        self._regime_task = asyncio.create_task(
                            self._update_regime(), name="update_regime"
                        )
                        self._regime_task.add_done_callback(self._log_task_exception)
                    self._last_regime_update = now

                # ── Periodic rotation refresh ──────────────────
                rotation_interval = cfg.system.get('rotation_interval', 600)
                if now - self._last_rotation_update >= rotation_interval:
                    if not getattr(self, '_rotation_task', None) or self._rotation_task.done():
                        self._rotation_task = asyncio.create_task(
                            self._update_rotation(), name="update_rotation"
                        )
                        self._rotation_task.add_done_callback(self._log_task_exception)
                    self._last_rotation_update = now

                # ── Universe refresh ───────────────────────────
                await scanner.build_universe()  # Internal TTL check

                # ── Periodic cooldown-state prune ──────────────
                # Bounds _loss_cooldown / _recent_symbol_direction memory growth.
                # Cheap (just dict iteration) so safe to run every cycle, but
                # actual mutation only happens for entries past TTL.
                if now - self._last_cooldown_prune > 600:  # at most every 10 min
                    self._prune_cooldown_state()
                    self._last_cooldown_prune = now

                # ── Daily summary ──────────────────────────────
                await self._check_daily_summary()
                await self._check_strategy_health()  # Fix A4

                # ── BTC News Intelligence: price feed + context decay ──
                # Feed current BTC price so the move analyzer can detect sharp moves,
                # and decay expired event contexts (e.g. a 4h MACRO_RISK_OFF that's now stale).
                try:
                    from core.price_cache import price_cache
                    _btc_px = price_cache.get("BTC/USDT")
                    if _btc_px:
                        btc_news_intelligence.record_btc_price(_btc_px)
                    btc_news_intelligence.decay_context()
                except Exception:
                    pass

                # ── Periodic leverage mapper feed ──────────────
                # Wire liquidation_analyzer data into leverage_mapper
                # so it can track cascade risks and pressure scores.
                try:
                    _lm_interval = 300  # 5 min, same as OI refresh
                    _lm_last = getattr(self, '_last_leverage_mapper_update', 0)
                    if now - _lm_last >= _lm_interval:
                        _btc_ci = liquidation_analyzer.get("BTC")
                        if _btc_ci and not _btc_ci.is_stale:
                            from analyzers.coingecko_client import coingecko
                            _btc_mcap = 0.0
                            try:
                                _cg_data = coingecko.get_cached("bitcoin")
                                if _cg_data:
                                    _btc_mcap = _cg_data.get("market_cap", 0) or 0
                            except Exception:
                                pass
                            leverage_mapper.update(
                                oi_usd=_btc_ci.total_oi_usd,
                                market_cap=_btc_mcap,
                                funding_rate=_btc_ci.funding_rate,
                                oi_change_pct=_btc_ci.oi_change_1h,
                                long_liq_clusters=[
                                    {"price": p, "usd": u}
                                    for p, u in (_btc_ci.liq_below or [])
                                ],
                                short_liq_clusters=[
                                    {"price": p, "usd": u}
                                    for p, u in (_btc_ci.liq_above or [])
                                ],
                                current_price=_btc_ci.current_price,
                            )
                        self._last_leverage_mapper_update = now
                except Exception as _lm_err:
                    logger.debug("LeverageMapper feed failed: %s", _lm_err)
                if now - self._last_console_status >= CONSOLE_STATUS_INTERVAL:
                    uptime = now - self._start_time
                    uptime_str = self._format_uptime(uptime)
                    regime_name = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
                    pending_inv = invalidation_monitor.pending_count
                    from signals.outcome_monitor import outcome_monitor
                    active_oc = outcome_monitor.get_active_count()

                    logger.info(
                        f"💓 [{uptime_str}] "
                        f"regime={regime_name} chop={regime_analyzer.chop_strength:.2f} | "
                        f"scans={self._scan_count} | "
                        f"signals={self._signal_count} | "
                        f"pending={pending_inv} | "
                        f"tracking={active_oc} | "
                        f"cycle={self._cycle_count}"
                    )
                    self._last_console_status = now

                    # Telegram quiet-bot heartbeat — fires if no signal published in 90+ min
                    # Tells you the bot is alive and the reason for silence
                    try:
                        _last_pub = getattr(self, '_last_signal_publish_time', 0)
                        _quiet_mins = (now - _last_pub) / 60 if _last_pub > 0 else 0
                        _sent_quiet_at = getattr(self, '_quiet_heartbeat_sent_at', 0)
                        _quiet_threshold = 90  # minutes
                        if _quiet_mins >= _quiet_threshold and (now - _sent_quiet_at) > _quiet_threshold * 60:
                            from tg.bot import telegram_bot
                            # Count active dedup blocks
                            _da = signal_aggregator._deduplicator
                            _dedup_active = sum(
                                1 for k, t in _da._recent.items()
                                if now - t < _da._effective_window(k)
                            )
                            _block_parts = []
                            if _dedup_active > 0: _block_parts.append(f"dedup: {_dedup_active} symbols")
                            _daily_syms = sum(
                                1 for sym, cnt in signal_aggregator._daily_count.items() if cnt > 0
                            )
                            if _daily_syms > 0: _block_parts.append(f"daily limit: {_daily_syms} symbols")
                            _msg = (
                                f"🔇 <b>Bot quiet for {_quiet_mins:.0f} min</b> — scanning normally\n"
                                f"Regime: <b>{regime_name}</b> | F&G: {regime_analyzer.fear_greed} | "
                                f"Chop: {regime_analyzer.chop_strength:.2f}\n"
                                f"Scans: {self._scan_count:,} | Total signals: {self._signal_count}\n"
                                + (f"Active filters: {', '.join(_block_parts)}" if _block_parts
                                   else "Filters clear — waiting for clean setups to emerge")
                            )
                            await telegram_bot.send_message(_msg)
                            self._quiet_heartbeat_sent_at = now
                            logger.info(f"📡 Sent quiet-bot heartbeat ({_quiet_mins:.0f} min silence)")
                    except Exception as _qhb_err:
                        logger.debug(f"Quiet heartbeat failed (non-fatal): {_qhb_err}")

                # ── Circuit breaker tick — fires on_cleared callback after auto-resume ──
                # BUG-2 FIX: tick() was defined but never called, so the on_cleared
                # callback (Telegram "trading resumed" notification) was silently dropped
                # every time the CB auto-cleared after cooldown. Call it once per loop
                # iteration — it is a no-op unless _auto_clear_scheduled is True.
                await risk_manager.circuit_breaker.tick()

                # ── Console full heartbeat (every 5 min) ───────
                if now - self._last_heartbeat >= HEARTBEAT_INTERVAL:
                    await self._console_heartbeat()
                    self._last_heartbeat = now

                # ── Telegram heartbeat (every 1 hour) ──────────
                if now - self._last_tg_heartbeat >= TG_HEARTBEAT_INTERVAL:
                    await self._telegram_heartbeat()
                    self._last_tg_heartbeat = now

                # ── Whale aggregator flush ─────────────────────
                from signals.whale_aggregator import whale_aggregator
                if whale_aggregator.should_flush():
                    summary = await whale_aggregator.flush()
                    if summary:
                        try:
                            await telegram_bot.send_signals_text(text=summary)
                        except Exception as e:
                            logger.debug(f"Whale summary send failed: {e}")

                # ── Scan due symbols ───────────────────────────
                from core.network_monitor import network_monitor
                # I1: During warmup period, scan but don't publish signals
                if self._warmup_active:
                    elapsed = time.time() - self._start_time
                    if elapsed >= self._warmup_secs:
                        self._warmup_active = False
                        self._warmup_end_time = time.time()  # Gate 5 reference point
                        logger.info(
                            f"✅ Warmup complete ({self._warmup_secs//60}min) — "
                            f"signal publishing now ACTIVE"
                        )
                    else:
                        remaining = self._warmup_secs - elapsed
                        logger.debug(
                            f"⏳ Warmup: {remaining:.0f}s remaining — "
                            f"scanning but not publishing"
                        )

                if not self._paused and network_monitor.is_online:
                    due_symbols = scanner.get_due_symbols()

                    if due_symbols:
                        batch_size = cfg.system.get('batch_size', 15)
                        batch = due_symbols[:batch_size]
                        batch_start = time.time()

                        logger.info(
                            f"📡 Scanning {len(batch)} symbols "
                            f"({', '.join(batch[:4])}{'...' if len(batch) > 4 else ''})"
                        )

                        tasks = [
                            self._scan_symbol(symbol, return_publish_candidate=True)
                            for symbol in batch
                        ]
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        candidates = [
                            result for result in results
                            if isinstance(result, PreparedPublishCandidate)
                        ]
                        errors = [r for r in results if isinstance(r, Exception)]

                        if candidates:
                            try:
                                regime_name = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
                                ranked_candidates, skipped_candidates = self._rank_publish_candidates(
                                    candidates,
                                    regime_name=regime_name,
                                )
                            except Exception as _rank_err:
                                logger.error(f"Batch signal ranking failed: {_rank_err}")
                                ranked_candidates, skipped_candidates = candidates, []

                            for skipped in skipped_candidates:
                                signal_aggregator.unmark_signal(skipped.symbol, skipped.direction)
                                logger.info(
                                    f"⏭ Skipped by batch ranking | {skipped.symbol} {skipped.direction} "
                                    f"[{skipped.signal.strategy}]"
                                )

                            for candidate in ranked_candidates:
                                await self._publish_prepared_candidate(candidate)

                        batch_time = time.time() - batch_start

                        logger.info(
                            f"⚡ Batch done: {len(batch)} symbols in {batch_time:.1f}s "
                            f"({len(batch)/batch_time:.1f} sym/sec)"
                            f"{f' | {len(errors)} errors' if errors else ''}"
                        )

                # Short sleep to prevent CPU spinning
                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                await telegram_bot.send_error_alert(str(e)[:200], recovering=True)
                await asyncio.sleep(10)

    async def _scan_symbol(self, symbol: str, return_publish_candidate: bool = False):
        """Full scan pipeline for a single symbol"""
        async with self._scan_semaphore:
            _scan_t0 = time.time()
            try:
                # Mark as scanned immediately
                scanner.mark_scanned(symbol)
                self._scan_count += 1
                self._cycle_symbols_scanned += 1  # Pass 7 FIX: was never incremented
                diagnostic_engine.record_scan()

                logger.info(f"🔍 Scanning {symbol}...")

                # ── OHLCV cooldown gate ────────────────────────
                # Skip symbols that recently failed data-quality checks
                if scanner.is_ohlcv_cooled_down(symbol):
                    logger.debug(f"   ⏳ {symbol}: OHLCV cooldown active — skipping")
                    return

                # ── Fetch OHLCV for all timeframes ─────────────
                timeframes_needed = self._get_required_timeframes()
                ohlcv_dict = {}

                fetch_tasks = [
                    api.fetch_ohlcv(symbol, tf, limit=200)
                    for tf in timeframes_needed
                ]
                # Timeout protection: prevent frozen workers if Binance stalls
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*fetch_tasks, return_exceptions=True),
                        timeout=15.0  # 15s max for all timeframes
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ {symbol}: OHLCV fetch timed out (15s)")
                    return

                _any_quality_fail = False
                for tf, result in zip(timeframes_needed, results):
                    if isinstance(result, Exception):
                        import ccxt
                        if isinstance(result, (ccxt.BadSymbol, ccxt.ExchangeError)):
                            # E3: Symbol may be delisted — exclude from future scans
                            logger.warning(
                                f"   ⚠️  {symbol}: exchange error ({result}) — "
                                f"symbol may be delisted, excluding from universe"
                            )
                            scanner.exclude_symbol(symbol)
                            return
                    elif isinstance(result, list) and result:
                        # Validate OHLCV quality before using it
                        try:
                            from core.health_monitor import health_monitor as _hm_ohlcv
                            _ok, _problems = _hm_ohlcv.validate_ohlcv(symbol, tf, result)
                            if not _ok:
                                _any_quality_fail = True
                        except Exception:
                            pass
                        ohlcv_dict[tf] = result

                if not ohlcv_dict:
                    logger.info(f"   ⚠️  {symbol}: no OHLCV data available — skipping")
                    # Record the failure for cooldown tracking
                    scanner.record_ohlcv_fail(symbol)
                    return

                # Track data-quality failures/successes for cooldown
                if _any_quality_fail:
                    scanner.record_ohlcv_fail(symbol)
                else:
                    scanner.record_ohlcv_success(symbol)

                logger.info(
                    f"   ✅ {symbol}: data ready [{', '.join(f'{tf}' for tf in ohlcv_dict)}]"
                )

                # ── Stalker scan ──────────────────────────────
                primary_tf = '1h' if '1h' in ohlcv_dict else list(ohlcv_dict.keys())[0]
                stalker_result = await scanner.stalker_scan(symbol, ohlcv_dict[primary_tf])

                if stalker_result:
                    # Only send to Telegram for high-quality setups.
                    # Cooldown is enforced in scanner.stalker_scan so this
                    # fires at most once per hour per symbol.
                    if stalker_result['score'] >= Penalties.STALKER_QUALITY_THRESHOLD:
                        await telegram_bot.send_watchlist_alert(
                            symbol,
                            stalker_result['score'],
                            stalker_result['reasons']
                        )
                    else:
                        logger.info(
                            f"👀 Stalker: {symbol} score={stalker_result['score']:.0f} "
                            f"({', '.join(stalker_result['reasons'][:2])})"
                        )

                # ── Whale detection ────────────────────────────
                try:
                    order_book = await asyncio.wait_for(
                        api.fetch_order_book(symbol, limit=20), timeout=8.0
                    )
                except asyncio.TimeoutError:
                    order_book = None
                if order_book:
                    whale = await scanner.check_whale_activity(symbol, order_book)
                    if whale:
                        # Buffer whale event — aggregator handles Telegram summary
                        from signals.whale_aggregator import whale_aggregator
                        whale_aggregator.add_event(whale)
                        # V2.09: Track whale flow for structural audit
                        try:
                            ai_analyst.record_whale_event(whale.get('side','buy'), whale.get('order_usd', 0.0))
                        except Exception:
                            pass

                # ── Volume-based tier promotion ───────────────
                if '1h' in ohlcv_dict:
                    bars = ohlcv_dict['1h']
                    # Estimate 24h volume in USDT
                    vol_24h_usdt = sum(float(b[5]) * float(b[4]) for b in bars[-24:])

                    promo = await scanner.check_promotions(symbol, vol_24h_usdt)
                    if promo:
                        from_tier, to_tier = promo
                        logger.info(
                            f"📊 Tier promotion: {symbol} T{from_tier}→T{to_tier} "
                            f"(vol: ${vol_24h_usdt/1e6:.1f}M)"
                        )

                # ── Circuit breaker check ─────────────────────
                if risk_manager.circuit_breaker.is_active:
                    return

                # ── Regime gate ───────────────────────────────
                # VOLATILE: restrict to whitelist strategies only (SMC, Wyckoff, FundingRateArb).
                # Previously a hard `return` silenced the bot entirely — but REGIME_THRESHOLDS
                # already defines a whitelist for exactly this scenario. Use it.
                _risk_off_whitelist = None
                _current_regime_val = regime_analyzer.regime.value
                if _current_regime_val in ('VOLATILE', 'VOLATILE_PANIC'):
                    from signals.regime_thresholds import REGIME_THRESHOLDS
                    _regime_cfg = REGIME_THRESHOLDS.get(_current_regime_val, {})
                    _risk_off_whitelist = _regime_cfg.get('strategies')
                    # strategies=None means ALL strategies allowed in this regime (normal VOLATILE)
                    # Only VOLATILE_PANIC has an explicit whitelist restriction
                    if _current_regime_val == 'VOLATILE_PANIC' and not _risk_off_whitelist:
                        logger.info(f"⚠️ VOLATILE_PANIC with empty whitelist — skipping {symbol}")
                        return
                    # VOLATILE with strategies=None: allow all — no skip

                # ── Pre-warm feature store + clear indicator cache ────
                # Compute features once per symbol/timeframe now so that:
                #   1. feature_store.compute() is already cached when probability
                #      engine calls it later (line ~1440)
                #   2. BaseStrategy.calculate_* caches are cleared so prior-symbol
                #      values never bleed into this symbol's strategies
                from strategies.base import BaseStrategy as _BS
                _BS.clear_indicator_cache()
                for _tf, _candles in ohlcv_dict.items():
                    if _candles:
                        feature_store.compute(symbol, _tf, _candles)

                # ── Run all enabled strategies (collect ALL signals) ──
                # FIX: Skip strategy dispatch entirely if symbol already at daily limit
                # Saves CPU on symbols that can't publish anyway (e.g. DOT/XRP hitting 3/day)
                from signals.aggregator import signal_aggregator as _agg_ref
                _sym_times = _agg_ref._daily_count_times.get(symbol, [])
                _sym_times = [t for t in _sym_times if t > time.time() - 86400]
                _max_sym = getattr(_agg_ref._agg_cfg, 'max_signals_per_symbol', 3) if hasattr(_agg_ref, '_agg_cfg') else 3
                if len(_sym_times) >= _max_sym:
                    logger.debug(
                        f"⏭ Skipping {symbol} strategy dispatch — daily symbol limit "
                        f"{len(_sym_times)}/{_max_sym} already reached"
                    )
                    return  # exits _scan_symbol() early — no strategies run for this symbol today

                all_signals = []
                for strategy in self._strategies:
                    try:
                        if not telegram_bot.is_strategy_enabled(self._strat_key_map.get(strategy.name, strategy.name.lower())):
                            continue
                        if performance_tracker.is_strategy_disabled(strategy.name):
                            continue
                        # VOLATILE whitelist gate: only permitted strategies run
                        if _risk_off_whitelist and strategy.name not in _risk_off_whitelist:
                            continue

                        signal = await strategy.analyze(symbol, ohlcv_dict)
                        if signal:
                            state = scanner._symbols.get(symbol)
                            signal.tier = state.tier.value if state else 2
                            if getattr(signal, 'raw_data', None) is None:
                                signal.raw_data = {}
                            signal.raw_data.setdefault('signal_created_ts', time.time())
                            # FIX #25: stamp regime_time so staleness check works
                            signal.regime_time = self._last_regime_update if self._last_regime_update > 0 else time.time()
                            all_signals.append(signal)
                            # BUG-10 FIX: record raw signal generation time (not just publish time).
                            # Health alert was firing "silent for 12h" even when strategies were
                            # generating dozens of signals per hour that died in filters. Now the
                            # alert only fires if a strategy fails to generate any raw signal at all
                            # (scan error, config issue) rather than "generates but all filtered".
                            _strat_key = self._strat_key_map.get(strategy.name, strategy.name.lower())
                            self._strategy_last_signal[_strat_key] = time.time()
                            # V2.09: Track per-strategy direction counts for structural audit
                            try:
                                ai_analyst.record_strategy_signal(signal.strategy, getattr(signal.direction, 'value', str(signal.direction)))
                            except Exception:
                                pass
                    except Exception as e:
                        logger.error(f"Strategy error ({strategy.name}, {symbol}): {e}")
                        continue

                if not all_signals:
                    logger.info(f"   ➖ {symbol}: no signals from any strategy")
                    return  # FIX: was falling through to momentum check and cooldown with empty list

                # ── Phase-1 Gap 3: Normalize confidence across strategies ─────
                # Different strategies produce confidence on different native
                # scales (e.g. RangeScalper caps at 88, SMC at 95).  Normalize
                # to a common [40, 95] range so downstream gates treat them
                # equally.
                for _ns in all_signals:
                    _ns.confidence = normalize_confidence(_ns.strategy, _ns.confidence)

                # V13 BUG 11: Short-term momentum sanity check
                # Penalize signals that fire against the last 3 candles' direction
                # This catches "catching a falling knife" scenarios
                if all_signals:
                    try:
                        _stm_tf = '15m' if '15m' in ohlcv_dict else ('1h' if '1h' in ohlcv_dict else None)
                        if _stm_tf and len(ohlcv_dict[_stm_tf]) >= 4:
                            _stm_bars = ohlcv_dict[_stm_tf]
                            _stm_close_now = float(_stm_bars[-1][4])
                            _stm_close_3ago = float(_stm_bars[-4][4])
                            _stm_pct = (_stm_close_now - _stm_close_3ago) / _stm_close_3ago
                            for _stm_sig in all_signals:
                                # LONG signal but last 3 bars falling > 1%
                                if getattr(_stm_sig.direction, 'value', str(_stm_sig.direction)) == "LONG" and _stm_pct < -0.01:
                                    _stm_sig.confidence -= 8
                                    _stm_sig.confluence.append(
                                        f"⚠️ Counter-momentum: price falling {_stm_pct:.1%} last 3 bars"
                                    )
                                # SHORT signal but last 3 bars rising > 1%
                                elif getattr(_stm_sig.direction, 'value', str(_stm_sig.direction)) == "SHORT" and _stm_pct > 0.01:
                                    _stm_sig.confidence -= 8
                                    _stm_sig.confluence.append(
                                        f"⚠️ Counter-momentum: price rising {_stm_pct:+.1%} last 3 bars"
                                    )
                    except Exception:
                        pass

                # FIX #2 / BUG-A FIX: Re-entry cooldown check.
                # After a loss on a symbol+direction, block same-direction signals
                # for cooldown period. Prevents 4x-size re-entry 24s after loss close.
                #
                # BUG-A: The original code had an unconditional `return` AFTER the
                # for-loop which caused the entire scan pipeline to be skipped for
                # every symbol once _loss_cooldown was non-empty (i.e. after ANY loss).
                # Fix: remove cooldown-blocked signals individually; only exit if ALL
                # signals are blocked. Unaffected signals continue through the pipeline.
                if all_signals and self._loss_cooldown:
                    _now_cd = time.time()
                    _blocked = []
                    for _pre_check_sig in all_signals:
                        _cd_key = (symbol, getattr(_pre_check_sig.direction, 'value', str(_pre_check_sig.direction)))
                        _last_loss = self._loss_cooldown.get(_cd_key, 0)
                        if _last_loss > 0 and _now_cd - _last_loss < self._reentry_cooldown_mins * 60:
                            _mins_left = int((_last_loss + self._reentry_cooldown_mins * 60 - _now_cd) / 60)
                            logger.info(
                                f"{symbol} {getattr(_pre_check_sig.direction, 'value', str(_pre_check_sig.direction))}"
                                f"| cooldown {_mins_left}min remaining after last loss"
                            )
                            _blocked.append(_pre_check_sig)
                    for _bs in _blocked:
                        all_signals.remove(_bs)
                    if not all_signals:
                        return  # All directions in cooldown — skip symbol entirely

                logger.info(
                    f"   🎯 {symbol}: {len(all_signals)} raw signal(s) → "
                    f"{', '.join(s.strategy + '(' + s.direction.value + ')' for s in all_signals)}"
                )

                # ── PRE-GATE: Counter-regime confidence penalty ──────────────────────────
                # Before is_hard_blocked runs, apply a 0.70× multiplier to signals that
                # are counter to the weekly HTF trend. This ensures strategies with raw
                # confidence just above the HTF threshold (e.g. Wyckoff LONG at 84 vs
                # threshold 87) are properly penalised before the check.
                # Evidence: KITE Wyckoff LONG published despite HTF BEARISH because
                # 84 > 83.8 (0.2pt margin). With 0.70×: 84→58.8, safely below 87.
                try:
                    from analyzers.htf_guardrail import htf_guardrail as _ctr_htf
                    for _ctr_sig in all_signals:
                        _is_counter_dir = (
                            (getattr(_ctr_sig.direction, 'value', str(_ctr_sig.direction)) == "LONG" and _ctr_htf._weekly_bias == "BEARISH") or
                            (getattr(_ctr_sig.direction, 'value', str(_ctr_sig.direction)) == "SHORT" and _ctr_htf._weekly_bias == "BULLISH")
                        )
                        if _is_counter_dir and _ctr_htf._weekly_adx >= Penalties.ADX_COUNTER_TREND_MIN:
                            _ctr_sig.confidence *= Penalties.COUNTER_TREND_CONF_MULT
                            logger.debug(
                                f"Counter-regime penalty ×0.80 | {symbol} "
                                f"{getattr(_ctr_sig.direction, 'value', str(_ctr_sig.direction))} [{_ctr_sig.strategy}] "
                                f"weekly={_ctr_htf._weekly_bias} ADX={_ctr_htf._weekly_adx:.0f}"
                            )
                except Exception as _ctr_err:
                    logger.debug(f"Counter-regime penalty skip: {_ctr_err}")

                # ── PHASE 1 FIX (P1-B): HTF pre-aggregation hard gate ─────────────────────
                # Run this BEFORE the aggregator so counter-trend signals are blocked
                # before they can accumulate confidence from other strategies.
                # is_hard_blocked() uses cached state — synchronous, no await needed.
                try:
                    from analyzers.htf_guardrail import htf_guardrail as _htf_pre
                    for _pre_sig in all_signals[:]:  # iterate copy to allow removal
                        _blocked, _block_reason = _htf_pre.is_hard_blocked(
                            signal_direction=getattr(_pre_sig.direction, 'value', str(_pre_sig.direction)),
                            raw_confidence=_pre_sig.confidence,
                            strategy_name=_pre_sig.strategy,
                        )
                        if _blocked:
                            all_signals.remove(_pre_sig)
                            logger.info(
                                f"🚫 Pre-agg HTF block | {symbol} {getattr(_pre_sig.direction, 'value', str(_pre_sig.direction))} "
                                f"[{_pre_sig.strategy}] | {_block_reason}"
                            )
                            # V2.09: Track HTF block direction for structural audit
                            try:
                                ai_analyst.record_htf_block(getattr(_pre_sig.direction, 'value', str(_pre_sig.direction)))
                            except Exception:
                                pass
                    if not all_signals:
                        logger.info(f"   ➖ {symbol}: all signals blocked by HTF pre-gate")
                        return
                except Exception as _htf_pre_err:
                    logger.debug(f"HTF pre-gate error (skipped): {_htf_pre_err}")

                # ── Confluence scoring (cross-strategy agreement) ──
                from signals.confluence import confluence_scorer
                try:
                    from signals.whale_aggregator import whale_aggregator
                    _whale_dominant_side = whale_aggregator.get_recent_dominant_side(
                        symbol=symbol,
                        max_age_secs=300,
                    )
                except Exception:
                    _whale_dominant_side = ""
                confluence = confluence_scorer.score(
                    all_signals,
                    whale_dominant_side=_whale_dominant_side,
                )
                if not confluence.should_publish or not confluence.best_signal:
                    return

                signal = confluence.best_signal

                # ══════════════════════════════════════════════════
                # SOFT PENALTY PIPELINE (was: hard rejection gates)
                # Each filter applies a confidence penalty instead
                # of vetoing. Only final min_confidence gate rejects.
                # This produces more trades + same quality.
                # ══════════════════════════════════════════════════

                death_reasons = []  # Track why confidence drops
                _raw_confidence = signal.confidence  # Preserve for HTF guardrail (Bug O fix)

                # ── Time filter (soft) ────────────────────────────
                from signals.time_filter import time_filter
                # Fix W4: use "B" as pre-score default; real grade enforced again after scoring below
                grade_pre = "A+" if signal.confidence > Grading.PREGRADE_A_PLUS_THRESHOLD else ("A" if signal.confidence > Grading.PREGRADE_A_THRESHOLD else "B")
                time_blocked_pre, time_adj, time_reason = time_filter.evaluate(grade_pre)
                if time_blocked_pre:
                    # Bug O fix: halve the time penalty for trend-aligned signals.
                    # A SHORT in a bearish weekly trend is the RIGHT trade — penalising
                    # it equally to a counter-trend signal is overly conservative.
                    from analyzers.htf_guardrail import htf_guardrail as _htf_ref
                    _is_aligned = (
                        (getattr(signal.direction, 'value', str(signal.direction)) == "SHORT" and _htf_ref._weekly_bias == "BEARISH")
                        or (getattr(signal.direction, 'value', str(signal.direction)) == "LONG" and _htf_ref._weekly_bias == "BULLISH")
                    )
                    _time_pen = Penalties.TIME_PENALTY_ALIGNED_PTS if _is_aligned else Penalties.TIME_PENALTY_COUNTER_PTS
                    signal.confidence -= _time_pen
                    death_reasons.append(f"time_penalty(-{_time_pen}): {time_reason}")
                    logger.info(f"   ⏰ {symbol}: time pre-penalty -{_time_pen} ({time_reason})")
                else:
                    signal.confidence = max(30, signal.confidence + time_adj)
                    # Sunday A-grade: mark for 0.6x position size reduction.
                    # The penalty already reduces confidence; size reduction is the
                    # second lever so risk exposure stays equivalent to a normal session.
                    if "Sunday early" in time_reason and "0.6x size" in time_reason:
                        signal._sunday_size_scale = 0.6
                        logger.info(f"   ⏰ {symbol}: Sunday A-grade — 0.6x size flag set")
                signal.confluence.append(time_reason)

                # ── Regime threshold filter (soft) ─────────────────
                from signals.regime_thresholds import apply_regime_filter, is_in_range_outer_zone
                from analyzers.htf_guardrail import htf_guardrail as _htf_regime_ref
                regime_name = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
                chop_str = regime_analyzer.chop_strength
                # FIX: initialise adj_conf before the try block. If apply_regime_filter raises
                # and is swallowed by the except, the subsequent _regime_alignment_real line
                # references adj_conf before assignment → silent NameError kills the scan.
                adj_conf = signal.confidence
                regime_blocked = False
                regime_reason = "regime_filter_skipped"
                # Deadlock fix: don't penalise the direction that is aligned with the
                # weekly HTF trend. If weekly=BEARISH and signal=SHORT, the regime's
                # short_bias penalty would fight the HTF — skip it.
                _htf_aligned = (
                    (getattr(signal.direction, 'value', str(signal.direction)) == "SHORT" and _htf_regime_ref._weekly_bias == "BEARISH")
                    or (getattr(signal.direction, 'value', str(signal.direction)) == "LONG"  and _htf_regime_ref._weekly_bias == "BULLISH")
                )
                try:
                    adj_conf, regime_blocked, regime_reason = apply_regime_filter(
                        signal.confidence, getattr(signal.direction, 'value', str(signal.direction)),
                        signal.strategy, regime_name,
                        chop_strength=chop_str,
                        htf_aligned=_htf_aligned,
                    )
                except Exception as _rf_err:
                    logger.debug(f"apply_regime_filter raised (using fallback values): {_rf_err}")
                if regime_blocked:
                    # R7-B1: Distinguish strategy hard-blocks from confidence penalties.
                    # Strategy hard-block (VOLATILE_PANIC allowlist) should kill the signal
                    # entirely — penalizing to conf=30 lets blocked strategies leak through.
                    if "not allowed in" in regime_reason:
                        logger.info(f"   🚫 {symbol}: strategy blocked by regime ({regime_reason})")
                        return  # Hard-kill: strategy not permitted in this regime
                    # Confidence below floor: penalize but keep alive
                    signal.confidence = max(30, adj_conf)
                    death_reasons.append(f"regime_penalty: {regime_reason}")
                    logger.info(f"   📊 {symbol}: regime penalty ({regime_reason})")
                else:
                    signal.confidence = adj_conf
                signal.confluence.append(regime_reason)

                # R7-B2: Enforce counter_long_max in bear trends.
                # regime_thresholds defines counter_long_max=2 but it was never enforced.
                _dir_str = getattr(signal.direction, 'value', str(signal.direction))
                try:
                    from signals.regime_thresholds import get_regime_threshold
                    _rt = get_regime_threshold(regime_name)
                    _counter_max = _rt.get("counter_long_max")
                except Exception:
                    _counter_max = None
                if _counter_max is not None and _dir_str == "LONG":
                    _current_longs = sum(
                        1 for p in portfolio_engine._positions.values()
                        if getattr(p, 'direction', '') == 'LONG'
                    )
                    if _current_longs >= _counter_max:
                        logger.info(
                            f"   🚫 {symbol}: counter-trend LONG cap reached "
                            f"({_current_longs}/{_counter_max} in {regime_name})"
                        )
                        return  # Hard-block: too many counter-trend longs

                # FIX (P2-B): Enforce counter_short_max in bull trends.
                # Mirror of counter_long_max — controls SHORT portfolio exposure
                # when intraday regime is BULL_TREND. Shorts are allowed (GPT:
                # "fine IF controlled") but capped to prevent the 65% SHORT
                # flooding seen in the April 6 audit.
                try:
                    _counter_short_max = _rt.get("counter_short_max")
                except Exception:
                    _counter_short_max = None
                if _counter_short_max is not None and _dir_str == "SHORT":
                    _current_shorts = sum(
                        1 for p in portfolio_engine._positions.values()
                        if getattr(p, 'direction', '') == 'SHORT'
                    )
                    if _current_shorts >= _counter_short_max:
                        logger.info(
                            f"   🚫 {symbol}: counter-trend SHORT cap reached "
                            f"({_current_shorts}/{_counter_short_max} in {regime_name})"
                        )
                        return  # Hard-block: too many counter-trend shorts

                # ── Chop range zone filter (soft) ──────────────────
                if chop_str >= Penalties.CHOP_RANGE_THRESHOLD:
                    entry_mid = (signal.entry_low + signal.entry_high) / 2
                    in_outer, zone = is_in_range_outer_zone(
                        entry_mid,
                        regime_analyzer.range_high,
                        regime_analyzer.range_low,
                    )
                    if not in_outer:
                        # Soft: penalty for mid-range trading during chop
                        signal.confidence -= Penalties.CHOP_MID_RANGE_PENALTY_PTS
                        death_reasons.append(f"chop_mid_range(-{Penalties.CHOP_MID_RANGE_PENALTY_PTS}): {zone}")
                        logger.debug(
                            f"Chop zone penalty {symbol}: -{Penalties.CHOP_MID_RANGE_PENALTY_PTS} (price in {zone})"
                        )
                    elif zone != "no_range":
                        signal.confluence.append(
                            f"📐 Trading {zone} (chop={chop_str:.2f})"
                        )

                # ── Sector rotation weight ─────────────────────────
                sector_weight, sector_reason = rotation_tracker.get_sector_weight(
                    symbol, getattr(signal.direction, 'value', str(signal.direction))
                )
                if sector_weight != 1.0:
                    signal.confidence *= sector_weight
                    if sector_reason:
                        signal.confluence.append(sector_reason)
                    logger.info(f"   🔄 {symbol}: sector weight {sector_weight:.2f}x — {sector_reason}")

                # ── Equilibrium zone enforcement (soft) ────────────
                # BUG-6 FIX: Pass this symbol's own recent range so eq_analyzer
                # compares the entry price against the symbol's own H/L, not BTC's.
                from analyzers.equilibrium import eq_analyzer
                entry_mid = (signal.entry_low + signal.entry_high) / 2
                _sym_bars = ohlcv_dict.get('1h') or ohlcv_dict.get('4h') or []
                _sym_range_high = float(max(b[2] for b in _sym_bars[-50:])) if len(_sym_bars) >= 20 else 0.0
                _sym_range_low  = float(min(b[3] for b in _sym_bars[-50:])) if len(_sym_bars) >= 20 else 0.0
                eq_result = eq_analyzer.assess(
                    entry_mid,
                    getattr(signal.direction, 'value', str(signal.direction)),
                    symbol_range_high=_sym_range_high,
                    symbol_range_low=_sym_range_low,
                )

                if eq_result.should_block:
                    # Fix W3: hard block when in wrong zone — LONG at premium, SHORT at discount
                    logger.info(
                        f"⚖️  Signal died | {symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                        f"| reason=EQ_ZONE_BLOCK | {eq_result.reason}"
                    )
                    return
                elif eq_result.confidence_mult != 1.0:
                    signal.confidence *= eq_result.confidence_mult
                    signal.confluence.append(eq_result.reason)

                if eq_result.zone in ('premium', 'discount'):
                    signal.confluence.append(
                        f"📐 Zone: {eq_result.zone.upper()} "
                        f"(depth: {eq_result.zone_depth:.0%}) | "
                        f"EQ: {fmt_price(eq_result.equilibrium)}"
                    )

                # ── Whale flow confirmation ────────────────────────
                # If whales are buying this symbol and we're going LONG → boost
                # If whales are selling and we're going SHORT → boost
                from signals.whale_aggregator import whale_aggregator
                whale_stats = whale_aggregator.get_session_stats()
                # Track regime vs signal direction for structural audit.
                # FIX: was `scored.regime` which causes UnboundLocalError — scored
                # is not assigned until signal_aggregator.process() ~400 lines later.
                # Use regime_name which is set at the top of _scan_symbol.
                try:
                    ai_analyst.record_regime_signal(regime_name, getattr(signal.direction, 'value', str(signal.direction)))
                except Exception:
                    pass
                # BUG-16 FIX: Use public get_recent_events() instead of accessing
                # private _buffer directly. Direct _buffer access is fragile — any
                # internal refactor of WhaleAggregator silently breaks this path.
                recent_whales = whale_aggregator.get_recent_events(symbol=symbol, max_age_secs=300)
                if recent_whales:
                    aligned = sum(
                        1 for w in recent_whales
                        if (w.side == 'buy' and getattr(signal.direction, 'value', str(signal.direction)) == 'LONG') or
                           (w.side == 'sell' and getattr(signal.direction, 'value', str(signal.direction)) == 'SHORT')
                    )
                    opposed = len(recent_whales) - aligned
                    if aligned > opposed:
                        whale_boost = min(1.12, 1.0 + 0.04 * aligned)
                        signal.confidence *= whale_boost
                        total_usd = sum(w.order_usd for w in recent_whales if 
                            (w.side == 'buy' and getattr(signal.direction, 'value', str(signal.direction)) == 'LONG') or
                            (w.side == 'sell' and getattr(signal.direction, 'value', str(signal.direction)) == 'SHORT'))
                        signal.confluence.append(
                            f"🐋 Whale flow aligned: {aligned} orders (${total_usd/1000:.0f}k)"
                        )
                    elif opposed > aligned and opposed >= 2:
                        signal.confidence *= Penalties.WHALE_OPPOSED_CONF_MULT  # Penalize trading against whales
                        signal.confluence.append(
                            f"⚠️ Whale flow opposed: {opposed} counter-orders"
                        )
                        # V2.09: Track whale conflict for structural audit
                        try:
                            ai_analyst.record_signal_vs_whale_conflict()
                        except Exception:
                            pass

                # ── Multi-Exchange OI + Liquidation Intelligence ───────────
                # Adjusts confidence based on aggregate OI trend, crowding,
                # and funding rates across 7 exchanges. Also suggests TP
                # based on nearest liquidation cluster.
                try:
                    _entry_mid_liq = (signal.entry_low + signal.entry_high) / 2
                    _liq_delta, _liq_note, _liq_tp = liquidation_analyzer.get_signal_intel(
                        symbol, getattr(signal.direction, 'value', str(signal.direction)), _entry_mid_liq
                    )
                    if _liq_delta != 0:
                        signal.confidence = max(10, signal.confidence + _liq_delta)
                        if _liq_note:
                            signal.confluence.append(_liq_note)
                    # Use liq cluster as TP — update signal.tp1 if cluster is
                    # a more precise target between entry and existing TP
                    if _liq_tp and _liq_tp > 0:
                        if getattr(signal.direction, 'value', str(signal.direction)) == "LONG":
                            # Cluster must be above entry, below or near current tp1
                            if _entry_mid_liq < _liq_tp < signal.tp1 * 1.3:
                                old_tp1 = signal.tp1
                                signal.tp1 = _liq_tp
                                signal.confluence.append(
                                    f"🎯 TP adjusted to liq cluster: {_liq_tp:.4f} "
                                    f"(was {old_tp1:.4f})"
                                )
                        else:  # SHORT
                            # Cluster must be below entry, above or near current tp1
                            if signal.tp1 * 0.85 < _liq_tp < _entry_mid_liq:
                                old_tp1 = signal.tp1
                                signal.tp1 = _liq_tp
                                signal.confluence.append(
                                    f"🎯 TP adjusted to liq cluster: {_liq_tp:.4f} "
                                    f"(was {old_tp1:.4f})"
                                )
                        # C-3 fix: recompute rr_ratio whenever tp1 was changed.
                        # rr is measured to tp2 (primary target); use tp1 as fallback
                        # if tp2 is not set. Also re-check the R:R floor so the
                        # aggregator geometry gate sees the correct number.
                        try:
                            _tp_for_rr = getattr(signal, 'tp2', None) or signal.tp1
                            _entry_mid_rr = (signal.entry_low + signal.entry_high) / 2
                            _risk_rr = abs(_entry_mid_rr - signal.stop_loss)
                            if _risk_rr > 0:
                                _new_rr = abs(_tp_for_rr - _entry_mid_rr) / _risk_rr
                                signal.rr_ratio = round(_new_rr, 3)
                        except Exception:
                            pass  # Non-fatal — keep existing rr_ratio if recompute fails
                except Exception:
                    pass

                # ── Smart Money (Hyperliquid top traders + institutional flow) ──
                # Layer 1: Hyperliquid top trader wallet positions
                try:
                    _sm_delta, _sm_note = smart_money.get_signal_intel(
                        symbol, getattr(signal.direction, 'value', str(signal.direction))
                    )
                    if _sm_delta != 0:
                        signal.confidence = max(10, signal.confidence + _sm_delta)
                        if _sm_note:
                            signal.confluence.append(_sm_note)
                except Exception:
                    pass

                # Layer 2: Institutional flow (CME·Coinbase·Binance·Bybit + macro)
                try:
                    from analyzers.institutional_flow import institutional_flow
                    _if_delta, _if_note = institutional_flow.get_signal_intel(
                        symbol, getattr(signal.direction, 'value', str(signal.direction))
                    )
                    if _if_delta != 0:
                        signal.confidence = max(10, signal.confidence + _if_delta)
                        if _if_note:
                            signal.confluence.append(_if_note)

                    # Macro score check: strong risk-off suppresses LONGs
                    _macro_score = institutional_flow.get_macro_score()
                    _is_long = getattr(signal.direction, 'value', str(signal.direction)) == "LONG"
                    if _macro_score < -40 and _is_long:
                        signal.confidence = max(10, signal.confidence - 8)
                        signal.confluence.append(
                            f"⚠️ Macro risk-off ({_macro_score:+.0f}): LONG headwind")
                    elif _macro_score > 40 and not _is_long:
                        signal.confidence = max(10, signal.confidence - 5)
                        signal.confluence.append(
                            f"⚠️ Macro risk-on ({_macro_score:+.0f}): SHORT headwind")
                except Exception:
                    pass

                # ── Market Microstructure (CVD · MaxPain · Netflow · Basis) ─
                # Four additional free signals that refine decision quality:
                #   CVD:     are buyers or sellers the aggressor right now?
                #   Basis:   how crowded are longs/shorts vs spot price?
                #   MaxPain: options expiry gravity for BTC/ETH
                #   Netflow: exchange flow as accumulation/distribution signal
                try:
                    _entry_ms = (signal.entry_low + signal.entry_high) / 2
                    _ms_delta, _ms_note = microstructure.get_signal_intel(
                        symbol, getattr(signal.direction, 'value', str(signal.direction)), _entry_ms
                    )
                    if _ms_delta != 0:
                        signal.confidence = max(10, signal.confidence + _ms_delta)
                        if _ms_note:
                            signal.confluence.append(_ms_note)
                except Exception:
                    pass

                # ── A+ Upgrade: New institutional-grade signal enrichment ──
                # Architecture: COLLECT-THEN-CAP with hierarchical guards.
                #   1. Valuation pre-gate — blocks bullish enrichment when overvalued
                #   2. Vol regime dampener — scales down in noisy environments
                #   3. Each module's delta is collected (not applied immediately)
                #   4. Aggregate cap clamps total before applying to confidence
                # This addresses the additive confidence problem: 7 modules × ±10-15
                # previously stacked to ±75 unbounded.  Now capped at ±25.
                from config.constants import Enrichment as _EC

                _sig_dir = getattr(signal.direction, 'value', str(signal.direction))
                _is_long_enrich = _sig_dir == "LONG"

                # ── Step 1: Valuation pre-gate ──────────────────────────
                _val_zone = "UNKNOWN"
                try:
                    _val_zone = onchain_analytics.get_valuation_zone()
                except Exception:
                    pass

                # ── Step 2: Vol regime dampener ─────────────────────────
                _vol_regime = "UNKNOWN"
                try:
                    _vol_regime = volatility_analyzer.get_vol_regime()
                except Exception:
                    pass
                _vol_dampen = (
                    _EC.HIGH_VOL_DAMPENER
                    if _vol_regime in ("HIGH", "HIGH_VOL", "EXTREME")
                    else 1.0
                )

                # ── Step 2b: Feed cross-module context ──────────────────
                # Stablecoin flow is a macro-level signal; its volume gate
                # (LOW_VOLUME_THRESHOLD_USD = $500M) is calibrated for
                # global 24h market volume, not per-symbol traded volume.
                # Passing per-symbol volume would incorrectly halve the
                # acceleration deltas for mid-cap coins even when global
                # market liquidity is ample.  Leave the value at 0.0 so
                # the gate is bypassed as documented in set_market_volume().
                stablecoin_analyzer.set_market_volume(0.0)

                # Vol compression timer needs OI/funding/leverage context
                try:
                    _lm_snap = leverage_mapper.get_snapshot()
                    volatility_analyzer.set_context(
                        oi_change_pct=_lm_snap.leverage.oi_change_pct,
                        funding_rate=_lm_snap.leverage.funding_weighted,
                        leverage_zone=_lm_snap.leverage.zone,
                    )
                except Exception as _ctx_err:
                    logger.debug("Phase3 vol context feed failed: %s", _ctx_err)

                # Vol compression directional bias (whale intent + flow direction)
                try:
                    _whale_int = wallet_profiler.get_whale_intent() if wallet_profiler else "NEUTRAL"
                    _flow_trend = stablecoin_analyzer.get_liquidity_signal() if stablecoin_analyzer else "NEUTRAL"
                    # Map stablecoin liquidity labels to flow direction labels
                    # get_liquidity_signal() returns AMPLE/DRAINING/NEUTRAL
                    # set_directional_bias() expects INFLOW/OUTFLOW/NEUTRAL
                    _flow_label_map = {"AMPLE": "INFLOW", "DRAINING": "OUTFLOW"}
                    _flow_dir = _flow_label_map.get(_flow_trend, _flow_trend)
                    volatility_analyzer.set_directional_bias(
                        whale_intent=_whale_int,
                        flow_direction=_flow_dir,
                    )
                except Exception as _bias_err:
                    logger.debug("Phase3 directional bias feed failed: %s", _bias_err)

                # Stablecoin-linked HTF threshold adjustment
                # When capital is flowing out, raise the HTF ADX bar (more restrictive)
                try:
                    if stablecoin_analyzer:
                        from analyzers.htf_guardrail import htf_guardrail as _htf_sc
                        _sc_signal = stablecoin_analyzer.get_liquidity_signal()
                        _htf_sc.update_stablecoin_adx_offset(_sc_signal)
                except Exception as _sc_htf_err:
                    logger.debug("Phase3 stablecoin→HTF feed failed: %s", _sc_htf_err)

                # Regime transition → risk manager Kelly reduction
                try:
                    _rt_kelly = risk_manager.get_regime_transition_kelly_mult()
                except Exception as _rt_kelly_err:
                    _rt_kelly = 1.0
                    logger.debug("Phase3 regime transition kelly check failed: %s", _rt_kelly_err)

                # VOLATILE_PANIC position review — check if open positions need tightening
                try:
                    _panic_review = risk_manager.check_panic_position_review()
                    if _panic_review["panic_active"] and _panic_review["positions_to_tighten"]:
                        logger.warning(
                            "🚨 VOLATILE_PANIC: %d positions past TP1 flagged for review",
                            len(_panic_review["positions_to_tighten"])
                        )
                except Exception as _panic_err:
                    logger.debug("Phase3 panic position review failed: %s", _panic_err)

                # Regime transition needs vol regime + flow acceleration + exhaustion
                # Gather all inputs first, then call the batched setter once
                # (avoids 3× _refresh_transition_warning per symbol).
                _rt_flow_accel = None
                _rt_oi_change_pct = None
                _rt_price_change_pct = None
                try:
                    _flow_dyn = stablecoin_analyzer.get_flow_dynamics()
                    if _flow_dyn:
                        _rt_flow_accel = _flow_dyn.acceleration
                except Exception as _rt2_err:
                    logger.debug("Phase3 flow accel transition feed failed: %s", _rt2_err)
                try:
                    _lm_snap_ex = leverage_mapper.get_snapshot()
                    _price_hist = leverage_mapper.get_price_history()
                    _rt_price_change_pct = 0.0
                    if len(_price_hist) >= 2 and _price_hist[0] > 0:
                        _rt_price_change_pct = (_price_hist[-1] - _price_hist[0]) / _price_hist[0] * 100
                    _rt_oi_change_pct = _lm_snap_ex.leverage.oi_change_pct
                except Exception as _rt3_err:
                    logger.debug("Phase3 exhaustion data feed failed: %s", _rt3_err)
                try:
                    regime_analyzer.update_transition_inputs(
                        vol_regime=_vol_regime,
                        flow_acceleration=_rt_flow_accel,
                        oi_change_pct=_rt_oi_change_pct,
                        price_change_pct=_rt_price_change_pct,
                    )
                except Exception as _rt_err:
                    logger.debug("Phase3 transition batch update failed: %s", _rt_err)

                # ── Step 3: Collect all enrichment deltas ───────────────
                _enrich_deltas: list = []   # list of (delta, note, icon)

                # On-chain analytics (MVRV · SOPR · NUPL · HODL · Puell)
                try:
                    _oc_delta, _oc_note = onchain_analytics.get_signal_intel(
                        symbol, _sig_dir
                    )
                    if _oc_delta != 0:
                        _enrich_deltas.append((_oc_delta, _oc_note, "📊"))
                except Exception:
                    pass

                # Stablecoin flows (supply · dominance)
                try:
                    _sc_delta, _sc_note = stablecoin_analyzer.get_signal_intel(
                        symbol, _sig_dir
                    )
                    if _sc_delta != 0:
                        _enrich_deltas.append((_sc_delta, _sc_note, ""))
                except Exception:
                    pass

                # Mining & validator data (hash ribbons · fees · miner revenue)
                try:
                    _mn_delta, _mn_note = mining_analyzer.get_signal_intel(
                        symbol, _sig_dir
                    )
                    if _mn_delta != 0:
                        _enrich_deltas.append((_mn_delta, _mn_note, ""))
                except Exception:
                    pass

                # Network activity (active addresses · transactions · NVT)
                try:
                    _na_delta, _na_note = network_activity.get_signal_intel(
                        symbol, _sig_dir
                    )
                    if _na_delta != 0:
                        _enrich_deltas.append((_na_delta, _na_note, ""))
                except Exception:
                    pass

                # Volatility structure (realized vol · IV · vol regime)
                try:
                    _vs_delta, _vs_note = volatility_analyzer.get_signal_intel(
                        symbol, _sig_dir
                    )
                    if _vs_delta != 0:
                        _enrich_deltas.append((_vs_delta, _vs_note, ""))
                except Exception:
                    pass

                # Wallet behavior profiler (accumulation/distribution · coordination)
                try:
                    _wb_delta, _wb_note = wallet_profiler.get_signal_intel(
                        symbol, _sig_dir
                    )
                    if _wb_delta != 0:
                        _enrich_deltas.append((_wb_delta, _wb_note, ""))
                except Exception:
                    pass

                # Leverage mapper (leverage concentration · cascade risk)
                try:
                    _lm_delta, _lm_note = leverage_mapper.get_signal_intel(
                        symbol, _sig_dir
                    )
                    if _lm_delta != 0:
                        _enrich_deltas.append((_lm_delta, _lm_note, ""))
                except Exception:
                    pass

                # Regime transition early warning
                try:
                    from config.constants import RegimeTransition as _RT
                    _tw = regime_analyzer.get_transition_warning()
                    if _tw["warning"]:
                        _tw_delta = _RT.WARNING_ENRICHMENT_DELTA
                        _tw_note = (
                            f"⚠️ Regime transition warning "
                            f"({_tw['factor_count']} factors: "
                            f"{', '.join(_tw['factors'])})"
                        )
                        _enrich_deltas.append((_tw_delta, _tw_note, ""))
                except Exception:
                    pass

                # ── Step 4: Apply hierarchical guards + aggregate cap ───
                _raw_total = 0
                _enrichment_notes: list = []

                for _ed, _en, _ei in _enrich_deltas:
                    d = _ed

                    # Valuation pre-gate: block bullish enrichment when overvalued
                    if _val_zone == "OVERVALUED" and _is_long_enrich and d > 0:
                        d = min(d, _EC.OVERVALUED_LONG_DELTA_CEIL)
                    elif _val_zone == "UNDERVALUED" and not _is_long_enrich and d > 0:
                        d = min(d, _EC.UNDERVALUED_SHORT_DELTA_CEIL)

                    # Vol regime dampener
                    if _vol_dampen < 1.0:
                        d = int(round(d * _vol_dampen))

                    _raw_total += d
                    if d != 0 and _en:
                        _enrichment_notes.append(f"{_ei} {_en}".strip() if _ei else _en)

                # Aggregate cap
                _capped_total = max(
                    -_EC.AGGREGATE_DELTA_CAP,
                    min(_EC.AGGREGATE_DELTA_CAP, _raw_total)
                )

                if _capped_total != 0:
                    signal.confidence = max(10, signal.confidence + _capped_total)
                    for _note in _enrichment_notes:
                        signal.confluence.append(_note)
                    if _capped_total != _raw_total:
                        signal.confluence.append(
                            f"🛡️ Enrichment capped {_raw_total:+d}→{_capped_total:+d}"
                        )
                    if _vol_dampen < 1.0:
                        signal.confluence.append(
                            f"📈 Vol dampener ×{_vol_dampen:.0%} ({_vol_regime})"
                        )
                    if _val_zone == "OVERVALUED" and _is_long_enrich:
                        signal.confluence.append(
                            "⚠️ Valuation pre-gate: LONG enrichment blocked (overvalued)"
                        )
                    elif _val_zone == "UNDERVALUED" and not _is_long_enrich:
                        signal.confluence.append(
                            "⚠️ Valuation pre-gate: SHORT enrichment blocked (undervalued)"
                        )

                # ══════════════════════════════════════════════════
                # ⭐ ENSEMBLE VOTE — Deterministic multi-source gate
                # Replaces AI for pre-publish decisions.
                # Sources vote; 3+ opposing = suppress/reduce.
                # AI explains results; rules make decisions.
                # Veto system (3-AI debate) runs separately every 2h
                # for strategic-level improvements.
                # ══════════════════════════════════════════════════
                try:
                    _coin_ev = symbol.replace("/USDT","").replace("/BUSD","").upper()

                    # Gather vote inputs from cached analyzer data
                    _ci  = liquidation_analyzer.get(_coin_ev)
                    _smb = smart_money.get_bias(_coin_ev)

                    # Institutional flow — CME + Coinbase + Binance + Bybit + Macro
                    _if_aligned: Optional[bool] = None
                    _if_note: str = ""
                    try:
                        from analyzers.institutional_flow import institutional_flow as _if_eng
                        _if_aligned, _if_note = _if_eng.get_ensemble_vote(symbol, getattr(signal.direction, 'value', str(signal.direction)))
                        _macro_regime = _if_eng.get_macro_regime()
                    except Exception:
                        _if_aligned = None
                        _macro_regime = "NEUTRAL"

                    _cvd = microstructure._cvd.get(_coin_ev)
                    _bas = microstructure._basis.get(_coin_ev)

                    # Whale alignment for this symbol (local import — whale_aggregator
                    # is not at module level to avoid circular import)
                    from signals.whale_aggregator import whale_aggregator as _wa
                    _recent_whales  = _wa.get_recent_events(symbol=symbol, max_age_secs=300)
                    _whale_aligned_val = None
                    if _recent_whales:
                        _wa_support = sum(1 for w in _recent_whales
                                         if (w.side=='buy'  and getattr(signal.direction, 'value', str(signal.direction))=='LONG') or
                                            (w.side=='sell' and getattr(signal.direction, 'value', str(signal.direction))=='SHORT'))
                        _wa_oppose  = sum(1 for w in _recent_whales
                                         if (w.side=='sell' and getattr(signal.direction, 'value', str(signal.direction))=='LONG') or
                                            (w.side=='buy'  and getattr(signal.direction, 'value', str(signal.direction))=='SHORT'))
                        if _wa_support + _wa_oppose > 0:
                            _whale_aligned_val = _wa_support > _wa_oppose

                    _ev_verdict = ensemble_voter.evaluate(
                        symbol      = symbol,
                        direction   = getattr(signal.direction, 'value', str(signal.direction)),
                        entry_price = (signal.entry_low + signal.entry_high) / 2,

                        cvd_signal  = _cvd.signal   if _cvd and not _cvd.is_stale  else None,
                        # Smart money: use institutional flow lean (CME+Coinbase+Binance+Bybit)
                        # if available, fall back to Hyperliquid wallet-level signal
                        sm_direction= (
                            getattr(signal.direction, 'value', str(signal.direction)) if _if_aligned else
                            (getattr(signal.direction, 'value', str(signal.direction)) if _smb and not _smb.is_stale and _smb.direction == getattr(signal.direction, 'value', str(signal.direction)) else
                             ("SHORT" if _smb and not _smb.is_stale and _smb.direction == "LONG" else
                              ("LONG" if _smb and not _smb.is_stale and _smb.direction == "SHORT" else None)))
                        ) if (_if_aligned is not None or (_smb and not _smb.is_stale)) else None,
                        oi_trend    = _ci.oi_trend   if _ci  and not _ci.is_stale  else None,
                        crowd_sentiment = _ci.crowd_sentiment if _ci and not _ci.is_stale else None,
                        whale_aligned   = _whale_aligned_val,
                        basis_pct   = _bas.basis_pct if _bas and not _bas.is_stale else None,
                        # A+ Upgrade: new data sources for ensemble voting
                        onchain_zone    = onchain_analytics.get_valuation_zone(),
                        stablecoin_trend = stablecoin_analyzer.get_liquidity_signal(),
                        mining_health   = mining_analyzer.get_network_health(),
                        network_demand  = network_activity.get_demand_signal(),
                        whale_intent    = wallet_profiler.get_whale_intent(),
                        vol_regime      = volatility_analyzer.get_vol_regime(),
                        # Signal quality info for high-confidence override
                        signal_confidence = signal.confidence,
                        signal_rr         = signal.rr_ratio or 0.0,
                    )

                    # Log the verdict + institutional context
                    _if_log = (f" | inst={_if_note[:50]}" if _if_note else "")
                    logger.info(
                        f"   🗳️  {symbol}: {_ev_verdict.as_log_line()}{_if_log}"
                    )

                    if _ev_verdict.hard_suppress:
                        # Hard suppress — signal dies regardless of confidence
                        logger.info(
                            f"   🗳️  {symbol} ENSEMBLE SUPPRESSED: "
                            f"{_ev_verdict.reason}"
                        )
                        try:
                            ai_analyst.buffer_dead_signal(
                                symbol, getattr(signal.direction, 'value', str(signal.direction)), signal.strategy,
                                "ENSEMBLE_SUPPRESS", signal.rr_ratio or 0,
                                signal.confidence, regime_name
                            )
                        except Exception:
                            pass
                        return  # kill signal

                    elif _ev_verdict.action == "REDUCE":
                        signal.confidence = max(10, signal.confidence + _ev_verdict.confidence_adj)
                        signal.confluence.append(
                            f"🗳️ Ensemble: {_ev_verdict.oppose_count}v{_ev_verdict.support_count} "
                            f"opposition → reduced"
                        )

                    elif _ev_verdict.action == "BOOST":
                        signal.confidence = min(99, signal.confidence + _ev_verdict.confidence_adj)
                        signal.confluence.append(
                            f"🗳️ Ensemble: {_ev_verdict.support_count}v{_ev_verdict.oppose_count} "
                            f"aligned → boosted"
                        )
                        if _ev_verdict.cvd_sm_confirm:
                            signal.confluence.append(
                                "✅ CVD + Smart Money confirmed"
                            )

                except Exception as _ev_err:
                    logger.debug(f"Ensemble voter error (non-fatal): {_ev_err}")

                # ══════════════════════════════════════════════════
                # ⭐ R8 ADAPTIVE LAYER — Whale/News/BTC.D/Rotation
                # Applied after ensemble but before institutional layer.
                # ══════════════════════════════════════════════════
                _dir_val = getattr(signal.direction, 'value', str(signal.direction))
                _pump_dump_raw_adj = 0.0

                # ── R8-F2: Whale deposit confidence penalty ──────────
                try:
                    from analyzers.whale_deposit_monitor import whale_deposit_monitor
                    _whale_penalty = whale_deposit_monitor.get_confidence_penalty()
                    if _whale_penalty != 0:
                        signal.confidence = max(10, signal.confidence + _whale_penalty)
                        signal.confluence.append(
                            f"🐋 Whale deposit alert: "
                            f"${whale_deposit_monitor.get_state().total_inflow_1h_usd/1e6:.0f}M "
                            f"inflow → conf {_whale_penalty:+d}"
                        )
                except Exception:
                    pass

                # ── R8-F2b: On-chain to news correlation ──────────
                # Feature 5: Cross-reference on-chain events (whale/onchain)
                # against recent headlines. Distinguish correlated (same event)
                # from independent (different events) for confidence scaling.
                try:
                    from config.feature_flags import ff as _ff
                    from config.shadow_mode import shadow_log as _sl
                    if _ff.is_active("ONCHAIN_NEWS_CORRELATION"):
                        _corr_result = self._correlate_onchain_news(
                            symbol=_sym,
                            signal_direction=_dir_val,
                        )
                        if _corr_result is not None:
                            _corr_mult, _corr_case, _corr_detail = _corr_result
                            if _ff.is_enabled("ONCHAIN_NEWS_CORRELATION"):
                                signal.confidence = max(10, min(100, signal.confidence * _corr_mult))
                                signal.confluence.append(
                                    f"🔗 On-chain↔News {_corr_case}: "
                                    f"conf ×{_corr_mult:.2f} ({_corr_detail})"
                                )
                                self._record_confidence_adjustment(
                                    signal,
                                    "onchain_news_correlation",
                                    multiplier=round(_corr_mult, 4),
                                    detail=_corr_detail,
                                    extra={"case": _corr_case},
                                )
                                from config.fcn_logger import fcn_log
                                fcn_log("ONCHAIN_NEWS_CORRELATION", f"live | {_sym} case={_corr_case} mult={_corr_mult:.3f} detail={_corr_detail}")
                            else:
                                _sl("ONCHAIN_NEWS_CORRELATION", {
                                    "symbol": _sym,
                                    "case": _corr_case,
                                    "multiplier": _corr_mult,
                                    "detail": _corr_detail,
                                    "live_confidence": signal.confidence,
                                    "shadow_confidence": signal.confidence * _corr_mult,
                                })
                                from config.fcn_logger import fcn_log
                                fcn_log("ONCHAIN_NEWS_CORRELATION", f"shadow | {_sym} case={_corr_case} mult={_corr_mult:.3f} detail={_corr_detail}")
                        else:
                            from config.fcn_logger import fcn_log
                            _onc_mode = "shadow" if _ff.is_shadow("ONCHAIN_NEWS_CORRELATION") else "live"
                            fcn_log("ONCHAIN_NEWS_CORRELATION", f"{_onc_mode} | {_sym} — no correlation detected")
                except Exception:
                    pass

                # ── R8-F4: Strategy rotation confidence adj ──────────
                try:
                    _rot_adj = learning_loop.get_rotation_confidence_adj(
                        signal.strategy, regime_name
                    )
                    if _rot_adj != 0:
                        signal.confidence = max(10, signal.confidence + _rot_adj)
                        _rot_status = learning_loop.get_strategy_regime_status(
                            signal.strategy, regime_name
                        )
                        signal.confluence.append(
                            f"🔄 Strategy rotation: {signal.strategy} is "
                            f"{_rot_status} in {regime_name} → conf {_rot_adj:+d}"
                        )
                except Exception:
                    pass

                # ── R8-F6: News storm confidence penalty (regime-aware) ─
                # FIX (P1-C): No longer hard-blocks LONGs. Instead applies
                # direction-aware confidence penalties scaled by regime.
                # Sentiment amplifies conviction, doesn't veto.
                try:
                    _storm_regime = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
                    _storm = news_scraper.detect_news_storm(current_regime=_storm_regime)
                    if _storm['is_storm']:
                        _storm_pen = _storm['long_penalty'] if _dir_val == "LONG" else _storm['short_penalty']
                        if _storm_pen != 0:
                            signal.confidence = max(10, signal.confidence + _storm_pen)
                            signal.confluence.append(
                                f"📰 News storm {_storm['severity']}: "
                                f"{_storm['negative_count']} negative items → "
                                f"{_dir_val} conf {_storm_pen:+d} (regime={_storm_regime})"
                            )
                except Exception:
                    pass

                # ── R8-F8/F9: Pump/dump + contextual cap runtime wiring ──
                try:
                    from analyzers.market_microstructure import detect_pump_dump
                    from analyzers.narrative_tracker import (
                        clamp_contextual_adjustments,
                        narrative_tracker,
                    )
                    from analyzers.sentiment import compute_fear_greed_raw_adjustment
                    from config.constants import NewsIntelligence as _ni
                    from config.feature_flags import ff as _ff

                    if not hasattr(signal, "raw_data") or signal.raw_data is None:
                        signal.raw_data = {}

                    _pd_metrics = self._compute_pump_dump_metrics(ohlcv_dict.get('15m'))
                    _pd_debug = {"status": "insufficient_data"}
                    if _pd_metrics is not None:
                        _move_direction = "LONG" if _pd_metrics["price_change_pct"] >= 0 else "SHORT"
                        _move_label = "PUMP" if _move_direction == "LONG" else "DUMP"
                        _news_explainer = self._find_recent_directional_news(
                            symbol,
                            _move_direction,
                            max_age_mins=_ni.PUMP_DUMP_NEWS_EXEMPT_WINDOW_MINUTES,
                        )
                        _pd_alert = detect_pump_dump(
                            symbol,
                            _pd_metrics["price_change_pct"],
                            _pd_metrics["volume_change_pct"],
                            has_correlated_news=_news_explainer is not None,
                        )
                        _pd_debug = {
                            **_pd_metrics,
                            "move_direction": _move_label,
                            "news_explained": _news_explainer is not None,
                            "news_headline": (
                                _news_explainer["article"].get("title", "")[:80]
                                if _news_explainer else ""
                            ),
                            "alert": (
                                {
                                    "risk_level": _pd_alert.risk_level,
                                    "is_no_trade": _pd_alert.is_no_trade,
                                    "direction": _pd_alert.direction,
                                }
                                if _pd_alert else None
                            ),
                        }
                        signal.raw_data["pump_dump_debug"] = _pd_debug
                        if _pd_alert is not None:
                            _pump_dump_raw_adj = self._pump_dump_raw_adjustment(_pd_alert)
                            if _pd_alert.is_no_trade:
                                _pd_reason = (
                                    f"{_pd_alert.direction} risk={_pd_alert.risk_level} "
                                    f"price={_pd_alert.price_change_pct:+.1f}% "
                                    f"volume={_pd_alert.volume_change_pct:+.0f}%"
                                )
                                signal.confluence.append(f"🚫 Pump/dump guard: {_pd_reason}")
                                self._record_confidence_adjustment(
                                    signal,
                                    "pump_dump_guard",
                                    detail=_pd_reason,
                                    extra=_pd_debug,
                                )
                                logger.warning("🚫 %s suppressed by pump/dump guard | %s", symbol, _pd_reason)
                                return

                    _fear_greed_adj = (
                        compute_fear_greed_raw_adjustment(regime_analyzer.fear_greed, _dir_val)
                        if _ff.is_active("FEAR_GREED")
                        else 0.0
                    )
                    if _ff.is_active("FEAR_GREED"):
                        from config.fcn_logger import fcn_log
                        _fg_mode = "shadow" if _ff.is_shadow("FEAR_GREED") else "live"
                        _fg_val = regime_analyzer.fear_greed
                        fcn_log("FEAR_GREED", f"{_fg_mode} | F&G={_fg_val} dir={_dir_val} adj={_fear_greed_adj:+.3f}")
                    # In shadow mode, compute for logging but don't apply
                    if _ff.is_shadow("FEAR_GREED"):
                        _fear_greed_adj = 0.0
                    _narrative_adj = narrative_tracker.get_raw_adjustment(symbol, _dir_val)
                    _context_mult = clamp_contextual_adjustments(
                        fear_greed_adj=_fear_greed_adj,
                        narrative_adj=_narrative_adj,
                        pump_dump_adj=_pump_dump_raw_adj,
                    )
                    _context_total = _fear_greed_adj + _narrative_adj + _pump_dump_raw_adj
                    signal.raw_data["contextual_overlay"] = {
                        "fear_greed_adj": round(_fear_greed_adj, 4),
                        "narrative_adj": round(_narrative_adj, 4),
                        "pump_dump_adj": round(_pump_dump_raw_adj, 4),
                        "raw_total": round(_context_total, 4),
                        "applied_multiplier": round(_context_mult, 4),
                    }
                    if _context_mult != 1.0:
                        signal.confidence = max(10, min(99, signal.confidence * _context_mult))
                        signal.confluence.append(
                            f"🧭 Context overlay ×{_context_mult:.2f} "
                            f"(F&G={_fear_greed_adj:+.2f}, narrative={_narrative_adj:+.2f}, "
                            f"pump/dump={_pump_dump_raw_adj:+.2f})"
                        )
                        self._record_confidence_adjustment(
                            signal,
                            "contextual_overlay",
                            multiplier=round(_context_mult, 4),
                            detail="Global cap across Fear&Greed, narratives, and pump/dump",
                            extra=signal.raw_data["contextual_overlay"],
                        )
                except Exception as _ctx_overlay_err:
                    logger.debug("Contextual overlay wiring failed for %s: %s", symbol, _ctx_overlay_err)

                # ── R8-F9: BTC dominance confidence adj ──────────────
                try:
                    from analyzers.btc_dominance import btc_dominance_tracker
                    _btcd_adj = btc_dominance_tracker.get_confidence_adj(
                        symbol, _dir_val
                    )
                    if _btcd_adj != 0:
                        signal.confidence = max(10, signal.confidence + _btcd_adj)
                        _btcd_meta = btc_dominance_tracker.get_metadata()
                        signal.confluence.append(
                            f"📊 BTC.D {_btcd_meta['btcd_trend']}: "
                            f"{_btcd_meta['btcd_change_24h']:+.1f}% → "
                            f"conf {_btcd_adj:+d}"
                        )
                except Exception:
                    pass

                # ══════════════════════════════════════════════════
                # ⭐ INSTITUTIONAL UPGRADE LAYER (V3)
                # Evaluates market environment before trade quality.
                # "What environment?" before "Is this a setup?"
                # ══════════════════════════════════════════════════

                # ── Market State Engine ────────────────────────────
                # Classify: TRENDING, EXPANSION, COMPRESSION,
                #           LIQUIDITY_HUNT, ROTATION, VOLATILE_PANIC
                _btc_adx = 20.0  # FIX P1-E: initialize before try block to prevent NameError
                _state_mult = 1.0  # FIX P2-A: initialize here too
                _state_note = None
                _mkt_state_name = "NEUTRAL"  # captured for sig_data (ParamTuner attribution)
                try:
                    from analyzers.market_state_engine import market_state_engine
                    mkt_state = await market_state_engine.get_state()
                    _mkt_state_name = mkt_state.state.value

                    # Soft-degrade for strategies suppressed in this state.
                    # A hard block (return) is only applied when the signal is
                    # already weak (confidence < 65 after the penalty) so that
                    # strong, high-conviction signals are never silently killed
                    # by a non-deterministic AI classification.
                    if mkt_state.blocks_strategy(signal.strategy):
                        # Penalty is proportional to state severity — a mild liquidity
                        # sweep is not the same risk as a full macro panic.
                        # Values come from the adaptive param store (defaults match the
                        # original hard-coded constants; ParamTuner adjusts them over time).
                        # FIX P4: Scale penalty by state classification confidence.
                        # A VOLATILE_PANIC at 55% confidence should not apply the same
                        # penalty as one at 90% confidence — treat regime as hypothesis,
                        # not fact. Clamped to [0.5, 1.0] so low confidence still matters.
                        from governance.adaptive_params import adaptive_params as _ap
                        _mse_penalty_map = {
                            "VOLATILE_PANIC": int(_ap.get("mse_penalty_volatile_panic")),
                            "LIQUIDITY_HUNT": int(_ap.get("mse_penalty_liquidity_hunt")),
                        }
                        _mse_penalty_raw = _mse_penalty_map.get(
                            mkt_state.state.value, int(_ap.get("mse_penalty_default"))
                        )
                        _state_conf_scale = max(Penalties.MSE_CONF_SCALE_MIN, min(Penalties.MSE_CONF_SCALE_MAX, mkt_state.confidence))
                        _mse_penalty = max(Penalties.MSE_PENALTY_MIN_PTS, round(_mse_penalty_raw * _state_conf_scale))
                        signal.confidence = max(0, signal.confidence - _mse_penalty)
                        logger.info(
                            f"🧠 {symbol}: market state soft-block {signal.strategy} "
                            f"−{_mse_penalty} pts (conf_scale={_state_conf_scale:.2f}) → conf={signal.confidence} "
                            f"| state={mkt_state.state.value}"
                        )
                        if signal.confidence < Penalties.MARKET_STATE_DEATH_THRESHOLD:
                            logger.info(
                                f"🧠 Signal died | {symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                                f"| reason=MARKET_STATE_BLOCK (conf={signal.confidence}<{Penalties.MARKET_STATE_DEATH_THRESHOLD}) "
                                f"| state={mkt_state.state.value} "
                                f"| strategy={signal.strategy} "
                                f"| {mkt_state.get_description()}"
                            )
                            return
                        # Re-anchor the AI delta cap baseline to the post-penalty value.
                        # The market state block is deterministic (technical analysis),
                        # not an AI opinion, so the ±12 AI cap should govern only the
                        # AI context adjustments that follow (news, BTC sentiment, etc.).
                        _raw_confidence = signal.confidence

                    # FIX P2-A: state_mult applied ONLY post-aggregation (not here) to avoid double-count
                    _state_mult = mkt_state.get_confidence_mult(signal.strategy)
                    _state_note = (
                        f"🧠 {mkt_state.state.value}: "
                        f"{'boost' if _state_mult > 1 else 'penalty'} "
                        f"×{_state_mult:.2f} on {signal.strategy}"
                    ) if _state_mult != 1.0 else None
                    # Note: do NOT apply _state_mult to signal.confidence here.
                    # It will be applied once after full aggregation below.
                    signal.confluence.append(_state_note) if _state_note else None

                    # FIX P3: Apply macro pressure confidence reduction on new alt longs.
                    # When a BTC drop has fired within the last hour, reduce confidence
                    # on new altcoin long signals to naturally reduce position sizing.
                    try:
                        from signals.proactive_alerts import proactive_alerts as _pa
                        if (_pa.is_macro_pressure_active()
                                and getattr(signal.direction, 'value', str(signal.direction)) == "LONG"
                                and "BTC" not in symbol):
                            _macro_penalty = 8
                            signal.confidence = max(0, signal.confidence - _macro_penalty)
                            signal.confluence.append(
                                f"₿ macro pressure active: −{_macro_penalty} pts (BTC drop <1h ago)"
                            )
                            logger.info(
                                f"   ₿ {symbol}: macro pressure penalty −{_macro_penalty} "
                                f"→ conf={signal.confidence}"
                            )
                    except Exception:
                        pass

                    # Regime transition risk
                    from analyzers.regime_transition import (
                        amplify_transition_with_news,
                        get_trade_type_transition_penalty,
                        has_active_breakout_exemption,
                        should_block_countertrend_pullback,
                        transition_detector,
                    )
                    trans_result = None
                    _transition_risk = 0.0
                    _transition_type = "stable"
                    trans_adx = mkt_state.vol_ratio  # proxy if no adx available
                    try:
                        from analyzers.regime import regime_analyzer as _ra_ref
                        _btc_adx = getattr(_ra_ref, '_btc_adx', 20.0)
                    except Exception:
                        _btc_adx = 20.0
                    transition_detector.record(
                        regime_name, chop_str, _btc_adx
                    )
                    trans_result = transition_detector.evaluate(
                        current_regime=regime_name,
                        chop_strength=chop_str,
                        vol_ratio=mkt_state.vol_ratio,
                        adx=_btc_adx,
                    )
                    try:
                        _btc_ctx_transition = btc_news_intelligence.get_event_context()
                        if _btc_ctx_transition.is_active:
                            _news_bias, _news_score = btc_news_intelligence.get_net_news_bias()
                            trans_result = amplify_transition_with_news(
                                transition_result=trans_result,
                                regime_name=regime_name,
                                news_bias=_news_bias,
                                news_score=_news_score,
                            )
                    except Exception as _transition_news_err:
                        logger.debug(f"Transition news amplification skipped: {_transition_news_err}")
                    if trans_result.in_transition:
                        signal.confidence -= trans_result.confidence_increase
                        death_reasons.append(
                            f"transition_penalty(-{trans_result.confidence_increase}): "
                            f"{trans_result.transition_type}"
                        )
                        signal.confluence.append(trans_result.warning)
                        logger.info(
                            f"   ⚠️  {symbol}: regime transition {trans_result.transition_type} "
                            f"-{trans_result.confidence_increase}"
                        )

                    # FIX P2-E: Store transition_risk for post-score gate.
                    # The pre-score grade_pre is inaccurate. The real gate runs after
                    # alpha_score is computed below using the true alpha grade.
                    # FIX P5: Skip suppression for breakout transitions — COMPRESSION→EXPANSION
                    # is the highest-expectancy setup; penalising it as "transition noise"
                    # is exactly the opposite of what the system should do.
                    _transition_risk = market_state_engine.get_transition_risk()
                    _transition_type = market_state_engine.get_transition_type()
                    # Pre-score hard block only for very low confidence signals during NOISE transitions
                    if (_transition_risk > 0.8 and signal.confidence < 70
                            and not has_active_breakout_exemption(_transition_type, trans_result)):
                        logger.info(
                            f"   ⚠️  {symbol}: very high transition risk ({_transition_risk:.2f}) "
                            f"[{_transition_type}] and low pre-score confidence — suppressed"
                        )
                        return

                except Exception as _mse_err:
                    logger.debug(f"Market state engine skip: {_mse_err}")

                # ── No-Trade Zone Engine ───────────────────────────
                # Suppress signals when market conditions are unfavorable
                try:
                    from analyzers.no_trade_zones import no_trade_engine

                    # mkt_state IS the MarketStateResult — access its fields directly.
                    # (Previous code read mkt_state._state which is an engine-internal attr,
                    # not a field on the result dataclass — causing silent AttributeError.)
                    _has_mkt = 'mkt_state' in locals() and mkt_state is not None
                    _vol_ratio   = mkt_state.vol_ratio          if _has_mkt else 1.0
                    _btc_mom     = mkt_state.btc_momentum_fast  if _has_mkt else 0.0
                    # Fix S1: use genuine 24-bar slow momentum, not same as 6-bar fast
                    _btc_mom_1h  = mkt_state.btc_momentum_slow  if _has_mkt else _btc_mom

                    # Get funding rate if available
                    _funding = None
                    try:
                        _fr_data = ohlcv_dict.get('funding')
                        if _fr_data and len(_fr_data):
                            _funding = float(_fr_data[-1])
                    except Exception:
                        pass

                    entry_mid_ntze = (signal.entry_low + signal.entry_high) / 2.0
                    ntze_result = no_trade_engine.evaluate(
                        symbol=symbol,
                        strategy=signal.strategy,
                        direction=getattr(signal.direction, 'value', str(signal.direction)),
                        vol_ratio=_vol_ratio,
                        btc_momentum_fast=_btc_mom,
                        btc_momentum_1h=_btc_mom_1h,  # Fix S1: use genuine slow momentum
                        funding_rate=_funding,
                        price=entry_mid_ntze,
                        range_high=regime_analyzer.range_high,
                        range_low=regime_analyzer.range_low,
                        session=regime_analyzer.session.value,  # Phase 9-12 fix: pass session for dead-zone rule
                        adx=_btc_adx,
                    )

                    if ntze_result.hard_block and ntze_result.blocks(signal.strategy):
                        reason_str = ntze_result.reasons[0] if ntze_result.reasons else "no_trade_zone"
                        logger.info(
                            f"⛔ Signal died | {symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                            f"| reason=NO_TRADE_ZONE | {reason_str}"
                        )
                        return

                    if ntze_result.confidence_penalty > 0:
                        signal.confidence -= ntze_result.confidence_penalty
                        for _nr in ntze_result.reasons:
                            death_reasons.append(
                                f"no_trade_penalty(-{ntze_result.confidence_penalty}): {_nr}"
                            )
                            signal.confluence.append(_nr)

                except Exception as _ntze_err:
                    logger.debug(f"No-trade zone check skip: {_ntze_err}")

                # ── HTF Weekly Guardrail (soft-penalty pass) ──────────
                # FIX HTF-DOUBLE: The old code ran htf_guardrail.check() here
                # AFTER is_hard_blocked() already ran at step P1-B (line ~887).
                # is_hard_blocked() uses threshold = min(88, 78 + 0.2*(ADX-25)).
                # check() used the legacy formula  = min(93, 80 + 0.2*(ADX-25)).
                # At ADX=53: pre-gate needed 83.6, old check() needed 85.6.
                # A signal with raw_conf 84 passed the pre-gate then got hard-
                # blocked here — the "Need 85.7+" log comes from this second check.
                #
                # Fix: signals that survived is_hard_blocked() are already HTF-
                # vetted. This pass only applies the soft PENALTY (alignment bonus/
                # penalty), never hard-blocks. Counter-trend strategies still get
                # their extra soft penalty of -8.
                try:
                    from analyzers.htf_guardrail import htf_guardrail
                    _COUNTER_TREND_STRATEGIES = {"MeanReversion", "RangeScalper"}
                    _is_counter_trend_strat = signal.strategy in _COUNTER_TREND_STRATEGIES

                    htf_result = await htf_guardrail.check(
                        signal_direction=getattr(signal.direction, 'value', str(signal.direction)),
                        signal_confidence=_raw_confidence,
                    )

                    # NEVER hard-block here — pre-gate already made the block decision.
                    # Only apply soft penalties for context/alignment signals.
                    if htf_result.penalty > 0:
                        # Extra penalty for counter-trend strategies that use softer threshold
                        _pen = (htf_result.penalty + 3) if _is_counter_trend_strat else htf_result.penalty
                        signal.confidence -= _pen
                        death_reasons.append(
                            f"htf_penalty(-{_pen}): weekly={htf_result.weekly_bias}"
                        )
                    signal.confluence.append(htf_result.reason)

                except Exception as _htf_err:
                    logger.debug(f"HTF guardrail skip: {_htf_err}")

                # ── Signal Clarity Score ───────────────────────────
                # Reject messy signals: wide entries, floating stops,
                # weak RR, conflicting indicators
                try:
                    from signals.signal_clarity import signal_clarity_scorer
                    _atr_for_clarity = 0.0
                    try:
                        _cl_ohlcv = ohlcv_dict.get(signal.entry_timeframe) or ohlcv_dict.get('1h') or []
                        if _cl_ohlcv and len(_cl_ohlcv) >= 14:
                            _cl_highs  = [float(c[2]) for c in _cl_ohlcv[-15:]]
                            _cl_lows   = [float(c[3]) for c in _cl_ohlcv[-15:]]
                            _cl_closes = [float(c[4]) for c in _cl_ohlcv[-15:]]
                            import numpy as _np_cl
                            _tr = _np_cl.maximum(
                                _np_cl.array(_cl_highs[1:]) - _np_cl.array(_cl_lows[1:]),
                                _np_cl.maximum(
                                    _np_cl.abs(_np_cl.array(_cl_highs[1:]) - _np_cl.array(_cl_closes[:-1])),
                                    _np_cl.abs(_np_cl.array(_cl_lows[1:]) - _np_cl.array(_cl_closes[:-1])),
                                )
                            )
                            _atr_for_clarity = float(_np_cl.mean(_tr[-14:]))
                    except Exception:
                        pass

                    clarity = signal_clarity_scorer.score(signal, _atr_for_clarity)

                    # Store clarity score on signal so aggregator can use it
                    # for hourly rate-limit bypass decisions.
                    signal.clarity_score = clarity.score

                    if clarity.grade == "REJECT":
                        logger.info(
                            f"🔍 Signal died | {symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                            f"| reason=CLARITY_REJECT | score={clarity.score} "
                            f"| {'; '.join(clarity.reasons[:3])}"
                        )
                        return

                    if clarity.confidence_adjustment != 1.0:
                        signal.confidence *= clarity.confidence_adjustment
                        clarity_note = (
                            f"🔍 Clarity: {clarity.grade} ({clarity.score}/100) "
                            f"×{clarity.confidence_adjustment:.2f}"
                        )
                        signal.confluence.append(clarity_note)
                        if clarity.confidence_adjustment < 1.0:
                            death_reasons.append(
                                f"clarity_penalty(×{clarity.confidence_adjustment:.2f}): grade={clarity.grade}"
                            )
                        logger.info(
                            f"   🔍 {symbol}: clarity {clarity.grade} score={clarity.score} "
                            f"×{clarity.confidence_adjustment:.2f}"
                        )

                except Exception as _clar_err:
                    logger.debug(f"Signal clarity skip: {_clar_err}")

                # ── Smart entry zone refinement (soft) ──────────────
                try:
                    refined = _entry_refiner.refine(signal, ohlcv_dict, signal.strategy)
                    if refined is None:
                        # Soft: penalty instead of rejection
                        signal.confidence -= 8
                        death_reasons.append("entry_refine_fail(-8)")
                        logger.info(f"   🎯 {symbol}: entry refiner penalty -8")
                    else:
                        signal = refined
                except Exception as e:
                    logger.debug(f"Entry refiner error for {symbol}: {e}")
                    # Continue with unrefined signal

                # ── V12: Regime-aware TP/SL adjustment ─────────────────────
                # Adjusts strategy-computed targets based on current regime,
                # range structure, and setup class. This is where CHOPPY tightens
                # targets to range boundaries and TRENDING widens them.
                try:
                    from signals.regime_levels import STRUCTURE_MAX_SCORE, adjust_levels
                    # FIX TP-CORRUPTION: pass the SYMBOL's own range, not BTC's global range.
                    # regime_analyzer.range_high/low is BTC's weekly range (~$70k/$60k).
                    # Passing that to an altcoin at $0.71 caused SL distances of ~$59,000
                    # and TP1 targets of $47,360 on a $0.71 coin.
                    _sym_bars_adj = ohlcv_dict.get('1h') or ohlcv_dict.get('4h') or []
                    _sym_rh = float(max(b[2] for b in _sym_bars_adj[-50:])) if len(_sym_bars_adj) >= 20 else 0.0
                    _sym_rl = float(min(b[3] for b in _sym_bars_adj[-50:])) if len(_sym_bars_adj) >= 20 else 0.0
                    _sym_req = (_sym_rh + _sym_rl) / 2 if _sym_rh > 0 and _sym_rl > 0 else 0.0
                    _struct_bars = _sym_bars_adj[-30:] if len(_sym_bars_adj) >= 14 else []
                    _adj = adjust_levels(
                        entry_low=signal.entry_low,
                        entry_high=signal.entry_high,
                        stop_loss=signal.stop_loss,
                        tp1=signal.tp1,
                        tp2=signal.tp2,
                        tp3=signal.tp3,
                        direction=getattr(signal.direction, 'value', str(signal.direction)),
                        setup_class=getattr(signal, 'setup_class', 'intraday'),
                        regime=regime_name,
                        chop_strength=chop_str,
                        range_high=_sym_rh,    # symbol's own range
                        range_low=_sym_rl,     # symbol's own range
                        range_eq=_sym_req,     # symbol's own equilibrium
                        recent_opens=[float(b[1]) for b in _struct_bars] if _struct_bars else None,
                        recent_highs=[float(b[2]) for b in _struct_bars] if _struct_bars else None,
                        recent_lows=[float(b[3]) for b in _struct_bars] if _struct_bars else None,
                        recent_closes=[float(b[4]) for b in _struct_bars] if _struct_bars else None,
                        recent_volumes=[float(b[5]) for b in _struct_bars] if _struct_bars else None,
                    )
                    # Safety check: reject adjustments that produce absurd TPs
                    # (e.g. if range data was still wrong for any reason)
                    _entry_mid_check = (signal.entry_low + signal.entry_high) / 2
                    _max_sane_tp = _entry_mid_check * 10.0  # TP can't be more than 10x entry
                    if _adj.tp1 > 0 and abs(_adj.tp1 - _entry_mid_check) < abs(_entry_mid_check) * 10.0:
                        signal.stop_loss = _adj.stop_loss
                        signal.tp1 = _adj.tp1
                        signal.tp2 = _adj.tp2
                        signal.tp3 = _adj.tp3
                        signal.rr_ratio = _adj.rr_ratio
                    else:
                        logger.warning(
                            f"   ⚠️  {symbol}: adjust_levels produced absurd TP1={_adj.tp1:.4f} "
                            f"for entry={_entry_mid_check:.4f} — keeping original levels"
                        )
                    if _adj.adjustments:
                        signal.confluence.append(
                            f"📐 Regime adjust: {', '.join(_adj.adjustments[:3])}"
                        )
                        logger.info(
                            f"   📐 {symbol}: regime TP/SL adj [{regime_name}]: "
                            f"{', '.join(_adj.adjustments)}"
                        )
                    # Store trail_pct on signal for outcome monitor
                    signal._regime_trail_pct = _adj.trail_pct
                    signal._regime_trade_type = _adj.trade_type
                    if signal.raw_data is None:
                        signal.raw_data = {}
                    signal.raw_data["regime_trade_type"] = _adj.trade_type
                    signal.raw_data["local_structure_bias"] = _adj.local_structure_bias
                    signal.raw_data["local_structure_score"] = _adj.local_structure_score
                    signal.raw_data["local_structure_reason"] = _adj.local_structure_reason
                    signal.raw_data["local_structure_bars"] = _adj.local_structure_bars
                    signal.raw_data["local_structure_used_vwap"] = _adj.local_structure_used_vwap
                    signal.raw_data["local_structure_used_rejections"] = _adj.local_structure_used_rejections
                    if _adj.trade_type.startswith(("LOCAL_CONTINUATION_", "PULLBACK_")):
                        signal.confluence.append(
                            f"🧭 Trade type: {_adj.trade_type} ({_adj.local_structure_bias}, strength {int(_adj.local_structure_score)}/{STRUCTURE_MAX_SCORE})"
                        )
                    if trans_result and trans_result.in_transition:
                        _adjusted_transition_penalty = get_trade_type_transition_penalty(
                            trade_type=_adj.trade_type,
                            transition_result=trans_result,
                        )
                        _transition_refund = trans_result.confidence_increase - _adjusted_transition_penalty
                        if _transition_refund > 0:
                            signal.confidence += _transition_refund
                            death_reasons.append(
                                f"transition_trade_type_adjust(+{_transition_refund}): {_adj.trade_type}"
                            )
                            logger.info(
                                f"   ⚠️  {symbol}: {_adj.trade_type} transition penalty scaled "
                                f"from -{trans_result.confidence_increase} to -{_adjusted_transition_penalty}"
                            )
                    if should_block_countertrend_pullback(
                        trade_type=_adj.trade_type,
                        transition_result=trans_result,
                        market_transition_type=_transition_type,
                    ):
                        death_reasons.append(
                            f"transition_pullback_block: {_adj.trade_type} during "
                            f"{getattr(trans_result, 'transition_type', 'transition')}"
                        )
                        logger.info(
                            f"   ⚠️  {symbol}: {_adj.trade_type} blocked during "
                            f"{getattr(trans_result, 'transition_type', 'transition')} "
                            f"[{_transition_type}]"
                        )
                        return
                except Exception as _rl_err:
                    logger.debug(f"Regime level adjustment skipped: {_rl_err}")

                # ── FIX #3/#9/#13/#17: Post-pipeline geometric validation ──────────
                # Ensures entry zone, SL, TP, and R:R are geometrically valid.
                # Catches zone inversions (#9), impossible SL/TP, and fixes R:R sign (#13).
                try:
                    # FIX #9: Auto-correct inverted entry zones
                    if signal.entry_low > signal.entry_high:
                        signal.entry_low, signal.entry_high = signal.entry_high, signal.entry_low
                        logger.info(f"   🔧 {symbol}: entry zone swapped (was inverted)")

                    # FIX SL-REPAIR-2: Regime levels adjustment can shift zones and re-invert SL.
                    # entry_refiner.repair runs earlier but regime_levels runs after it.
                    # Run a second repair here as the definitive pre-aggregation gate.
                    try:
                        _rbs = ohlcv_dict.get(getattr(signal,'entry_timeframe','1h')) or ohlcv_dict.get('1h') or []
                        _sl_atr2 = float(sum(float(_rbs[i][2])-float(_rbs[i][3]) for i in range(max(-14,-len(_rbs)),0)) / max(1,min(14,len(_rbs)))) if _rbs else signal.entry_low*0.003
                        _sl_buf2 = max(_sl_atr2*0.5, signal.entry_low*0.002)
                        if signal.direction == SignalDirection.LONG and signal.stop_loss >= signal.entry_low:
                            signal.stop_loss = signal.entry_low - _sl_buf2
                            logger.info(f"   🔧 {symbol}: SL repaired post-regime-levels (LONG)")
                        elif signal.direction == SignalDirection.SHORT and signal.stop_loss <= signal.entry_high:
                            signal.stop_loss = signal.entry_high + _sl_buf2
                            logger.info(f"   🔧 {symbol}: SL repaired post-regime-levels (SHORT)")
                    except Exception:
                        pass

                    # FIX #17: Enforce TP ordering (TP1 closest, TP3 farthest)
                    _entry_mid_val = (signal.entry_low + signal.entry_high) / 2
                    if signal.direction == SignalDirection.LONG:
                        signal.tp1 = max(signal.tp1, _entry_mid_val + 1e-8)
                        signal.tp2 = max(signal.tp2, signal.tp1 + 1e-8)
                        if signal.tp3 is not None:
                            signal.tp3 = max(signal.tp3, signal.tp2 + 1e-8)
                        # SL must be below entry for LONG
                        if signal.stop_loss >= signal.entry_low:
                            logger.info(f"   ❌ {symbol}: SL above entry zone for LONG — skipping")
                            return
                    else:  # SHORT
                        signal.tp1 = min(signal.tp1, _entry_mid_val - 1e-8)
                        signal.tp2 = min(signal.tp2, signal.tp1 - 1e-8)
                        if signal.tp3 is not None:
                            signal.tp3 = min(signal.tp3, signal.tp2 - 1e-8)
                        # SL must be above entry for SHORT
                        if signal.stop_loss <= signal.entry_high:
                            logger.info(f"   ❌ {symbol}: SL below entry zone for SHORT — skipping")
                            return

                    # FIX #13: Ensure R:R is always positive
                    _risk_val = abs(_entry_mid_val - signal.stop_loss)
                    if _risk_val > 0:
                        signal.rr_ratio = abs(signal.tp2 - _entry_mid_val) / _risk_val
                    else:
                        signal.rr_ratio = 0.0
                except Exception as _val_err:
                    logger.debug(f"Post-pipeline validation error: {_val_err}")

                # ── BUG-3 FIX: Inject OHLCV into signal.raw_data for candlestick detection ──
                # aggregator.py step 7b reads raw_data['ohlcv_1h'] / raw_data['ohlcv_15m']
                # to run patterns/candlestick.py. No strategy populates these keys —
                # they store domain-specific data (OB zones, wave counts, etc.).
                # Without this injection, _candle_bonus is ALWAYS 0.0 and the entire
                # candlestick confluence layer is dead code.
                if 'ohlcv_1h' not in signal.raw_data and '1h' in ohlcv_dict:
                    signal.raw_data['ohlcv_1h'] = ohlcv_dict['1h']
                if 'ohlcv_15m' not in signal.raw_data and '15m' in ohlcv_dict:
                    signal.raw_data['ohlcv_15m'] = ohlcv_dict['15m']

                # ── Inject news context for sentiment scoring ─────────────
                try:
                    # RSS-based news (real-time, no API key needed)
                    _rss_news = news_scraper.get_news_for_symbol(symbol, max_age_mins=60)
                    if _rss_news:
                        signal.raw_data['news_headlines'] = _rss_news[:10]
                    # Keyword sentiment score from RSS headlines
                    _cp_score = news_scraper.get_symbol_sentiment_score(symbol, max_age_mins=60)
                    if _cp_score is not None:
                        signal.raw_data['cp_sentiment_score'] = _cp_score
                        # Clamp to ±3 (was ±5) to prevent keyword noise from over-influencing
                        _sent_delta = round((_cp_score - Penalties.SENTIMENT_SCORE_CENTER) / Penalties.SENTIMENT_SCORE_DIVISOR)
                        _sent_delta = max(-Penalties.SENTIMENT_DELTA_MAX, min(Penalties.SENTIMENT_DELTA_MAX, _sent_delta))
                        _n_headlines = len(_rss_news) if _rss_news else 0
                        # Guard: only apply if signal is already confident (>=70)
                        # OR multiple headlines agree (>=3 distinct sources)
                        if _sent_delta != 0 and (signal.confidence >= Penalties.SENTIMENT_CONF_GATE or _n_headlines >= Penalties.SENTIMENT_HEADLINES_GATE):
                            signal.confidence = max(0, min(99, signal.confidence + _sent_delta))
                            if 'confidence_adjustments' not in signal.raw_data:
                                signal.raw_data['confidence_adjustments'] = []
                            signal.raw_data['confidence_adjustments'].append({
                                'source': 'news_sentiment',
                                'delta': _sent_delta,
                                'score': round(_cp_score, 1),
                                'headlines': _n_headlines,
                            })
                            if abs(_sent_delta) >= 2:
                                direction_word = "bullish" if _sent_delta > 0 else "bearish"
                                signal.confluence.append(
                                    f"📰 {'Strong' if abs(_sent_delta) >= 3 else 'Moderate'} {direction_word} news sentiment ({_sent_delta:+d})"
                                )
                    # Check for dangerous upcoming events (token unlocks etc)
                    if news_scraper.has_dangerous_event_soon(symbol, within_hours=Penalties.DANGEROUS_EVENT_LOOKAHEAD_HOURS):
                        _ev = news_scraper.get_upcoming_events(symbol)
                        signal.confidence -= Penalties.DANGEROUS_EVENT_PENALTY_PTS
                        if 'confidence_adjustments' not in signal.raw_data:
                            signal.raw_data['confidence_adjustments'] = []
                        signal.raw_data['confidence_adjustments'].append({
                            'source': 'dangerous_event',
                            'delta': -8,
                            'detail': _ev[0]['name'] if _ev else 'unlock/listing',
                        })
                        signal.confluence.append(
                            f"⚠️ Major event in <48h: {_ev[0]['name'] if _ev else 'unlock/listing'}"
                        )
                except Exception:
                    pass

                # ── CoinGecko trending boost ──────────────────────────────
                try:
                    _cg_boost, _cg_note = coingecko.get_trending_boost(symbol)
                    if _cg_boost > 0:
                        signal.confidence = min(signal.confidence + _cg_boost, 99)
                        signal.confluence.append(_cg_note)
                        signal.raw_data['coingecko_trending'] = True
                        if 'confidence_adjustments' not in signal.raw_data:
                            signal.raw_data['confidence_adjustments'] = []
                        signal.raw_data['confidence_adjustments'].append({
                            'source': 'coingecko_trending',
                            'delta': _cg_boost,
                            'detail': _cg_note,
                        })
                except Exception:
                    pass

                # ── Aggregate and score ────────────────────────────
                try:
                    scored = await signal_aggregator.process(signal, raw_signals=all_signals)  # BUG-E fix: pass raw list for SMC veto
                    if not scored:
                        return

                    # Record that a signal passed aggregation (for diagnostic reports)
                    try:
                        diagnostic_engine.record_signal_generated()
                    except Exception:
                        pass

                    # ── Risk validation ────────────────────────────
                    # FIX #4: Don't count C-grade signals against daily limit.
                    # We can't know the grade yet, but we defer the daily counter
                    # increment to AFTER alpha grading (see below after alpha_score).
                    # For now, only check circuit breaker and daily loss — not trade count.
                    if risk_manager.circuit_breaker.is_active:
                        logger.info(
                            f"❌ Signal died | {symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                            f"| reason=CIRCUIT_BREAKER"
                        )
                        return

                    # ── Apply performance weight ───────────────────
                    weight = performance_tracker.get_strategy_weight(signal.strategy)
                    scored.final_confidence = min(99, scored.final_confidence * weight)

                    # ── Regime staleness check ─────────────────────
                    # If regime data is >10 min old, penalize confidence
                    # (prevents acting on stale regime classification)
                    if signal.regime_time > 0:
                        regime_age = time.time() - signal.regime_time
                        if regime_age > Timing.REGIME_MAX_AGE_SECS:  # 10 minutes
                            stale_penalty = min(Penalties.REGIME_STALE_PENALTY_FLOOR, 1.0 - (regime_age - Timing.REGIME_MAX_AGE_SECS) / Timing.REGIME_STALE_PENALTY_WINDOW_SECS)
                            scored.final_confidence *= stale_penalty
                            logger.debug(
                                f"Regime stale ({regime_age/60:.0f}min) — "
                                f"confidence penalized {stale_penalty:.2f}x"
                            )

                    # Adaptive confidence threshold (adjusts to regime + win rate)
                    base_min = cfg.aggregator.get('min_confidence', 72)
                    ll_stats = learning_loop.get_stats()
                    recent_wr = ll_stats.get('win_rate', -1)
                    adaptive_min = regime_analyzer.get_adaptive_min_confidence(
                        base_min, recent_wr,
                        direction=getattr(signal.direction, 'value', str(signal.direction)),  # FIX #15
                    )

                    # BUG-8/10 FIX: HTF-aligned signals get the fear/greed bonus reversed.
                    # get_adaptive_min_confidence() penalises ALL signals by +5 when F&G<20
                    # and +3 when F&G>80. For an HTF-aligned SHORT during extreme fear
                    # (F&G=15), that +5 is wrong — extreme fear is the REASON to short,
                    # not a reason to block the trade. Similarly a LONG in extreme greed
                    # during a weekly bull trend should not be blocked by the greed penalty.
                    # Subtract the fear/greed portion of the penalty for aligned signals.
                    if _htf_aligned:
                        _fg = regime_analyzer.fear_greed
                        _fear_adj = 5 if _fg < 20 else (3 if _fg > 80 else 0)
                        if _fear_adj > 0:
                            adaptive_min = max(60, adaptive_min - _fear_adj)
                            logger.debug(
                                f"HTF-aligned {getattr(signal.direction, 'value', str(signal.direction))}: adaptive_min reduced "
                                f"by fear/greed bonus ({_fear_adj}) → {adaptive_min} "
                                f"(F&G={_fg})"
                            )

                    # ── PHASE 2 AUDIT FIX: Apply context multipliers BEFORE gate ──
                    # Previously _state_mult and _btc_ctx.confidence_mult were applied
                    # AFTER the adaptive_min gate, so they could never rescue/kill a
                    # borderline signal at the primary pass/fail decision point.

                    # Market-state multiplier (computed earlier at ~line 2490)
                    if _state_mult != 1.0:
                        scored.final_confidence = min(100, max(0, scored.final_confidence * _state_mult))
                        if _state_note:
                            logger.info(f"   🧠 {symbol}: market state pre-gate ×{_state_mult:.2f}")

                    # BTC news confidence multiplier (pre-gate)
                    try:
                        _btc_ctx_pregate = btc_news_intelligence.get_event_context()
                        if _btc_ctx_pregate.is_active and _btc_ctx_pregate.confidence_mult != 1.0:
                            scored.final_confidence = min(
                                100, scored.final_confidence * _btc_ctx_pregate.confidence_mult
                            )
                            logger.info(
                                f"   🗞️ {symbol}: BTC news pre-gate conf ×{_btc_ctx_pregate.confidence_mult:.2f}"
                            )
                        # ── Net news score bias adjustment ─────────────
                        # Aggregate of all recent headlines: boost longs in
                        # bullish regime, penalize in bearish, reduce size in mixed.
                        if _btc_ctx_pregate.is_active:
                            from config.constants import NewsIntelligence as _NI_net
                            _net_bias, _net_score = btc_news_intelligence.get_net_news_bias()
                            _sig_dir_pre = getattr(signal.direction, 'value', str(signal.direction)) if hasattr(signal, 'direction') else ""
                            if _net_bias == "BULLISH" and _sig_dir_pre == "LONG":
                                scored.final_confidence = min(
                                    100, scored.final_confidence * _NI_net.NET_SCORE_BULLISH_CONF_BOOST
                                )
                            elif _net_bias == "BEARISH" and _sig_dir_pre == "LONG":
                                scored.final_confidence = min(
                                    100, scored.final_confidence * _NI_net.NET_SCORE_BEARISH_CONF_PENALTY
                                )
                            elif _net_bias == "BEARISH" and _sig_dir_pre == "SHORT":
                                scored.final_confidence = min(
                                    100, scored.final_confidence * _NI_net.NET_SCORE_BULLISH_CONF_BOOST
                                )
                    except Exception:
                        _btc_ctx_pregate = None

                    # PHASE 3 AUDIT FIX (P3-5): Re-anchor the AI delta cap baseline
                    # AFTER the deterministic context multipliers (market-state, BTC news).
                    # These mults are technical/deterministic — not AI opinion — so they
                    # must not count against the ±12pt AI influence cap.  Previously
                    # _raw_confidence was stale from ~line 2467, causing the cap to
                    # incorrectly clip market-state boosts/penalties as if they were AI.
                    _raw_confidence = scored.final_confidence

                    if scored.final_confidence < adaptive_min:
                        # ── SIGNAL DEATH LOG ──────────────────────────
                        # Mandatory death reason — never lose track of why
                        all_reasons = ", ".join(death_reasons) if death_reasons else "base_confidence_too_low"
                        logger.info(
                            f"❌ Signal died | {symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                            f"| conf={scored.final_confidence:.0f}/{adaptive_min} "
                            f"| strategy={signal.strategy} "
                            f"| death={all_reasons}"
                        )
                        return

                    # ── Global AI confidence delta cap ────────────────────────
                    # Prevent stacked AI adjustments (market state × news × BTC context)
                    # from erasing a high-conviction signal.  Each layer is individually
                    # small, but together they can drop a 78-confidence signal to 44.
                    # Cap: total AI influence is clamped to ±12 pts from the base score
                    # that existed at the start of the soft-penalty pipeline.
                    _AI_DELTA_CAP = 12
                    _ai_delta = scored.final_confidence - _raw_confidence
                    if _ai_delta < -_AI_DELTA_CAP:
                        _capped = _raw_confidence - _AI_DELTA_CAP
                        logger.debug(
                            f"   🛡️ {symbol}: AI delta cap — total AI impact "
                            f"{_ai_delta:.1f} pts capped to −{_AI_DELTA_CAP} "
                            f"({scored.final_confidence:.0f} → {_capped:.0f})"
                        )
                        if 'confidence_adjustments' not in signal.raw_data:
                            signal.raw_data['confidence_adjustments'] = []
                        signal.raw_data['confidence_adjustments'].append({
                            'source': 'ai_delta_cap',
                            'delta': int(_capped - scored.final_confidence),
                            'detail': f'capped at base{-_AI_DELTA_CAP:+d}',
                        })
                        scored.final_confidence = _capped
                    elif _ai_delta > _AI_DELTA_CAP:
                        _capped = _raw_confidence + _AI_DELTA_CAP
                        logger.debug(
                            f"   🛡️ {symbol}: AI delta cap — total AI boost "
                            f"+{_ai_delta:.1f} pts capped to +{_AI_DELTA_CAP} "
                            f"({scored.final_confidence:.0f} → {_capped:.0f})"
                        )
                        scored.final_confidence = _capped

                    # ── Feature Store: compute standardized features ──
                    # Derive entry_timeframe from setup_class (not analysis tf)
                    from strategies.base import SETUP_CLASS_ENTRY_TF
                    signal.entry_timeframe = SETUP_CLASS_ENTRY_TF.get(signal.setup_class, '15m')
                    primary_tf = signal.entry_timeframe if signal.entry_timeframe in ohlcv_dict else '1h'
                    if primary_tf in ohlcv_dict:
                        features = feature_store.compute(
                            symbol, primary_tf, ohlcv_dict[primary_tf]
                        )
                    else:
                        features = None

                    # ── Probability Engine: P(win | all evidence) ──
                    evidence = {}
                    if features:
                        evidence = features.to_evidence_dict()
                    # Add signal-level evidence
                    # FIX EVIDENCE-KEYS: Match actual confluence note text from each module.
                    # Old checks used strings that never appeared together, making all keys False
                    # → absent-evidence cascade → p_win always below 0.45 gate.
                    _cl = [c.lower() for c in signal.confluence]
                    # R6-F2: _has_word must be defined BEFORE any evidence key that uses it.
                    # Defining it after its first use causes Python to raise
                    # "cannot access free variable '_has_word'" at runtime because the
                    # name is seen as a local (assigned later in the same scope) but the
                    # generator expression inside any() tries to close over it before the
                    # assignment executes.
                    def _has_word(word: str, text: str) -> bool:
                        """Check if word appears with word boundaries in text."""
                        idx = text.find(word)
                        while idx != -1:
                            before_ok = (idx == 0 or not text[idx - 1].isalpha())
                            after_ok = (idx + len(word) >= len(text)
                                        or not text[idx + len(word)].isalpha())
                            if before_ok and after_ok:
                                return True
                            idx = text.find(word, idx + 1)
                        return False

                    # htf_alignment: HTF guardrail writes '✅ Weekly trend aligned' or '✅ HTF aligned'
                    # SMC writes '✅ 4H Bullish BOS' — either indicates HTF structure support
                    evidence['htf_alignment'] = any(
                        ('weekly' in c and 'align' in c) or
                        ('htf align' in c) or
                        ('4h' in c and (_has_word('bos', c) or _has_word('choch', c) or 'bullish' in c or 'bearish' in c))
                        for c in _cl
                    )
                    # mtf_alignment: SMC writes '✅ 1H structure aligns bullish/bearish'
                    evidence['mtf_alignment'] = any(
                        ('1h' in c and 'align' in c) or ('mtf' in c and 'align' in c)
                        for c in _cl
                    )
                    # ob_fvg_confluence: SMC writes '✅ OB + FVG confluence' when both exist,
                    # but also '✅ Fresh Bullish OB' or '✅ FVG fill zone' separately
                    # Use word-boundary-aware check for 'ob' — raw substring matches inside
                    # "job", "knob", "global", "prob" etc.
                    evidence['ob_fvg_confluence'] = any(
                        _has_word('ob', c) and (_has_word('fvg', c) or 'order block' in c) or
                        ('ob + fvg' in c)
                        for c in _cl
                    )
                    evidence['liquidity_sweep'] = any('sweep' in c for c in _cl)
                    evidence['liq_cluster_overlap'] = any(
                        'liquidation cluster' in c or 'liq cluster' in c for c in _cl
                    )
                    evidence['multi_strategy_agree'] = len(confluence.agreeing_strategies) >= 2
                    # premium_discount_zone: equilibrium writes 'discount zone' or 'premium zone'
                    # SMC writes '✅ Buying in discount zone' or '✅ Selling in premium zone'
                    evidence['premium_discount_zone'] = any(
                        'discount' in c or 'premium' in c for c in _cl
                    )
                    # bar_confirmation: entry_refiner writes '✅ Price action confirmed'
                    evidence['bar_confirmation'] = any(
                        'confirm' in c or 'confirmed' in c for c in _cl
                    )
                    # volume_confirmation: volume analyzer writes notes about volume
                    evidence['volume_confirmation'] = any(
                        'volume' in c and ('high' in c or 'above' in c or 'spike' in c or '✅' in c)
                        for c in _cl
                    )
                    # killzone_active: time filter writes killzone notes
                    evidence['killzone_active'] = any('killzone' in c for c in _cl)

                    # V17: Populate negative evidence keys so Bayesian engine can penalize
                    evidence['counter_regime'] = (
                        (getattr(signal.direction, 'value', str(signal.direction)) == "LONG" and "BEAR" in regime_name.upper()) or
                        (getattr(signal.direction, 'value', str(signal.direction)) == "SHORT" and "BULL" in regime_name.upper())
                    )
                    evidence['low_volume'] = scored.volume_score < 35
                    evidence['dead_zone_time'] = any('dead zone' in c.lower() for c in signal.confluence)
                    evidence['weekend'] = any('saturday' in c.lower() or 'sunday' in c.lower() for c in signal.confluence)

                    # BUG-NEW-2 FIX: wire strategy_cold_streak (LR=-0.15) — last remaining
                    # missing key. Check alpha model's Bayesian weight for this strategy.
                    try:
                        _sw = alpha_model._get_weight(signal.strategy)
                        evidence['strategy_cold_streak'] = (
                            _sw.count >= 15 and _sw.ev < -0.05
                        )
                    except Exception:
                        evidence['strategy_cold_streak'] = False

                    # Sync engine-resolved evidence back into FeatureSet so the
                    # learning loop tracks these keys and can update their LR values.
                    if features:
                        features.htf_aligned              = evidence.get('htf_alignment', False)
                        features.mtf_aligned              = evidence.get('mtf_alignment', False)
                        features.ob_fvg_confluence        = evidence.get('ob_fvg_confluence', False)
                        features.liquidity_sweep_present  = evidence.get('liquidity_sweep', False)
                        features.liq_cluster_overlap      = evidence.get('liq_cluster_overlap', False)
                        features.bar_confirmation         = evidence.get('bar_confirmation', False)
                        features.multi_strategy_agree     = evidence.get('multi_strategy_agree', False)
                        features.premium_discount_zone    = evidence.get('premium_discount_zone', False)
                        features.counter_regime           = evidence.get('counter_regime', False)
                        features.is_dead_zone             = evidence.get('dead_zone_time', False)
                        features.strategy_on_cold_streak  = evidence.get('strategy_cold_streak', False)

                    prob_estimate = probability_engine.estimate(
                        strategy=signal.strategy,
                        regime=regime_name,
                        direction=getattr(signal.direction, 'value', str(signal.direction)),
                        confidence=scored.final_confidence,
                        rr_ratio=signal.rr_ratio,
                        evidence=evidence,
                        confluence_count=len(confluence.agreeing_strategies),
                    )

                    # ── Alpha Model: EV gate ───────────────────────
                    is_killzone = features.is_killzone if features else False
                    # FIX P2-C: Compute regime_alignment from actual regime filter result
                    # adj_conf captures the regime-adjusted confidence. The ratio tells us
                    # how aligned this signal is with the current regime (1.0 = perfectly aligned).
                    _base_conf_for_align = max(1.0, signal.confidence)
                    _regime_alignment_real = min(1.0, max(0.0, adj_conf / _base_conf_for_align))

                    alpha_score = alpha_model.evaluate(
                        strategy=signal.strategy,
                        p_win=prob_estimate.p_win,
                        rr_ratio=signal.rr_ratio,
                        regime=regime_name,
                        direction=getattr(signal.direction, 'value', str(signal.direction)),
                        is_killzone=is_killzone,
                        confluence_count=len(confluence.agreeing_strategies),
                        regime_alignment=_regime_alignment_real,  # FIX P2-C: real computed value
                        chop_strength=chop_str,  # Gap 1: continuous chop context
                    )

                    # NOTE: market_state mult now applied pre-gate (Phase 2 audit fix)

                    # ── BTC News Intelligence gate ─────────────────────────
                    # Consult the BTC event context to adjust altcoin signals
                    # based on WHY BTC is moving, not just THAT it's moving.
                    try:
                        _btc_ctx = btc_news_intelligence.get_event_context()
                        if _btc_ctx.is_active:
                            _sig_dir = getattr(signal.direction, 'value', str(signal.direction))  # "LONG" or "SHORT"

                            # Hard block check
                            if _btc_ctx.block_longs and _sig_dir == "LONG":
                                logger.info(
                                    f"   🗞️ {symbol}: LONG blocked by BTC news "
                                    f"[{_btc_ctx.event_type.value}] — {_btc_ctx.explanation[:60]}"
                                )
                                return

                            if _btc_ctx.block_shorts and _sig_dir == "SHORT":
                                logger.info(
                                    f"   🗞️ {symbol}: SHORT blocked by BTC news "
                                    f"[{_btc_ctx.event_type.value}]"
                                )
                                return

                            # Low-correlation filter: BTC_FUNDAMENTAL bullish wants alts
                            # that aren't just riding BTC — let them develop independently
                            if _btc_ctx.require_low_corr:
                                _feat_corr = features.btc_correlation if features else 0.7
                                if _feat_corr > 0.6:
                                    logger.info(
                                        f"   🗞️ {symbol}: skipped (BTC-corr={_feat_corr:.2f} > 0.6 "
                                        f"during BTC fundamental event — waiting for alt to diverge)"
                                    )
                                    return

                            # Confidence mult already applied pre-gate (Phase 2 audit fix).
                            # Just record the confluence note and size mult here.
                            if _btc_ctx.confidence_mult != 1.0:
                                signal.confluence.append(
                                    f"🗞️ BTC {_btc_ctx.event_type.value} "
                                    f"({_btc_ctx.direction}) conf×{_btc_ctx.confidence_mult:.2f}"
                                )
                                if 'confidence_adjustments' not in signal.raw_data:
                                    signal.raw_data['confidence_adjustments'] = []
                                signal.raw_data['confidence_adjustments'].append({
                                    'source': 'btc_news_event',
                                    'mult': _btc_ctx.confidence_mult,
                                    'event': _btc_ctx.event_type.value,
                                    'direction': _btc_ctx.direction,
                                })

                            # Stash size multiplier for portfolio engine
                            if _btc_ctx.reduce_size_mult < 1.0:
                                signal.raw_data['btc_news_size_mult'] = _btc_ctx.reduce_size_mult

                    except Exception as _bni_err:
                        logger.debug(f"BTC news intelligence gate error (non-fatal): {_bni_err}")

                    if not alpha_score.should_trade:
                        logger.info(
                            f"   🚫 {symbol}: Alpha rejected — {alpha_score.reject_reason} "
                            f"(EV={alpha_score.expected_value_r:.3f}R, P(win)={alpha_score.p_win:.3f})"
                        )
                        return

                    # FIX #4/#27: Daily trade limit check AFTER alpha grading.
                    # Only A/B grade signals consume the daily trade counter.
                    # C-grade signals are informational and should NOT block real trades.
                    # BUG-FIX v2.10: skip validate_signal during warmup — the counter was
                    # being incremented for every warmup signal, exhausting the daily limit
                    # (default 10) before publishing was even active. Result: RISK_LIMIT on
                    # every signal immediately after warmup ended.
                    if alpha_score.grade != "C" and not self._warmup_active:
                        approved, reject_reason = await risk_manager.validate_signal(scored)
                        if not approved:
                            logger.info(
                                f"❌ Signal died | {symbol} {getattr(signal.direction, 'value', str(signal.direction))} "
                                f"| reason=RISK_LIMIT | {reject_reason}"
                            )
                            return

                    # ── Macro regime penalty ──────────────────────────────────────────
                    # If institutional data shows VOLATILE environment (strong dollar,
                    # high VIX, rising yields), suppress LONG confidence and boost SHORT.
                    # Keep penalty small (max 5 pts) — macro is slow-moving context,
                    # not a hard block. Hard blocks are HTF guardrail's job.
                    # PHASE 2 AUDIT FIX: was modifying signal.confidence (never read
                    # again); now correctly modifies scored.final_confidence.
                    if _macro_regime == "VOLATILE" and getattr(signal.direction, 'value', str(signal.direction)) == "LONG":
                        scored.final_confidence = max(0, scored.final_confidence - 5)
                        logger.debug(f"   🌍 {symbol}: macro VOLATILE — LONG final_confidence -5")
                    elif _macro_regime == "RISK_ON" and getattr(signal.direction, 'value', str(signal.direction)) == "SHORT":
                        scored.final_confidence = max(0, scored.final_confidence - 3)
                        logger.debug(f"   🌍 {symbol}: macro RISK_ON — SHORT final_confidence -3")

                    # FIX TRANSITION-GATE: was blocking B and C at risk>0.6, killing most
                    # valid signals during choppy/transitioning markets. Only block C-grade
                    # signals at very high transition risk (>0.80).
                    # FIX P5: Exempt breakout transitions — COMPRESSION→EXPANSION is the
                    # best entry point, not a reason to suppress.
                    try:
                        if (_transition_risk > 0.80 and alpha_score.grade == "C"
                                and _transition_type != "breakout"):
                            logger.info(
                                f"   ⚠️  {symbol}: very high transition risk ({_transition_risk:.2f}) "
                                f"[{_transition_type}] — C-grade suppressed post-score"
                            )
                            return
                    except NameError:
                        pass  # _transition_risk/_transition_type not set (market state engine unavailable)

                    # Fix W4: re-run time filter with REAL grade after full scoring
                    # This catches C/B signals slipping through in dead zones via grade_est
                    try:
                        time_blocked_real, _, _ = time_filter.evaluate(alpha_score.grade)
                        if time_blocked_real:
                            logger.info(
                                f"   ⏰ Signal died | {symbol} | reason=TIME_BLOCK_POST_SCORE "
                                f"| grade={alpha_score.grade}"
                            )
                            return
                    except Exception:
                        pass

                    # ── Execution Quality Gate ─────────────────────────
                    # Separates "good setup" from "good trade."  Evaluates
                    # execution-specific factors (session, trigger quality,
                    # spread, whale alignment, entry position, volume) as a
                    # group.  When too many execution killers stack up, the
                    # signal is hard-blocked regardless of setup confidence.
                    try:
                        from analyzers.execution_gate import execution_gate as _exec_gate
                        from signals.context_contracts import (
                            build_execution_context,
                            enrich_setup_context_with_eq,
                        )
                        _dir_exec = getattr(signal.direction, 'value', str(signal.direction))
                        if getattr(signal, 'raw_data', None) is None:
                            signal.raw_data = {}
                        _raw_exec = signal.raw_data

                        # Session context
                        _session_name_exec = time_filter.get_session_name()
                        _is_killzone_exec = time_filter.is_killzone()
                        _is_dead_zone_exec = 'dead zone' in _session_name_exec.lower()
                        _is_weekend_exec = datetime.now(timezone.utc).weekday() >= 5

                        # EQ zone from earlier assessment (survives in scope)
                        _eq_zone_exec = ''
                        _eq_depth_exec = 0.0
                        try:
                            _eq_zone_exec = eq_result.zone
                            _eq_depth_exec = eq_result.zone_depth
                        except (NameError, AttributeError):
                            pass

                        # Volume context from raw_data
                        _vol_ctx_exec = _raw_exec.get('trigger_quality_volume_context', 'NORMAL')
                        _vol_quality_ctx = _raw_exec.get('volume_quality_context', _vol_ctx_exec)

                        # Trade type from raw_data
                        _trade_type_exec = _raw_exec.get('regime_trade_type', '')
                        _trade_type_strength = _raw_exec.get('local_structure_score', 0)
                        _vol_regime_exec = "UNKNOWN"
                        _transition_type_exec = "stable"
                        _transition_risk_exec = 0.0
                        try:
                            _vol_regime_exec = _vol_regime
                        except NameError:
                            pass
                        try:
                            _transition_type_exec = _transition_type
                            _transition_risk_exec = _transition_risk
                        except NameError:
                            pass

                        _exec_context = build_execution_context(
                            signal,
                            session_name=_session_name_exec,
                            is_killzone=_is_killzone_exec,
                            is_dead_zone=_is_dead_zone_exec,
                            is_weekend=_is_weekend_exec,
                            eq_zone=_eq_zone_exec,
                            eq_zone_depth=_eq_depth_exec,
                            volume_context=_vol_quality_ctx,
                            volume_score=scored.volume_score,
                            derivatives_score=scored.derivatives_score,
                            sentiment_score=scored.sentiment_score,
                            volatility_regime=_vol_regime_exec,
                            transition_type=_transition_type_exec,
                            transition_risk=_transition_risk_exec,
                            trade_type=_trade_type_exec,
                            trade_type_strength=_trade_type_strength,
                        )

                        signal.execution_context = _exec_context
                        _raw_exec['execution_context'] = _exec_context
                        _raw_exec['spread_bps'] = _exec_context.get('liquidity', {}).get('spread_bps', 0.0)
                        _raw_exec['whale_buy_ratio'] = _exec_context.get('whales', {}).get('buy_ratio', 0.5)
                        _raw_exec['whale_aligned'] = _exec_context.get('whales', {}).get('aligned')
                        signal.setup_context = enrich_setup_context_with_eq(
                            getattr(signal, 'setup_context', None) or _raw_exec.get('setup_context'),
                            eq_zone=_eq_zone_exec,
                            eq_distance=_eq_depth_exec,
                        )
                        _raw_exec['setup_context'] = signal.setup_context

                        _exec_assessment = _exec_gate.evaluate(
                            context=_exec_context,
                            setup_context=signal.setup_context,
                            direction=_dir_exec,
                            grade=alpha_score.grade,
                            confidence=scored.final_confidence,
                            symbol=symbol,
                        )

                        # Store execution assessment in raw_data for transparency
                        _raw_exec['execution_score'] = _exec_assessment.execution_score
                        _raw_exec['execution_block_threshold'] = _exec_assessment.block_threshold_used
                        _raw_exec['execution_factors'] = _exec_assessment.factors
                        _raw_exec['execution_bad_factors'] = _exec_assessment.bad_factors
                        if _exec_assessment.kill_combo:
                            _raw_exec['execution_kill_combo'] = _exec_assessment.kill_combo
                        _raw_exec['execution_context_snapshot'] = _exec_assessment.context_snapshot
                        _exec_event = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "score": round(_exec_assessment.execution_score, 2),
                            "block_threshold": round(_exec_assessment.block_threshold_used, 2),
                            "trigger_quality": _exec_context.get("trigger", {}).get("score"),
                            "volume_score": _exec_context.get("liquidity", {}).get("volume_score"),
                            "volume_context": _exec_context.get("liquidity", {}).get("volume_context"),
                            "volatility_regime": _exec_context.get("market", {}).get("volatility_regime"),
                            "kill_combo": _exec_assessment.kill_combo,
                        }
                        signal.execution_history.append(_exec_event)
                        signal.execution_history = signal.execution_history[-EG.EXECUTION_HISTORY_MAX_ENTRIES:]
                        _raw_exec.setdefault('execution_history', []).append(_exec_event)
                        _raw_exec['execution_history'] = _raw_exec['execution_history'][-EG.EXECUTION_HISTORY_MAX_ENTRIES:]

                        if _exec_assessment.should_block:
                            _exec_event["action"] = "BLOCK"
                            logger.info(
                                f"⛔ Signal died | {symbol} {_dir_exec} "
                                f"| reason=EXECUTION_GATE | {_exec_assessment.reason}"
                            )
                            for _note in _exec_assessment.notes:
                                logger.info(f"   {_note}")
                            # Record in diagnostic death log
                            try:
                                diagnostic_engine.record_signal_death(
                                    symbol=symbol,
                                    direction=_dir_exec,
                                    strategy=signal.strategy,
                                    kill_reason="EXECUTION_GATE",
                                    rr=signal.rr_ratio if hasattr(signal, 'rr_ratio') else 0,
                                    confidence=scored.final_confidence,
                                    regime=signal.regime if hasattr(signal, 'regime') else "UNKNOWN",
                                    setup_class=_raw_exec.get('setup_class', 'intraday'),
                                )
                            except Exception:
                                pass
                            # Record near-miss + block count for feedback metrics
                            try:
                                from analyzers.near_miss_tracker import near_miss_tracker
                                near_miss_tracker.record_block()
                                if _exec_assessment.is_near_miss:
                                    _entry_p = _raw_exec.get('entry_price', 0)
                                    _tp1_p = _raw_exec.get('tp1')
                                    _tp_p = _tp1_p if _tp1_p is not None else _raw_exec.get('take_profit', 0)
                                    _sl_p = _raw_exec.get('stop_loss', 0)
                                    near_miss_tracker.record_near_miss(
                                        symbol=symbol,
                                        direction=_dir_exec,
                                        strategy=signal.strategy,
                                        execution_score=_exec_assessment.execution_score,
                                        block_threshold=_exec_assessment.block_threshold_used,
                                        factors=_exec_assessment.factors,
                                        bad_factors=_exec_assessment.bad_factors,
                                        kill_combo=_exec_assessment.kill_combo,
                                        grade=alpha_score.grade,
                                        confidence=scored.final_confidence,
                                        entry_price=float(_entry_p) if _entry_p else 0.0,
                                        tp_price=float(_tp_p) if _tp_p else 0.0,
                                        sl_price=float(_sl_p) if _sl_p else 0.0,
                                    )
                            except Exception:
                                pass
                            # Push to SSE stream for live dashboard
                            try:
                                from web.app import push_notification
                                push_notification({
                                    "type": "exec_gate",
                                    "symbol": symbol,
                                    "direction": _dir_exec,
                                    "action": "BLOCK",
                                    "score": round(_exec_assessment.execution_score, 1),
                                    "factors": _exec_assessment.factors,
                                    "bad_factors": _exec_assessment.bad_factors,
                                    "kill_combo": _exec_assessment.kill_combo,
                                    "reason": _exec_assessment.reason,
                                    "grade": alpha_score.grade,
                                    "strategy": signal.strategy,
                                    "is_near_miss": _exec_assessment.is_near_miss,
                                })
                            except Exception:
                                pass
                            # Rollback dedup so symbol isn't suppressed
                            signal_aggregator.unmark_signal(symbol, _dir_exec)
                            return

                        if _exec_assessment.should_penalize:
                            _exec_event["action"] = "PENALIZE"
                            scored.final_confidence *= _exec_assessment.penalty_mult
                            signal.confluence.append(_exec_assessment.reason)
                            logger.info(
                                f"   ⚠️ {symbol}: execution gate penalty "
                                f"×{_exec_assessment.penalty_mult:.2f} "
                                f"(exec_score={_exec_assessment.execution_score:.0f})"
                            )
                            # Push penalize event to SSE stream
                            try:
                                from web.app import push_notification
                                push_notification({
                                    "type": "exec_gate",
                                    "symbol": symbol,
                                    "direction": _dir_exec,
                                    "action": "PENALIZE",
                                    "score": round(_exec_assessment.execution_score, 1),
                                    "factors": _exec_assessment.factors,
                                    "bad_factors": _exec_assessment.bad_factors,
                                    "penalty_mult": _exec_assessment.penalty_mult,
                                    "reason": _exec_assessment.reason,
                                    "grade": alpha_score.grade,
                                    "strategy": signal.strategy,
                                })
                            except Exception:
                                pass
                        else:
                            _exec_event["action"] = "PASS"
                            # Log passing assessment for transparency
                            logger.debug(
                                f"   ✅ {symbol}: execution gate passed "
                                f"(exec_score={_exec_assessment.execution_score:.0f})"
                            )

                        # Record pass in near-miss tracker for feedback metrics
                        if not _exec_assessment.should_block:
                            try:
                                from analyzers.near_miss_tracker import near_miss_tracker
                                near_miss_tracker.record_pass()
                            except Exception:
                                pass
                    except Exception as _exec_err:
                        logger.warning(
                            "Execution gate error (non-fatal) for %s: %s",
                            symbol,
                            _exec_err,
                        )

                    # ── Portfolio Engine: position sizing (A/B only) ──
                    # BEH-2/3: C grades never run portfolio sizing — keeps slots clean
                    entry_mid = (signal.entry_low + signal.entry_high) / 2
                    if alpha_score.grade == "C":
                        # Create a dummy approved sizing so the pipeline doesn't crash
                        from core.portfolio_engine import SizingDecision
                        sizing = SizingDecision(
                            approved=True, position_size_usdt=0, risk_amount_usdt=0,
                            risk_pct=0, kelly_fraction=0, leverage_suggested=1,
                        )
                    else:
                        sizing = await portfolio_engine.size_position(
                        symbol=symbol,
                        direction=getattr(signal.direction, 'value', str(signal.direction)),
                        strategy=signal.strategy,
                        entry_price=entry_mid,
                        stop_loss=signal.stop_loss,
                        kelly_fraction=alpha_score.kelly_fraction,
                        p_win=prob_estimate.p_win,
                        rr_ratio=signal.rr_ratio,
                        sector=getattr(signal, 'sector', ''),
                            # PHASE 2 FIX (VOL-DEFAULT): tier-based fallback instead of
                            # 0.03 universal default. T3 alts can have vol 3-5x baseline.
                            symbol_volatility=(features.realized_vol_20 if (features and features.realized_vol_20 > 0)
                                else (0.025 if getattr(signal, 'tier', 2) == 1
                                      else 0.05 if getattr(signal, 'tier', 2) == 2
                                      else 0.08)),  # T1/T2/T3 fallback
                        )  # end size_position

                    if alpha_score.grade != "C" and not sizing.approved:
                        logger.info(
                            f"   🚫 {symbol}: Portfolio rejected — {sizing.reject_reason}"
                        )
                        return

                    # ── T7: Circuit breaker drawdown size reduction ──────
                    try:
                        _cb_mult = risk_manager.circuit_breaker.position_size_multiplier
                        if _cb_mult < 1.0:
                            sizing.position_size_usdt *= _cb_mult
                            sizing.risk_amount_usdt *= _cb_mult
                            logger.info(
                                f"   🔻 {symbol}: CB drawdown reduction {_cb_mult:.1f}x "
                                f"→ ${sizing.position_size_usdt:,.0f}"
                            )
                    except Exception:
                        pass

                    # ── Chop position size reduction ──────────────
                    from signals.regime_thresholds import get_chop_size_multiplier
                    chop_size_mult = get_chop_size_multiplier(chop_str)
                    if chop_size_mult < 0.95:
                        sizing.position_size_usdt *= chop_size_mult
                        sizing.risk_amount_usdt *= chop_size_mult
                        logger.info(
                            f"   📉 {symbol}: Chop size reduction {chop_size_mult:.2f}x → ${sizing.position_size_usdt:,.0f}"
                        )

                    # Sunday A-grade: apply 0.6x size reduction flagged by time filter.
                    # Thin weekend liquidity = same confidence floor but smaller exposure.
                    _sunday_scale = getattr(signal, '_sunday_size_scale', None)
                    if _sunday_scale is not None:
                        sizing.position_size_usdt *= _sunday_scale
                        sizing.risk_amount_usdt *= _sunday_scale
                        logger.info(
                            f"   📅 {symbol}: Sunday size reduction {_sunday_scale:.1f}x → ${sizing.position_size_usdt:,.0f}"
                        )

                    # ══════════════════════════════════════════════
                    # ⭐ END QUANT PIPELINE
                    # ══════════════════════════════════════════════

                    # ── BEH-1: Upgrade tracker gate for C grade signals ──────
                    # upgrade_tracker.evaluate() was never called — the entire
                    # Phase 1+2 Bayesian C-grade gating system was dead code.
                    if alpha_score.grade == "C":
                        try:
                            from signals.upgrade_tracker import upgrade_tracker as _ut_gate
                            send_to_user, up_score = _ut_gate.evaluate(
                                scored,        # ScoredSignal object (first positional arg)
                                signal_id=0,   # pre-save placeholder
                            )
                            if not send_to_user:
                                logger.info(
                                    f"   📋 {symbol}: C grade filtered by upgrade_tracker "
                                    f"(UP={up_score:.2f} < {0.45}) — admin only"
                                )
                                return  # Don't publish to signals channel
                        except Exception as _ut_e:
                            logger.debug(f"upgrade_tracker gate error: {_ut_e}")
                            # On error, allow C signal through (fail-open)

                    # ── BEH-2/3: Skip portfolio sizing for C grade signals ────
                    # C signals are informational — sizing is meaningless and
                    # portfolio slot checks must not block real A/B signals.

                    # ── I1: Warmup gate — observation only ────────────────────
                    # FIX: gate moved BEFORE db.save_signal. The old code saved the signal
                    # first, then returned. Warmup signals were written to the signals table
                    # as real entries, polluting the learning loop training data and the
                    # daily signal counter loaded on restart via load_daily_count().
                    if self._warmup_active:
                        logger.info(
                            f"⏳ WARMUP: {symbol} {signal.direction} signal scored "
                            f"({alpha_score.grade}) — not saving or publishing (warmup active)"
                        )
                        return

                    # ── Save to database ───────────────────────────
                    # FIX BUG-3: 'closes' was referenced at publish_price but never
                    # assigned in this scope. Extract close prices from the already-
                    # loaded OHLCV so publish_price records the exact market price.
                    _closes_tf = ohlcv_dict.get('1h') or ohlcv_dict.get('15m') or ohlcv_dict.get('4h') or []
                    closes = [float(c[4]) for c in _closes_tf] if _closes_tf else []

                    sig_data = {
                        'symbol': signal.symbol,
                        'direction': getattr(signal.direction, 'value', str(signal.direction)),
                        'strategy': signal.strategy,
                        'confidence': scored.final_confidence,
                        'entry_low': signal.entry_low,
                        'entry_high': signal.entry_high,
                        'stop_loss': signal.stop_loss,
                        'tp1': signal.tp1,
                        'tp2': signal.tp2,
                        'tp3': signal.tp3,
                        'rr_ratio': signal.rr_ratio,
                        'setup_class':  signal.setup_class,
                        'entry_timeframe': signal.entry_timeframe,
                        'timeframe': signal.entry_timeframe,  # back-compat DB column
                        'regime': regime_name,
                        'sector': signal.sector,
                        'tier': signal.tier,
                        'confluence': scored.all_confluence,
                        'confluence_strategies': confluence.agreeing_strategies,
                        'raw_scores': {
                            'technical':   scored.technical_score,
                            'volume':      scored.volume_score,
                            'derivatives': scored.derivatives_score,
                            'orderflow':   scored.orderflow_score,
                            'sentiment':   scored.sentiment_score,
                            # Raw signal data for logic panel
                            'has_ob':      getattr(signal, 'raw_data', {}).get('has_ob', False) if hasattr(signal, 'raw_data') and signal.raw_data else False,
                            'has_fvg':     getattr(signal, 'raw_data', {}).get('has_fvg', False) if hasattr(signal, 'raw_data') and signal.raw_data else False,
                            'has_sweep':   getattr(signal, 'raw_data', {}).get('has_sweep', False) if hasattr(signal, 'raw_data') and signal.raw_data else False,
                            'wave_type':   getattr(signal, 'raw_data', {}).get('wave_type', '') if hasattr(signal, 'raw_data') and signal.raw_data else '',
                            'wyckoff_event': getattr(signal, 'raw_data', {}).get('wyckoff_event', '') if hasattr(signal, 'raw_data') and signal.raw_data else '',
                            'htf_structure': getattr(signal, 'raw_data', {}).get('htf_structure', '') if hasattr(signal, 'raw_data') and signal.raw_data else '',
                            'adx':         getattr(signal, 'raw_data', {}).get('adx', 0) if hasattr(signal, 'raw_data') and signal.raw_data else 0,
                            'rsi':         getattr(signal, 'raw_data', {}).get('rsi', 0) if hasattr(signal, 'raw_data') and signal.raw_data else 0,
                            'funding_rate': getattr(signal, 'raw_data', {}).get('funding_rate', 0) if hasattr(signal, 'raw_data') and signal.raw_data else 0,
                            'vol_ratio':   getattr(signal, 'raw_data', {}).get('vol_ratio', 0) if hasattr(signal, 'raw_data') and signal.raw_data else 0,
                        },
                        # Fix L1+L2+A1: quant fields stored at top level for easy retrieval at outcome time
                        'p_win': prob_estimate.p_win,
                        'ev_r': alpha_score.expected_value_r,
                        'alpha_grade': alpha_score.grade,  # PHASE 2 AUDIT FIX: store actual alpha model grade (used by execution gating)
                        'agg_grade': scored.grade,  # Aggregator grade (matches Telegram display) — for audit trail
                        'evidence': evidence,   # Fix L2: full evidence dict stored
                        # Fix M1: sizing stored at top level so engine can retrieve at entry
                        'risk_amount': sizing.risk_amount_usdt,
                        'position_size': sizing.position_size_usdt,
                        # FIX 3A: publish_price — market price at exact moment of publication.
                        # Critical for AI to calculate entry-zone-distance (fill rate predictor).
                        # Use the closes[-1] from the OHLCV already loaded in this scan cycle.
                        'publish_price': float(closes[-1]) if closes is not None and len(closes) > 0 else None,
                        # Legacy quant blob
                        'quant': {
                            'p_win': prob_estimate.p_win,
                            'ev_r': alpha_score.expected_value_r,
                            'alpha': alpha_score.total_alpha,
                            'alpha_grade': alpha_score.grade,  # PHASE 2 AUDIT FIX: actual alpha model grade
                            'kelly': alpha_score.kelly_fraction,
                            'sharpe_est': alpha_score.sharpe_estimate,
                            'position_size': sizing.position_size_usdt,
                            'risk_amount': sizing.risk_amount_usdt,
                            'vol_adj': sizing.vol_adjustment,
                        },
                        # CTX-2 / CTX-7: Market context fields for richer signal cards
                        # features may be None for non-primary-TF paths — guard all with fallback
                        'funding_rate':   float(features.funding_rate)   if features else 0.0,
                        'funding_annual': float(features.funding_annualized) if features else 0.0,
                        'atr_pct':        float(features.atr_pct)        if features else 0.0,
                        'vol_percentile': float(features.vol_percentile) if features else 0.5,
                        'vol_expanding':  bool(features.is_expanding)    if features else False,
                        'adx':            float(features.adx)            if features else 0.0,
                        # AI influence: net confidence delta from AI context adjustments.
                        # Baseline is post-market-state-penalty (_raw_confidence); this
                        # captures only the contribution of news / BTC context / sentiment.
                        # Stored here for post-hoc attribution: "did AI help or hurt?"
                        'ai_influence':   round(scored.final_confidence - _raw_confidence, 1),
                        # Market state at signal time — enables per-state attribution by ParamTuner
                        'market_state':   _mkt_state_name,
                        # Phase 9-12 audit fix: persist per-source vote breakdown
                        # so adaptive_weights can compute true per-source accuracy
                        # instead of using overall signal outcome as a proxy.
                        'ensemble_votes': {
                            v.source: {'value': int(v.value), 'weight': v.weight}
                            for v in _ev_verdict.votes
                        } if '_ev_verdict' in locals() and hasattr(_ev_verdict, 'votes') else {},
                    }
                    # ── Pre-save gates ─────────────────────────────
                    # All publish-blocking checks run BEFORE db.save_signal()
                    # so that signals which will never be published are not
                    # written to the database. Previously these ran AFTER save,
                    # leaving phantom "WATCHING / NOT SENT" rows in Signal History.

                    # Helper: direction string needed by dedup unmark on gate-kill
                    _dir_str = getattr(signal.direction, 'value', str(signal.direction))

                    # Gate 1: Circuit breaker — check again in case breaker
                    # tripped during the scan/scoring pipeline
                    if risk_manager.circuit_breaker.is_active:
                        logger.warning(
                            f"🚫 Circuit breaker tripped — {symbol} signal "
                            f"not saved or published"
                        )
                        # PHASE 3 FIX (P3-2): rollback dedup mark so this symbol
                        # isn't suppressed for the rest of the dedup window
                        signal_aggregator.unmark_signal(symbol, _dir_str)
                        return

                    # Gate 2: Directional stability — don't rapid-fire the opposite side
                    _directional_block_reason = await self._directional_stability_block_reason(
                        symbol,
                        signal,
                    )
                    if _directional_block_reason:
                        logger.info(
                            f"🧭 {symbol}: signal blocked by directional stability gate — "
                            f"{_directional_block_reason}"
                        )
                        signal_aggregator.unmark_signal(symbol, _dir_str)
                        return

                    # Gate 3: Health validation
                    from core.health_monitor import health_monitor as _hm
                    _current_p = 0.0
                    try:
                        from core.price_cache import price_cache as _pc
                        _current_p = _pc.get(symbol) or 0.0
                    except Exception:
                        pass
                    _hm_ok, _hm_issues = _hm.validate_signal(
                        symbol=symbol,
                        direction=_dir_str,
                        entry_low=signal.entry_low,
                        entry_high=signal.entry_high,
                        stop_loss=signal.stop_loss,
                        tp1=signal.tp1,
                        tp2=signal.tp2,
                        tp3=signal.tp3,
                        rr_ratio=signal.rr_ratio,
                        strategy=signal.strategy,
                        current_price=_current_p,
                    )
                    if not _hm_ok:
                        logger.warning(
                            f"   🚨 {symbol}: signal BLOCKED by health monitor — "
                            f"{len(_hm_issues)} validation failures (see above)"
                        )
                        signal_aggregator.unmark_signal(symbol, _dir_str)
                        return  # skip save & publish

                    # Gate 4: Burst throttle pre-check — avoid saving signals
                    # that would be immediately throttled by the publisher
                    from signals.signal_publisher import signal_publisher
                    _grade = getattr(alpha_score, 'grade', None) or scored.grade
                    if signal_publisher.would_throttle(_grade):
                        logger.info(
                            f"⏸️ {symbol}: signal not saved — would be throttled "
                            f"(grade={_grade})"
                        )
                        signal_aggregator.unmark_signal(symbol, _dir_str)
                        return

                    # Gate 5: Post-startup extension filter
                    # After warmup ends, block signals for POST_STARTUP_FILTER_SECS where
                    # the recent 4-bar 1h price move is already ≥ threshold in the signal
                    # direction.  Prevents "catch-up" signals published at extended prices
                    # when moves happened before the bot was running.
                    # Only active during the post-startup window; no impact after that.
                    # Note: falls back to 4h bars when 1h is unavailable, which widens the
                    # measured window to ~16h — intentionally conservative for that case.
                    _psfilt_active = (
                        self._warmup_end_time > 0
                        and time.time() - self._warmup_end_time < Timing.POST_STARTUP_FILTER_SECS
                        and _current_p > 0
                    )
                    if _psfilt_active:
                        try:
                            _chk_bars = ohlcv_dict.get('1h') or ohlcv_dict.get('4h') or []
                            if len(_chk_bars) >= 8:
                                _c_now  = float(_chk_bars[-1][4])
                                _c_4ago = float(_chk_bars[-5][4])
                                _atr_period = 14
                                _atr = (
                                    sum(float(_chk_bars[i][2]) - float(_chk_bars[i][3])
                                        for i in range(-_atr_period, 0)) / _atr_period
                                ) if len(_chk_bars) >= _atr_period else 0.0
                                if _c_4ago > 0 and _atr > 0 and _c_now > 0:
                                    _4h_chg = (_c_now - _c_4ago) / _c_4ago * 100
                                    _atr_pct = _atr / _c_now * 100
                                    _ext_thresh = max(
                                        _atr_pct * Timing.POST_STARTUP_EXT_ATR_MULT,
                                        Timing.POST_STARTUP_EXT_FLOOR_PCT,
                                    )
                                    _is_extended = (
                                        (_dir_str == "LONG"  and _4h_chg >  _ext_thresh) or
                                        (_dir_str == "SHORT" and _4h_chg < -_ext_thresh)
                                    )
                                    if _is_extended:
                                        logger.info(
                                            f"⏳ {symbol}: {_dir_str} blocked by post-startup "
                                            f"extension filter — 4h move {_4h_chg:+.1f}% "
                                            f"exceeds ±{_ext_thresh:.1f}% threshold "
                                            f"(ATR={_atr_pct:.2f}%)"
                                        )
                                        signal_aggregator.unmark_signal(symbol, _dir_str)
                                        return
                        except Exception:
                            pass  # data issue — never silently block a signal

                    candidate = PreparedPublishCandidate(
                        symbol=symbol,
                        signal=signal,
                        scored=scored,
                        sig_data=sig_data,
                        alpha_score=alpha_score,
                        prob_estimate=prob_estimate,
                        sizing=sizing,
                        confluence=confluence,
                        regime_name=regime_name,
                    )
                    if return_publish_candidate:
                        return candidate
                    await self._publish_prepared_candidate(candidate)

                except Exception as e:
                    logger.error(f"Signal processing error ({symbol}): {e}")
                    return

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Scan error for {symbol}: {e}")
            finally:
                # Record scan timing for diagnostic engine
                try:
                    diagnostic_engine.record_scan_timing(
                        symbol=symbol,
                        duration_ms=(time.time() - _scan_t0) * 1000,
                    )
                except Exception:
                    pass

    # ── Periodic tasks ────────────────────────────────────────

    async def _update_regime(self):
        """Update market regime analysis + market state engine"""
        try:
            old_regime = regime_analyzer.regime.value
            await regime_analyzer.update()
            new_regime = regime_analyzer.regime.value
            if old_regime != new_regime:
                logger.info(f"🔄 REGIME CHANGE: {old_regime} → {new_regime}")
                alpha_model.reset_regime()
                # Fix G7: also reset probability engine regime counters and performance tracker
                try:
                    from core.probability_engine import probability_engine
                    probability_engine.reset_regime()
                except Exception:
                    pass
                try:
                    from governance.performance_tracker import performance_tracker
                    performance_tracker.reset_regime(new_regime)
                except Exception:
                    pass
                try:
                    await telegram_bot.send_signals_text(
                        text=f"🔄 <b>Regime Change</b>\n{old_regime} → <b>{new_regime}</b>",
                    )
                except Exception:
                    pass
                # FIX L2: invalidate execution engine's regime stats cache on regime change
                try:
                    from core.execution_engine import execution_engine
                    execution_engine.invalidate_regime_cache()
                except Exception:
                    pass

                # FIX-6: Mid-trade TP upgrade on CHOPPY → trending regime transition.
                # When market breaks out of range, active trades entered during CHOPPY
                # have TPs capped at EQ and range boundaries. Those caps are now wrong —
                # the range has resolved into a trend. Upgrade TP2 and TP3 by the same
                # multipliers the trending regime would have applied at entry.
                _choppy_to_trend = (
                    old_regime == "CHOPPY"
                    and new_regime in ("BULL_TREND", "BEAR_TREND", "VOLATILE")
                )
                if _choppy_to_trend:
                    try:
                        from signals.outcome_monitor import outcome_monitor as _om_fix6
                        _upgraded = 0
                        for _sig_id, _tracked in list(_om_fix6.get_active_signals().items()):
                            # Only upgrade ACTIVE trades (entry reached, not yet TP1)
                            from signals.outcome_monitor import SignalState as _SS
                            if _tracked.state not in (_SS.ACTIVE,):
                                continue
                            _entry = _tracked.entry_price or ((_tracked.entry_low + _tracked.entry_high) / 2)
                            _risk = abs(_entry - _tracked.stop_loss)
                            if _risk <= 0:
                                continue
                            # Apply the new regime's tp2_mult to the remaining distance
                            _new_tp2_mult = 1.40 if new_regime in ("BULL_TREND","BEAR_TREND") else 1.30
                            _is_long = _tracked.direction == "LONG"
                            _old_tp2_dist = abs(_tracked.tp2 - _entry)
                            _new_tp2_dist = _old_tp2_dist * _new_tp2_mult
                            if _is_long:
                                _tracked.tp2 = _entry + _new_tp2_dist
                                if _tracked.tp3:
                                    _tracked.tp3 = _entry + _new_tp2_dist * 1.30
                            else:
                                _tracked.tp2 = _entry - _new_tp2_dist
                                if _tracked.tp3:
                                    _tracked.tp3 = _entry - _new_tp2_dist * 1.30
                            _upgraded += 1
                            logger.info(
                                f"📈 FIX-6: TP upgraded for #{_sig_id} {_tracked.symbol} "
                                f"{_tracked.direction} — CHOPPY→{new_regime}: "
                                f"TP2 {_old_tp2_dist/_risk:.1f}R → {_new_tp2_dist/_risk:.1f}R"
                            )
                        if _upgraded:
                            logger.info(f"📈 FIX-6: Upgraded TPs on {_upgraded} active trade(s) after CHOPPY→{new_regime}")
                    except Exception as _f6_err:
                        logger.debug(f"FIX-6 TP upgrade failed (non-fatal): {_f6_err}")

                # T12: Re-validate pending signals after regime change.
                # Signals published during CHOPPY may have wrong TP/SL for BEAR_TREND.
                # Alert user about stale signals and cancel ones that conflict with new regime.
                try:
                    from signals.invalidation_monitor import invalidation_monitor as _im
                    _pending = _im.get_pending_signals() if hasattr(_im, 'get_pending_signals') else []
                    _stale_count = 0
                    for _sig in _pending:
                        _sig_regime = _sig.get('regime', old_regime) if isinstance(_sig, dict) else old_regime
                        if _sig_regime != new_regime:
                            _stale_count += 1
                    if _stale_count > 0:
                        logger.warning(
                            f"T12: {_stale_count} pending signals were published under {old_regime} "
                            f"regime — now in {new_regime}. Consider reviewing open setups."
                        )
                        try:
                            await telegram_bot.send_signals_text(
                                text=(
                                    f"⚠️ <b>Regime changed to {new_regime}</b>\n"
                                    f"{_stale_count} pending signal(s) were set up under {old_regime} conditions.\n"
                                    f"Review open setups — TP/SL levels may need adjustment."
                                ),
                            )
                        except Exception:
                            pass
                except Exception as _t12_err:
                    logger.debug(f"T12 regime revalidation check failed: {_t12_err}")
            else:
                logger.debug(f"Regime: {new_regime} (unchanged)")

            # ── Also refresh Market State Engine ────────────────────
            try:
                from analyzers.market_state_engine import market_state_engine
                mkt_state = await market_state_engine.get_state()
                logger.info(
                    f"🧠 Market State: {mkt_state.state.value} | "
                    f"bias={mkt_state.direction_bias} | "
                    f"vol_ratio={mkt_state.vol_ratio:.2f} | "
                    f"compress={mkt_state.compression_bars}bars"
                )
            except Exception as _mse_err:
                logger.debug(f"Market state engine update failed: {_mse_err}")

        except Exception as e:
            logger.error(f"Regime update failed: {e}")

    async def _update_rotation(self):
        """Update sector rotation"""
        try:
            await rotation_tracker.update()
        except Exception as e:
            logger.error(f"Rotation update failed: {e}")

    async def _check_strategy_health(self):
        """Fix A4: alert if any active strategy has gone silent for too long.
        BUG-9 FIX: _strategy_last_signal is keyed by _strat_key_map values (e.g. 'smc',
        'breakout'). The health check iterated the same short-key list so the lookup
        was correct — BUT strategies not in _strat_key_map never got a key written,
        so they would silently appear as 'silent since start_time'. This fix explicitly
        uses _strat_key_map to derive keys, skipping anything not registered.
        """
        try:
            now = time.time()
            if now - self._last_health_check < self._strategy_health_alert_interval:
                return
            self._last_health_check = now

            # Use _strat_key_map values (the canonical short keys that _strategy_last_signal uses)
            # Deduplicate since some display names map to the same key (e.g. HarmonicDetector + HarmonicPattern)
            all_strats = list(dict.fromkeys(self._strat_key_map.values()))
            silent_strats = []
            for s in all_strats:
                if not cfg.is_strategy_enabled(s):
                    continue
                last = self._strategy_last_signal.get(s, self._start_time)
                hours_silent = (now - last) / 3600
                # Only alert for strategies that have been running long enough
                if hours_silent > 8 and now - self._start_time > 3600 * 4:
                    silent_strats.append((s, hours_silent))

            if silent_strats:
                # Cross-check: is silence expected in the current regime?
                try:
                    from analyzers.regime import regime_analyzer as _hra
                    _current_regime = _hra.regime.value if _hra.regime else "UNKNOWN"
                except Exception:
                    _current_regime = "UNKNOWN"
                # Strategies that are expected to be silent in this regime
                _regime_expected_silent = {
                    "VOLATILE":       {"mean_reversion", "range_scalper", "momentum", "wyckoff",
                                       "institutional_breakout", "geometric", "harmonic"},
                    "VOLATILE_PANIC": {"mean_reversion", "range_scalper", "momentum", "wyckoff",
                                       "institutional_breakout", "geometric", "harmonic",
                                       "elliott_wave", "price_action"},
                    "BEAR_TREND":     {"mean_reversion"},
                    "CHOPPY":         {"momentum", "institutional_breakout"},
                }.get(_current_regime, set())

                # Split into real alerts vs expected
                _real_alerts   = [(s, h) for s, h in silent_strats if s not in _regime_expected_silent]
                _regime_normal = [(s, h) for s, h in silent_strats if s in _regime_expected_silent]

                if not _real_alerts:
                    # All silences are regime-expected — log only, no Telegram spam
                    logger.info(
                        f"A4 health: {len(_regime_normal)} strategies silent but expected in "
                        f"{_current_regime}: {[s for s,_ in _regime_normal]}"
                    )
                    return  # Skip Telegram message

                lines = [
                    "⚠️ <b>Strategy Health Alert</b>",
                    f"<i>Regime: {_current_regime} — these strategies should be active but aren't:</i>",
                    "",
                ]
                for s, h in _real_alerts:
                    lines.append(f"  📉 <b>{s}</b> — no raw signal for {h:.0f}h")
                if _regime_normal:
                    lines += ["",
                        f"<i>ℹ️ {len(_regime_normal)} other strategies are silent but expected in {_current_regime} regime.</i>"]
                lines += ["", "<i>Check logs for scan errors or config issues.</i>"]
                try:
                    await telegram_bot.send_admin_text(text="\n".join(lines))
                except Exception:
                    pass
                logger.warning(f"A4 health alert: silent strategies: {[s for s,_ in silent_strats]}")
        except Exception as e:
            logger.debug(f"Strategy health check error: {e}")

    async def _check_daily_summary(self):
        """Send daily summary at configured hour"""
        now = datetime.now(timezone.utc)
        summary_hour = cfg.telegram.get('daily_summary_hour', 8)

        # Check if it's time for daily summary (within first minute of the hour)
        if (now.hour == summary_hour and
                now.minute == 0 and
                time.time() - self._last_daily_summary > 3600):
            await telegram_bot.send_daily_summary()
            signal_aggregator.reset_daily_counts()
            # BUG-NEW-1 FIX: reset_daily() existed but was never called here.
            # _daily_loss_pct accumulated across calendar days, meaning a 4% Monday
            # loss + 4% Tuesday loss = 8% running total triggering the 8% CB limit
            # on Tuesday even though each day was individually within limits.
            await risk_manager.circuit_breaker.reset_daily()
            logger.info("Daily counters reset (signal_aggregator + circuit_breaker)")
            self._last_daily_summary = time.time()

            # D4-FIX: daily DB backup at midnight reset
            try:
                await db.backup()
            except Exception as _bkp_err:
                logger.debug(f"Daily backup skipped (non-fatal): {_bkp_err}")

    # ── Observability methods ─────────────────────────────────

    async def _send_welcome_telegram(self, regime: str, num_symbols: int, strategies: list):
        """Send compact engine notice to admin channel only.
        Silently skipped when no dedicated admin channel is configured — avoids
        polluting the signals channel with engine chatter.
        """
        try:
            _signals_id = telegram_bot.signals_chat_id
            _admin_id   = telegram_bot.admin_chat_id
            if not _admin_id or _admin_id == _signals_id:
                logger.info("Engine started (no admin channel configured — skipping notice)")
                return
            now_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            strat_list = ", ".join(strategies)
            text = (
                f"🔧 <b>Engine started</b>  {now_str}\n"
                f"Regime: {regime}  ·  Symbols: {num_symbols}\n"
                f"Strategies: {strat_list}"
            )
            await telegram_bot.send_text(_admin_id, text)
        except Exception as e:
            logger.debug(f"Admin startup note failed: {e}")

    async def _console_heartbeat(self):
        """Detailed console heartbeat every 5 minutes"""
        try:
            uptime = time.time() - self._start_time
            regime = regime_analyzer.regime.value
            from signals.outcome_monitor import outcome_monitor

            # Learning loop stats
            ll_stats = learning_loop.get_stats()

            # Portfolio state
            pf_state = portfolio_engine.get_state()

            logger.info("")
            logger.info("╔═══════════════ HEARTBEAT ═══════════════╗")
            logger.info(f"║  Uptime     : {self._format_uptime(uptime):<28}║")
            logger.info(f"║  Regime     : {regime:<28}║")
            logger.info(f"║  Chop       : {regime_analyzer.chop_strength:.2f}{'':<25}║")
            # Show range + EQ if in chop
            if regime_analyzer.chop_strength > 0.40 and regime_analyzer.range_high > 0:
                logger.info(
                    f"║  Range      : {fmt_price(regime_analyzer.range_low)} — "
                    f"{fmt_price(regime_analyzer.range_high)}{'':<6}║"
                )
                logger.info(f"║  EQ         : {fmt_price(regime_analyzer.range_eq):<28}║")
            logger.info(f"║  Scans      : {self._scan_count:<28}║")
            logger.info(f"║  Signals    : {self._signal_count:<28}║")
            logger.info(f"║  Tracking   : {outcome_monitor.get_active_count()} active outcomes{'':<14}║")
            logger.info(f"║  Pending    : {invalidation_monitor.pending_count} pre-entry signals{'':<11}║")
            # Execution engine status
            from core.execution_engine import execution_engine
            exec_summary = execution_engine.get_status_summary()
            logger.info(f"║  Execution  : {exec_summary:<28}║")
            logger.info(f"║  Positions  : {pf_state.position_count} ({pf_state.net_exposure:+,.0f} net){'':<10}║")
            logger.info(f"║  Learning   : {ll_stats.get('total_trades', 0)} trades recorded{'':<12}║")
            if ll_stats.get('total_trades', 0) > 0:
                logger.info(
                    f"║  Win Rate   : {ll_stats['win_rate']:.1%} | "
                    f"Avg R: {ll_stats['avg_r']:+.3f}{'':<8}║"
                )
            # Whale session stats
            from signals.whale_aggregator import whale_aggregator
            w_stats = whale_aggregator.get_session_stats()
            if w_stats['total_events'] > 0:
                logger.info(
                    f"║  Whales     : {w_stats['total_events']} events | "
                    f"Buy ${w_stats['total_buy_volume']/1e6:.1f}M "
                    f"Sell ${w_stats['total_sell_volume']/1e6:.1f}M ║"
                )
            # API telemetry
            api_stats = api.get_request_stats()
            logger.info(
                f"║  API        : {api_stats['total_requests']} reqs | "
                f"avg {api_stats['avg_latency_ms']:.0f}ms | "
                f"{api_stats['total_retries']} retries{'':<4}║"
            )
            # Network monitor stats
            from core.network_monitor import network_monitor
            net_stats = network_monitor.get_stats()
            net_state = net_stats['state']
            net_offline = net_stats['total_offline_time']
            if net_offline > 0 or net_state != 'ONLINE':
                logger.info(
                    f"║  Network    : {net_state} | "
                    f"offline: {net_offline:.0f}s | "
                    f"drops: {net_stats['offline_count']}{'':<4}║"
                )
            logger.info("╚═════════════════════════════════════════╝")
            logger.info("")
        except Exception as e:
            logger.debug(f"Heartbeat error: {e}")

    async def _telegram_heartbeat(self):
        """Send a heartbeat status to Telegram every hour"""
        try:
            uptime = time.time() - self._start_time
            regime = regime_analyzer.regime.value
            from signals.outcome_monitor import outcome_monitor
            ll_stats = learning_loop.get_stats()

            text = (
                f"💓 <b>Heartbeat</b> — {self._format_uptime(uptime)}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Regime: <b>{regime}</b> (chop: {regime_analyzer.chop_strength:.2f})\n"
                f"🔍 Scans: {self._scan_count} | Signals: {self._signal_count}\n"
                f"📡 Tracking: {outcome_monitor.get_active_count()} | "
                f"Pending: {invalidation_monitor.pending_count}\n"
            )

            # ── Active trades section ─────────────────────
            active_sigs = {
                sid: t for sid, t in outcome_monitor.get_active_signals().items()
                if t.state.value in ("ACTIVE", "BE_ACTIVE", "TP1_HIT")
                and t.entry_price is not None
            }
            if active_sigs:
                text += "\n<b>OPEN TRADES</b>\n"
                for sid, t in list(active_sigs.items())[:5]:  # cap at 5
                    current_r = outcome_monitor._calc_pnl_r(
                        t, getattr(t, 'last_price', t.entry_price)
                    )
                    pnl_emoji = "📈" if current_r >= 0 else "📉"
                    be_marker = " 🛡BE" if t.be_stop else ""
                    text += (
                        f"{pnl_emoji} {t.symbol} {t.direction}  "
                        f"<b>{current_r:+.2f}R</b>{be_marker}\n"
                    )
                text += "<i>Tap 📈 Live P&L on any trade card for details.</i>\n"

            # Sector rotation summary
            rot_summary = rotation_tracker.get_rotation_summary()
            if rot_summary.get('hot'):
                hot_names = [s for s, _ in rot_summary['hot']]
                text += f"\n🔥 Hot sectors: {', '.join(hot_names)}\n"
            if rot_summary.get('cold'):
                cold_names = [s for s, _ in rot_summary['cold']]
                text += f"❄️ Cold sectors: {', '.join(cold_names)}\n"
            if ll_stats.get('real_trades', ll_stats.get('total_trades', 0)) > 0:
                _rt = ll_stats.get('real_trades', ll_stats['total_trades'])
                _exp = ll_stats.get('expired_count', 0)
                # Only show WR when we have enough closed trades to be meaningful
                # (open positions don't count — they're not in the WR denominator)
                _wr_display = (
                    f"WR: {ll_stats['win_rate']:.1%}"
                    if _rt >= 5
                    else f"WR: — (need {5 - _rt} more closed)"
                )
                text += (
                    f"🧠 {_wr_display} | "
                    f"R: {ll_stats['total_r']:+.1f} ({_rt} trades"
                )
                if _exp > 0:
                    text += f", {_exp} expired"
                text += ")\n"
            # Flush batch-expired signals as a single summary (not N individual messages)
            if self._pending_expired_batch:
                _expired = self._pending_expired_batch[:]
                self._pending_expired_batch = []
                self._last_expiry_flush = time.time()
                if len(_expired) == 1:
                    sym, dr, sid = _expired[0]
                    _exp_text = (
                        f"⏰\n\n📊 <b>EXPIRED — {sym} {dr}</b>\n"
                        f"Result: 0.0R\n"
                        f"<i>⏰ Signal expired — entry zone never reached</i>"
                    )
                else:
                    _exp_lines = "\n".join(
                        f"  • {sym} {dr}  #{sid}" for sym, dr, sid in _expired
                    )
                    _exp_text = (
                        f"⏰ <b>{len(_expired)} signals expired</b>\n\n"
                        f"{_exp_lines}\n\n"
                        f"<i>Entry zones not reached. Edge has diminished.</i>"
                    )
                try:
                    await telegram_bot.send_signals_text(
                        text=_exp_text,
                        disable_web_page_preview=True,
                    )
                except Exception:
                    pass

            text += f"\n<i>Next heartbeat in 1 hour</i>"

            await telegram_bot.send_signals_text(text=text)
        except Exception as e:
            logger.debug(f"Telegram heartbeat failed: {e}")

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format seconds into human-readable uptime"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        return f"{s}s"

    # ── Control methods ───────────────────────────────────────

    async def _handle_signal_outcome(
        self, signal_id, outcome, pnl_r, strategy, message_id, message,
        symbol="", direction="", confidence=0, stop_loss=0.0
    ):
        """
        Called by OutcomeMonitor when a signal resolves.
        Feeds into performance tracker, learning loop, and notifies via Telegram.
        """
        try:
            won = outcome == "WIN"

            # PHASE 2 FIX (P3-B): Record loss time for re-entry cooldown
            if outcome == "LOSS" and symbol and direction:
                async with self._get_loss_cooldown_lock():
                    self._loss_cooldown[(symbol, direction)] = time.time()
                logger.info(f"⏸️  Re-entry cooldown started: {symbol} {direction} for {self._reentry_cooldown_mins}min")

            # Pull execution-quality fields from DB so the performance tracker
            # can classify the loss (GOOD_LOSS / BORDERLINE / BAD_EXECUTION /
            # SYSTEM_FAILURE) without a second round-trip later.
            _exec_entry_status: Optional[str] = None
            _exec_max_r: Optional[float] = None
            try:
                _eq_row = await db.get_signal(signal_id)
                if _eq_row:
                    _exec_entry_status = _eq_row.get("entry_status")
                    _exec_max_r = _eq_row.get("max_r")
            except Exception:
                pass

            # Record in performance tracker → updates EWMA weights
            if outcome == "WIN":
                await performance_tracker.record_outcome(strategy, "WIN", pnl_r=pnl_r)
                risk_manager.circuit_breaker.record_win()
            elif outcome == "LOSS":
                await performance_tracker.record_outcome(
                    strategy, "LOSS", pnl_r=pnl_r,
                    entry_status=_exec_entry_status, max_r=_exec_max_r,
                )
                # PHASE 1 FIX (P1-C): pass signal_id for dedup guard in circuit_breaker
                # BUG-1 FIX: pass current_capital so CB can compute peak-equity drawdown
                # internally — the check was dead when actual_loss_pct was never passed.
                await risk_manager.circuit_breaker.record_loss(
                    loss_r=abs(pnl_r) if pnl_r else 1.0,
                    signal_id=str(signal_id),
                    current_capital=portfolio_engine._capital,
                )
            elif outcome == "BREAKEVEN":
                await performance_tracker.record_outcome(strategy, "BREAKEVEN", pnl_r=0.0)
                risk_manager.circuit_breaker.record_breakeven()  # FIX P3-E: BE != win

            try:
                from analyzers.near_miss_tracker import near_miss_tracker
                if outcome == "WIN":
                    near_miss_tracker.record_pass_outcome(won=True)
                elif outcome == "LOSS":
                    near_miss_tracker.record_pass_outcome(won=False)
            except Exception:
                pass

            # FIX P1-C: also call risk_manager.record_outcome() to update position history
            # and ensure circuit_breaker dedup runs through the manager path too.
            # FIX AUDIT-10: pass current_capital so peak-equity drawdown check uses fresh value.
            await risk_manager.record_outcome(
                pnl_r=pnl_r, symbol=symbol or "", signal_id=str(signal_id),
                current_capital=portfolio_engine._capital,
            )

            # ⭐ Feed learning loop with real signal data (Fix L1+L2+L4+L6)
            try:
                # Reuse the row already fetched above for execution quality;
                # fall back to a fresh fetch if that attempt failed.
                sig_row = _eq_row if _eq_row is not None else await db.get_signal(signal_id)
                if sig_row:
                    symbol = symbol or sig_row.get("symbol", "")
                    direction = direction or sig_row.get("direction", "")
                    confidence = confidence or sig_row.get("confidence", 0)
                    # Fix L1: real p_win from DB, not hardcoded 0.55
                    p_win_real = sig_row.get("p_win") or 0.55
                    # Fix L4+L6: original rr_ratio from DB, not realized pnl_r
                    rr_original = sig_row.get("rr_ratio") or (abs(pnl_r) if pnl_r != 0 else 2.0)
                    # Fix L2: evidence dict from DB
                    evidence_raw = sig_row.get("evidence") or "{}"
                    import json as _json
                    evidence_dict = _json.loads(evidence_raw) if isinstance(evidence_raw, str) else evidence_raw
                else:
                    p_win_real = 0.55
                    rr_original = abs(pnl_r) if pnl_r != 0 else 2.0
                    evidence_dict = {}

                learning_loop.record_trade(
                    signal_id=signal_id,
                    symbol=symbol,
                    strategy=strategy,
                    direction=direction,
                    regime=regime_analyzer.regime.value,
                    confidence=confidence,
                    p_win_predicted=p_win_real,      # Fix L1
                    rr_ratio=rr_original,             # Fix L4+L6
                    won=won,
                    pnl_r=pnl_r,
                    outcome=outcome,
                    evidence=evidence_dict,            # Fix L2
                )
            except Exception as e:
                logger.debug(f"Learning loop update failed: {e}")

            # Save outcome to database
            await db.update_signal_outcome(signal_id, outcome, pnl_r)

            # Clean up execution tracking
            from core.execution_engine import execution_engine
            execution_engine.untrack(signal_id)

            # PHASE 1 FIX (NO-RESTART): Remove from persistent open_positions table
            try:
                await db.close_open_position(signal_id)
            except Exception:
                pass

            # Fix L3: Release portfolio position slot so limits update
            # Bug 7 fix: also update capital so subsequent Kelly sizing uses real balance
            try:
                closed_pos = await portfolio_engine.close_position(signal_id)
                if closed_pos is not None:
                    # Convert pnl_r (R-multiples) back to USDT using the position's own risk
                    pnl_usdt = pnl_r * closed_pos.risk_usdt
                    # T1-FIX: deduct accumulated funding fees so the Bayesian learning
                    # loop sees the true net P&L, not just the price-based P&L.
                    funding_cost = portfolio_engine.calculate_funding_cost(closed_pos)
                    if funding_cost > 0:
                        logger.info(
                            f"💸 Funding fee deducted: ${funding_cost:.2f} "
                            f"({closed_pos.symbol} held "
                            f"{(time.time() - closed_pos.opened_at)/3600:.1f}h)"
                        )
                        pnl_usdt -= funding_cost
                    new_capital = portfolio_engine._capital + pnl_usdt
                    portfolio_engine.update_capital(max(1.0, new_capital))
                    # T3-FIX: keep the circuit breaker's intraday peak up to date
                    # so the peak-equity drawdown check has a valid reference.
                    risk_manager.circuit_breaker.update_peak_capital(portfolio_engine._capital)
                    logger.info(
                        f"💰 Capital updated: ${portfolio_engine._capital:,.2f} "
                        f"(trade PnL: {pnl_r:+.2f}R / ${pnl_usdt:+,.2f})"
                    )
            except Exception as _pe:
                logger.debug(f"portfolio close_position failed: {_pe}")

            # ⭐ Update the original message silently (edit in place)
            if message_id:
                await telegram_bot.edit_signal_outcome(message_id, outcome, pnl_r, message)

            # V14: Send SL-specific message for LOSS outcomes
            if outcome == "LOSS" and stop_loss > 0 and message_id:
                try:
                    await telegram_bot.update_signal_sl_hit(signal_id, stop_loss)
                except Exception:
                    pass

            # ⭐ Send a NEW push notification for the close (so it surfaces in chat)
            outcome_emoji = {"WIN": "✅", "LOSS": "🔴", "BREAKEVEN": "➡️"}.get(outcome, "📊")
            pnl_str = f"{pnl_r:+.2f}R" if pnl_r != 0 else "0.0R"
            sym_str = symbol or f"#{signal_id}"
            dir_str = f" {direction}" if direction else ""

            if outcome == "EXPIRED":
                # Batch EXPIRED messages — accumulate and flush as one summary
                # Avoids: 6 individual "EXPIRED" messages cluttering the chat
                self._pending_expired_batch.append((sym_str, direction, signal_id))
                logger.debug(f"📦 Queued expiry #{signal_id} {sym_str} — batch now {len(self._pending_expired_batch)}")
            else:
                await telegram_bot.send_signals_text(
                    text=(
                        f"{outcome_emoji} <b>{outcome}</b> — <b>{sym_str}{dir_str}</b>\n"
                        f"Result: <b>{pnl_str}</b>\n"
                        f"<i>{message}</i>"
                    ),
                    reply_to_message_id=message_id,
                    disable_web_page_preview=True,
                )

            logger.info(
                f"📊 Outcome recorded: signal #{signal_id} → {outcome} "
                f"({pnl_r:+.1f}R) | {symbol} {direction} | Strategy: {strategy}"
            )

        except Exception as e:
            logger.error(f"Error handling signal outcome: {e}", exc_info=True)

    async def _force_scan(self, symbol: str = None):
        """Force an immediate scan (called from Telegram /scan)"""
        if symbol:
            await self._scan_symbol(symbol)
        else:
            symbols = scanner.get_all_symbols()[:20]  # Scan top 20
            tasks = [self._scan_symbol(s, return_publish_candidate=True) for s in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            candidates = [
                result for result in results
                if isinstance(result, PreparedPublishCandidate)
            ]
            if candidates:
                regime_name = getattr(regime_analyzer.regime, 'value', 'UNKNOWN')
                ranked_candidates, skipped_candidates = self._rank_publish_candidates(
                    candidates,
                    regime_name=regime_name,
                )
                for skipped in skipped_candidates:
                    signal_aggregator.unmark_signal(skipped.symbol, skipped.direction)
                for candidate in ranked_candidates:
                    await self._publish_prepared_candidate(candidate)

    # ── Invalidation monitor callbacks ────────────────────────

    async def _handle_signal_invalidated(self, signal_id, reason, message_id):
        """Signal invalidated before entry — price hit SL before entry zone."""
        try:
            await db.update_signal_outcome(signal_id, "INVALIDATED", 0.0)
            if message_id:
                # Use rich formatter for a clear, instructional message
                sig_row = await db.get_signal(signal_id)
                symbol = sig_row.get('symbol', '?') if sig_row else '?'
                direction = sig_row.get('direction', '?') if sig_row else '?'
                strategy = sig_row.get('strategy', '') if sig_row else ''
                rich_text = formatter.format_invalidated(symbol, direction, reason, strategy)
                await telegram_bot.send_reply(message_id, rich_text)

            # FIX P1-D: feed invalidation into learning loop as weak negative
            try:
                sig_row = sig_row if message_id else await db.get_signal(signal_id)
                if sig_row:
                    import json as _json
                    _ev = sig_row.get('evidence') or '{}'
                    _ev_dict = _json.loads(_ev) if isinstance(_ev, str) else _ev
                    learning_loop.record_trade(
                        signal_id=signal_id,
                        symbol=sig_row.get('symbol', ''),
                        strategy=sig_row.get('strategy', ''),
                        direction=sig_row.get('direction', ''),
                        regime=sig_row.get('regime') or regime_analyzer.regime.value,
                        confidence=sig_row.get('confidence', 0),
                        p_win_predicted=sig_row.get('p_win') or 0.55,
                        rr_ratio=sig_row.get('rr_ratio') or 2.0,
                        won=False,
                        pnl_r=0.0,
                        outcome="INVALIDATED",
                        evidence=_ev_dict,
                    )
            except Exception as _ll_e:
                logger.debug(f"Learning loop invalidation update failed: {_ll_e}")

            # Remove from outcome monitor (no longer needs post-entry tracking)
            from signals.outcome_monitor import outcome_monitor
            from core.execution_engine import execution_engine
            outcome_monitor.remove_signal(signal_id)
            execution_engine.untrack(signal_id)
            logger.info(f"❌ Signal #{signal_id} invalidated: {reason}")
        except Exception as e:
            logger.error(f"Error handling invalidation: {e}")

    async def _handle_entry_reached(self, signal_id, price, message_id):
        """Entry zone reached — transition card to ENTRY_ACTIVE with live trade buttons."""
        try:
            # update_signal_entry: edits the card in place + swaps in active keyboard + sends push reply
            await telegram_bot.update_signal_entry(signal_id, price)
            logger.info(f"✅ Signal #{signal_id} entry reached at {fmt_price(price)}")

            # Position created at EXECUTE state (ENTER NOW), not zone touch.
            # Zone touch = price arrived. EXECUTE = all triggers confirmed = actual entry.
            # This ensures P&L tracks from when you actually go to TradingView to trade.
        except Exception as e:
            logger.error(f"Error handling entry reached: {e}")

    async def _handle_signal_conflict(self, signal_id, conflicting_direction, message_id):
        """Counter-signal fired on same symbol — send full conflict advisory."""
        try:
            sig_row = await db.get_signal(signal_id)
            symbol = sig_row.get('symbol', '?') if sig_row else '?'
            original_dir = sig_row.get('direction', '?') if sig_row else '?'
            strategy = sig_row.get('strategy', '') if sig_row else ''
            if message_id:
                rich_text = formatter.format_conflict(
                    symbol, original_dir, conflicting_direction, strategy
                )
                await telegram_bot.send_reply(message_id, rich_text)
            logger.info(f"⚠️ Signal #{signal_id} conflicted by {conflicting_direction}")
        except Exception as e:
            logger.error(f"Error handling conflict: {e}")

    async def _handle_signal_expired(self, signal_id, message_id):
        """Signal timed out without entry — send rich expiry message."""
        try:
            await db.update_signal_outcome(signal_id, "EXPIRED", 0.0)
            sig_row = await db.get_signal(signal_id)
            symbol = sig_row.get('symbol', '?') if sig_row else '?'
            direction = sig_row.get('direction', '?') if sig_row else '?'
            strategy = sig_row.get('strategy', '') if sig_row else ''
            if message_id:
                rich_text = formatter.format_expired(symbol, direction, strategy, signal_id=signal_id)
                await telegram_bot.send_reply(message_id, rich_text)

            # FIX P1-D: feed expiry into learning loop as weak negative
            try:
                if sig_row:
                    import json as _json
                    _ev = sig_row.get('evidence') or '{}'
                    _ev_dict = _json.loads(_ev) if isinstance(_ev, str) else _ev
                    learning_loop.record_trade(
                        signal_id=signal_id,
                        symbol=sig_row.get('symbol', ''),
                        strategy=sig_row.get('strategy', ''),
                        direction=sig_row.get('direction', ''),
                        regime=sig_row.get('regime') or regime_analyzer.regime.value,
                        confidence=sig_row.get('confidence', 0),
                        p_win_predicted=sig_row.get('p_win') or 0.55,
                        rr_ratio=sig_row.get('rr_ratio') or 2.0,
                        won=False,
                        pnl_r=0.0,
                        outcome="EXPIRED",
                        evidence=_ev_dict,
                    )
            except Exception as _ll_e:
                logger.debug(f"Learning loop expiry update failed: {_ll_e}")

            # Remove from outcome monitor
            from signals.outcome_monitor import outcome_monitor
            from core.execution_engine import execution_engine
            outcome_monitor.remove_signal(signal_id)
            execution_engine.untrack(signal_id)
            logger.info(f"⏰ Signal #{signal_id} expired")
        except Exception as e:
            logger.error(f"Error handling expiry: {e}")

    async def _handle_signal_aging(
        self,
        signal_id: int,
        pct_elapsed: float,
        remaining_mins: float,
        message_id: Optional[int],
    ):
        """
        Gap 5: Staleness warning — fired at 75% of the signal's validity window.
        Sends a Telegram alert so the trader knows to re-verify the setup before acting.
        """
        try:
            if not message_id:
                return
            sig_row = await db.get_signal(signal_id)
            if not sig_row:
                return
            symbol    = sig_row.get('symbol', '?')
            direction = sig_row.get('direction', '?')
            strategy  = sig_row.get('strategy', '')
            rich_text = formatter.format_aging(
                symbol=symbol,
                direction=direction,
                strategy=strategy,
                pct_elapsed=pct_elapsed,
                remaining_mins=remaining_mins,
                signal_id=signal_id,
            )
            await telegram_bot.send_reply(message_id, rich_text)
            logger.info(
                f"⚠️ Aging alert sent for signal #{signal_id} "
                f"({pct_elapsed:.0f}% elapsed, {remaining_mins:.0f}min remaining)"
            )
        except Exception as e:
            logger.error(f"Error handling signal aging: {e}")

    async def _handle_trade_checkin(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        current_r: float,
        max_r: float,
        state: str,
        tp1: float,
        tp2: float,
        tp3,
        be_stop,
        stop_loss: float,
        message_id: int,
    ):
        """
        Hourly silent edit on the active trade card showing live P&L.
        No new message sent — updates in place so no spam.
        Trader can tap 📊 Update button for full details any time.
        """
        try:
            from utils.formatting import fmt_price
            dir_emoji = "🟢" if direction == "LONG" else "🔴"
            pnl_emoji = "📈" if current_r >= 0 else "📉"
            active_sl = fmt_price(be_stop) if be_stop else fmt_price(stop_loss)
            be_label = "BE Stop" if be_stop else "SL"

            # Next target
            if state == "BE_ACTIVE":
                next_target = f"TP2: <code>{fmt_price(tp2)}</code>"
            else:
                next_target = f"TP1: <code>{fmt_price(tp1)}</code>"

            now_utc = __import__('datetime').datetime.now(
                __import__('datetime').timezone.utc
            ).strftime("%H:%M UTC")

            text = (
                f"🟡 <b>TRADE ACTIVE — {symbol} {direction}</b>\n"
                f"{dir_emoji} {pnl_emoji} P&amp;L: <b>{current_r:+.2f}R</b>  "
                f"(peak: {max_r:+.2f}R)\n\n"
                f"Next: {next_target}\n"
                f"{be_label}: <code>{active_sl}</code>\n\n"
                f"<i>Updated {now_utc}</i>"
            )

            if message_id:
                # Silent edit on existing message — no notification
                await telegram_bot.edit_signals_text(
                    message_id=message_id,
                    text=text,
                    reply_markup=keyboards.signal_entry_active(signal_id, symbol),
                )
                logger.debug(f"⏱ Check-in edit: #{signal_id} {symbol} {current_r:+.2f}R")
            else:
                # PHASE 3 FIX (CHECKIN-DARK): API trade with no Telegram message_id —
                # send a new message so trader knows the position is alive.
                await telegram_bot.send_signals_text(text=text)
                logger.debug(f"⏱ Check-in new msg: #{signal_id} {symbol} {current_r:+.2f}R (no msg_id)")
        except Exception as e:
            logger.debug(f"Check-in failed (non-critical): {e}")

    async def _handle_signal_state_change(
        self,
        signal_id: int,
        state: str,
        symbol: str,
        direction: str,
        entry_price: float,
        be_stop: float,
        strategy: str,
        message_id: int,
    ):
        """
        Outcome monitor state changes — currently handles TP1_HIT.
        Called by OutcomeMonitor.on_state_change when TP1 is hit.

        Sends a clear instructional follow-up:
          - Congratulates on TP1
          - Shows new breakeven stop level
          - Tells trader to let the rest run
        """
        try:
            if state == "TP1_HIT":
                # update_signal_tp1: edits card + sends push reply with BE stop
                await telegram_bot.update_signal_tp1(signal_id, entry_price, be_stop)
                logger.info(
                    f"🎯 Signal #{signal_id} TP1 hit — "
                    f"BE stop now at {fmt_price(be_stop)}"
                )
        except Exception as e:
            logger.error(f"Error handling state change {state}: {e}")

    async def _handle_signal_approaching(
        self, signal_id: int, price: float, pct_away: float, message_id: int
    ):
        """
        Price is within threshold of entry zone — fire a heads-up alert.
        FIX-SL-TP: Uses PendingSignal (in-memory) for all price levels.
        Never re-reads from DB for SL/TP to avoid column mismatch issues.
        """
        try:
            from signals.invalidation_monitor import invalidation_monitor
            sig = invalidation_monitor._pending.get(signal_id)

            if not sig:
                # Signal not in memory — try DB as fallback
                sig_row = await db.get_signal(signal_id)
                if not sig_row:
                    return
                symbol    = sig_row.get('symbol', '?')
                direction = sig_row.get('direction', '?')
                entry_low  = float(sig_row.get('entry_low', 0) or 0)
                entry_high = float(sig_row.get('entry_high', 0) or 0)
                stop_loss  = float(sig_row.get('stop_loss', 0) or 0)
                tp1        = float(sig_row.get('tp1', 0) or 0)
                tp2        = float(sig_row.get('tp2', 0) or 0)
                tp3_val    = sig_row.get('tp3')
                tp3        = float(tp3_val) if tp3_val else None
                rr         = float(sig_row.get('rr_ratio', 0) or 0)
                strategy   = sig_row.get('strategy', '')
                grade      = sig_row.get('alpha_grade', 'B')
            else:
                # Use in-memory PendingSignal — guaranteed correct price levels
                symbol    = sig.symbol
                direction = sig.direction
                entry_low  = sig.entry_low
                entry_high = sig.entry_high
                stop_loss  = sig.stop_loss
                tp1        = sig.tp1
                tp2        = sig.tp2
                tp3        = sig.tp3
                rr         = sig.rr_ratio
                strategy   = sig.strategy
                grade      = sig.grade

            dir_emoji = "🟢" if direction == "LONG" else "🔴"

            # Determine candle duration hint from strategy
            from tg.formatter import STRATEGY_EXPIRY_CANDLES, TF_MINUTES
            from strategies.base import BaseStrategy
            _tf_hint = "1h"
            try:
                # Get the timeframe used by this strategy for a human duration hint
                from config.loader import cfg as _cfg
                _strat_cfg = getattr(_cfg.strategies, strategy.lower().replace(' ', '_'), None)
                if _strat_cfg:
                    _tf_hint = getattr(_strat_cfg, 'timeframe', '1h')
            except Exception:
                pass
            _tf_mins  = TF_MINUTES.get(_tf_hint, 60)
            _candle_duration = f"{_tf_mins}min" if _tf_mins < 60 else f"{_tf_mins//60}h"

            # Build price level strings (only show if valid price levels)
            _sl_str  = f"SL:    <code>{fmt_price(stop_loss)}</code>\n" if stop_loss > 0 else ""
            _tp1_str = f"TP1:   <code>{fmt_price(tp1)}</code>\n" if tp1 > 0 else ""
            _tp2_str = f"TP2:   <code>{fmt_price(tp2)}</code>\n" if tp2 > 0 else ""
            _tp3_str = f"TP3:   <code>{fmt_price(tp3)}</code>\n" if tp3 and tp3 > 0 else ""
            _rr_str  = f"R/R:   <b>{rr:.1f}R</b>\n" if rr > 0 else ""

            # Grade badge for approaching — user needs to know urgency level
            _grade_badge = ""
            try:
                from core.signal_lifecycle import signal_lifecycle as _slc
                _slc_sig = _slc.get_signal(signal_id)
                if _slc_sig:
                    _g = getattr(_slc_sig, 'grade', None) or getattr(_slc_sig, 'alpha_grade', None)
                    _grade_map = {"A+": "⚡ A+", "A": "🔥 A", "B+": "📊 B+", "B": "📊 B"}
                    _grade_badge = f"  {_grade_map.get(_g, _g)}" if _g else ""
            except Exception:
                pass
            text = (
                f"📍 <b>APPROACHING ENTRY ZONE</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{dir_emoji} <b>{symbol} {direction}{_grade_badge}</b>  #{signal_id}\n\n"
                f"Price <code>{fmt_price(price)}</code> is "
                f"<b>{pct_away:.1f}%</b> from entry zone\n"
                f"Zone: <code>{fmt_price(entry_low)}</code> — "
                f"<code>{fmt_price(entry_high)}</code>\n"
                f"{_sl_str}{_tp1_str}{_tp2_str}{_tp3_str}"
                f"{_rr_str}\n"
                f"<i>Get your limit order ready. "
                f"Entry likely within 1-2 candles (~{_candle_duration} each).</i>"
            )
            if message_id:
                await telegram_bot.send_reply(message_id, text)
            logger.info(
                f"📍 Signal #{signal_id} approaching alert: "
                f"{pct_away:.1f}% from zone | "
                f"SL={fmt_price(stop_loss)} TP1={fmt_price(tp1)}"
            )
        except Exception as e:
            logger.error(f"Error handling approaching alert: {e}")

    async def _pause(self):
        """Pause signal generation"""
        self._paused = True
        logger.info("Engine paused")

    async def _resume(self):
        """Resume signal generation"""
        self._paused = False
        logger.info("Engine resumed")

    async def _on_network_offline(self):
        """Called when connectivity is lost — pause execution, keep signals"""
        logger.warning("🌐 Network offline — pausing execution (signals preserved)")
        self._paused = True
        # B6: Also pause outcome monitor to avoid false SL hits on stale prices
        try:
            from signals.outcome_monitor import outcome_monitor as _om
            _om.pause()
        except Exception:
            pass

    async def _on_network_reconnect(self):
        """
        Called when connectivity is restored.
        Revalidates pending signals instead of discarding them.
        FIX: Runs an immediate SL reconciliation sweep BEFORE resuming
        normal monitoring — closes the gap where positions could breach
        SL during an outage but not be caught until the next 15s cycle.
        """
        logger.info("🌐 Reconnecting — revalidating pending signals...")
        self._paused = False

        # B6-FIX: Immediate SL reconciliation sweep on reconnect.
        # Fetch fresh prices for ALL active positions and check SL BEFORE
        # resuming normal outcome monitoring. This ensures zero-gap coverage:
        # a position that breached SL during the 11s (or 11min) outage is
        # caught instantly instead of waiting up to 15s for the next cycle.
        try:
            from signals.outcome_monitor import outcome_monitor as _om
            from core.price_cache import price_cache as _recon_pc
            _active_signals = list(_om._active.values())
            if _active_signals:
                _reconciled = 0
                _breached = 0
                for _sig in _active_signals:
                    try:
                        _fresh_price = _recon_pc.get(_sig.symbol)
                        if _fresh_price is None:
                            # Price cache may still be stale; try API
                            from data.api_client import api as _recon_api
                            _ticker = await _recon_api.fetch_ticker(_sig.symbol)
                            _fresh_price = _ticker.get('last') if _ticker else None
                        if _fresh_price is None:
                            continue
                        _reconciled += 1

                        # Check SL breach (mirrors outcome_monitor._check_signal logic)
                        _active_sl = _sig.be_stop if _sig.be_stop is not None else _sig.stop_loss
                        _is_long = getattr(_sig.direction, 'value', str(_sig.direction)) == "LONG"
                        _sl_breached = (
                            (_is_long and _fresh_price <= _active_sl) or
                            (not _is_long and _fresh_price >= _active_sl)
                        )
                        if _sl_breached:
                            _breached += 1
                            logger.warning(
                                f"🚨 SL breached during outage: #{_sig.signal_id} "
                                f"{_sig.symbol} {'LONG' if _is_long else 'SHORT'} "
                                f"price={_fresh_price:.6f} vs SL={_active_sl:.6f}"
                            )
                            # Trigger immediate exit via outcome monitor's check
                            await _om._check_signal(_sig, _fresh_price)
                    except Exception as _sig_err:
                        logger.debug(f"Reconciliation skip #{_sig.signal_id}: {_sig_err}")

                if _breached:
                    logger.warning(
                        f"🌐 SL reconciliation: {_breached}/{_reconciled} position(s) "
                        f"breached SL during outage — exits triggered"
                    )
                elif _reconciled:
                    logger.info(
                        f"🌐 SL reconciliation: {_reconciled} position(s) checked — all OK"
                    )

            # Resume normal monitoring AFTER reconciliation
            _om.resume()
        except Exception as _recon_err:
            logger.debug(f"SL reconciliation error: {_recon_err}")
            # Still resume even if reconciliation fails
            try:
                from signals.outcome_monitor import outcome_monitor as _om2
                _om2.resume()
            except Exception:
                pass

        # Revalidate pending invalidation signals
        try:
            from signals.invalidation_monitor import invalidation_monitor
            pending = invalidation_monitor.get_pending_signals()
            revalidated = 0
            expired = 0

            for sig in pending:
                # Check entry window: reject if too old (>20 min)
                age = time.time() - sig.get('created_at', 0)
                if age > 1200:  # 20 minutes
                    invalidation_monitor.cancel_signal(
                        sig['signal_id'], reason="ENTRY_TIMEOUT"
                    )
                    expired += 1
                    logger.info(
                        f"❌ Signal #{sig['signal_id']} expired after reconnect "
                        f"(age={age/60:.0f}min)"
                    )
                else:
                    revalidated += 1

            logger.info(
                f"🌐 Revalidation complete: {revalidated} active, {expired} expired"
            )
        except Exception as e:
            logger.debug(f"Revalidation error: {e}")

    async def _on_execution_stage_change(self, exec_sig, old_state, new_state):
        """
        Send staged Telegram messages based on execution state transitions.
        
        Psychological staging:
          🟡 Yellow → Observe (setup forming)
          🟠 Orange → Prepare (price in zone)
          🟢 Green  → Act (enter now)
          ❌ Red    → Cancel (expired/invalidated)
        """
        from core.execution_engine import SignalState
        from utils.formatting import fmt_price
        from tg.keyboards import stage_approaching, stage_execute

        symbol_short = exec_sig.symbol.replace('/USDT', '')

        if new_state == SignalState.WATCHING:
            # Suppress — fires too early with no actionable info.
            # ENTRY_ZONE fires when price actually enters the zone.
            return

        elif new_state == SignalState.ENTRY_ZONE:
            text = (
                f"🟠 <b>WATCHING ENTRY</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>{symbol_short}</b> {exec_sig.direction}\n"
                f"✅ Price IN entry zone!\n"
                f"💰 Zone: {fmt_price(exec_sig.entry_low)} — {fmt_price(exec_sig.entry_high)}\n"
                f"🛑 SL: {fmt_price(exec_sig.stop_loss)}\n\n"
                f"<i>⏳ Watching for {exec_sig.min_triggers} trigger points (score 0/6)...</i>"
            )

        elif new_state == SignalState.ALMOST:
            triggers = []
            if exec_sig.has_rejection_candle:
                triggers.append("🕯️ Rejection candle")
            if exec_sig.has_structure_shift:
                triggers.append("📐 Structure shift")
            if exec_sig.has_momentum_expansion:
                triggers.append("🚀 Momentum expansion")
            if exec_sig.has_liquidity_reaction:
                triggers.append("💧 Liquidity reaction")
            trigger_text = "\n".join(triggers)

            # V11: For A-grade (needs 1 trigger), ALMOST = about to fire
            _remaining = max(0, exec_sig.min_triggers - exec_sig.triggers_met)
            _almost_note = f"⏳ Need {_remaining} more trigger{'s' if _remaining != 1 else ''}..." if _remaining > 0 else "⚡ Executing..."
            _max_score = 6  # struct=2 + mom=2 + liq=1 + rej=1
            text = (
                f"🟠 <b>ALMOST READY</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>{symbol_short}</b> {exec_sig.direction}\n"
                f"✅ Score: {exec_sig.triggers_met:.0f}/{_max_score}  (need ≥ {exec_sig.min_triggers} to execute)\n"
                f"{trigger_text}\n\n"
                f"<i>{_almost_note}</i>"
            )

        elif new_state == SignalState.EXECUTE:
            triggers = []
            if exec_sig.has_rejection_candle:
                triggers.append("✅ Rejection candle")
            if exec_sig.has_structure_shift:
                triggers.append("✅ Structure shift")
            if exec_sig.has_momentum_expansion:
                triggers.append("✅ Momentum expansion")
            if exec_sig.has_liquidity_reaction:
                triggers.append("✅ Liquidity reaction")
            trigger_text = "\n".join(triggers)

            # Build TP lines
            tp_lines = f"TP1:   <code>{fmt_price(exec_sig.tp1)}</code> → BE\n" if hasattr(exec_sig, 'tp1') and exec_sig.tp1 else ""
            tp_lines += f"TP2:   <code>{fmt_price(exec_sig.tp2)}</code>\n" if hasattr(exec_sig, 'tp2') and exec_sig.tp2 else ""
            tp_lines += f"TP3:   <code>{fmt_price(exec_sig.tp3)}</code>\n" if hasattr(exec_sig, 'tp3') and exec_sig.tp3 else ""
            rr_line = f"R/R:   {exec_sig.rr_ratio:.1f}R\n" if hasattr(exec_sig, 'rr_ratio') and exec_sig.rr_ratio else ""

            text = (
                f"🟢 <b>ENTER NOW</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>{symbol_short}</b> {exec_sig.direction}\n\n"
                f"Zone:  <code>{fmt_price(exec_sig.entry_low)}</code> — <code>{fmt_price(exec_sig.entry_high)}</code>\n"
                f"SL:    <code>{fmt_price(exec_sig.stop_loss)}</code>\n"
                f"{tp_lines}"
                f"{rr_line}\n"
                f"<b>Triggers: Score {exec_sig.triggers_met:.0f} ✓  (threshold ≥ {exec_sig.min_triggers}):</b>\n"
                f"{trigger_text}\n\n"
                f"<b>⚡ EXECUTION CONFIRMED</b>"
            )

        elif new_state == SignalState.EXPIRED:
            # Write outcome to DB so it doesn't accumulate as a ghost pending signal
            try:
                from data.database import db as _db_ref
                await _db_ref.update_signal_outcome(exec_sig.signal_id, "EXPIRED", 0.0)
            except Exception as _db_err:
                logger.debug(f"Could not write EXPIRED outcome to DB: {_db_err}")
            # PHANTOM-LEARNING FIX: tell outcome_monitor to stop tracking this signal.
            # Without this, outcome_monitor keeps running after execution_engine gives up
            # in ALMOST state — it calls TP1_HIT, closes at BREAKEVEN, and feeds a ghost
            # trade into the learning_loop.  Confirmed in logs: UNI #2 expired in ALMOST
            # at 18:55 but outcome_monitor logged "TP1 hit → BREAKEVEN +0.0R" at 22:51.
            try:
                from signals.outcome_monitor import outcome_monitor as _om_exp
                _om_exp.remove_signal(exec_sig.signal_id)
            except Exception as _om_e:
                logger.debug(f"outcome_monitor.remove_signal on EXPIRED failed (non-fatal): {_om_e}")
            # Release the hourly rate-limit slot so a fresh signal can take this one's place
            try:
                from signals.aggregator import signal_aggregator as _sa
                _sa.release_hourly_slot()
            except Exception:
                pass
            text = (
                f"⏰ <b>ENTRY EXPIRED</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol_short} {exec_sig.direction}\n"
                f"<i>Entry window closed. No action needed.</i>"
            )

        elif new_state == SignalState.INVALIDATED:
            # Dedup: don't send if we already sent a CANCELLED for this signal recently
            _cancel_key = f"_cancelled_{exec_sig.signal_id}"
            _last_cancel = getattr(self, _cancel_key, 0)
            if time.time() - _last_cancel < 120:  # 2 min suppress window
                return  # Duplicate cancellation — skip silently
            setattr(self, _cancel_key, time.time())
            # Release the hourly rate-limit slot so a fresh signal can take this one's place
            try:
                from signals.aggregator import signal_aggregator as _sa
                _sa.release_hourly_slot()
            except Exception:
                pass
            # PHANTOM-LEARNING FIX: same as EXPIRED — stop outcome_monitor tracking
            try:
                from signals.outcome_monitor import outcome_monitor as _om_inv
                _om_inv.remove_signal(exec_sig.signal_id)
            except Exception as _om_e:
                logger.debug(f"outcome_monitor.remove_signal on INVALIDATED failed (non-fatal): {_om_e}")
            # Show brief reason if available
            _cancel_reason = getattr(exec_sig, 'invalidation_reason', None)
            _reason_line = f"\nReason: <i>{_cancel_reason}</i>" if _cancel_reason else ""
            text = (
                f"❌ <b>SIGNAL CANCELLED</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol_short} {exec_sig.direction}{_reason_line}\n"
                f"<i>Structure changed. Do NOT enter.</i>"
            )
        else:
            return  # No message for other transitions

        # Send to Telegram — V10: threaded to original signal card
        try:
            _reply_id = getattr(exec_sig, 'message_id', None)
            _reply_kb = None
            if new_state == SignalState.EXECUTE:
                _reply_kb = stage_execute(exec_sig.signal_id, exec_sig.symbol)
            elif new_state in (SignalState.ENTRY_ZONE, SignalState.ALMOST):
                _reply_kb = stage_approaching(exec_sig.signal_id, exec_sig.symbol)
            await telegram_bot.send_signals_text(
                text=text,
                reply_to_message_id=_reply_id,
                reply_markup=_reply_kb,
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.debug(f"Stage message send failed: {e}")

        # ── Push notification to dashboard SSE ────────────────────────────────────
        try:
            from web.app import push_notification
            _notif_type = {
                "WATCHING":    "watching",
                "ENTRY_ZONE":  "entry_zone",
                "ALMOST":      "almost",
                "EXECUTE":     "execute",
                "EXPIRED":     "expired",
                "INVALIDATED": "invalidated",
                "FILLED":      "filled",
            }.get(new_state.value if hasattr(new_state, "value") else str(new_state), "info")
            push_notification({
                "type":      _notif_type,
                "symbol":    exec_sig.symbol,
                "direction": exec_sig.direction,
                "strategy":  exec_sig.strategy,
                "signal_id": exec_sig.signal_id,
                "grade":     getattr(exec_sig, 'grade', 'B'),
                "entry_low": exec_sig.entry_low,
                "entry_high": exec_sig.entry_high,
                "sl":        exec_sig.stop_loss,
                "tp1":       exec_sig.tp1,
                "tp2":       exec_sig.tp2,
                "rr":        exec_sig.rr_ratio,
                "message":   text[:120] if "text" in dir() else "",
            })
        except Exception:
            pass  # Never let notification push crash the engine

        # ── EXECUTE: create position at ENTER NOW (correct entry point) ──────────
        # This fires when ALL confirmation triggers are met — this is when you
        # actually go to TradingView to trade. P&L tracks from this price.
        if new_state == SignalState.EXECUTE:
            try:
                from core.price_cache import price_cache as _pc
                _entry_now = _pc.get(exec_sig.symbol) or exec_sig.entry_mid
                _stop      = exec_sig.stop_loss
                _stop_dist = abs(_entry_now - _stop) / _entry_now if _entry_now > 0 else 0.02

                sig_row = await db.get_signal(exec_sig.signal_id)
                _risk_usdt = (sig_row.get('risk_amount') if sig_row else None) or (portfolio_engine._capital * 0.01)
                _size_usdt = (sig_row.get('position_size') if sig_row else None) or (_risk_usdt / max(_stop_dist, 0.001))
                _sector    = (sig_row.get('sector', '') if sig_row else '') or ''
                # T1-FIX: pass funding_rate so P&L on close is net of funding fees
                _funding_rate_8h = float((sig_row.get('funding_rate', 0) or 0) if sig_row else 0)

                await portfolio_engine.open_position(
                    symbol=exec_sig.symbol,
                    direction=exec_sig.direction,
                    strategy=exec_sig.strategy,
                    entry_price=_entry_now,
                    size_usdt=_size_usdt,
                    risk_usdt=_risk_usdt,
                    stop_loss=_stop,
                    sector=_sector,
                    signal_id=exec_sig.signal_id,
                    funding_rate_8h=_funding_rate_8h,
                )
                await db.save_open_position(
                    signal_id=exec_sig.signal_id,
                    symbol=exec_sig.symbol,
                    direction=exec_sig.direction,
                    strategy=exec_sig.strategy,
                    entry_price=_entry_now,
                    size_usdt=_size_usdt,
                    risk_usdt=_risk_usdt,
                    stop_loss=_stop,
                    tp1=exec_sig.tp1,
                    tp2=exec_sig.tp2,
                    tp3=exec_sig.tp3,
                    rr_ratio=exec_sig.rr_ratio,
                    sector=_sector,
                    message_id=getattr(exec_sig, 'message_id', None),
                    correlation_to_btc=getattr(exec_sig, 'correlation_to_btc', 0.7),
                    alpha_grade=getattr(exec_sig, 'alpha_grade', 'B'),
                    confidence=getattr(exec_sig, 'confidence', 0.0),
                )
                # Mark signal as ACTIVE in DB so dashboard shows "IN TRADE" status
                try:
                    from data.database import db as _db_ref
                    await _db_ref._exec(
                        "UPDATE signals SET outcome = 'ACTIVE' WHERE id = ? AND outcome IS NULL",
                        (exec_sig.signal_id,)
                    )
                    await _db_ref._conn.commit()
                except Exception as _ae:
                    logger.debug(f"Could not mark signal #{exec_sig.signal_id} ACTIVE: {_ae}")
                logger.info(
                    f"✅ Position opened at ENTER NOW | {exec_sig.symbol} {exec_sig.direction} "
                    f"@ {fmt_price(_entry_now)} | risk=${_risk_usdt:.0f}"
                )
            except Exception as _pos_err:
                logger.warning(f"Position creation at EXECUTE failed: {_pos_err}")

    @staticmethod
    def _record_confidence_adjustment(
        signal,
        source: str,
        *,
        delta: Optional[float] = None,
        multiplier: Optional[float] = None,
        detail: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """Append operator-friendly confidence debug metadata to the signal."""
        if not hasattr(signal, "raw_data") or signal.raw_data is None:
            signal.raw_data = {}
        signal.raw_data.setdefault("confidence_adjustments", [])
        payload = {"source": source}
        if delta is not None:
            payload["delta"] = delta
        if multiplier is not None:
            payload["multiplier"] = multiplier
        if detail:
            payload["detail"] = detail
        if extra:
            payload.update(extra)
        signal.raw_data["confidence_adjustments"].append(payload)

    @staticmethod
    def _compute_pump_dump_metrics(ohlcv_15m: Optional[List]) -> Optional[Dict]:
        """Compute simple 15m pump/dump metrics from recent OHLCV candles."""
        if not ohlcv_15m or len(ohlcv_15m) < 5:
            return None
        try:
            prev_close = float(ohlcv_15m[-2][4])
            last_close = float(ohlcv_15m[-1][4])
            last_volume = float(ohlcv_15m[-1][5])
            baseline_volumes = [float(c[5]) for c in ohlcv_15m[-5:-1] if float(c[5]) >= 0]
            baseline_volume = sum(baseline_volumes) / len(baseline_volumes) if baseline_volumes else 0.0
            price_change_pct = ((last_close - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
            volume_change_pct = (
                ((last_volume - baseline_volume) / baseline_volume) * 100.0
                if baseline_volume > 0 else 0.0
            )
            return {
                "price_change_pct": price_change_pct,
                "volume_change_pct": volume_change_pct,
                "baseline_volume": baseline_volume,
                "last_volume": last_volume,
            }
        except Exception:
            return None

    def _find_recent_directional_news(
        self,
        symbol: str,
        direction: str,
        *,
        max_age_mins: int,
        keywords: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Find the best recent news item aligned with a direction and optional keywords."""
        now = time.time()
        best_match = None
        best_score = float("-inf")
        aligned_count = 0

        try:
            recent_news = news_scraper.get_news_for_symbol(symbol, max_age_mins=max_age_mins)
        except Exception:
            recent_news = []

        for article in recent_news:
            title = article.get("title", "")
            score = news_scraper._score_headline(title)
            article_direction = "LONG" if score > 0 else "SHORT" if score < 0 else "NEUTRAL"
            overlap = sum(1 for kw in (keywords or []) if kw and kw.lower() in title.lower())
            if article_direction == direction:
                aligned_count += 1
            if article_direction != direction and overlap == 0:
                continue
            published = article.get("published_at", now)
            delta_min = abs(now - published) / 60.0
            urgency = news_scraper._urgency_level(title)
            match_score = (
                overlap * 10
                + (5 if article_direction == direction else 0)
                + (2 if urgency == "HIGH" else 0)
                - delta_min / 10.0
            )
            if match_score > best_score:
                best_score = match_score
                best_match = {
                    "article": article,
                    "delta_min": delta_min,
                    "overlap": overlap,
                    "urgency": urgency,
                    "article_direction": article_direction,
                }

        if best_match is not None:
            best_match["aligned_count"] = aligned_count
        return best_match

    @staticmethod
    def _pump_dump_raw_adjustment(alert) -> float:
        """Translate pump/dump risk into a raw contextual adjustment."""
        if not alert:
            return 0.0
        if alert.risk_level in {"HIGH", "VERY_HIGH"}:
            return -0.10
        if alert.risk_level == "MEDIUM":
            return -0.05
        return 0.0

    # ── On-Chain to News Correlation (Feature 5) ────────────────
    def _correlate_onchain_news(
        self,
        symbol: str,
        signal_direction: str,
    ) -> Optional[tuple]:
        """
        Cross-reference on-chain events against recent directional headlines.
        """
        from config.constants import NewsIntelligence as NI
        try:
            from analyzers.onchain_analytics import onchain_analytics
            from analyzers.whale_deposit_monitor import whale_deposit_monitor
        except ImportError:
            return None

        now = time.time()
        onchain_events = []

        try:
            oc_state = onchain_analytics.get_state()
            if getattr(oc_state, "last_anomaly_time", 0):
                onchain_events.append({
                    "type": "onchain_anomaly",
                    "time": oc_state.last_anomaly_time,
                    "keywords": getattr(oc_state, "anomaly_keywords", []),
                    "bias": getattr(oc_state, "anomaly_bias", "NEUTRAL"),
                    "detail": getattr(oc_state, "anomaly_detail", ""),
                })
        except Exception:
            pass

        try:
            whale_state = whale_deposit_monitor.get_state()
            if getattr(whale_state, "last_event_time", 0):
                onchain_events.append({
                    "type": "whale_event",
                    "time": whale_state.last_event_time,
                    "keywords": getattr(whale_state, "last_event_keywords", []),
                    "bias": getattr(whale_state, "last_event_bias", "NEUTRAL"),
                    "detail": getattr(whale_state, "last_event_detail", ""),
                })
        except Exception:
            pass

        aligned_events = [
            event for event in onchain_events
            if event.get("bias") in {signal_direction, "NEUTRAL"}
        ]
        if not aligned_events:
            return None

        latest_oc = max(aligned_events, key=lambda e: e["time"])
        oc_age_minutes = (now - latest_oc["time"]) / 60.0
        if oc_age_minutes > NI.CORRELATION_PARTIAL_WINDOW_MINUTES:
            return None

        match = self._find_recent_directional_news(
            symbol,
            signal_direction,
            max_age_mins=NI.CORRELATION_PARTIAL_WINDOW_MINUTES,
            keywords=latest_oc.get("keywords", []),
        )

        if match is not None:
            delta = min(match["delta_min"], oc_age_minutes)
            time_weight = 1.0 if delta <= NI.CORRELATION_FULL_WINDOW_MINUTES else NI.CORRELATION_PARTIAL_WEIGHT
            headline = match["article"].get("title", "")[:70]
            if match["overlap"] > 0:
                effective_mult = 1.0 + (NI.CORRELATION_SAME_EVENT_BOOST - 1.0) * time_weight
                return (
                    effective_mult,
                    "SAME_EVENT",
                    f"{latest_oc['detail']} ↔ '{headline}' overlap={match['overlap']} delta={delta:.0f}m",
                )
            if match["aligned_count"] >= 2 or match["urgency"] == "HIGH":
                effective_mult = 1.0 + (NI.CORRELATION_INDEPENDENT_BOOST - 1.0) * time_weight
                return (
                    effective_mult,
                    "INDEPENDENT",
                    f"{latest_oc['detail']} + directional headline '{headline}' delta={delta:.0f}m",
                )

        return (
            NI.UNEXPLAINED_ONCHAIN_PENALTY,
            "UNEXPLAINED",
            f"{latest_oc['detail']} without aligned news inside {NI.CORRELATION_PARTIAL_WINDOW_MINUTES}m",
        )

    # ── Setup ─────────────────────────────────────────────────

    def _register_strategies(self):
        """Register all strategy instances"""
        self._strategies = [
            SMCStrategy(),
            BreakoutStrategy(),
            ReversalStrategy(),
            MeanReversionStrategy(),
            PriceActionStrategy(),
            MomentumStrategy(),
            IchimokuStrategy(),
            ElliottWaveStrategy(),
            FundingArbStrategy(),
            RangeScalperStrategy(),
            WyckoffStrategy(),
            HarmonicDetector(),
            GeometricPatterns(),
        ]
        enabled = [s.name for s in self._strategies
                   if cfg.is_strategy_enabled(self._strat_key_map.get(s.name, s.name.lower()))]
        logger.info(f"Registered {len(self._strategies)} strategies: "
                    f"{[s.name for s in self._strategies]}")

    def _get_required_timeframes(self) -> List[str]:
        """Get all timeframes needed by active strategies"""
        tfs = set()
        strategy_tfs = {
            'smc':          ['4h', '1h', '15m'],
            'breakout':     ['4h', '1h'],
            'reversal':     ['1h', '15m'],
            'mean_reversion': ['1h'],
            'price_action': ['4h', '1h'],
            'momentum':     ['1h'],
            'ichimoku':     ['4h'],
            'elliott_wave': ['4h'],
            'funding_arb':  ['1h'],
            'range_scalper': ['1h', '15m'],
            'wyckoff':      ['4h'],   # P1 fix: was missing, silently failed if other 4h strats disabled
            'harmonic':     ['4h', '1h'],
            'geometric':    ['4h', '1h'],
        }
        for strat_name, tf_list in strategy_tfs.items():
            if cfg.is_strategy_enabled(strat_name):
                tfs.update(tf_list)
        return list(tfs) if tfs else ['1h', '4h', '15m']


# ── Singleton ──────────────────────────────────────────────
engine = Engine()
