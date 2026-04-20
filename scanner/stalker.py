"""
TitanBot Pro — Stalker Engine
================================
Watches symbols that are setting up but haven't triggered yet.
Sends pre-breakout alerts to give you time to prepare.

Detects:
  - Bollinger Band squeeze (volatility compression → expansion coming)
  - Volume building under resistance (accumulation pattern)
  - RSI coiling near 50 (indecision before breakout)
  - ATR compression (price range narrowing)
  - Multiple timeframe alignment without full confirmation

When a symbol scores high enough, it goes on the watchlist.
The scanner then elevates its scan frequency.

All detectors operate on *closed* bars only (the last OHLCV row is the
currently-forming candle and is excluded from every comparison). The
earlier version silently included the live candle in BB / ATR / RSI /
consolidation detectors, producing tick-driven false alerts.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.loader import cfg
from data.api_client import api
from data.database import db
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


def _get_regime() -> str:
    """Return the current regime string, or 'UNKNOWN' if not yet initialised."""
    try:
        from analyzers.regime import regime_analyzer
        return str(regime_analyzer.regime.value)
    except Exception:
        return "UNKNOWN"


def _regime_recently_changed(within_secs: int = 600) -> bool:
    """Return True iff the regime changed within the last ``within_secs`` seconds."""
    try:
        from analyzers.regime import regime_analyzer
        return regime_analyzer.is_recently_transitioned(within_secs)
    except Exception:
        return False


def _wilder_rsi_last(closes, period: int = 14) -> float:
    """Single-pass Wilder RSI — O(N)."""
    n = len(closes)
    if n < period + 1:
        return 50.0
    gains_sum = 0.0
    losses_sum = 0.0
    for i in range(1, period + 1):
        d = float(closes[i]) - float(closes[i - 1])
        if d > 0:
            gains_sum += d
        else:
            losses_sum += -d
    avg_g = gains_sum / period
    avg_l = losses_sum / period
    for i in range(period + 1, n):
        d = float(closes[i]) - float(closes[i - 1])
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)


def _wilder_rsi_tail_series(closes, tail: int, period: int = 14) -> List[float]:
    """
    Return the last `tail` Wilder RSI values using a single-pass streaming
    computation. Replaces the previous O(N²) `[rsi(closes[:i]) for i in ...]`
    pattern in ``_detect_rsi_coiling``.
    """
    n = len(closes)
    out: List[float] = []
    if n < period + 1:
        return out
    gains_sum = 0.0
    losses_sum = 0.0
    for i in range(1, period + 1):
        d = float(closes[i]) - float(closes[i - 1])
        if d > 0:
            gains_sum += d
        else:
            losses_sum += -d
    avg_g = gains_sum / period
    avg_l = losses_sum / period
    # RSI is defined starting at index `period`; iterate over all updates and
    # keep the final `tail` values.
    buf: List[float] = []
    for i in range(period, n):
        if i > period:
            d = float(closes[i]) - float(closes[i - 1])
            g = d if d > 0 else 0.0
            l = -d if d < 0 else 0.0
            avg_g = (avg_g * (period - 1) + g) / period
            avg_l = (avg_l * (period - 1) + l) / period
        if avg_l == 0:
            buf.append(100.0 if avg_g > 0 else 50.0)
        else:
            buf.append(100.0 - 100.0 / (1.0 + avg_g / avg_l))
    out = buf[-tail:] if tail > 0 else buf
    return out


class StalkerEngine:
    """
    Monitors potential pre-breakout setups and populates the watchlist.
    Runs on its own interval, independent of the main scanner.
    """

    def __init__(self):
        self._watched: Dict[str, float] = {}     # symbol -> last_alert_time
        self._alert_cooldown = 3600              # 1 hour between repeat alerts
        self._min_watch_score = 55               # Min score to add to watchlist (base)

    async def scan_symbol(self, symbol: str, ohlcv_1h: List, ohlcv_15m: List) -> Optional[Dict]:
        """
        Analyze a symbol for pre-breakout conditions.
        Returns a watchlist entry dict or None.

        Minimum 51 rows (50 closed + 1 forming) is a sanity floor. Individual
        detectors may silently skip when their own data requirements aren't
        met (e.g. `_detect_bb_squeeze` needs `period*3=60` closed bars, so on
        inputs between 51–60 bars only a subset of detectors contribute).
        """
        # Need at least 51 rows so every detector has enough closed bars to
        # run; detectors that need more short-circuit locally.
        if not ohlcv_1h or len(ohlcv_1h) < 51:
            return None

        # Check alert cooldown *before* any expensive computation
        last_alert = self._watched.get(symbol, 0)
        if time.time() - last_alert < self._alert_cooldown:
            return None

        _regime = _get_regime()
        # VOLATILE_PANIC: same rationale as scanner.stalker_scan — pre-breakout
        # compression alerts in panic regimes are bull traps.
        if _regime == "VOLATILE_PANIC":
            return None

        # Dynamic threshold — mirrors scanner.stalker_scan logic
        # Hysteresis: damp by 25% within 10 min of a regime flip.
        _recently_flipped = _regime_recently_changed(600)
        if _regime == "CHOPPY":
            _raw_adj = 8
        elif _regime in ("VOLATILE", "VOLATILE_PANIC"):
            _raw_adj = 5
        elif _regime in ("BULL_TREND", "BEAR_TREND"):
            _raw_adj = -5
        else:
            _raw_adj = 0
        if _recently_flipped:
            _raw_adj = round(_raw_adj * 0.75)
        min_score = self._min_watch_score + _raw_adj

        # Regime-dependent dedup TTL (mirrors scanner.stalker_scan).
        # Read the base TTL from the scanner singleton so both paths stay in sync.
        # Hysteresis: within 10 min of a regime flip, damp the TTL reduction by
        # 25% so we don't instantly halve the dedup window on a noisy transition.
        _scanner_singleton = None
        try:
            from scanner.scanner import scanner as _scanner_singleton
            _base_dedup_ttl = _scanner_singleton._signal_dedup_ttl
        except Exception:
            _base_dedup_ttl = 1800  # 30 min fallback
        if _regime in ("BULL_TREND", "BEAR_TREND"):
            _dedup_ttl = _base_dedup_ttl        # 30 min — signals evolve slowly
        else:
            _raw_ttl_reduction = _base_dedup_ttl // 2
            if _recently_flipped:
                _damped_ttl = _base_dedup_ttl - round(_raw_ttl_reduction * 0.75)
                _dedup_ttl = _damped_ttl
            else:
                _dedup_ttl = _base_dedup_ttl - _raw_ttl_reduction  # 15 min in chop/volatile

        df = pd.DataFrame(ohlcv_1h, columns=['ts','open','high','low','close','volume'])
        df = df.astype({'open': float,'high': float,'low': float,'close': float,'volume': float})

        # Drop the still-forming bar from every detector.
        closes_all  = df['close'].values
        highs_all   = df['high'].values
        lows_all    = df['low'].values
        volumes_all = df['volume'].values
        opens_all   = df['open'].values
        closes  = closes_all[:-1]
        highs   = highs_all[:-1]
        lows    = lows_all[:-1]
        volumes = volumes_all[:-1]
        opens   = opens_all[:-1]

        score   = 0
        reasons = []

        # ── Bollinger Band Squeeze ────────────────────────────
        bb_squeeze = self._detect_bb_squeeze(closes)
        if bb_squeeze:
            score   += 25
            reasons.append("🌀 BB squeeze — volatility compression")

        # ── ATR Compression ───────────────────────────────────
        atr_compression = self._detect_atr_compression(highs, lows, closes)
        if atr_compression:
            score   += 20
            reasons.append("📉 ATR compression — price range narrowing")

        # ── Volume Build ──────────────────────────────────────
        vol_build = self._detect_volume_build(volumes, closes, opens)
        if vol_build == 'accumulation':
            score   += 15
            reasons.append("📊 Volume building with up-bars — accumulation")
        elif vol_build == 'distribution':
            # Still structurally interesting for shorts but downgraded —
            # volume rising on red bars is distribution, not accumulation.
            score   += 5
            reasons.append("📊 Volume building with down-bars — distribution")

        # ── RSI Coiling ───────────────────────────────────────
        rsi_coil = self._detect_rsi_coiling(closes)
        if rsi_coil:
            score   += 10
            reasons.append("🎯 RSI coiling near 50 — indecision before breakout")

        # ── Consolidation Range ───────────────────────────────
        consolidation = self._detect_consolidation(highs, lows, closes)
        if consolidation:
            score   += 15
            reasons.append("📐 Consolidation pattern forming")

        # ── LTF alignment check ───────────────────────────────
        if ohlcv_15m and len(ohlcv_15m) >= 30:
            df15 = pd.DataFrame(ohlcv_15m, columns=['ts','open','high','low','close','volume'])
            df15 = df15.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            ltf_alert = self._check_ltf_alignment(df15)
            if ltf_alert:
                score   += 15
                reasons.append("⚡ 15m structure aligning with 1h")

        if score < min_score or not reasons:
            return None

        # Cross-path dedup via Scanner.try_mark_signal_dedup — single source of
        # truth, so StalkerEngine and scanner.stalker_scan can't double-fire.
        setup_type = "pre_breakout"
        dedup_key  = (symbol, setup_type)
        if _scanner_singleton is not None:
            try:
                if not _scanner_singleton.try_mark_signal_dedup(dedup_key, _dedup_ttl):
                    return None
            except Exception:
                pass  # dedup unavailable — allow through rather than block

            # Global alert-rate circuit breaker — share the scanner's token
            # bucket so the combined stalker/scanner output is rate-limited.
            try:
                if not _scanner_singleton._rate_allow('watchlist', max_alerts=20, window_secs=600):
                    logger.info(f"Stalker rate-limit hit — suppressing {symbol} (score={score})")
                    return None
            except Exception:
                pass

        return {
            'symbol':  symbol,
            'score':   score,
            'reasons': reasons,
            'regime':  _regime,
        }

    async def process_result(self, result: Dict, publisher):
        """Save to watchlist DB and optionally alert Telegram"""
        symbol  = result['symbol']
        score   = result['score']
        reasons = result['reasons']

        # Decouple DB + Telegram: a DB hiccup must not swallow the Telegram alert.
        try:
            await db.upsert_watchlist(symbol, score, reasons)
        except Exception as e:
            logger.warning(f"Stalker: watchlist DB write failed for {symbol}: {e}")
        self._watched[symbol] = time.time()

        # Alert Telegram for high-score setups
        if score >= 70 and cfg.telegram.get('send_watchlist_alerts', True):
            try:
                await publisher.publish_watchlist(symbol, score, reasons)
                logger.info(f"Watchlist alert sent: {symbol} score={score}")
            except Exception as e:
                logger.warning(f"Stalker: telegram publish failed for {symbol}: {e}")
        else:
            logger.debug(f"Watchlist updated silently: {symbol} score={score}")

    # ── Detection methods ─────────────────────────────────────

    def _detect_bb_squeeze(self, closes, period: int = 20) -> bool:
        """
        Bollinger Band squeeze: bandwidth in the bottom 20% of recent history.
        History now uses up to 200 closed bars with stride 2 → ~90 samples,
        stable percentile (prior 100/5 gave ~20 samples).
        """
        if len(closes) < period * 3:
            return False

        def bandwidth(data):
            m = np.mean(data)
            s = np.std(data)
            return (s * 4) / m if m > 0 else 0

        current_bw = bandwidth(closes[-period:])

        hist_start = max(period, len(closes) - 200)
        historical_bw = [
            bandwidth(closes[i:i + period])
            for i in range(hist_start, len(closes) - period, 2)
        ]

        if not historical_bw:
            return False

        percentile_20 = np.percentile(historical_bw, 20)
        return current_bw <= percentile_20

    def _detect_atr_compression(self, highs, lows, closes, period: int = 14) -> bool:
        """
        ATR compression: recent ATR significantly below 50-bar average.
        Caller passes 15-bar tail so period=14 averages 14 TRs (fixes prior
        off-by-one where a 14-bar slice produced only 13 TRs).
        """
        if len(closes) < 60:
            return False

        def atr(h, l, c, p):
            trs = []
            for i in range(1, len(c)):
                trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
            return np.mean(trs[-p:]) if trs else 0

        recent_atr = atr(highs[-15:], lows[-15:], closes[-15:], period)
        long_atr   = atr(highs[-60:], lows[-60:], closes[-60:], 59)   # 60 bars → 59 TRs, average all

        if long_atr == 0:
            return False

        return recent_atr < long_atr * 0.65   # ATR contracted to 65% of normal

    def _detect_volume_build(self, volumes, closes, opens) -> Optional[str]:
        """
        Volume building while price is flat/consolidating — classic pre-breakout.

        Returns:
          'accumulation'  — vol rising with net bullish bars
          'distribution'  — vol rising with net bearish bars
          None            — no volume build detected

        Previously returned a bool that treated both regimes identically; a
        high-vol contraction with 70% red candles is distribution and the
        expected break direction is *down*, not up — scoring them the same
        was producing unreliable bull-biased alerts.
        """
        if len(volumes) < 20 or len(closes) < 10:
            return None

        recent_vol = np.mean(volumes[-5:])
        prior_vol  = np.mean(volumes[-20:-5])

        if prior_vol == 0:
            return None

        # Volume increasing
        vol_increasing = recent_vol > prior_vol * 1.3

        # But price not moving much (consolidation). Use the window's midpoint
        # as the denominator rather than the arbitrary `closes[-10]` bar,
        # which can distort on sharp recent moves.
        window = closes[-10:]
        mid = (max(window) + min(window)) / 2.0
        price_range = (max(window) - min(window)) / mid if mid > 0 else 1.0
        price_flat  = price_range < 0.04   # Less than 4% range in 10 bars

        if not (vol_increasing and price_flat):
            return None

        # Up/down volume split (Wyckoff-style). Use open-to-close to classify.
        # Iterate the last 5 bars; guard index bounds so a short `opens`
        # array (edge case) falls back to closing-price sign.
        up_vol = 0.0
        dn_vol = 0.0
        n_opens = len(opens)
        for i in range(-5, 0):
            c = float(closes[i])
            # Index `i` (negative) is valid iff n_opens >= -i.
            o = float(opens[i]) if n_opens >= -i else c
            v = float(volumes[i])
            if c >= o:
                up_vol += v
            else:
                dn_vol += v
        total = up_vol + dn_vol
        if total <= 0:
            return 'accumulation'  # can't classify; be charitable
        up_share = up_vol / total
        if up_share >= 0.55:
            return 'accumulation'
        if up_share <= 0.45:
            return 'distribution'
        return 'accumulation'  # neutral → treat as mild accumulation

    def _detect_rsi_coiling(self, closes) -> bool:
        """
        RSI sitting in the 40–60 band repeatedly — indecision before breakout.

        Relaxed from 8/8 bars in-band to 7/8, since a single stray print to
        62 used to reset a genuine 15-bar coil. O(N) instead of O(N²).
        """
        if len(closes) < 30:
            return False

        rsi_vals = _wilder_rsi_tail_series(list(closes), tail=15, period=14)
        if len(rsi_vals) < 8:
            return False
        tail = rsi_vals[-8:]
        in_band = sum(1 for v in tail if 40 < v < 60)
        return in_band >= 7

    def _detect_consolidation(self, highs, lows, closes) -> bool:
        """
        Price in a tight range for 10+ bars (consolidation box).

        Denominator is the range midpoint (not the low) so a 5 %-range coin
        trading 30 % down doesn't look tighter than one trading 30 % up.
        """
        if len(closes) < 15:
            return False

        lookback = 15
        h_range = max(highs[-lookback:])
        l_range = min(lows[-lookback:])

        mid = (h_range + l_range) / 2.0
        if mid <= 0:
            return False

        range_pct = (h_range - l_range) / mid
        return range_pct < 0.05   # Less than 5% range

    def _check_ltf_alignment(self, df15: pd.DataFrame) -> bool:
        """
        Check if 15m is showing early breakout from the consolidation.
        Uses matched 5-bar lookback windows (prior vs recent) for a fair
        higher-high comparison, and excludes the still-forming candle.
        """
        if len(df15) < 20:
            return False

        closes  = df15['close'].values
        volumes = df15['volume'].values
        highs   = df15['high'].values

        avg_vol = np.mean(volumes[-20:-1])   # exclude still-forming
        cur_vol = volumes[-2]                 # last closed candle

        # Volume spike on last-closed bar
        if avg_vol <= 0:
            return False
        vol_spike = cur_vol > avg_vol * 1.8

        # Higher-high test on matched windows:
        #   recent = [-6:-1]  (5 closed bars)
        #   prior  = [-11:-6] (5 closed bars)
        if len(highs) < 12:
            return False
        recent_high = max(highs[-6:-1])
        prior_high  = max(highs[-11:-6])
        higher_high = recent_high > prior_high

        return vol_spike and higher_high


# ── Singleton ──────────────────────────────────────────────
stalker = StalkerEngine()
