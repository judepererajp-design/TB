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
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.loader import cfg
from data.api_client import api
from data.database import db
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class StalkerEngine:
    """
    Monitors potential pre-breakout setups and populates the watchlist.
    Runs on its own interval, independent of the main scanner.
    """

    def __init__(self):
        self._watched: Dict[str, float] = {}     # symbol -> last_alert_time
        self._alert_cooldown = 3600              # 1 hour between repeat alerts
        self._min_watch_score = 55               # Min score to add to watchlist

    async def scan_symbol(self, symbol: str, ohlcv_1h: List, ohlcv_15m: List) -> Optional[Dict]:
        """
        Analyze a symbol for pre-breakout conditions.
        Returns a watchlist entry dict or None.
        """
        if not ohlcv_1h or len(ohlcv_1h) < 50:
            return None

        df = pd.DataFrame(ohlcv_1h, columns=['ts','open','high','low','close','volume'])
        df = df.astype({'open': float,'high': float,'low': float,'close': float,'volume': float})

        closes  = df['close'].values
        highs   = df['high'].values
        lows    = df['low'].values
        volumes = df['volume'].values

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
        vol_build = self._detect_volume_build(volumes, closes)
        if vol_build:
            score   += 15
            reasons.append("📊 Volume building — accumulation detected")

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
            df15 = df15.astype(float)
            ltf_alert = self._check_ltf_alignment(df15)
            if ltf_alert:
                score   += 15
                reasons.append("⚡ 15m structure aligning with 1h")

        if score < self._min_watch_score or not reasons:
            return None

        # Check cooldown
        last_alert = self._watched.get(symbol, 0)
        if time.time() - last_alert < self._alert_cooldown:
            return None

        return {
            'symbol':  symbol,
            'score':   score,
            'reasons': reasons,
        }

    async def process_result(self, result: Dict, publisher):
        """Save to watchlist DB and optionally alert Telegram"""
        symbol  = result['symbol']
        score   = result['score']
        reasons = result['reasons']

        await db.upsert_watchlist(symbol, score, reasons)
        self._watched[symbol] = time.time()

        # Alert Telegram for high-score setups
        if score >= 70 and cfg.telegram.get('send_watchlist_alerts', True):
            await publisher.publish_watchlist(symbol, score, reasons)
            logger.info(f"Watchlist alert sent: {symbol} score={score}")
        else:
            logger.debug(f"Watchlist updated silently: {symbol} score={score}")

    # ── Detection methods ─────────────────────────────────────

    def _detect_bb_squeeze(self, closes: np.ndarray, period: int = 20) -> bool:
        """
        Bollinger Band squeeze: bandwidth at a 6-month low.
        Tight bands → explosive move coming.
        """
        if len(closes) < period * 3:
            return False

        def bandwidth(data):
            m = np.mean(data)
            s = np.std(data)
            return (s * 4) / m if m > 0 else 0

        current_bw = bandwidth(closes[-period:])

        # Historical bandwidths over last 100 bars
        historical_bw = [
            bandwidth(closes[i:i+period])
            for i in range(max(0, len(closes)-100), len(closes)-period, 5)
        ]

        if not historical_bw:
            return False

        # Squeeze = current BW in bottom 20% of historical range
        percentile_20 = np.percentile(historical_bw, 20)
        return current_bw <= percentile_20

    def _detect_atr_compression(self, highs, lows, closes, period: int = 14) -> bool:
        """
        ATR compression: recent ATR significantly below 50-bar average.
        """
        if len(closes) < 60:
            return False

        def atr(h, l, c, p):
            trs = []
            for i in range(1, len(c)):
                trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
            return np.mean(trs[-p:]) if trs else 0

        recent_atr = atr(highs[-20:], lows[-20:], closes[-20:], period)
        long_atr   = atr(highs[-60:], lows[-60:], closes[-60:], period)

        if long_atr == 0:
            return False

        return recent_atr < long_atr * 0.65   # ATR contracted to 65% of normal

    def _detect_volume_build(self, volumes, closes) -> bool:
        """
        Volume building while price is flat/consolidating.
        Classic accumulation signature.
        """
        if len(volumes) < 20:
            return False

        recent_vol = np.mean(volumes[-5:])
        prior_vol  = np.mean(volumes[-20:-5])

        if prior_vol == 0:
            return False

        # Volume increasing
        vol_increasing = recent_vol > prior_vol * 1.3

        # But price not moving much (consolidation)
        price_range = (max(closes[-10:]) - min(closes[-10:])) / closes[-10] if closes[-10] > 0 else 1
        price_flat  = price_range < 0.04   # Less than 4% range in 10 bars

        return vol_increasing and price_flat

    def _detect_rsi_coiling(self, closes) -> bool:
        """
        RSI bouncing between 45-55 repeatedly — indecision before breakout.
        """
        if len(closes) < 30:
            return False

        def rsi(c, p=14):
            if len(c) < p+1:
                return 50.0
            d = np.diff(c[-(p+1):])
            gains  = np.where(d > 0, d, 0)
            losses = np.where(d < 0, -d, 0)
            avg_g  = np.mean(gains)  or 1e-10
            avg_l  = np.mean(losses) or 1e-10
            return 100 - (100 / (1 + avg_g/avg_l))

        rsi_vals = [rsi(closes[:i]) for i in range(len(closes)-15, len(closes))]
        coiling  = all(40 < v < 60 for v in rsi_vals[-8:])
        return coiling

    def _detect_consolidation(self, highs, lows, closes) -> bool:
        """
        Price in a tight range for 10+ bars (consolidation box).
        """
        if len(closes) < 15:
            return False

        lookback = 15
        h_range = max(highs[-lookback:])
        l_range = min(lows[-lookback:])

        if l_range == 0:
            return False

        range_pct = (h_range - l_range) / l_range
        return range_pct < 0.05   # Less than 5% range

    def _check_ltf_alignment(self, df15: pd.DataFrame) -> bool:
        """
        Check if 15m is showing early breakout from the consolidation.
        """
        if len(df15) < 20:
            return False

        closes  = df15['close'].values
        volumes = df15['volume'].values

        avg_vol = np.mean(volumes[-20:])
        cur_vol = volumes[-1]

        # Volume spike on recent bars
        vol_spike = cur_vol > avg_vol * 1.8

        # Price making a higher high on 15m
        recent_high = max(df15['high'].values[-5:])
        prior_high  = max(df15['high'].values[-20:-5])
        higher_high = recent_high > prior_high

        return vol_spike and higher_high


# ── Singleton ──────────────────────────────────────────────
stalker = StalkerEngine()
