"""
TitanBot Pro — Market Microstructure Analyzer
==============================================
Four genuine signal enhancers from completely free public APIs:

  1. CVD  — Cumulative Volume Delta (Binance aggTrades)
            Distinguishes real breakouts from fakeouts by measuring
            whether buyers or sellers are the aggressor.

  2. Max Pain + PCR  — Deribit options (BTC + ETH only)
            Price gravitates toward max pain before weekly/monthly expiry.
            Put/call ratio measures directional options sentiment.

  3. Exchange Netflow  — BTC/ETH moving to/from exchanges
            CryptoQuant free API + Glassnode free tier.
            Large inflows = selling incoming. Outflows = accumulation.

  4. Perp/Spot Basis  — premium of perp over spot price
            Directly measures leverage crowding more precisely than
            funding rate alone. High basis = overleveraged longs.

No API keys required for any of these.

Signal impact per feature:
  CVD aligned:       +5 confidence  |  CVD opposed: -7 (fakeout penalty)
  Max pain aligned:  +4 confidence  |  Against:     -4
  Netflow negative:  +4 (accum)     |  Positive:    -5 (sell pressure)
  Basis neutral:     +3             |  Extreme:     -6
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

from config.constants import NewsIntelligence
from config.feature_flags import ff
from config.shadow_mode import shadow_log

logger = logging.getLogger(__name__)

_BINANCE_FAPI  = "https://fapi.binance.com"
_BINANCE_API   = "https://api.binance.com"
_DERIBIT       = "https://www.deribit.com/api/v2/public"
_CRYPTOQUANT   = "https://api.cryptoquant.com/v1"

# Refresh intervals
_CVD_WINDOW_SECS    = 300     # 5 min CVD lookback
_CVD_REFRESH        = 120     # refresh every 2 min
_BASIS_REFRESH      = 60      # refresh every 1 min (fast-moving)
_OPTIONS_REFRESH    = 3600    # refresh every hour (slow-moving)
_NETFLOW_REFRESH    = 1800    # refresh every 30 min


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class CVDSnapshot:
    coin:           str
    cvd_usd:        float   # net buy - sell volume in USD (positive = net buying)
    buy_vol_usd:    float
    sell_vol_usd:   float
    total_vol_usd:  float
    cvd_pct:        float   # cvd / total_vol — normalized strength
    fetched_at:     float = field(default_factory=time.time)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.fetched_at > 600

    @property
    def signal(self) -> str:
        if self.total_vol_usd < 10_000:
            return "NEUTRAL"  # insufficient volume
        pct = self.cvd_pct
        if pct > 0.15:   return "STRONG_BUY"
        if pct > 0.05:   return "BUY"
        if pct < -0.15:  return "STRONG_SELL"
        if pct < -0.05:  return "SELL"
        return "NEUTRAL"

    def confidence_delta(self, direction: str) -> Tuple[int, str]:
        sig = self.signal
        if sig == "NEUTRAL":
            return 0, ""
        aligned   = (direction == "LONG"  and sig in ("BUY", "STRONG_BUY")) or \
                    (direction == "SHORT" and sig in ("SELL", "STRONG_SELL"))
        opposed   = (direction == "LONG"  and sig in ("SELL", "STRONG_SELL")) or \
                    (direction == "SHORT" and sig in ("BUY", "STRONG_BUY"))
        strong    = "STRONG" in sig

        buy_pct  = self.buy_vol_usd / max(self.total_vol_usd, 1) * 100
        sell_pct = self.sell_vol_usd / max(self.total_vol_usd, 1) * 100
        vol_fmt  = _fmt_usd(self.total_vol_usd)

        if aligned and strong:
            return +5, f"📊 CVD strong {'buy' if direction=='LONG' else 'sell'} pressure ({buy_pct if direction=='LONG' else sell_pct:.0f}% aggressor, vol={vol_fmt})"
        if aligned:
            return +3, f"📊 CVD {'buy' if direction=='LONG' else 'sell'} pressure ({vol_fmt})"
        if opposed and strong:
            return -7, f"⚠️ CVD FAKEOUT: {'sellers' if direction=='LONG' else 'buyers'} dominating ({sell_pct if direction=='LONG' else buy_pct:.0f}% aggressor)"
        if opposed:
            return -4, f"⚠️ CVD against {direction}: {'sell' if direction=='LONG' else 'buy'} pressure"
        return 0, ""


@dataclass
class OptionsSnapshot:
    """BTC or ETH Deribit options data."""
    coin:          str
    max_pain:      float    # price where most options expire worthless
    put_call_ratio: float   # > 1 = more puts (bearish), < 1 = more calls (bullish)
    next_expiry_h: float    # hours until next major expiry
    total_oi_usd:  float    # total options OI in USD
    fetched_at:    float = field(default_factory=time.time)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.fetched_at > 7200  # 2 hours

    def confidence_delta(
        self, direction: str, current_price: float
    ) -> Tuple[int, str]:
        if self.max_pain <= 0 or current_price <= 0:
            return 0, ""

        deviation = (current_price - self.max_pain) / self.max_pain * 100
        # Expiry gravity: only applies within 48h of expiry
        expiry_gravity = self.next_expiry_h <= 48

        # PCR signal: > 1.3 = bearish hedge, < 0.7 = bullish speculation
        pcr_bearish = self.put_call_ratio > 1.3
        pcr_bullish = self.put_call_ratio < 0.7

        delta = 0
        notes = []

        if expiry_gravity:
            hrs = int(self.next_expiry_h)
            if direction == "SHORT" and deviation > 2.0:
                # Positive deviation means price is ABOVE max_pain.
                # AUDIT FIX: previously this note read "% below", inverting
                # the semantics shown to the operator (delta logic below
                # remains correct).
                delta += 4
                notes.append(
                    f"🎯 Max pain ${self.max_pain:,.0f} "
                    f"({deviation:+.1f}% above, expiry {hrs}h)"
                )
            elif direction == "LONG" and deviation < -2.0:
                delta += 4
                notes.append(
                    f"🎯 Max pain ${self.max_pain:,.0f} "
                    f"({abs(deviation):.1f}% above, expiry {hrs}h)"
                )
            elif direction == "SHORT" and deviation < -2.0:
                delta -= 4
                notes.append(
                    f"⚠️ Against max pain ${self.max_pain:,.0f} "
                    f"(price {abs(deviation):.1f}% below)"
                )
            elif direction == "LONG" and deviation > 2.0:
                delta -= 4
                notes.append(
                    f"⚠️ Against max pain ${self.max_pain:,.0f} "
                    f"(price {deviation:.1f}% above)"
                )

        # PCR sentiment
        if direction == "SHORT" and pcr_bearish:
            delta += 2
            notes.append(f"🐻 PCR {self.put_call_ratio:.2f} bearish hedging")
        elif direction == "LONG" and pcr_bullish:
            delta += 2
            notes.append(f"🐂 PCR {self.put_call_ratio:.2f} bullish positioning")
        elif direction == "LONG" and pcr_bearish:
            delta -= 2
            notes.append(f"⚠️ PCR {self.put_call_ratio:.2f} hedging against long")
        elif direction == "SHORT" and pcr_bullish:
            delta -= 2
            notes.append(f"⚠️ PCR {self.put_call_ratio:.2f} calls dominating")

        return delta, " | ".join(notes)


@dataclass
class NetflowSnapshot:
    """Exchange inflow/outflow for BTC or ETH."""
    coin:           str
    netflow_24h:    float   # USD — positive = to exchange (bearish), negative = out (bullish)
    inflow_24h:     float
    outflow_24h:    float
    ma7_netflow:    float   # 7-day moving average for context
    source:         str     # "cryptoquant" | "estimated"
    fetched_at:     float = field(default_factory=time.time)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.fetched_at > 3600

    def confidence_delta(self, direction: str) -> Tuple[int, str]:
        # Normalise: compare today vs 7d MA
        if self.ma7_netflow == 0:
            return 0, ""

        ratio = self.netflow_24h / max(abs(self.ma7_netflow), 1)
        # > 2.0 = 2x normal inflow (selling pressure)
        # < -2.0 = 2x normal outflow (accumulation)

        inflow_fmt  = _fmt_usd(abs(self.inflow_24h))
        outflow_fmt = _fmt_usd(abs(self.outflow_24h))

        if direction == "LONG":
            if ratio > 2.0:
                return -5, f"⚠️ {self.coin} exchange inflow 2x avg ({inflow_fmt}) — sell pressure"
            if ratio > 1.3:
                return -2, f"⚠️ {self.coin} elevated inflow ({inflow_fmt})"
            if ratio < -2.0:
                return +4, f"🏦 {self.coin} heavy outflow ({outflow_fmt}) — accumulation"
            if ratio < -1.3:
                return +2, f"🏦 {self.coin} outflow above avg ({outflow_fmt})"
        else:  # SHORT
            if ratio > 2.0:
                return +4, f"🏦 {self.coin} exchange inflow 2x avg ({inflow_fmt}) — sell pressure aligns"
            if ratio > 1.3:
                return +2, f"🏦 {self.coin} elevated inflow ({inflow_fmt})"
            if ratio < -2.0:
                return -4, f"⚠️ {self.coin} heavy outflow ({outflow_fmt}) — accumulation opposes short"
            if ratio < -1.3:
                return -2, f"⚠️ {self.coin} outflow above avg"

        return 0, ""


@dataclass
class BasisSnapshot:
    """Perp vs spot price premium for a coin."""
    coin:         str
    perp_price:   float
    spot_price:   float
    basis_pct:    float    # (perp - spot) / spot * 100
    fetched_at:   float = field(default_factory=time.time)

    @property
    def is_stale(self) -> bool:
        return time.time() - self.fetched_at > 300  # 5 min

    def confidence_delta(self, direction: str) -> Tuple[int, str]:
        b = self.basis_pct

        if direction == "LONG":
            if b > 0.5:
                return -6, f"⚠️ Perp premium {b:+.2f}% — overleveraged longs"
            if b > 0.2:
                return -3, f"⚠️ Perp at {b:+.2f}% premium"
            if b < -0.3:
                return +3, f"📊 Perp discount {b:.2f}% — shorts overcrowded"
        else:  # SHORT
            if b < -0.5:
                return -6, f"⚠️ Perp discount {b:.2f}% — overleveraged shorts"
            if b < -0.2:
                return -3, f"⚠️ Perp at {b:.2f}% discount"
            if b > 0.3:
                return +3, f"📊 Perp premium {b:+.2f}% — longs crowded, short favored"

        return 0, ""


# ── Main Analyzer ─────────────────────────────────────────────────────────────

class MarketMicrostructure:
    """
    Manages CVD, options, netflow, and basis data.
    All data is optional — engine degrades gracefully if any source fails.
    """

    def __init__(self):
        self._cvd:      Dict[str, CVDSnapshot]     = {}
        self._options:  Dict[str, OptionsSnapshot] = {}
        self._netflow:  Dict[str, NetflowSnapshot] = {}
        self._basis:    Dict[str, BasisSnapshot]   = {}
        self._session:  Optional[aiohttp.ClientSession] = None
        self._running   = False
        self._task:     Optional[asyncio.Task] = None

        # Spot prices cache (fetched alongside perp for basis)
        self._spot_prices: Dict[str, float] = {}

        # Timing
        self._next_cvd      = 0.0
        self._next_basis    = 0.0
        self._next_options  = 0.0
        self._next_netflow  = 0.0

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._main_loop())
            logger.info(
                "🔬 MarketMicrostructure started "
                "(CVD · Deribit MaxPain · Netflow · Basis — all free)"
            )

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Main public interface ──────────────────────────────────────────────────

    def get_signal_intel(
        self, symbol: str, direction: str, entry_price: float = 0.0
    ) -> tuple:
        """
        Main interface for engine. Returns (confidence_delta, note).
        Aggregates CVD + Basis signals for the given coin.
        Engine expects: delta, note = microstructure.get_signal_intel(symbol, direction, entry)
        """
        coin = symbol.replace("/USDT", "").replace("/BUSD", "").upper()
        total_delta = 0
        notes = []

        # CVD signal for this coin
        cvd = self._cvd.get(coin)
        if cvd and not cvd.is_stale:
            d, n = cvd.confidence_delta(direction)
            total_delta += d
            if n:
                notes.append(n)

        # Basis/crowding signal for this coin
        basis = self._basis.get(coin)
        if basis and not basis.is_stale:
            d, n = basis.confidence_delta(direction)
            total_delta += d
            if n:
                notes.append(n)

        # Netflow signal (BTC/ETH only — applies market-wide)
        btc_coin = "BTC" if coin in ("BTC", "ETH") else None
        if btc_coin:
            nf = self._netflow.get(btc_coin)
            if nf and not nf.is_stale:
                d, n = nf.confidence_delta(direction)
                total_delta += d
                if n:
                    notes.append(n)

        # Cap total delta
        total_delta = max(-15, min(12, total_delta))
        note = " · ".join(notes) if notes else ""
        return total_delta, note

    def get_dashboard_data(self) -> dict:
        """Returns all microstructure data for dashboard display."""
        return {
            "cvd":     [_cvd_to_dict(c) for c in self._cvd.values()
                        if not c.is_stale],
            "basis":   [_basis_to_dict(b) for b in self._basis.values()
                        if not b.is_stale],
            "options": [_opts_to_dict(o) for o in self._options.values()
                        if not o.is_stale],
            "netflow": [_nf_to_dict(n) for n in self._netflow.values()
                        if not n.is_stale],
        }

    # ── Session ────────────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "TitanBot/3.0"},
            )
        return self._session

    async def _get(self, url: str, params: dict = None) -> Optional[any]:
        try:
            s = await self._get_session()
            async with s.get(url, params=params) as r:
                if r.status == 200:
                    return await r.json()
        except Exception:
            pass
        return None

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def _main_loop(self):
        await asyncio.sleep(15)  # let bot settle

        while self._running:
            try:
                now = time.time()

                if now >= self._next_basis:
                    await self._refresh_basis()
                    self._next_basis = now + _BASIS_REFRESH

                if now >= self._next_cvd:
                    await self._refresh_cvd()
                    self._next_cvd = now + _CVD_REFRESH

                if now >= self._next_options:
                    await self._refresh_options()
                    self._next_options = now + _OPTIONS_REFRESH

                if now >= self._next_netflow:
                    await self._refresh_netflow()
                    self._next_netflow = now + _NETFLOW_REFRESH

                await asyncio.sleep(15)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"MarketMicrostructure loop error: {e}")
                await asyncio.sleep(60)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. CVD — Cumulative Volume Delta
    # ══════════════════════════════════════════════════════════════════════════

    async def _refresh_cvd(self):
        """
        Fetch recent aggTrades from Binance futures for active coins.
        We track the top 30 coins by OI (which we already have from liq_analyzer).
        """
        # Get top coins to track
        try:
            from analyzers.liquidation_analyzer import liquidation_analyzer
            all_ci = liquidation_analyzer.get_all()
            top_coins = sorted(
                all_ci.keys(),
                key=lambda c: all_ci[c].total_oi_usd,
                reverse=True
            )[:30]
        except Exception:
            # Fallback: major coins
            top_coins = [
                "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
                "ARB", "OP", "SUI", "APT", "LINK", "DOT", "MATIC",
            ]

        for coin in top_coins:
            try:
                await self._fetch_cvd_for_coin(coin)
                await asyncio.sleep(0.1)  # polite
            except Exception as e:
                logger.debug(f"CVD error {coin}: {e}")

        logger.debug(f"🔬 CVD refreshed for {len(top_coins)} coins")

    async def _fetch_cvd_for_coin(self, coin: str):
        """
        Fetch last 500 aggTrades for coin, compute CVD.
        aggTrade has: m=true means buyer is market maker (sell aggressor),
                      m=false means buyer is market taker (buy aggressor).
        """
        symbol = f"{coin}USDT"
        data   = await self._get(
            f"{_BINANCE_FAPI}/fapi/v1/aggTrades",
            params={"symbol": symbol, "limit": 500}
        )
        if not data or not isinstance(data, list):
            return

        buy_vol  = 0.0
        sell_vol = 0.0
        cutoff   = time.time() * 1000 - _CVD_WINDOW_SECS * 1000  # 5 min ago

        for trade in data:
            try:
                ts  = int(trade.get("T", 0))
                if ts < cutoff:
                    continue
                qty = float(trade.get("q", 0))
                px  = float(trade.get("p", 0))
                usd = qty * px
                # m=True: the buyer was the market maker → seller is aggressor
                if trade.get("m", False):
                    sell_vol += usd
                else:
                    buy_vol  += usd
            except (ValueError, TypeError):
                pass

        total = buy_vol + sell_vol
        if total < 1000:
            return  # skip low-volume coins

        cvd     = buy_vol - sell_vol
        cvd_pct = cvd / total if total > 0 else 0

        self._cvd[coin] = CVDSnapshot(
            coin          = coin,
            cvd_usd       = cvd,
            buy_vol_usd   = buy_vol,
            sell_vol_usd  = sell_vol,
            total_vol_usd = total,
            cvd_pct       = cvd_pct,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Perp/Spot Basis
    # ══════════════════════════════════════════════════════════════════════════

    async def _refresh_basis(self):
        """
        Fetch spot prices for coins we have perp prices for.
        Basis = (perp - spot) / spot * 100
        """
        # Get perp prices from liq_analyzer (already fetched)
        try:
            from analyzers.liquidation_analyzer import liquidation_analyzer
            ci_all = liquidation_analyzer.get_all()
            perp_prices = {
                coin: ci.current_price
                for coin, ci in ci_all.items()
                if ci.current_price > 0
            }
        except Exception:
            return

        if not perp_prices:
            return

        # Fetch spot prices in bulk from Binance spot API
        data = await self._get(f"{_BINANCE_API}/api/v3/ticker/price")
        if not data:
            return

        spot_map: Dict[str, float] = {}
        for item in (data if isinstance(data, list) else []):
            sym = item.get("symbol", "")
            if sym.endswith("USDT"):
                coin = sym[:-4]
                try:
                    spot_map[coin] = float(item.get("price", 0))
                except (ValueError, TypeError):
                    pass

        # Compute basis for each coin
        updated = 0
        for coin, perp_px in perp_prices.items():
            spot_px = spot_map.get(coin, 0)
            if spot_px <= 0 or perp_px <= 0:
                continue

            basis_pct = (perp_px - spot_px) / spot_px * 100

            self._basis[coin] = BasisSnapshot(
                coin        = coin,
                perp_price  = perp_px,
                spot_price  = spot_px,
                basis_pct   = round(basis_pct, 4),
            )
            updated += 1

        logger.debug(f"🔬 Basis updated for {updated} coins")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Deribit Options — Max Pain + Put/Call Ratio
    # ══════════════════════════════════════════════════════════════════════════

    async def _refresh_options(self):
        """Fetch BTC and ETH options data from Deribit."""
        for coin in ["BTC", "ETH"]:
            try:
                await self._fetch_deribit_options(coin)
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.debug(f"Deribit options error {coin}: {e}")

    async def _fetch_deribit_options(self, coin: str):
        """
        Fetch all active options, find next expiry, compute max pain and PCR.

        Max pain = strike price where total option buyer loss is maximised
                 = where the most options (by OI) expire worthless
                 = MMs are incentivised to pin price here at expiry

        PCR = total put OI / total call OI
        """
        # Get all active instruments
        instruments = await self._get(
            f"{_DERIBIT}/get_instruments",
            params={"currency": coin, "kind": "option", "expired": "false"}
        )
        if not instruments:
            return

        instr_list = instruments.get("result", [])
        if not instr_list:
            return

        # Find next weekly/monthly expiry
        now_ts = time.time() * 1000
        expiries = sorted(set(
            i.get("expiration_timestamp", 0)
            for i in instr_list
            if i.get("expiration_timestamp", 0) > now_ts
        ))

        if not expiries:
            return

        next_expiry_ts = expiries[0]
        next_expiry_h  = (next_expiry_ts - now_ts) / 3600 / 1000

        # Filter to next expiry instruments only
        next_instr = [
            i for i in instr_list
            if i.get("expiration_timestamp") == next_expiry_ts
        ]

        # Get book summaries for all next-expiry instruments
        # Deribit allows fetching by currency
        summaries = await self._get(
            f"{_DERIBIT}/get_book_summary_by_currency",
            params={"currency": coin, "kind": "option"}
        )
        if not summaries:
            return

        summary_map: Dict[str, dict] = {
            s.get("instrument_name", ""): s
            for s in summaries.get("result", [])
        }

        # Build strike → {call_oi, put_oi} for next expiry
        strike_data: Dict[float, dict] = {}
        total_put_oi  = 0.0
        total_call_oi = 0.0

        for instr in next_instr:
            name         = instr.get("instrument_name", "")
            strike       = float(instr.get("strike", 0))
            option_type  = instr.get("option_type", "")  # "call" or "put"
            summary      = summary_map.get(name, {})
            oi           = float(summary.get("open_interest", 0) or 0)

            if strike <= 0:
                continue

            if strike not in strike_data:
                strike_data[strike] = {"call_oi": 0.0, "put_oi": 0.0}

            if option_type == "call":
                strike_data[strike]["call_oi"] += oi
                total_call_oi += oi
            elif option_type == "put":
                strike_data[strike]["put_oi"] += oi
                total_put_oi  += oi

        if not strike_data:
            return

        # Get current underlying price
        index_data = await self._get(
            f"{_DERIBIT}/get_index_price",
            params={"index_name": f"{coin.lower()}_usd"}
        )
        current_price = 0.0
        if index_data:
            current_price = float(
                index_data.get("result", {}).get("index_price", 0) or 0
            )

        if current_price <= 0:
            return

        # Compute max pain at each strike
        # Pain for call holders at strike S, given expiry at P:
        #   If P > S: call buyer gains (P-S)*oi, so call writer loses (P-S)*oi
        #   If P <= S: call is worthless
        # Pain for put holders at strike S, given expiry at P:
        #   If P < S: put buyer gains (S-P)*oi, so put writer loses (S-P)*oi
        #   If P >= S: put is worthless
        # Max pain = strike where total writer pain (buyer gain) is minimized
        # i.e., where buyers lose the most

        max_pain_strike = current_price
        min_buyer_profit = float("inf")
        strikes = sorted(strike_data.keys())

        for test_price in strikes:
            buyer_profit = 0.0
            for strike, data in strike_data.items():
                # Call buyers profit if test_price > strike
                if test_price > strike:
                    buyer_profit += (test_price - strike) * data["call_oi"]
                # Put buyers profit if test_price < strike
                if test_price < strike:
                    buyer_profit += (strike - test_price) * data["put_oi"]

            if buyer_profit < min_buyer_profit:
                min_buyer_profit  = buyer_profit
                max_pain_strike   = test_price

        pcr = total_put_oi / max(total_call_oi, 1)
        total_oi_usd = (total_put_oi + total_call_oi) * current_price

        self._options[coin] = OptionsSnapshot(
            coin           = coin,
            max_pain       = max_pain_strike,
            put_call_ratio = round(pcr, 3),
            next_expiry_h  = next_expiry_h,
            total_oi_usd   = total_oi_usd,
        )

        logger.info(
            f"🔬 {coin} options: max_pain=${max_pain_strike:,.0f} "
            f"PCR={pcr:.2f} expiry={next_expiry_h:.1f}h "
            f"OI={_fmt_usd(total_oi_usd)}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Exchange Netflow
    # ══════════════════════════════════════════════════════════════════════════

    async def _refresh_netflow(self):
        """
        Fetch BTC + ETH exchange netflow.
        Primary: CryptoQuant free API (no key needed for basic metrics).
        Fallback: estimate from Binance large-order flow proxy.
        """
        for coin in ["BTC", "ETH"]:
            try:
                snap = await self._fetch_cryptoquant_netflow(coin)
                if snap:
                    self._netflow[coin] = snap
                    logger.info(
                        f"🔬 {coin} netflow: {_fmt_usd(snap.netflow_24h)} "
                        f"(in={_fmt_usd(snap.inflow_24h)} "
                        f"out={_fmt_usd(snap.outflow_24h)}) "
                        f"[{snap.source}]"
                    )
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.debug(f"Netflow error {coin}: {e}")

    async def _fetch_cryptoquant_netflow(
        self, coin: str
    ) -> Optional[NetflowSnapshot]:
        """
        Exchange netflow estimation.
        Primary: Binance large taker buy/sell imbalance as proxy.
        Fallback: CryptoQuant if accessible, then funding-rate estimate.
        """
        # Primary: Use Binance aggTrades large-order imbalance as netflow proxy
        # Large sellers = coins moving to exchange; large buyers = accumulation
        try:
            symbol = f"{coin}USDT"
            data = await self._get(
                f"{_BINANCE_FAPI}/fapi/v1/aggTrades",
                params={"symbol": symbol, "limit": 1000}
            )
            if data and isinstance(data, list):
                cutoff = time.time() * 1000 - 3600 * 1000  # last hour
                large_buy = 0.0
                large_sell = 0.0
                threshold = 100_000  # $100k+ = large order

                for trade in data:
                    try:
                        ts  = int(trade.get("T", 0))
                        if ts < cutoff:
                            continue
                        qty = float(trade.get("q", 0))
                        px  = float(trade.get("p", 0))
                        usd = qty * px
                        if usd < threshold:
                            continue
                        if trade.get("m", False):
                            large_sell += usd  # seller aggressor
                        else:
                            large_buy  += usd  # buyer aggressor
                    except (ValueError, TypeError):
                        pass

                if large_buy + large_sell > 0:
                    # Net: positive = more large selling (bearish/inflow signal)
                    #      negative = more large buying (bullish/outflow signal)
                    netflow = large_sell - large_buy
                    # AUDIT FIX: this path previously stored
                    # ``ma7 = (large_sell + large_buy) / 2`` and labelled it a
                    # "7-day moving average".  It is in fact the current-day
                    # mean of sell/buy volumes, which bounds the downstream
                    # ``netflow / ma7`` ratio to [-2, +2] — making the
                    # ``> 2.0`` / ``< -2.0`` "2× historical" thresholds in
                    # ``confidence_delta`` numerically unreachable from this
                    # source.  Emit ``ma7_netflow=0`` so ``confidence_delta``
                    # early-returns with zero adjustment until a real 7-day
                    # baseline is available (e.g. via the Glassnode path,
                    # which still populates a proper MA).
                    return NetflowSnapshot(
                        coin        = coin,
                        netflow_24h = netflow,
                        inflow_24h  = large_sell,
                        outflow_24h = large_buy,
                        ma7_netflow = 0.0,
                        source      = "binance_flow",
                    )
        except Exception as e:
            logger.debug(f"Binance flow proxy error {coin}: {e}")

        # Fallback: funding rate estimate
        return await self._estimate_netflow_from_binance(coin)

    async def _fetch_glassnode_netflow(
        self, coin: str
    ) -> Optional[NetflowSnapshot]:
        """
        Glassnode free API — exchange net position change.
        No API key needed for daily resolution.
        """
        gl_asset = "btc" if coin == "BTC" else "eth"
        # Exchange net flow (inflow - outflow) from Glassnode
        data = await self._get(
            f"https://api.glassnode.com/v1/metrics/transactions/transfers_volume_exchanges_net",
            params={"a": gl_asset, "i": "24h", "f": "JSON", "timestamp_format": "humanized"}
        )

        if data and isinstance(data, list) and data:
            try:
                # Most recent entry
                latest = data[-1]
                netflow_btc = float(latest.get("v", 0) or 0)

                # Get BTC price for USD conversion
                btc_price = 83000  # fallback
                try:
                    from analyzers.liquidation_analyzer import liquidation_analyzer
                    ci = liquidation_analyzer.get(coin)
                    if ci and ci.current_price > 0:
                        btc_price = ci.current_price
                except Exception:
                    pass

                netflow_usd = netflow_btc * btc_price

                # Get some history for MA
                hist_data  = data[-8:] if len(data) >= 8 else data
                values     = [float(r.get("v", 0) or 0) for r in hist_data[:-1]]
                ma7_flow   = (sum(values) / len(values) * btc_price) if values else 0

                return NetflowSnapshot(
                    coin        = coin,
                    netflow_24h = netflow_usd,
                    inflow_24h  = max(netflow_usd, 0),
                    outflow_24h = max(-netflow_usd, 0),
                    ma7_netflow = ma7_flow,
                    source      = "glassnode",
                )
            except Exception as e:
                logger.debug(f"Glassnode parse error {coin}: {e}")

        return None

    async def _estimate_netflow_from_binance(
        self, coin: str
    ) -> Optional[NetflowSnapshot]:
        """
        Rough netflow estimate from Binance large trade direction.
        Very approximate but better than nothing.
        Uses the funding rate as a proxy: high funding = longs crowded = inflow.
        """
        try:
            from analyzers.liquidation_analyzer import liquidation_analyzer
            ci = liquidation_analyzer.get(coin)
            if not ci:
                return None

            # Crude proxy: positive funding → market buying perps → inflow likely
            # Negative funding → market selling → outflow likely
            funding = ci.funding_rate
            price   = ci.current_price or 1

            # Estimate: 0.01% funding ≈ $100M net flow imbalance (very rough)
            estimated_flow = funding * 1e13  # scale to USD
            ma7_estimate   = 0.0  # no history for this estimate

            return NetflowSnapshot(
                coin        = coin,
                netflow_24h = estimated_flow,
                inflow_24h  = max(estimated_flow, 0),
                outflow_24h = max(-estimated_flow, 0),
                ma7_netflow = ma7_estimate,
                source      = "estimated",
            )
        except Exception:
            return None


# ── Dashboard serializers ──────────────────────────────────────────────────────

def _cvd_to_dict(c: CVDSnapshot) -> dict:
    buy_pct  = round(c.buy_vol_usd  / max(c.total_vol_usd, 1) * 100, 1)
    sell_pct = round(c.sell_vol_usd / max(c.total_vol_usd, 1) * 100, 1)
    return {
        "coin":          c.coin,
        "cvd_usd":       c.cvd_usd,
        "cvd_fmt":       _fmt_usd(abs(c.cvd_usd)),
        "cvd_direction": "BUY" if c.cvd_usd > 0 else "SELL",
        "cvd_pct":       round(c.cvd_pct * 100, 1),
        "buy_pct":       buy_pct,
        "sell_pct":      sell_pct,
        "signal":        c.signal,
        "vol_fmt":       _fmt_usd(c.total_vol_usd),
    }

def _basis_to_dict(b: BasisSnapshot) -> dict:
    return {
        "coin":      b.coin,
        "basis_pct": b.basis_pct,
        "perp":      b.perp_price,
        "spot":      b.spot_price,
        "signal":    ("CROWDED_LONG" if b.basis_pct > 0.3
                      else "CROWDED_SHORT" if b.basis_pct < -0.3
                      else "NEUTRAL"),
    }

def _opts_to_dict(o: OptionsSnapshot) -> dict:
    return {
        "coin":          o.coin,
        "max_pain":      o.max_pain,
        "pcr":           o.put_call_ratio,
        "expiry_h":      round(o.next_expiry_h, 1),
        "oi_fmt":        _fmt_usd(o.total_oi_usd),
        "pcr_signal":    ("BEARISH" if o.put_call_ratio > 1.3
                          else "BULLISH" if o.put_call_ratio < 0.7
                          else "NEUTRAL"),
    }

def _nf_to_dict(n: NetflowSnapshot) -> dict:
    return {
        "coin":         n.coin,
        "netflow_fmt":  _fmt_usd(abs(n.netflow_24h)),
        "direction":    "INFLOW" if n.netflow_24h > 0 else "OUTFLOW",
        "signal":       ("BEARISH" if n.netflow_24h > abs(n.ma7_netflow) * 1.3
                         else "BULLISH" if n.netflow_24h < -abs(n.ma7_netflow) * 1.3
                         else "NEUTRAL"),
        "source":       n.source,
    }


def _fmt_usd(v: float) -> str:
    v = abs(v)
    if v >= 1e9:  return f"${v/1e9:.2f}B"
    if v >= 1e6:  return f"${v/1e6:.1f}M"
    if v >= 1e3:  return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


# ── Feature 9: Pump/Dump Detection ────────────────────────────────────────────

@dataclass
class PumpDumpAlert:
    """Detected pump/dump event."""
    symbol: str
    price_change_pct: float
    volume_change_pct: float
    risk_level: str       # LOW, MEDIUM, HIGH, VERY_HIGH
    direction: str        # PUMP or DUMP
    timestamp: float
    is_no_trade: bool     # True if risk is HIGH or VERY_HIGH


def detect_pump_dump(
    symbol: str,
    price_change_pct: float,
    volume_change_pct: float,
    window_minutes: int = 15,
    has_correlated_news: bool = False,
) -> Optional[PumpDumpAlert]:
    """
    Detect pump/dump events based on price and volume spike rules.

    Rules:
      - price_change > 10% AND volume_change > 300% in 15min → HIGH RISK
      - price_change > 5% AND volume_change > 200% → MEDIUM RISK
      - price_change > 15% AND volume_change > 500% → VERY_HIGH RISK

    Nuance: If has_correlated_news=True (a news event within the news-exempt
    window explains the move), downgrade the risk tier by one level.
    This prevents blocking legitimate breakouts caused by major news.

    During HIGH or VERY_HIGH risk, signals enter a no-trade zone.
    This is a no-trade zone, NOT a signal generator.

    Returns:
        PumpDumpAlert if detected, None otherwise.

    Fallback: if feature flag is off, returns None.
    """
    if not ff.is_active("PUMP_DUMP_DETECTION"):
        return None

    abs_price = abs(price_change_pct)
    abs_volume = abs(volume_change_pct)

    risk_level = "LOW"
    is_no_trade = False

    if abs_price >= 15.0 and abs_volume >= 500.0:
        risk_level = "VERY_HIGH"
        is_no_trade = True
    elif abs_price >= NewsIntelligence.PUMP_DUMP_PRICE_CHANGE_THRESHOLD and \
         abs_volume >= NewsIntelligence.PUMP_DUMP_VOLUME_CHANGE_THRESHOLD:
        risk_level = "HIGH"
        is_no_trade = True
    elif abs_price >= 4.0 and abs_volume >= 150.0:
        risk_level = "MEDIUM"
    else:
        # No pump/dump detected
        from config.fcn_logger import fcn_log
        _mode = "shadow" if ff.is_shadow("PUMP_DUMP_DETECTION") else "live"
        fcn_log("PUMP_DUMP_DETECTION", f"{_mode} | {symbol} price={price_change_pct:+.1f}% vol={volume_change_pct:+.0f}% — no alert")
        return None

    # Nuance: if a correlated news event explains the move, downgrade risk
    # tier by one. A news-driven breakout (e.g. ETF approval, Fed decision)
    # is fundamentally different from a manipulative pump/dump.
    if has_correlated_news:
        _original_risk = risk_level
        _tier_map = {"VERY_HIGH": "HIGH", "HIGH": "MEDIUM", "MEDIUM": "LOW"}
        risk_level = _tier_map.get(risk_level, risk_level)
        is_no_trade = risk_level in ("HIGH", "VERY_HIGH")
        if _original_risk != risk_level:
            logger.info(
                f"📰 Pump/dump risk downgraded {_original_risk}→{risk_level} "
                f"for {symbol} due to correlated news event"
            )

    direction = "PUMP" if price_change_pct > 0 else "DUMP"

    alert = PumpDumpAlert(
        symbol=symbol,
        price_change_pct=price_change_pct,
        volume_change_pct=volume_change_pct,
        risk_level=risk_level,
        direction=direction,
        timestamp=time.time(),
        is_no_trade=is_no_trade,
    )

    logger.warning(
        f"🚨 PUMP/DUMP {direction}: {symbol} price={price_change_pct:+.1f}% "
        f"volume={volume_change_pct:+.0f}% risk={risk_level} "
        f"no_trade={is_no_trade} news_correlated={has_correlated_news}"
    )

    if ff.is_shadow("PUMP_DUMP_DETECTION"):
        shadow_log("PUMP_DUMP_DETECTION", {
            "symbol": symbol,
            "direction": direction,
            "price_change_pct": price_change_pct,
            "volume_change_pct": volume_change_pct,
            "risk_level": risk_level,
            "is_no_trade": is_no_trade,
            "has_correlated_news": has_correlated_news,
        })

    from config.fcn_logger import fcn_log
    import logging as _logging
    _mode = "shadow" if ff.is_shadow("PUMP_DUMP_DETECTION") else "live"
    fcn_log("PUMP_DUMP_DETECTION",
            f"{_mode} | {symbol} {direction} price={price_change_pct:+.1f}% vol={volume_change_pct:+.0f}% "
            f"risk={risk_level} no_trade={is_no_trade} news_corr={has_correlated_news}",
            level=_logging.WARNING)

    return alert


# ── Singleton ──────────────────────────────────────────────────────────────────
microstructure = MarketMicrostructure()
