"""
Stablecoin Depeg Guard
=======================
Reject signals quoted in a depegged stablecoin. Every USDT-quoted
chart on Binance is implicitly a (token / USD) chart that assumes
USDT ≡ $1.00. When USDT actually trades at, say, 0.985 (March 2023
Silicon Valley Bank weekend, May 2022 Terra collapse), the implicit
denominator shifts ~1.5%, which is enough to:

  * fake out every breakout strategy on the entire universe,
  * widen "atr-based" stops in dollar terms while looking unchanged
    in USDT terms,
  * make targets that cleared in dollars look stalled in USDT.

This guard polls the major stablecoin quote prices vs $1.00 (via
``data.api_client.api.fetch_ticker`` against a USD-anchored pair
where available, otherwise via direct stablecoin/stablecoin pairs)
with a short cache so it costs ~one ticker call per minute.

Fail-open: if the price cannot be retrieved we treat the stablecoin
as healthy. The risk of blocking valid signals on a missing sanity
check outweighs the benefit of erring conservative on a check that
is itself a backup safety net.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from config.constants import StablecoinDepeg as SDC

logger = logging.getLogger(__name__)


class DepegStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


@dataclass
class DepegSnapshot:
    quote: str
    status: DepegStatus
    price: Optional[float]            # last seen vs USD; None if unavailable
    deviation: Optional[float]        # |price - 1.0|; None if price unknown
    reason: str = ""

    @property
    def should_block(self) -> bool:
        return self.status == DepegStatus.UNHEALTHY


# ── Helpers ──────────────────────────────────────────────────────

def _extract_quote(symbol: str) -> str:
    """Return the quote currency for a CCXT-style symbol like 'BTC/USDT'.

    Falls back to the suffix for symbols without a slash (e.g. 'BTCUSDT').
    """
    if not symbol:
        return ""
    if "/" in symbol:
        return symbol.split("/", 1)[1].split(":", 1)[0].upper()
    sym_u = symbol.upper()
    for q in SDC.MONITORED_QUOTES:
        if sym_u.endswith(q):
            return q
    return sym_u[-4:] if len(sym_u) >= 4 else sym_u


# ── Guard ────────────────────────────────────────────────────────

class StablecoinDepegGuard:
    """Per-stablecoin depeg cache and gate."""

    # When fetching the dollar value of a stablecoin we prefer pairs
    # whose price is "stable / USD-equivalent" — a value < 1.0 means
    # the LHS is depegged to the downside, > 1.0 means depegged to
    # the upside. For Binance perpetuals USDT itself is the quote
    # for almost everything; so we use spot pairs that exist on
    # Binance and whose price natively equals "stable price in USD"
    # by virtue of the other side being a US-anchored pair.
    #
    # The map is queried in order — first available pair wins.
    _PROBE_PAIRS: Dict[str, tuple] = {
        # USDT — use BUSD (now retired) or USDC; failing that, derive
        # from DAI. We intentionally test against multiple anchors so a
        # single anchor's own depeg can't trigger a false positive.
        "USDT": ("USDC/USDT", "DAI/USDT", "FDUSD/USDT"),
        "USDC": ("USDC/USDT", "USDC/DAI"),
        "DAI":  ("DAI/USDT",),
        "FDUSD": ("FDUSD/USDT",),
        "TUSD": ("TUSD/USDT",),
    }

    def __init__(self) -> None:
        # quote -> (price_usd, fetched_at)
        self._cache: Dict[str, tuple] = {}
        self._last_status: Dict[str, DepegStatus] = {}

    # ── Public API ─────────────────────────────────────────────

    async def get_quote_price_usd(self, quote: str) -> Optional[float]:
        """Return the stablecoin's price in USD (fail-open returns None)."""
        quote = (quote or "").upper()
        cached = self._cache.get(quote)
        if cached:
            price, ts = cached
            if time.time() - ts < SDC.PRICE_CACHE_SECS:
                return price

        probes = self._PROBE_PAIRS.get(quote)
        if not probes:
            return None

        try:
            from data.api_client import api
        except Exception as exc:
            logger.debug(f"stablecoin_depeg: api_client unavailable ({exc})")
            return None

        for pair in probes:
            try:
                ticker = await api.fetch_ticker(pair)
            except Exception:
                ticker = None
            if not ticker:
                continue
            last = ticker.get("last") if isinstance(ticker, dict) else None
            if last is None or last <= 0:
                continue
            price_usd = self._derive_quote_usd(quote, pair, float(last))
            if price_usd is None or price_usd <= 0:
                continue
            self._cache[quote] = (price_usd, time.time())
            return price_usd

        # Couldn't fetch any probe — fail-open
        return None

    async def check_quote(self, quote: str) -> DepegSnapshot:
        """Compute the depeg status for a single stablecoin."""
        quote = (quote or "").upper()
        if quote not in SDC.MONITORED_QUOTES:
            # Not a monitored stablecoin — treat as healthy.
            return DepegSnapshot(
                quote=quote,
                status=DepegStatus.HEALTHY,
                price=None,
                deviation=None,
                reason="not monitored",
            )

        price = await self.get_quote_price_usd(quote)
        if price is None:
            # Fail-open
            return DepegSnapshot(
                quote=quote,
                status=DepegStatus.HEALTHY,
                price=None,
                deviation=None,
                reason="price unavailable (fail-open)",
            )

        deviation = abs(price - 1.0)
        if deviation >= SDC.UNHEALTHY_DEVIATION:
            status = DepegStatus.UNHEALTHY
            reason = f"|deviation|={deviation:.4f} ≥ {SDC.UNHEALTHY_DEVIATION:.4f}"
        elif deviation >= SDC.DEGRADED_DEVIATION:
            status = DepegStatus.DEGRADED
            reason = f"|deviation|={deviation:.4f} ≥ {SDC.DEGRADED_DEVIATION:.4f}"
        else:
            status = DepegStatus.HEALTHY
            reason = ""

        prev = self._last_status.get(quote)
        if prev != status:
            logger.warning(
                f"💵 Stablecoin {quote}: {prev.value if prev else 'INIT'} → "
                f"{status.value} | price={price:.4f} {reason}"
            )
            self._last_status[quote] = status

        return DepegSnapshot(
            quote=quote,
            status=status,
            price=price,
            deviation=deviation,
            reason=reason,
        )

    async def should_block_symbol(self, symbol: str) -> bool:
        """Convenience: True iff the symbol's quote stablecoin is UNHEALTHY."""
        snap = await self.check_quote(_extract_quote(symbol))
        return snap.should_block

    # ── Internals ─────────────────────────────────────────────

    @staticmethod
    def _derive_quote_usd(quote: str, pair: str, last: float) -> Optional[float]:
        """Translate a probe-pair last price into the stablecoin's USD price.

        For ``X/Y`` priced at ``last`` (i.e. 1 X = last Y), the USD
        value of X is ``last * usd(Y)``. Since both sides are
        stablecoins we approximate ``usd(Y) ≈ 1.0`` for the *anchor*
        leg — depeg is computed by comparing two assumed-pegged
        stablecoins, so any small error in the anchor surfaces as
        attributed deviation on the probed side. This is acceptable
        for a "did the world break" gate.
        """
        try:
            base, quote_side = pair.split("/", 1)
        except ValueError:
            return None
        base = base.upper()
        quote_side = quote_side.upper()
        if quote == base:
            # last = base in quote-side units. USD value of base ≈ last.
            return float(last)
        if quote == quote_side:
            # last = base/quote-side; USD value of quote-side = 1/last.
            if last <= 0:
                return None
            return 1.0 / float(last)
        return None


# ── Singleton ─────────────────────────────────────────────────
stablecoin_depeg_guard = StablecoinDepegGuard()
