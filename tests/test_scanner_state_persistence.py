from unittest.mock import AsyncMock

import pytest

from data.database import db
from scanner.scanner import Scanner, SymbolState, Tier


@pytest.mark.asyncio
async def test_check_whale_activity_resets_missing_snapshot_sides_and_persists(monkeypatch):
    scanner = Scanner()
    scanner._symbols["BTC/USDT"] = SymbolState(
        symbol="BTC/USDT",
        tier=Tier.TIER1,
        volume_24h=0.0,
    )
    scanner._whale_snapshot["BTC/USDT"] = {
        "buy": 1_500_000.0,
        "sell": 900_000.0,
        "iceberg_buy": 1_400_000.0,
        "iceberg_sell": 850_000.0,
    }
    db.save_learning_state = AsyncMock()

    monkeypatch.setattr("scanner.scanner.time.time", lambda: 1_000.0)

    result = await scanner.check_whale_activity(
        "BTC/USDT",
        {"bids": [[100.0, 20_000.0]], "asks": []},
    )

    assert result is not None
    assert scanner._whale_snapshot["BTC/USDT"]["sell"] == 0.0
    assert scanner._whale_snapshot["BTC/USDT"]["iceberg_sell"] == 0.0
    db.save_learning_state.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_promotions_persists_temp_exclusion_state(monkeypatch):
    scanner = Scanner()
    scanner._symbols["ETH/USDT"] = SymbolState(
        symbol="ETH/USDT",
        tier=Tier.TIER3,
        volume_24h=100_000.0,
        last_scan=900.0,
        tier3_underfloor_since=0.0,
    )
    db.save_learning_state = AsyncMock()

    monkeypatch.setattr("scanner.scanner.time.time", lambda: 1_000.0)

    state = scanner._symbols["ETH/USDT"]
    state.tier3_underfloor_since = 1_000.0 - (181 * 60.0)

    result = await scanner.check_promotions("ETH/USDT", current_volume=100_000.0)

    assert result is None
    assert scanner.is_temporarily_excluded("ETH/USDT") is True
    db.save_learning_state.assert_awaited_once()
