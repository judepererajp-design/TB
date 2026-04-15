import pytest

from tg.search import TelegramSearchService


class FakeDb:
    async def get_signal(self, signal_id):
        if signal_id == 42:
            return {
                "id": 42,
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "alpha_grade": "A",
                "outcome": "WIN",
            }
        return None

    async def get_recent_signals(self, hours=0, symbol=None, exclude_c_grade=False):
        if symbol == "BTC/USDT":
            return [
                {
                    "id": 42,
                    "symbol": "BTC/USDT",
                    "direction": "LONG",
                    "alpha_grade": "A",
                    "outcome": "WIN",
                },
                {
                    "id": 77,
                    "symbol": "BTC/USDT",
                    "direction": "SHORT",
                    "alpha_grade": "B+",
                    "outcome": "PENDING",
                },
            ]
        return []

    async def get_recent_signals_all(self, hours=0):
        return [
            {
                "id": 42,
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "strategy": "Wyckoff",
                "alpha_grade": "A",
                "outcome": "WIN",
            },
            {
                "id": 55,
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "strategy": "Wyckoff",
                "alpha_grade": "B",
                "outcome": "PENDING",
            },
        ]

    async def get_signal_notes(self, limit=0):
        return [{"signal_id": 42, "symbol": "BTC/USDT", "notes": "BTC breakout note"}]

    async def get_recent_events(self, limit=0):
        return [{"event_type": "telegram", "message": "telegram timeout recovered"}]


@pytest.mark.asyncio
async def test_universal_search_merges_unique_signals():
    service = TelegramSearchService(database=FakeDb())

    results = await service.search("any", "42")

    assert [sig["id"] for sig in results["signals"]] == [42]
    assert results["notes"][0]["signal_id"] == 42


@pytest.mark.asyncio
async def test_symbol_search_normalizes_symbol():
    service = TelegramSearchService(database=FakeDb())

    results = await service.search("symbol", "btc")

    assert len(results["signals"]) == 2
    assert all(sig["symbol"] == "BTC/USDT" for sig in results["signals"])


@pytest.mark.asyncio
async def test_event_search_returns_matching_health_events():
    service = TelegramSearchService(database=FakeDb())

    results = await service.search("event", "telegram")

    assert results["events"][0]["event_type"] == "telegram"


def test_format_results_includes_signals_notes_and_events():
    service = TelegramSearchService(database=FakeDb())

    text = service.format_results(
        {
            "mode": "any",
            "query": "btc",
            "signals": [{"id": 42, "symbol": "BTC/USDT", "direction": "LONG", "alpha_grade": "A", "outcome": "WIN"}],
            "notes": [{"signal_id": 42, "symbol": "BTC/USDT", "notes": "BTC breakout note"}],
            "events": [{"event_type": "telegram", "message": "telegram timeout recovered"}],
        }
    )

    assert "Universal Search" in text
    assert "#42" in text
    assert "Notes" in text
    assert "Events" in text
