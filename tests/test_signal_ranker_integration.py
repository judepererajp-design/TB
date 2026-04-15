from types import SimpleNamespace
from unittest.mock import MagicMock

from signals.signal_ranker import rank_publish_candidates
from strategies.base import SignalDirection


def _candidate(symbol: str, direction: str, confidence: float, rr: float, volume_score: float):
    signal = MagicMock()
    signal.direction = SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT
    signal.strategy = "Momentum"

    base_signal = SimpleNamespace(
        symbol=symbol,
        direction=signal.direction,
        rr_ratio=rr,
    )
    scored = SimpleNamespace(
        base_signal=base_signal,
        final_confidence=confidence,
        volume_score=volume_score,
    )

    return SimpleNamespace(
        symbol=symbol,
        signal=signal,
        scored=scored,
        sig_data={},
        alpha_score=SimpleNamespace(grade="A"),
        prob_estimate=None,
        sizing=None,
        confluence=None,
        regime_name="BULL_TREND",
    )


def test_rank_publish_candidates_orders_and_limits_batch():
    candidates = [
        _candidate("AVAXUSDT", "LONG", 60.0, 1.5, 40.0),
        _candidate("BTCUSDT", "LONG", 92.0, 3.0, 80.0),
        _candidate("DOGEUSDT", "LONG", 75.0, 2.0, 60.0),
        _candidate("XRPUSDT", "LONG", 55.0, 1.2, 35.0),
    ]

    ranked, skipped = rank_publish_candidates(candidates, regime="BULL_TREND")

    assert [candidate.symbol for candidate in ranked] == [
        "BTCUSDT",
        "DOGEUSDT",
        "AVAXUSDT",
    ]
    assert [candidate.symbol for candidate in skipped] == ["XRPUSDT"]


def test_rank_publish_candidates_applies_correlation_filter():
    candidates = [
        _candidate("BTCUSDT", "LONG", 90.0, 3.0, 80.0),
        _candidate("ETHUSDT", "LONG", 85.0, 2.5, 70.0),
        _candidate("DOGEUSDT", "LONG", 78.0, 2.0, 65.0),
    ]

    ranked, skipped = rank_publish_candidates(candidates, regime="BULL_TREND")

    assert [candidate.symbol for candidate in ranked] == ["BTCUSDT", "DOGEUSDT"]
    assert [candidate.symbol for candidate in skipped] == ["ETHUSDT"]
